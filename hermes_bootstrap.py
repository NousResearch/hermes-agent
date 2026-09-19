"""Process bootstrap for Hermes entry points: Windows UTF-8 stdio, import-path
hardening, durable lazy-install target, and dual-stack (Happy Eyeballs) connects.

Windows binds stdio to the console code page (cp1252), so ``print("café")`` raises
``UnicodeEncodeError``, and Python children inherit the same default unless
``PYTHONUTF8``/``PYTHONIOENCODING`` are set. Import this module first in every entry
point (``hermes``, ``hermes-agent``, ``hermes-acp``, ``gateway.run``, ``batch_runner``,
``cron/scheduler``). It does NOT re-exec with ``-X utf8``: ``open()`` in the current
process still needs an explicit ``encoding="utf-8"`` (ruff ``PLW1514``). POSIX is left
alone deliberately — users' ``LANG``/``LC_*`` choices are respected.

Stdlib only: entry points import this before ``harden_import_path()`` runs, so nothing
here may pull in a Hermes package that a project-local directory could shadow.
"""

from __future__ import annotations

import errno
import importlib.abc
import importlib.util
import os
import selectors
import socket
import sys
import time

_IS_WINDOWS = sys.platform == "win32"
_bootstrap_applied = False
_HAPPY_EYEBALLS_DELAY_SECONDS = 0.25
_URLLIB3_CONNECTION_MODULE = "urllib3.util.connection"


def _interleave_addrinfos(addrinfos: list[tuple]) -> list[tuple]:
    """Round-robin the resolved address families (deduped), preserving resolver order within each."""
    queues: dict[int, list[tuple]] = {}
    seen: set[tuple] = set()
    for addrinfo in addrinfos:
        family, socktype, proto, _canonname, sockaddr = addrinfo
        if (family, socktype, proto, sockaddr) not in seen:
            seen.add((family, socktype, proto, sockaddr))
            queues.setdefault(family, []).append(addrinfo)
    interleaved: list[tuple] = []
    while any(queues.values()):
        interleaved.extend(queue.pop(0) for queue in queues.values() if queue)
    return interleaved


def _quiet_unregister(selector, sock) -> None:
    try:
        selector.unregister(sock)
    except Exception:
        pass


def _happy_eyeballs_create_connection(address: tuple[str, int], timeout: float | None,
                                      source_address: tuple[str, int] | None = None, socket_options=()):
    """RFC 8305-style connect: staggered non-blocking attempts across families.

    ``socket.create_connection`` tries addresses serially, so broken-but-
    advertised IPv6 can burn the whole timeout per AAAA record before IPv4.
    """
    host, port = address
    addrinfos = _interleave_addrinfos(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
    if not addrinfos:
        raise OSError(f"getaddrinfo returned no addresses for {host}")

    selector = selectors.DefaultSelector()
    active: set[socket.socket] = set()
    winner = None
    last_error: OSError | None = None
    deadline = None if timeout is None else time.monotonic() + max(timeout, 0.0)
    next_launch = time.monotonic()
    pending = list(addrinfos)
    in_progress = {0, errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY, errno.EINTR, getattr(errno, "WSAEWOULDBLOCK", 10035)}

    def start_attempt(addrinfo):
        family, socktype, proto, _canonname, sockaddr = addrinfo
        candidate = socket.socket(family, socktype, proto)
        try:
            if source_address is not None:
                local_infos = socket.getaddrinfo(source_address[0], source_address[1], family=family, type=socktype)
                if not local_infos:
                    raise OSError(f"getaddrinfo returned no local {family} address for {source_address[0]}")
                candidate.bind(local_infos[0][4])
            candidate.setblocking(False)
            result = candidate.connect_ex(sockaddr)
            if result in (0, errno.EISCONN):
                return candidate
            if result not in in_progress:
                raise OSError(result, os.strerror(result))
            selector.register(candidate, selectors.EVENT_WRITE)
            active.add(candidate)
            return None
        except Exception:
            candidate.close()
            raise

    try:
        while pending or active:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                raise socket.timeout("timed out")
            if pending and now >= next_launch:
                try:
                    winner = start_attempt(pending.pop(0))
                except OSError as exc:
                    last_error = exc
                    if not active:
                        next_launch = now
                    continue
                if winner is not None:
                    break
                next_launch = now + _HAPPY_EYEBALLS_DELAY_SECONDS
            wait_timeout = None if deadline is None else max(0.0, deadline - now)
            if pending:
                until_launch = max(0.0, next_launch - now)
                wait_timeout = until_launch if wait_timeout is None else min(wait_timeout, until_launch)
            for key, _mask in selector.select(wait_timeout):
                candidate = key.fileobj
                error_code = candidate.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                selector.unregister(candidate)
                active.discard(candidate)
                if error_code == 0:
                    winner = candidate
                    break
                candidate.close()
                last_error = OSError(error_code, os.strerror(error_code))
            if winner is not None:
                break
            if not active and pending:
                next_launch = time.monotonic()

        if winner is None:
            raise last_error if last_error is not None else OSError(f"Could not connect to {host}:{port}")
        _quiet_unregister(selector, winner)
        active.discard(winner)
        winner.settimeout(timeout)
        for option in socket_options or ():
            winner.setsockopt(*option)
        winner.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return winner
    finally:
        for candidate in active:
            _quiet_unregister(selector, candidate)
            candidate.close()
        selector.close()


def _patch_urllib3_create_connection(module) -> None:
    """Point ``urllib3.util.connection.create_connection`` (its own serial walker) at the racer."""
    if getattr(module.create_connection, "_hermes_happy_eyeballs", False):
        return
    urllib3_sentinel = module._DEFAULT_TIMEOUT

    def _urllib3_racer(address, timeout=urllib3_sentinel, source_address=None, socket_options=None):
        effective = socket.getdefaulttimeout() if timeout is urllib3_sentinel else timeout
        # OSError = every candidate failed (identical to the serial original); anything else is a
        # racer bug and must surface rather than silently fall back to the serial stall.
        return _happy_eyeballs_create_connection(
            address, effective, source_address=source_address, socket_options=tuple(socket_options or ()))

    _urllib3_racer._hermes_happy_eyeballs = True  # type: ignore[attr-defined]
    module.create_connection = _urllib3_racer


class _Urllib3ConnectionPatcher(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """One-shot import hook: patch urllib3's connect walker the moment the module loads.

    Importing urllib3 eagerly costs ~50 ms on every CLI start, and ``hermes`` / the TUI
    gateway never load it unless something actually calls ``requests``.
    """

    def find_spec(self, fullname, path, target=None):
        if fullname != _URLLIB3_CONNECTION_MODULE:
            return None
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return None
        self._inner = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _patch_urllib3_create_connection(module)


def install_happy_eyeballs_socket_connect() -> None:
    """Race IPv6/IPv4 for every sync TCP connect in the process (RFC 8305, #114265).

    The startup path does not build its HTTP clients in one place: the model catalog
    fetch goes through ``requests``/``urllib3``, sync LLM and OAuth clients through
    httpcore, plugins through ``urllib``/``http.client``. All of them funnel their TCP
    connect into ``socket.create_connection`` (``http.client`` re-reads it per connection;
    httpcore looks it up at call time) or into urllib3's own serial copy in
    ``urllib3.util.connection``. The stock implementations walk the ``getaddrinfo``
    results serially — on a network whose advertised IPv6 route is blackholed, each AAAA
    record burns the full connect timeout before IPv4 answers. Idempotent, best-effort.
    """
    if getattr(socket.create_connection, "_hermes_happy_eyeballs", False):
        return

    def _socket_racer(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, *, all_errors=False):
        # Stock create_connection leaves the sentinel alone, so the socket keeps the
        # process default from socket.setdefaulttimeout(); the racer re-applies the
        # timeout on the winner, so it must resolve the sentinel the same way.
        effective = socket.getdefaulttimeout() if timeout is socket._GLOBAL_DEFAULT_TIMEOUT else timeout
        # OSError = every candidate failed (identical to the serial original); anything else is a
        # racer bug and must surface rather than silently fall back to the serial stall.
        return _happy_eyeballs_create_connection(address, effective, source_address=source_address)

    _socket_racer._hermes_happy_eyeballs = True  # type: ignore[attr-defined]
    socket.create_connection = _socket_racer

    urllib3_connection = sys.modules.get(_URLLIB3_CONNECTION_MODULE)
    if urllib3_connection is not None:
        _patch_urllib3_create_connection(urllib3_connection)
    elif not any(isinstance(finder, _Urllib3ConnectionPatcher) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Urllib3ConnectionPatcher())


def apply_windows_utf8_bootstrap() -> bool:
    """Apply the Windows UTF-8 bootstrap once; True only when it was applied this call."""
    global _bootstrap_applied

    if not _IS_WINDOWS or _bootstrap_applied:
        return False

    # setdefault() so a user can opt out with PYTHONUTF8=0 / PYTHONIOENCODING=...
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # os.environ changes don't rebind streams bound at interpreter startup, so
    # reconfigure them in-process. errors="replace" keeps a non-UTF-8 legacy
    # pipe on stdin from crashing us (U+FFFD instead of an exception).
    # Non-TextIOWrapper streams (BytesIO in tests, embedded hosts) have no
    # reconfigure(): skip — the env-var fix for children is the bigger win.
    for stream_name in ("stdout", "stderr", "stdin"):
        reconfigure = getattr(getattr(sys, stream_name, None), "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass  # closed, or replaced with something non-reconfigurable

    _bootstrap_applied = True
    return True


def suppress_platform_ver_console() -> None:
    """Stub ``platform._syscmd_ver`` on Windows — decode-crash + console-flash guard.

    ``platform.win32_ver()`` (reached via ``platform.platform()``, which the OpenAI SDK
    calls) shells out ``cmd /c ver`` with ``shell=True`` and no ``CREATE_NO_WINDOW``: a
    windowless parent (pythonw gateway, slash/kanban workers) flashes a console per call,
    and Python 3.11.0/3.11.1 (no ``encoding="locale"`` fix) strict-utf-8-decodes the OEM
    code page output under PEP 540 mode and raises (#69413). Returning the inputs makes
    ``win32_ver()`` fall back to ``sys.getwindowsversion()`` — same data, no subprocess.
    Mirrors ``hermes_cli._subprocess_compat.suppress_platform_ver_console`` for callers
    that never import ``hermes_cli.main``; double application is harmless.
    """
    if not _IS_WINDOWS:
        return
    try:
        import platform

        if hasattr(platform, "_syscmd_ver"):
            def _quiet_syscmd_ver(system="", release="", version="",
                                  supported_platforms=("win32", "win16", "dos")):
                return system, release, version

            platform._syscmd_ver = _quiet_syscmd_ver
    except Exception:
        pass  # hardening only — never break an entry point


def harden_import_path(src_root: str | None = None) -> None:
    """Stop a package in the current directory from shadowing Hermes modules.

    Hermes ships top-level modules with common names (``utils``, ``proxy``, ``ui``); a
    project with its own ``utils/`` launched from its directory would win the import.
    The cwd reaches ``sys.path`` as ``""``/``"."`` (script/``-m`` launches) AND as an
    absolute path (venv activation, PYTHONPATH), so both are handled: relative forms are
    dropped and the Hermes root is *relocated* to the front, not merely inserted when
    absent. ``src_root`` defaults to this module's directory (the repo root for every
    shipped entry point), so no spawner env var is required.
    """
    root = src_root or os.environ.get("HERMES_PYTHON_SRC_ROOT") or os.path.dirname(
        os.path.abspath(__file__)
    )

    sys.path[:] = [p for p in sys.path if p not in ("", ".")]

    root_abs = os.path.abspath(root)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) != root_abs]
    sys.path.insert(0, root)


# Apply on import — entry points just need ``import hermes_bootstrap``
# (or ``from hermes_bootstrap import apply_windows_utf8_bootstrap``) at
# the very top of their module, before importing anything else.  The
# import side effect does the right thing.
apply_windows_utf8_bootstrap()
suppress_platform_ver_console()

# Every entry point imports this module before its dependency graph.
from pathlib import Path
from pm.environments import activate_dependencies
from hermes_cli._early_recovery import recover_if_needed

from hermes_cli._parser import command_argv

_root = Path(__file__).resolve().parent
# Repair needs only stdlib. Do not activate the damaged tree to reach it.
_pm_repair = command_argv(sys.argv[1:])[:2] == ["pm", "repair"]
if not _pm_repair:
    from hermes_cli.venv_sync import prepare_launch, relaunch_command

    try:
        _launch_python = prepare_launch(_root, sys.argv[1:])
        if _launch_python is not None:
            _main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
            _command = relaunch_command(
                _launch_python, _root, sys.argv, sys.orig_argv,
                getattr(_main_spec, "name", None),
            )
            if os.name == "nt":
                import subprocess

                raise SystemExit(subprocess.call(_command))
            os.execv(str(_launch_python), _command)
    except Exception as exc:
        print(f"hermes: source-update completion failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    recover_if_needed(_root)
    try:
        activate_dependencies(_root)
    except (RuntimeError, OSError) as exc:
        if command_argv(sys.argv[1:])[:1] != ["pm"]:
            print(f"hermes: {exc}; run `hermes pm repair`", file=sys.stderr)
            raise SystemExit(1) from None
install_happy_eyeballs_socket_connect()

