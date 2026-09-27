"""Process bootstrap for Hermes entry points: Windows UTF-8 stdio and ANSI console, import-path
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


_ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004


def enable_windows_vt(streams=None) -> bool:
    """Opt the console behind stdout/stderr in to ANSI escape processing.

    ``hermes_cli.colors`` and the skins emit raw SGR codes whenever stdout is a TTY. A
    conhost console (PowerShell 5.1, cmd.exe, the installer's ``hermes setup``) prints
    them as ``←[35m`` until the output handle has ENABLE_VIRTUAL_TERMINAL_PROCESSING, and
    shells hand native children a console with it off. The mode belongs to the console
    buffer, so setting it here also covers the relaunched child. Handles that are not a
    console (pipes, files, NUL, a windowless pythonw) fail GetConsoleMode and are left
    alone. A console that refuses VT (pre-Windows 10) gets NO_COLOR instead, which
    ``should_use_color`` and rich honour, so it shows plain text rather than garbage.
    Returns False only in that fallback case.
    """
    if not _IS_WINDOWS:
        return True
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    enabled = True
    for stream in (sys.stdout, sys.stderr) if streams is None else streams:
        try:
            handle = msvcrt.get_osfhandle(stream.fileno())
        except (AttributeError, OSError, ValueError):
            continue  # no fd (None under pythonw, StringIO in embedders)
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            continue
        if mode.value & _ENABLE_VIRTUAL_TERMINAL_PROCESSING:
            continue
        if not kernel32.SetConsoleMode(handle, mode.value | _ENABLE_VIRTUAL_TERMINAL_PROCESSING):
            enabled = False
    if not enabled:
        os.environ.setdefault("NO_COLOR", "1")
    return enabled


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


def _glibc_frees_environ() -> bool:
    """True on glibc < 2.41, whose ``setenv`` of a NEW name reallocs the ``environ`` array
    and frees the old one (2.41+ never frees it, so a concurrent ``getenv`` stays safe)."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc, _, version = (os.confstr("CS_GNU_LIBC_VERSION") or "").partition(" ")
        return libc == "glibc" and tuple(int(p) for p in version.split(".")[:2]) < (2, 41)
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def install_never_free_environ() -> None:
    """Make ``os.environ`` writes safe against native ``getenv`` in other threads.

    On glibc < 2.41 adding a name reallocs ``environ`` and frees the old array while a
    thread that dropped the GIL (``getaddrinfo``, OpenSSL's ``SSL_CERT_FILE`` lookup) may
    still be walking it; the freed slots hold tcache pointers, so the walk segfaults the
    whole process. Hermes writes new names at runtime from many places (``session.create``
    turns on gateway prompts, the agent build sets ``HERMES_SESSION_ID``) while background
    threads fetch catalogs, so the tui_gateway died with SIGSEGV. This is glibc 2.41's own
    fix: entry strings are cached per ``NAME=value`` and never freed; a new name is
    appended in place to an array with spare room, and only a full array is replaced by
    a bigger one, the old one kept forever. Set/del churn of the same names therefore
    allocates nothing after the first cycle.

    Residual: a NEW-name ``setenv`` from native code (a C or Rust extension, not
    ``os.environ``) bypasses the lock and still reallocs glibc's own last array, so a
    ``getenv`` that started walking that array before our swap can still fault. None of
    the gateway's crash paths do this.
    """
    if getattr(os.putenv, "_hermes_never_free_environ", False) or not _glibc_frees_environ():
        return
    import _thread
    import ctypes

    try:
        libc = ctypes.CDLL(None)
        environ = ctypes.c_void_p.in_dll(libc, "environ")
        getenv = libc.getenv
    except (AttributeError, OSError, ValueError):
        return
    getenv.restype, getenv.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
    real_putenv, real_unsetenv = os.putenv, os.unsetenv
    # Two unserialized writers both copy the live array and the later publish drops the
    # other's new name or undoes its replacement. Reentrant: audit hooks run inside it.
    lock = _thread.RLock()
    # A fork while another thread holds the lock would leave it held forever in the child.
    os.register_at_fork(before=lock.acquire, after_in_parent=lock.release, after_in_child=lock.release)
    lines: dict[bytes, ctypes.Array] = {}  # b"NAME=value" -> its C string (glibc's known_values)
    arrays: list[tuple[ctypes.Array, int]] = []  # every array we published + its address; only the last grows
    gen = [0]  # bumped by every publish, so a nested write inside an audited call forces a redo

    def _putenv(key, value) -> None:
        name, val = os.fsencode(key), os.fsencode(value)
        if not name or b"=" in name or b"\0" in name + val:
            real_putenv(key, value)  # the usual OSError/ValueError
            return
        sys.audit("os.putenv", name, val)
        prefix = name + b"="
        with lock:
            if (line := lines.get(prefix + val)) is None:
                line = lines[prefix + val] = ctypes.create_string_buffer(prefix + val)
            entry = ctypes.addressof(line)
            # create_string_buffer/addressof above are audited and a hook may write os.environ
            # re-entrantly (RLock): redo the read if any write was published before ours. The loop
            # itself makes no audited call, so a hook that writes on every event cannot spin it.
            while True:
                start = gen[0]
                # getenv returns a pointer just past "NAME=" inside the matching entry, so the
                # walk compares pointers instead of reading every string.
                found = getenv(name)
                target = found - len(prefix) if found else None
                live = ctypes.cast(environ.value, ctypes.POINTER(ctypes.c_void_p)) if environ.value else None
                n, hit = 0, False
                while live and (current := live[n]):
                    if current == target:
                        hit = True
                        break
                    n += 1
                own, own_addr = arrays[-1] if arrays else (None, None)
                grow = not hit and not (own is not None and environ.value == own_addr and n + 2 <= len(own))
                if grow:
                    fresh = (ctypes.c_void_p * max(2 * (n + 2), 64))(*(live[:n] if live else ()), entry)
                    fresh_addr = ctypes.cast(fresh, ctypes.c_void_p).value  # unaudited, unlike addressof
                if gen[0] == start:
                    break
            # No audited call from here on. Plain aligned stores: a concurrent walker sees them in
            # order on x86-64 (TSO). aarch64 may reorder them, which is theoretical there and
            # matches glibc < 2.41's own plain-store publish; Python has no cheap portable fence.
            gen[0] += 1
            if hit:
                live[n] = entry  # replace in place, as glibc does
            elif not grow:
                own[n + 1] = None  # terminator first, so a walker never runs past the new entry
                own[n] = entry
            else:
                arrays.append((fresh, fresh_addr))
                environ.value = fresh_addr

    def _unsetenv(key) -> None:
        with lock:  # glibc shifts the entries of the live array (ours included) in place
            real_unsetenv(key)
            gen[0] += 1

    _putenv._hermes_never_free_environ = True  # type: ignore[attr-defined]
    _unsetenv._hermes_never_free_environ = True  # type: ignore[attr-defined]
    os.putenv, os.unsetenv = _putenv, _unsetenv


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


def export_scratch_tmp_env() -> None:
    """Point ``TMPDIR``/``TMP``/``TEMP`` at ``HERMES_HOME/cache/scratch`` unless the user set them.

    System temp is tmpfs on most Linux hosts and containers; Hermes' browser profiles, PTY
    probes and every ``tempfile`` default a child script makes would eat RAM there. Runs at
    import so every entry point and every child they spawn inherits it; ``hermes_cli.main``
    re-runs it after ``--profile`` re-homes the process. Never raises.
    """
    try:
        from hermes_constants import export_scratch_tmp_env as _export
        _export()
    except Exception:
        pass  # a missing/unwritable home just leaves the system temp dir in place


# Apply on import — entry points just need ``import hermes_bootstrap``
# (or ``from hermes_bootstrap import apply_windows_utf8_bootstrap``) at
# the very top of their module, before importing anything else.  The
# import side effect does the right thing.
apply_windows_utf8_bootstrap()
enable_windows_vt()
suppress_platform_ver_console()
install_never_free_environ()

# Every entry point imports this module before its dependency graph.
from pathlib import Path

_root = Path(__file__).resolve().parent
try:
    os.getcwd()
except FileNotFoundError:
    # Reaped workspaces leave children in a deleted cwd. PM resolves relative
    # import paths before the CLI's guards, so recover before any PM work.
    os.chdir(_root)


def _legacy_post_swap_invocation(argv: list[str]) -> tuple[Path, list[str]] | None:
    """Recognize the exact fresh-checkout command emitted by shipped updaters."""
    if not argv or argv[0] != "update":
        return None
    try:
        marker = argv.index("--post-swap", 1)
    except ValueError:
        return None
    if marker + 2 != len(argv):
        return None
    return Path(argv[marker + 1]), argv[1:marker]


# Everything below imports Hermes packages, so the root goes on sys.path first. A venv
# editable-installed from a pre-PM tree maps only the top-level packages it knew then:
# without this, ``pm`` is unimportable and the launch silently skips PM adoption.
harden_import_path(str(_root))

_legacy_post_swap = _legacy_post_swap_invocation(sys.argv[1:])
if _legacy_post_swap is not None:
    # This continuation exists precisely because the replacement tree may not
    # run under the old release's dependency graph. Take it over before PM
    # activation, launch preparation, or argparse imports any of that graph.
    from hermes_cli.update_handoff import _continue_legacy_post_swap

    _handoff_path, _argv_tail = _legacy_post_swap
    raise SystemExit(_continue_legacy_post_swap(_handoff_path, argv_tail=_argv_tail))


from pm.environments import activate_dependencies
from hermes_cli._early_recovery import recover_if_needed

from hermes_cli._parser import command_argv

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
        # Degrade, never brick the CLI: the previous dependency generation is still selected
        # (a failed sync commits nothing), so an offline or half-finished update leaves a
        # usable Hermes plus a warning. Activation below is the real gate — a tree whose
        # dependencies cannot load still exits with the repair remedy.
        print(f"hermes: source-update completion failed: {exc}; "
              "running with the previous dependencies — run `hermes update` to finish it",
              file=sys.stderr)
    recover_if_needed(_root)
    try:
        activate_dependencies(_root)
    except (RuntimeError, OSError) as exc:
        if command_argv(sys.argv[1:])[:1] != ["pm"]:
            print(f"hermes: {exc}; run `hermes pm repair`", file=sys.stderr)
            raise SystemExit(1) from None
install_happy_eyeballs_socket_connect()
export_scratch_tmp_env()
