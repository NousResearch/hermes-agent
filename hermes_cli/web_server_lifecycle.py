"""Serve-process lifecycle: parent death watchdog, port-conflict preflight, READY announcement, browser open, trusted proxies.
"""

import logging
import ipaddress
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover - annotation only
    import uvicorn

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


def _process_start_marker(pid: int) -> str:
    """Return a cross-runtime marker for the current incarnation of ``pid``.

    ``ProcessLookupError`` means the process is absent; other failures stay
    distinct so callers fail safe rather than killing a healthy backend.
    """
    if sys.platform == "linux":
        try:
            stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ProcessLookupError(pid) from exc

        # Field 2 (comm) may contain spaces/parens; split after its final ')'
        # so field 3 is index 0 and field 22 (starttime) is index 19.
        fields = stat_line.rsplit(")", 1)[1].strip().split()
        if len(fields) < 20 or not fields[19].isdigit():
            raise OSError(f"invalid /proc stat data for PID {pid}")
        return f"linux:{fields[19]}"

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error in (87, 1168):  # invalid parameter / not found
                raise ProcessLookupError(pid)
            raise OSError(error, f"OpenProcess failed for PID {pid}")

        creation, exit_time, kernel, user = (wintypes.FILETIME() for _ in range(4))
        try:
            if not kernel32.GetProcessTimes(
                handle, ctypes.byref(creation), ctypes.byref(exit_time), ctypes.byref(kernel), ctypes.byref(user)
            ):
                raise OSError(ctypes.get_last_error(), f"GetProcessTimes failed for PID {pid}")
        finally:
            kernel32.CloseHandle(handle)

        filetime = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        return f"win:{filetime + 504911232000000000}"

    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True, encoding="utf-8",
                            errors="replace", check=False)
    marker = result.stdout.strip()
    if result.returncode == 0 and marker:
        return f"ps:{marker}"
    if result.returncode == 1 and not marker:
        raise ProcessLookupError(pid)
    raise OSError(f"ps could not inspect PID {pid}: {result.stderr.strip()}")


def _valid_parent_start_marker(marker: str) -> bool:
    prefix, separator, value = marker.partition(":")
    if not separator or not value or value != value.strip():
        return False
    if prefix in ("linux", "win", "winms"):
        return value.isdigit()
    if prefix != "ps":
        return False
    # ``ps -o lstart=`` renders as ``Sat Aug 29 15:04:31 2026``. A marker split
    # on whitespace somewhere in the env plumbing arrives as ``ps:Sat`` -- a
    # real-looking identity that can never match, so the watchdog would exit a
    # live backend. Require a full value (#98132).
    tokens = value.split()
    has_year = any(token.isdigit() and len(token) == 4 for token in tokens)
    has_time = any(":" in token for token in tokens)
    return len(tokens) >= 4 and has_year and has_time


def _parent_start_marker_mismatch_is_conclusive(actual: str, expected: str) -> bool:
    """Whether a marker mismatch proves the Desktop parent was replaced.

    ``linux:`` jiffies and ``win:``/``winms:`` FILETIME markers are machine
    values; a mismatch there is PID-reuse evidence. The POSIX fallback (macOS
    has no ``/proc``) is ``ps -o lstart=`` -- a wall-clock string rendered in
    the current TZ/locale. Electron caches it once per app lifetime while the
    backend re-renders it per spawn, so a timezone change (or DST, or column
    padding) makes the SAME instant differ byte-for-byte. Any ``ps:`` mismatch
    is therefore inconclusive (#95693, #93958).
    """
    return not (actual.startswith("ps:") or expected.startswith("ps:"))


def _parent_start_markers_match(actual: str, expected: str) -> bool:
    """Compare parent markers across Desktop generations.

    Old Windows Desktop sends .NET ticks (``win:``); new builds send Electron's
    creation time in Unix ms (``winms:``) to avoid launching PowerShell. The
    backend reads the exact FILETIME and normalizes only for ``winms``.
    """
    if actual == expected:
        return True
    if not actual.startswith("win:") or not expected.startswith("winms:"):
        return False

    try:
        dotnet_ticks = int(actual.removeprefix("win:"))
        expected_unix_ms = int(expected.removeprefix("winms:"))
    except ValueError:
        return False

    dotnet_ticks_at_unix_epoch = 621_355_968_000_000_000
    actual_unix_ms = (dotnet_ticks - dotnet_ticks_at_unix_epoch) // 10_000
    return actual_unix_ms == expected_unix_ms


def _warm_gateway_module() -> None:
    """Pre-import heavy modules so the event loop is not stalled on first use.

    Cold Windows installs pay .pyc compilation + Defender scans (15-30s) on
    these chains; the first WS RPC burst (setup.status, setup.runtime_check,
    gateway.ready→resolve_skin, model.options) pulled them in on the loop
    thread (#60800). Warm them all off-loop while the socket is already open.
    """
    for mod in (
        "hermes_cli.gateway",
        "hermes_cli.auth",  # provider auth state → copilot_auth → subprocess
        "hermes_cli.copilot_auth",
        "hermes_cli.runtime_provider",
        "hermes_cli.skin_engine",  # resolve_skin() config + skin engine init
        "hermes_cli.inventory",  # provider catalogs + models.dev cache
        "hermes_cli.model_switch",
    ):
        try:
            __import__(mod)
        except Exception:
            pass


def _resolve_restart_drain_timeout() -> float:
    try:
        from hermes_cli.gateway import _get_restart_drain_timeout
        return _get_restart_drain_timeout()
    except ImportError:
        from gateway.restart import DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
        return DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT


def _eager_reconcile_own_session_db() -> None:
    """One writable open of this process's own state.db at startup.

    ``SessionDB.__init__`` runs ``_init_schema`` → ``_reconcile_columns`` with
    open-time lock patience. Never raises: an unfixable store still gets the
    per-poll read-probe heal in :func:`_open_session_db_at_path`.
    """
    try:
        from hermes_state import SessionDB, _default_db_path

        SessionDB(db_path=Path(_default_db_path()), read_only=False).close()
    except Exception as exc:
        _log.warning(
            "startup schema reconcile of state.db failed (%s); session "
            "reads will retry the heal per poll", exc,
        )


def _read_bound_port(server: "uvicorn.Server", fallback: int) -> int:
    """Read the OS-assigned port from the live uvicorn socket (ephemeral port-0 discovery)."""
    if server.servers and server.servers[0].sockets:
        return server.servers[0].sockets[0].getsockname()[1]
    return fallback


def _write_dashboard_ready_file(actual_port: int) -> None:
    """Publish the port through an atomic ready file when ``HERMES_DESKTOP_READY_FILE`` is set.

    Windows Desktop launches via ``pythonw.exe`` (no console flash) cannot use
    stdout for the port announcement, so Electron waits for this JSON instead.
    """
    target = os.environ.get("HERMES_DESKTOP_READY_FILE")
    if not target:
        return

    tmp_name = ""
    try:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"port": int(actual_port)}, separators=(",", ":"))
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp", delete=False
        ) as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
            tmp_name = fh.name
        os.replace(tmp_name, path)
    except Exception as exc:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except Exception:
                pass
        _log.warning("Failed to write dashboard ready file %r: %s", target, exc)


def _maybe_open_browser(host: str, actual_port: int, open_browser: bool, initial_profile: str) -> None:
    """Open the dashboard URL in the user's browser if appropriate.

    Skips headless Linux (no DISPLAY/WAYLAND_DISPLAY) so a TUI browser can't
    SIGHUP the server; maps ``0.0.0.0``/``::`` binds to ``127.0.0.1``.
    """
    if not open_browser:
        return

    import webbrowser

    _has_display = sys.platform != "linux" or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not _has_display:
        _log.debug(
            "Skipping browser-open: no DISPLAY or WAYLAND_DISPLAY detected "
            "(headless Linux). Pass --no-open to suppress this detection."
        )
        return

    _display_host = host if host not in ("0.0.0.0", "::") else "127.0.0.1"
    _open_url = f"http://{_display_host}:{actual_port}"
    if initial_profile:
        from urllib.parse import quote
        _open_url += f"/?profile={quote(initial_profile)}"

    def _open():
        try:
            time.sleep(1.0)
            webbrowser.open(_open_url)
        except Exception:
            pass

    threading.Thread(target=_open, daemon=True).start()


def _is_serve_orphaned(
    desktop_pid: int,
    expected_start_marker: Optional[str] = None,
    *,
    pid_exists=None,
    process_start_marker=None,
) -> bool:
    """True when the exact Desktop process that owns this backend is gone.

    ``HERMES_PARENT_PID`` is the Electron PID, not necessarily our PPID (the
    Windows ``hermes.exe`` launcher adds shims), so never compare getppid().
    The start marker (newer Desktops) defeats PID recycling; PID-only probing
    stays for older ones. Any inconclusive failure keeps serving (fail-safe).
    """
    try:
        if expected_start_marker is not None:
            probe = process_start_marker or _process_start_marker
            actual_marker = probe(int(desktop_pid))
            if _parent_start_markers_match(actual_marker, expected_start_marker):
                return False
            if _parent_start_marker_mismatch_is_conclusive(actual_marker, expected_start_marker):
                return True
            # Inconclusive marker: degrade to PID liveness instead of exiting.

        if pid_exists is None:
            from gateway.status import _pid_exists

            pid_exists = _pid_exists
        return not bool(pid_exists(int(desktop_pid)))
    except ProcessLookupError:
        return True
    except Exception:
        return False


def _start_parent_death_watchdog() -> None:
    """Exit when the exact desktop parent that spawned this backend dies.

    Desktop passes its PID and, in newer versions, a start marker (defeats PID
    reuse) plus a per-spawn nonce (makes mixed-version plumbing fail safe:
    marker without nonce or vice versa disables the watchdog).
    """
    raw_pid = os.environ.get("HERMES_PARENT_PID")
    # Empty inherited values mean "absent", not "marker present but blank".
    start_marker = os.environ.get("HERMES_PARENT_START_MARKER") or None
    nonce = os.environ.get("HERMES_PARENT_NONCE") or None

    try:
        desktop_pid = int(raw_pid or "")
    except (TypeError, ValueError):
        return
    if desktop_pid <= 0:
        return

    has_marker = start_marker is not None
    if has_marker != (nonce is not None):
        return
    if has_marker and (not _valid_parent_start_marker(start_marker or "") or not nonce or nonce != nonce.strip()):
        # Disarming is fail-safe (the backend keeps serving) but must not be
        # traceless: this backend will never reap itself if the Desktop dies.
        _log.warning(
            "Parent-death watchdog disabled: unusable HERMES_PARENT_START_MARKER=%r / nonce; "
            "falling back to no parent tracking for desktop PID %s.",
            start_marker,
            desktop_pid,
        )
        return

    try:
        poll = max(0.5, float(os.environ.get("HERMES_SERVE_WATCHDOG_POLL_S", "2.0")))
    except (TypeError, ValueError):
        poll = 2.0

    def _loop() -> None:
        while not _is_serve_orphaned(desktop_pid, start_marker):
            time.sleep(poll)
        try:
            _log.warning(
                "Parent-death watchdog: desktop PID %s appears orphaned (expected_start_marker=%r); exiting.",
                desktop_pid,
                start_marker,
            )
        except Exception:
            pass
        os._exit(0)

    threading.Thread(target=_loop, daemon=True, name="serve-parent-watchdog").start()


# Port-conflict sentinel (#93608): uvicorn's bind_socket() turns EADDRINUSE into
# a bare ERROR + exit 1, indistinguishable from a crash for the Desktop spawn.
# We probe the exact bind first and emit ONE machine-readable stdout line plus a
# distinct exit code. 75 == BSD EX_TEMPFAIL, the codebase's "transient
# environmental condition" convention (gateway/restart.py, kanban_db.py).
PORT_IN_USE_EXIT_CODE = 75
_PORT_IN_USE_SENTINEL = "BACKEND_PORT_IN_USE port={port}"


def _is_addr_in_use_error(exc: OSError) -> bool:
    """True when ``exc`` is the platform's address-in-use bind failure."""
    import errno

    # POSIX, Linux, macOS, WinSock; WSAEADDRINUSE also surfaces as winerror.
    return exc.errno in {errno.EADDRINUSE, 98, 48, 10048} or getattr(exc, "winerror", None) == 10048


def _port_bind_conflict(host: str, port: int) -> bool:
    """Probe whether binding ``host:port`` would fail with EADDRINUSE.

    ``port == 0`` (ephemeral) never conflicts, so it is skipped. Any probe
    error other than address-in-use returns ``False`` so uvicorn surfaces it
    with its normal diagnostics (bad host, EACCES, …).
    """
    if not port:
        return False
    import socket as _socket

    family = _socket.AF_INET6 if ":" in host else _socket.AF_INET
    try:
        probe = _socket.socket(family, _socket.SOCK_STREAM)
    except OSError:
        return False
    try:
        _exclusive = getattr(_socket, "SO_EXCLUSIVEADDRUSE", None)
        if sys.platform == "win32" and _exclusive is not None:
            # Windows SO_REUSEADDR binds over a live LISTEN socket and can never
            # detect a conflict; SO_EXCLUSIVEADDRUSE fails with 10048 exactly
            # when another socket holds the port (#93608).
            probe.setsockopt(_socket.SOL_SOCKET, _exclusive, 1)
        else:
            # Match uvicorn's own bind flags so the probe conflicts exactly when
            # its bind would: TIME_WAIT remnants pass, a live LISTEN fails.
            probe.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
    except OSError as exc:
        return _is_addr_in_use_error(exc)
    except Exception:
        return False
    finally:
        probe.close()
    return False


def _write_machine_sentinel_line(line: str) -> None:
    """Write a machine-parsed sentinel line to the REAL stdout (fd 1).

    ``tui_gateway.server`` redirects ``sys.stdout`` to stderr at import (#94724),
    so a ``print()`` sentinel never reaches the Desktop's stdout pipe; fd 1 is
    untouched. If fd 1 is unwritable (pythonw.exe) fall back to ``print()`` for
    humans only — pythonw spawns discover the port via the ready file. Never
    raises.
    """
    try:
        os.write(1, (line + "\n").encode())
    except OSError:
        try:
            print(line, flush=True)
        except Exception:
            pass


def _report_port_in_use(host: str, port: int) -> None:
    """Print the machine sentinel + a human hint naming likely holders."""
    _write_machine_sentinel_line(_PORT_IN_USE_SENTINEL.format(port=port))
    print(
        f"  Port {port} on {host} is already in use — likely another "
        "'hermes serve' / 'hermes dashboard' backend or the Hermes gateway. "
        "Stop the other process, or pass --port <other> "
        "(--port 0 picks a free ephemeral port).",
        flush=True,
    )


_DEFAULT_DASHBOARD_FORWARDED_ALLOW_IPS = ("127.0.0.1", "::1")


def _dashboard_forwarded_allow_ips(dashboard_config: dict[str, Any]) -> list[str]:
    """Return the bounded proxy addresses uvicorn may trust: loopback plus valid config entries.

    Invalid or unbounded (/0, '*') ``dashboard.trusted_proxies`` entries fail
    closed so client-supplied forwarding headers never become request metadata.
    """
    configured = dashboard_config.get("trusted_proxies", [])
    if configured in (None, ""):
        configured = []
    elif isinstance(configured, str):
        configured = [configured]
    elif not isinstance(configured, (list, tuple)):
        _log.warning(
            "dashboard.trusted_proxies must be a list of IP addresses or CIDR networks; "
            "ignoring %r",
            configured,
        )
        configured = []

    trusted = list(_DEFAULT_DASHBOARD_FORWARDED_ALLOW_IPS)
    for raw_entry in configured:
        if not isinstance(raw_entry, str) or not raw_entry.strip():
            _log.warning(
                "Ignoring invalid dashboard.trusted_proxies entry %r; expected an IP "
                "address or CIDR network",
                raw_entry,
            )
            continue

        entry = raw_entry.strip()
        try:
            if "/" in entry:
                network = ipaddress.ip_network(entry, strict=False)
                if network.prefixlen == 0:
                    raise ValueError("unbounded network")
                normalized = str(network)
            else:
                normalized = str(ipaddress.ip_address(entry))
        except ValueError:
            _log.warning(
                "Ignoring unsafe dashboard.trusted_proxies entry %r; use a bounded IP "
                "address or CIDR network, never '*' or a /0 network",
                raw_entry,
            )
            continue

        if normalized not in trusted:
            trusted.append(normalized)

    if trusted != list(_DEFAULT_DASHBOARD_FORWARDED_ALLOW_IPS):
        _log.info("Dashboard trusted proxies: %s", ", ".join(trusted))

    return trusted
