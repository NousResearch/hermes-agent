"""Dashboard/serve support: managed-service restart (systemd/respawn), status/listening probes, SSH session token file, named-profile routing, web-dist resolution, update stdio hangup protection.

Split out of ``hermes_cli/main.py``. Names that still live in main (``PROJECT_ROOT``, ...)
are imported lazily inside the functions that use them (avoids an import cycle).
"""

import contextlib
import os
import re
import shlex
import subprocess
import sys

from pathlib import Path
from typing import NoReturn
from hermes_cli.cli_output import line_input

_PRE_BUILD_HINT = "  Pre-build first:  npm install --workspace web && npm run build -w web"


def _find_stale_dashboard_pids(*, exclude_pids: set[int] | None = None) -> list[int]:
    """Return PIDs of stale ``dashboard``/``serve`` processes for update cleanup."""
    from hermes_cli.dashboard_procs import _scan_dashboard_processes
    return [pid for pid, _cmd in _scan_dashboard_processes(exclude_pids=exclude_pids)]


def _parse_dashboard_runtime(command: str) -> tuple[str, str, int] | None:
    """Best-effort parse of a dashboard/server cmdline into mode, host, and port."""
    mode = None
    for candidate in ("dashboard", "serve"):
        patterns = (f"hermes {candidate}", f"hermes_cli.main {candidate}", f"hermes_cli/main.py {candidate}")
        if any(pattern in command for pattern in patterns):
            mode = candidate
            break
    if mode is None:
        return None

    port = 9119
    host = "127.0.0.1"

    port_match = re.search(r"(?:^|\s)--port(?:=|\s+)(\d+)", command)
    if port_match:
        try:
            port = int(port_match.group(1))
        except ValueError:
            return None

    host_match = re.search(r"(?:^|\s)--host(?:=|\s+)(\"[^\"]+\"|'[^']+'|\S+)", command)
    if host_match:
        host = host_match.group(1).strip("\"'") or "127.0.0.1"

    return mode, host, port


def _dashboard_probe_host(host: str | None) -> str:
    """Map wildcard binds to a loopback address suitable for local probing."""
    normalized = (host or "127.0.0.1").strip().strip("[]")
    if normalized in {"", "0.0.0.0", "::"}:
        return "127.0.0.1"
    return normalized


_DASHBOARD_SYSTEMD_UNIT = "hermes-dashboard.service"

_SYSTEMCTL_ERRORS = (FileNotFoundError, subprocess.TimeoutExpired, OSError)


def _run_probe(cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess:
    """Captured, text-decoded ``subprocess.run`` for short local probes (systemctl, ps)."""
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def _restart_managed_dashboard_service(reason: str, unit: str = _DASHBOARD_SYSTEMD_UNIT) -> bool:
    """Restart a systemd-managed dashboard instead of raw-killing its PID.

    True when a unit was found and handled (success or printed failure) — which
    deliberately stops the caller's ``os.kill`` fallback: systemd treats a direct
    SIGTERM as a clean stop, so ``Restart=on-failure`` won't bring it back.
    """
    if sys.platform == "win32":
        return False

    def _systemctl(*args: str, timeout: int = 10) -> subprocess.CompletedProcess:
        return _run_probe(["systemctl", *args], timeout=timeout)

    # User manager first (Hermes installs Linux services in the user scope by
    # default), system manager only when the unit isn't there. Keep the selected
    # scope for ALL probes and the restart — a user unit must never be restarted
    # through the system manager (or raw-killed).
    scope: tuple[str, ...] | None = None
    for candidate in (("--user",), ()):
        try:
            result = _systemctl(*candidate, "list-unit-files", unit, "--no-legend", "--no-pager")
        except _SYSTEMCTL_ERRORS:
            continue
        if result.returncode != 0:
            continue
        unit_rows = (result.stdout or "").splitlines()
        if any(row.split()[0:1] == [unit] for row in unit_rows if row.split()):
            scope = candidate
            break

    if scope is None:
        return False

    try:
        active = _systemctl(*scope, "is-active", unit)
        enabled = _systemctl(*scope, "is-enabled", unit)
    except _SYSTEMCTL_ERRORS:
        return False

    active_state = (active.stdout or "").strip()
    enabled_state = (enabled.stdout or "").strip()
    if active_state != "active" and enabled_state not in {
        "enabled", "enabled-runtime", "linked", "linked-runtime", "static", "generated",
    }:
        return False

    print(f"\n⟲ Restarting managed dashboard service ({reason})")

    scope_label = "systemctl --user" if scope else "sudo systemctl"
    commands = [("systemctl", *scope, "restart", unit)]
    if not scope:
        # System units may require privilege escalation; user units must use
        # the user manager directly and never prompt for sudo.
        commands.append(("sudo", "-n", "systemctl", "restart", unit))

    errors: list[str] = []
    for command in commands:
        try:
            result = _run_probe(list(command), timeout=60)
        except _SYSTEMCTL_ERRORS as e:
            errors.append(f"{' '.join(command)}: {e}")
            continue
        if result.returncode == 0:
            print(f"    ✓ restarted {unit}")
            return True
        errors.append(f"{' '.join(command)}: {(result.stderr or result.stdout or '').strip()}")

    print(f"    ✗ failed to restart {unit}")
    for err in errors:
        if err.strip():
            print(f"      {err}")
    print(
        "  Dashboard is managed by systemd; not raw-killing its PID because "
        "systemd would treat that as a clean stop."
    )
    print(f"  Restart manually: {scope_label} restart {unit}")
    return True


def _pid_unified_cgroup_entries(pid: int):
    """Yield the ``0::<path>`` cgroup paths from ``/proc/<pid>/cgroup``; nothing when unreadable."""
    try:
        cgroup_path = Path(f"/proc/{pid}/cgroup")
        if not cgroup_path.is_file():
            return
        text = cgroup_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return
    for line in text.splitlines():
        parts = line.strip().split("::", 1)
        if len(parts) == 2:
            yield parts[1]


def _get_systemd_service_for_pid(pid: int) -> str | None:
    """The systemd service unit name *pid* belongs to (``hermes-serve.service``), or None.

    None when the PID isn't part of a service, the file is unreadable, or off Linux.
    """
    for cg_path in _pid_unified_cgroup_entries(pid):
        if cg_path.endswith(".service"):
            svc_name = cg_path.rsplit("/", 1)[-1]
            if svc_name:
                return svc_name
    return None


def _extract_scope_from_cgroup(cgroup_entry: str) -> str | None:
    """``user`` / ``system`` from a cgroup path (``/user.slice/…`` vs ``/system.slice/…``), else None."""
    if "/system.slice/" in cgroup_entry:
        return "system"
    if "/user.slice/" in cgroup_entry:
        return "user"
    return None


def _get_pid_cgroup_path(pid: int) -> str | None:
    """The unified (``0::``) cgroup path from ``/proc/<pid>/cgroup``, or None."""
    return next(_pid_unified_cgroup_entries(pid), None)


def _try_restart_systemd_service(svc_name: str, cgroup_path: str | None = None) -> bool:
    """Restart *svc_name* via systemctl (``--user`` for user-scope units). True on success.

    Unknown scope tries system first, then user.
    """
    scope = _extract_scope_from_cgroup(cgroup_path) if cgroup_path else None
    system_cmd = ["systemctl", "restart", svc_name]
    user_cmd = ["systemctl", "--user", "restart", svc_name]
    candidates = {"user": [user_cmd], "system": [system_cmd]}.get(scope, [system_cmd, user_cmd])
    for cmd in candidates:
        try:
            if _run_probe(cmd, timeout=15).returncode == 0:
                return True
        except _SYSTEMCTL_ERRORS:
            continue
    return False


def _dashboard_cmdline_for_pid(pid: int) -> list[str] | None:
    """Exact argv of a running process: ``/proc/<pid>/cmdline`` (Linux), ``ps -o command=`` + shlex
    (macOS), None on Windows (no graceful taskkill window; Desktop manages its backend)."""
    if sys.platform == "win32":
        return None
    try:
        cmdline_path = f"/proc/{pid}/cmdline"
        if os.path.exists(cmdline_path):
            with open(cmdline_path, "rb") as f:
                raw = f.read()
            argv = [part.decode("utf-8", errors="replace") for part in raw.split(b"\x00") if part]
            return argv or None
        result = _run_probe(["ps", "-p", str(pid), "-o", "command="], timeout=10)
        if result.returncode != 0:
            return None
        command = (result.stdout or "").strip()
        if not command:
            return None
        try:
            argv = shlex.split(command)
        except ValueError:
            argv = command.split()
        return argv or None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _respawn_dashboard_processes(commands: list[list[str]]) -> list[list[str]]:
    """Respawn manually-started dashboards after ``hermes update``, detached, logging to
    ``logs/dashboard-restart.log``; returns the argvs that failed to spawn. Callers pre-filter via
    ``_filter_dashboard_respawn_candidates`` (no Desktop ``--port 0`` backends, capped per profile).

    See #78821.
    """
    from hermes_constants import get_hermes_home
    respawned: list[list[str]] = []
    failed: list[tuple[list[str], str]] = []
    log_path = get_hermes_home() / "logs" / "dashboard-restart.log"
    with contextlib.suppress(OSError):
        log_path.parent.mkdir(parents=True, exist_ok=True)

    for command in commands:
        try:
            # Keep restarted dashboards headless; reopening a browser after a
            # background update is noisy and fails in SSH/headless sessions.
            if "dashboard" in command and "--no-open" not in command:
                command = [*command, "--no-open"]
            with open(log_path, "ab") as log_f:
                subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=log_f, stderr=subprocess.STDOUT,
                    start_new_session=True, close_fds=True)
            respawned.append(command)
        except (OSError, ValueError) as exc:
            failed.append((command, str(exc)))

    for command in respawned:
        print(f"    ✓ restarted: {shlex.join(command)}")
    for command, err_msg in failed:
        print(f"    ✗ failed to restart ({shlex.join(command)}): {err_msg}")
    return [command for command, _ in failed]


class _UpdateOutputStream:
    """stdout/stderr wrapper for ``hermes update``: mirrors to ``logs/update.log`` and, once the
    terminal vanishes (BrokenPipe/OSError/ValueError), drops screen output instead of the update."""

    _BROKEN = (BrokenPipeError, OSError, ValueError)

    def __init__(self, original, log_file):
        self._original = original
        self._log = log_file
        self._original_broken = False

    def write(self, data):
        # Mirror to the log file first — it's the most reliable destination.
        if self._log is not None:
            with contextlib.suppress(Exception):
                self._log.write(data)
        if not self._original_broken:
            try:
                return self._original.write(data)
            except self._BROKEN:
                self._original_broken = True  # terminal vanished; keep updating
        return len(data) if isinstance(data, (str, bytes)) else 0

    def flush(self):
        if self._log is not None:
            with contextlib.suppress(Exception):
                self._log.flush()
        if self._original_broken:
            return
        try:
            self._original.flush()
        except self._BROKEN:
            self._original_broken = True

    def isatty(self):
        if self._original_broken:
            return False
        try:
            return self._original.isatty()
        except Exception:
            return False

    def fileno(self):
        # Defer to the underlying stream; callers handle failures as when unwrapped.
        return self._original.fileno()

    def __getattr__(self, name):
        return getattr(self._original, name)


def _install_hangup_protection(gateway_mode: bool = False):
    """Protect ``cmd_update`` from SIGHUP (→ SIG_IGN, inherited by pip/git children) and broken pipes
    (stdio wrapped in ``_UpdateOutputStream``). SIGINT/SIGTERM are left alone — legitimate cancels.
    No-op in gateway mode (already detached). Returns state for ``_finalize_update_output``."""
    state = {
        "prev_stdout": sys.stdout, "prev_stderr": sys.stderr, "log_file": None, "installed": False}

    if gateway_mode:
        return state

    import signal as _signal

    if hasattr(_signal, "SIGHUP"):
        # Non-main thread: update still runs, just without hangup protection.
        with contextlib.suppress(ValueError, OSError):
            _signal.signal(_signal.SIGHUP, _signal.SIG_IGN)

    # Any failure here is non-fatal; we just skip the wrap.
    try:
        # Late-bound import so tests can monkeypatch
        # hermes_cli.config.get_hermes_home to simulate setup failure.
        from hermes_cli.config import get_hermes_home as _get_hermes_home
        logs_dir = _get_hermes_home() / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(logs_dir / "update.log", "a", buffering=1, encoding="utf-8")

        import datetime as _dt

        log_file.write(f"\n=== hermes update started {_dt.datetime.now().isoformat(timespec='seconds')} ===\n")

        state["log_file"] = log_file
        sys.stdout = _UpdateOutputStream(state["prev_stdout"], log_file)
        sys.stderr = _UpdateOutputStream(state["prev_stderr"], log_file)
        state["installed"] = True
    except Exception:
        state["log_file"] = None

    return state


def _finalize_update_output(state):
    """Restore stdio and close the update.log handle opened by ``_install_hangup_protection``."""
    if not state:
        return
    if state.get("installed"):
        with contextlib.suppress(Exception):
            sys.stdout = state.get("prev_stdout", sys.stdout)
        with contextlib.suppress(Exception):
            sys.stderr = state.get("prev_stderr", sys.stderr)
    log_file = state.get("log_file")
    if log_file is not None:
        with contextlib.suppress(Exception):
            log_file.flush()
            log_file.close()


def _report_dashboard_status() -> int:
    """Print live listening dashboard/serve processes and return the count.

    Serve-mode backends are INCLUDED: ``--stop`` kills them, so hiding them from
    ``--status`` let an operator kill what they couldn't see.

    Ledger-registered serves (profiled launches the argv scan can't match) surface via the spawn-ledger
    augmentation in _scan_dashboard_processes. See #81564.
    """
    from hermes_cli.dashboard_procs import _scan_dashboard_processes
    from gateway.status import _pid_exists
    live: list[tuple[int, str, str]] = []
    for pid, command in _scan_dashboard_processes():
        runtime = _parse_dashboard_runtime(command)
        if runtime is None:
            continue
        mode, host, port = runtime
        if port <= 0 or not _pid_exists(pid) or not _dashboard_listening(host, port):
            continue
        live.append((pid, command, mode))

    if not live:
        print("No hermes dashboard or serve processes running.")
        return 0

    print(f"{len(live)} hermes dashboard/serve process(es) running:")
    for pid, command, mode in live:
        print(f"    PID {pid} [{mode}]: {command}")
    return len(live)


def _dashboard_listening(host: str, port: int) -> bool:
    """True when something accepts TCP connections at host:port (even a 401 proves a dashboard is up)."""
    import socket
    try:
        with socket.create_connection((_dashboard_probe_host(host), port), timeout=1.5):
            return True
    except OSError:
        return False


def _cancel(message: str = "  Cancelled.") -> NoReturn:
    print(message)
    sys.exit(1)


def _maybe_setup_dashboard_auth_interactively(args) -> None:
    """Offer to configure dashboard auth when the gate engages and no provider exists.

    ``start_server`` fails closed for a non-loopback bind / ``dashboard.public_url``
    without a ``DashboardAuthProvider``; prompt an interactive operator first.
    No-op (fail-closed backstop stays) when the gate doesn't engage, a provider
    exists, or stdin/stdout isn't a TTY.
    """
    host = getattr(args, "host", "127.0.0.1") or "127.0.0.1"

    try:
        from hermes_cli.web_server import should_require_dashboard_auth
        if not should_require_dashboard_auth(host):
            return
    except Exception:
        return  # if we can't tell, defer to start_server's own gate

    try:
        from hermes_cli.dashboard_auth import list_providers
        if list_providers():
            return
    except Exception:
        return

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return

    print(f"\n⚠ Dashboard authentication is required for this configuration ({host}).")
    print(
        "  Non-loopback binds and configured external dashboard.public_url "
        "values require authentication (--insecure does not bypass this)."
    )
    print()
    print("  How do you want to authenticate the dashboard?")
    print("    [1] Username & password (quickest; for a trusted LAN / VPN)")
    print("    [2] OAuth via Nous Portal (run `hermes dashboard register`)\n    [3] Cancel\n")

    try:
        choice = input("  Choice [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        _cancel("\n  Cancelled.")

    if choice == "2":
        print()
        print(
            "  Run this on the host where the dashboard lives, then start "
            "the dashboard again:\n"
            "    hermes dashboard register\n"
            "  It provisions a Nous Portal OAuth client and writes "
            "HERMES_DASHBOARD_OAUTH_CLIENT_ID into ~/.hermes/.env for you.\n"
            "  Docs: https://hermes-agent.nousresearch.com/docs/"
            "user-guide/features/web-dashboard#authentication-gated-mode"
        )
        sys.exit(0)

    if choice != "1":
        _cancel()

    import getpass
    import secrets
    print()
    try:
        username = line_input("  Username [admin]: ").strip() or "admin"
        password = getpass.getpass("  Password: ")
        confirm = getpass.getpass("  Confirm password: ")
    except (EOFError, KeyboardInterrupt):
        _cancel("\n  Cancelled.")

    if not password:
        _cancel("  ✗ Empty password — aborting.")
    if password != confirm:
        _cancel("  ✗ Passwords don't match — aborting.")

    try:
        from plugins.dashboard_auth.basic import hash_password
    except Exception as exc:
        _cancel(f"  ✗ Could not load the password provider: {exc}")

    password_hash = hash_password(password)
    # A stable token-signing secret so sessions survive a dashboard restart.
    secret = secrets.token_urlsafe(32)

    try:
        from hermes_cli.config import load_config, save_config
        from hermes_cli.plugins_cmd import ensure_basic_auth_plugin_enabled_in_config
        cfg = load_config()
        basic = cfg.setdefault("dashboard", {}).setdefault("basic_auth", {})
        basic["username"] = username
        basic["password_hash"] = password_hash
        basic["password"] = ""  # never persist plaintext
        if not str(basic.get("secret", "") or "").strip():
            basic["secret"] = secret
        # The bundled basic provider is a backend plugin that honours
        # plugins.disabled; unblock it so discover_plugins below registers it,
        # and tell an operator who deliberately disabled it.
        if ensure_basic_auth_plugin_enabled_in_config(cfg):
            print("  ✓ Re-enabled the bundled 'basic' auth plugin (was in plugins.disabled)")
        save_config(cfg)
    except Exception as exc:
        _cancel(f"  ✗ Failed to write config.yaml: {exc}")

    # Re-run plugin discovery so the provider registers before start_server's gate.
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins(force=True)
    except Exception as exc:
        print(f"  ⚠ Plugin re-discovery failed ({exc}); the gate may still "
              "fail closed. Set the password again or restart the dashboard.")

    print()
    print(f"  ✓ Username/password auth configured (user: {username}).")
    print("    Saved to config.yaml under dashboard.basic_auth.")
    print("    Sign in at the dashboard with these credentials.\n")


def _read_ssh_session_token_file(path: str) -> str:
    """Read and unlink a Desktop SSH token from its private runtime directory."""
    if sys.platform == "win32":
        from hermes_cli.windows_ssh_runtime import read_token
        return read_token(path)

    import stat as _stat

    if not os.path.isabs(path):
        raise SystemExit("--ssh-session-token-file must be absolute")

    # Token validation and ownership-lock scanning must share the same
    # machine-scoped authority, independent of HERMES_HOME and the active
    # profile (#69551, #94032).
    from hermes_constants import get_desktop_ssh_runtime_root
    token_root = get_desktop_ssh_runtime_root()
    try:
        relative = Path(path).relative_to(token_root)
    except ValueError as exc:
        raise SystemExit("--ssh-session-token-file must be under the desktop-ssh directory") from exc
    if len(relative.parts) != 2 or not re.fullmatch(r"[0-9a-f]{32}", relative.parts[0]):
        raise SystemExit("--ssh-session-token-file has an invalid runtime path")
    if not re.fullmatch(r"[0-9a-f]{16}\.token", relative.parts[1]):
        raise SystemExit("--ssh-session-token-file has an invalid filename")

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    uid = os.getuid() if hasattr(os, "getuid") else None

    def _check_dir(fd: int, what: str) -> None:
        st = os.fstat(fd)
        if not _stat.S_ISDIR(st.st_mode):
            raise SystemExit(f"--ssh-session-token-file has an unsafe {what}")
        if uid is not None and st.st_uid != uid:
            raise SystemExit(f"--ssh-session-token-file {what} has the wrong owner")
        if what == "parent directory" and (st.st_mode & 0o777) != 0o700:
            raise SystemExit("--ssh-session-token-file parent has unsafe permissions")

    root_fd = -1
    directory_fd = -1
    file_fd = -1
    try:
        try:
            root_fd = os.open(token_root, directory_flags)
            _check_dir(root_fd, "runtime root")
            directory_fd = os.open(relative.parts[0], directory_flags, dir_fd=root_fd)
            _check_dir(directory_fd, "parent directory")
            file_fd = os.open(relative.parts[1], file_flags, dir_fd=directory_fd)
        except SystemExit:
            raise
        except OSError as exc:
            if exc.errno == getattr(__import__("errno"), "ELOOP", -1):
                raise SystemExit("--ssh-session-token-file is a symlink") from exc
            raise SystemExit("--ssh-session-token-file is not accessible") from exc

        file_stat = os.fstat(file_fd)
        if not _stat.S_ISREG(file_stat.st_mode):
            raise SystemExit("--ssh-session-token-file is not a regular file")
        if file_stat.st_size != 64:
            raise SystemExit("--ssh-session-token-file contains an invalid token")
        if uid is not None and file_stat.st_uid != uid:
            raise SystemExit("--ssh-session-token-file has the wrong owner")
        if uid is not None and (file_stat.st_mode & 0o777) & ~0o600:
            raise SystemExit("--ssh-session-token-file has unsafe permissions")

        with os.fdopen(file_fd, "r", encoding="utf-8") as token_stream:
            file_fd = -1
            token = token_stream.read(65)

        if not re.fullmatch(r"[0-9a-f]{64}", token):
            raise SystemExit("--ssh-session-token-file contains an invalid token")
        return token
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if directory_fd >= 0:
            with contextlib.suppress(OSError):
                os.unlink(relative.parts[1], dir_fd=directory_fd)
            os.close(directory_fd)
        if root_fd >= 0:
            os.close(root_fd)


def _is_electron_packaged_web_dist(path: str) -> bool:
    """True when *path* is an Electron-packaged renderer dist (``app.asar[.unpacked]/dist``).

    A standalone ``hermes dashboard`` inheriting that ``HERMES_WEB_DIST`` would
    serve the desktop frontend in the browser ("Desktop IPC bridge is unavailable").
    """
    if not path:
        return False
    return "app.asar" in path.replace("\\", "/")


def _route_named_profile_dashboard(
    args, _headless_backend: bool, _ssh_owner_nonce: str, _token_file: str) -> None:
    """Route a named-profile launch to the single MACHINE dashboard (per-request ``?profile=`` scoping
    makes one server per profile pure fragmentation).

    Already listening → open ``?profile=<name>`` and exit; else re-exec pinned to
    ``-p default`` (so ``_apply_profile_override`` can't re-route via the sticky
    active_profile file). ``--isolated`` opts out; Desktop pool backends
    (HERMES_DESKTOP=1) stay per-profile. Returns normally when no routing applies.
    """
    try:
        from hermes_cli.profiles import get_active_profile_name
        _launch_profile = get_active_profile_name()
    except Exception:
        _launch_profile = "default"

    if (
        _launch_profile in ("default", "custom")
        or getattr(args, "isolated", False)
        or getattr(args, "open_profile", "")
        or os.environ.get("HERMES_DESKTOP") == "1"
    ):
        return

    url = f"http://{args.host or '127.0.0.1'}:{args.port}/?profile={_launch_profile}"
    if _dashboard_listening(args.host, args.port):
        print(f"Machine dashboard already running on port {args.port}.")
        print(f"  Managing profile '{_launch_profile}': {url}")
        if not args.no_open:
            with contextlib.suppress(Exception):
                import webbrowser
                webbrowser.open(url)
        sys.exit(0)

    print(
        f"Routing to the machine dashboard (profile '{_launch_profile}' "
        f"preselected). Use --isolated for a dedicated per-profile server."
    )
    reexec_argv = [
        sys.executable, "-m", "hermes_cli.main",
        "-p", "default",
        # Preserve the lean serve path so a named-profile `serve` doesn't
        # silently rebuild the UI as `dashboard`.
        "serve" if _headless_backend else "dashboard",
        "--port", str(args.port),
        "--host", args.host,
        "--open-profile", _launch_profile]
    for enabled, extra in (
        (_ssh_owner_nonce, ["--ssh-owner-nonce", _ssh_owner_nonce]),
        (_token_file, ["--ssh-session-token-file", _token_file]),
        (args.no_open, ["--no-open"]),
        (getattr(args, "insecure", False), ["--insecure"]),
        (getattr(args, "skip_build", False), ["--skip-build"])):
        if enabled:
            reexec_argv.extend(extra)
    from tools.environments.local import build_subprocess_env
    # HERMES_HOME is pinned to the machine root below — the factory must not
    # re-inject a profile home.
    env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=False)
    # Pin the child to the machine ROOT, resolved explicitly rather than by
    # dropping HERMES_HOME: in the Docker layout the root is /opt/data, and an
    # unset HERMES_HOME would fall back to $HOME/.hermes = /opt/data/.hermes — an
    # empty auto-seeded home with only the default profile and no install stamp.
    # get_default_hermes_root() strips a trailing profiles/<name> for both layouts.
    try:
        from hermes_constants import get_default_hermes_root
        env["HERMES_HOME"] = str(get_default_hermes_root())
    except Exception:
        env.pop("HERMES_HOME", None)  # prior behaviour rather than blocking the reroute
    # On Windows os.execvpe() spawns via CreateProcess then exits, which under
    # Python 3.14+ can crash with STATUS_ACCESS_VIOLATION; use Popen + exit.
    if sys.platform == "win32":
        proc = subprocess.Popen(reexec_argv, env=env)
        sys.exit(proc.wait())
    else:
        os.execvpe(sys.executable, reexec_argv, env)


def _resolve_dashboard_web_dist(args, _headless_backend: bool) -> None:
    """Build or validate the web UI dist before the server imports.

    ``serve`` sets HERMES_SERVE_HEADLESS so mount_spa() stays off. Otherwise build
    unless HERMES_WEB_DIST / --skip-build promise a dist — then verify index.html
    (else the server serves 404s). --skip-build on the default location gets ONE
    recovery build; a caller-managed HERMES_WEB_DIST can't be populated.
    """
    from hermes_cli.main import PROJECT_ROOT
    from hermes_cli.main_web_build import _build_web_ui
    skip_build = getattr(args, "skip_build", False)
    if _headless_backend:
        os.environ["HERMES_SERVE_HEADLESS"] = "1"  # set before web_server import
    elif "HERMES_WEB_DIST" not in os.environ and not skip_build:
        if not _build_web_ui(PROJECT_ROOT / "web", fatal=True):
            sys.exit(1)
    elif skip_build:
        _dist_root = (
            # --build-mode skip trusts the caller to have pre-built the web UI. Verify the dist actually
            # exists; otherwise the server will start and serve 404s with no obvious cause (issue #23817).
            Path(os.environ["HERMES_WEB_DIST"])
            if "HERMES_WEB_DIST" in os.environ
            else PROJECT_ROOT / "hermes_cli" / "web_dist"
        )
        if not (_dist_root / "index.html").exists():
            # Only the default dist location is recoverable (desktop launches with
            # --build-mode skip after a wipe of web_dist); a custom HERMES_WEB_DIST
            # is a caller-managed directory the build cannot populate.
            # The caller promised a pre-built dist but there isn't one. Instead of hard-failing (issue
            # #59288 — desktop launches with --build-mode skip after a wipe of web_dist), warn and attempt
            # ONE recovery build through the normal build path.
            _recoverable = "HERMES_WEB_DIST" not in os.environ
            if _recoverable:
                print(f"⚠ --skip-build was passed but no web dist found at: {_dist_root}")
                print("  Attempting one recovery build of the web UI...")
                _build_web_ui(PROJECT_ROOT / "web", fatal=True)
            if not (_dist_root / "index.html").exists():
                print(f"✗ --skip-build was passed but no web dist found at: {_dist_root}")
                if _recoverable:
                    print("  The recovery build did not produce a usable dist.")
                print(_PRE_BUILD_HINT)
                print("  Or drop --skip-build to build automatically.")
                sys.exit(1)
            print("  ✓ Recovery build produced a web dist")
        print(f"→ Skipping web UI build (--skip-build); using dist at {_dist_root}")
    else:
        # HERMES_WEB_DIST without --skip-build: the env var points at a
        # caller-managed dist, so validate it like the --skip-build branch.
        # HERMES_WEB_DIST is set without --skip-build: the build is skipped (the env var points at a
        # caller-managed dist), so validate it the same way the --skip-build branch does — otherwise the
        # server starts and serves 404s with no obvious cause (same failure mode as #23817, via the env-var
        # path).
        _dist_root = Path(os.environ["HERMES_WEB_DIST"]).expanduser()
        if not (_dist_root / "index.html").exists():
            print(f"✗ HERMES_WEB_DIST is set but no web dist found at: {_dist_root}")
            print(_PRE_BUILD_HINT)
            print("  Or unset HERMES_WEB_DIST to build and use the default web UI dist.")
            sys.exit(1)
        # web_server reads HERMES_WEB_DIST raw at import (no expanduser), so a
        # validated "~/dist" would otherwise pass here and still 404 there.
        os.environ["HERMES_WEB_DIST"] = str(_dist_root)
        print(f"→ Using web dist from HERMES_WEB_DIST: {_dist_root}")
