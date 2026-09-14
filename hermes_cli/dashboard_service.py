"""Dashboard launchd service core for macOS lifecycle management (issue #44106).

WHY first-class macOS LaunchAgent lifecycle: the web dashboard needs the same
supervised restart / install / status path the gateway already has on macOS,
rather than being managed only by the Linux-only systemd branch.

No `--detach`: the dashboard already runs foreground (`cmd_dashboard`) — detach
would exit before launchd could supervise it, and KeepAlive would respawn the
launcher, not the server (the PR #40636 defect). The degraded fallback is a
manual nohup hint (not a detached gateway spawn — `_spawn_detached_gateway()`
spawns a GATEWAY, which is the wrong service for the dashboard).
"""

import argparse
import os
import plistlib
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from hermes_cli.gateway import (
    _launchctl_bootstrap,
    _launchctl_domain_unsupported,
    _launchctl_kickstart_current,
    _launchd_domain,
    _launchd_error_indicates_unloaded,
    _launchd_print_service_pid,
    _profile_suffix,
    get_python_path,
    is_macos,
    _service_venv_dir,
    _build_service_path_dirs,
    _append_node_dir_for_service,
    _stable_service_working_dir,
)
from hermes_cli.main_dashboard import _dashboard_probe_host
from xml.sax.saxutils import escape

from hermes_constants import get_hermes_home


def find_service_managed_dashboard_pids() -> dict[int, str]:
    """Discover OUR dashboard LaunchAgent PIDs from plists (not argv heuristics).

    WHY launchd, not argv substrings: the scanner's ``_DASHBOARD_PATTERNS``
    matches generic ``serve``/``dashboard`` strings that also appear in
    unrelated commands (#87594 class). Our own plists have exact ``Label``
    values, so ``launchctl print`` against the label yields a verified PID.
    The mapping is exact (pid → label) and the exclusion uses the label,
    not a string-match.

    On non-macOS: returns {} (pure query, no lifecycle command, no sys.exit).
    """
    if not is_macos():
        return {}
    import pwd

    result: dict[int, str] = {}
    # Use real account home via pwd (same resolution as get_dashboard_launchd_plist_path) —
    # tests mock Path.home() to redirect the glob target.
    try:
        home = Path(pwd.getpwuid(os.getuid()).pw_dir)  # windows-footgun: ok — POSIX launchd (macOS) helper, never invoked on Windows (is_macos gate above)
    except KeyError:
        home = Path.home()
    launchd_dir = home / "Library" / "LaunchAgents"
    if not launchd_dir.is_dir():
        return result
    for plist_path in launchd_dir.glob("ai.hermes.dashboard*.plist"):
        try:
            with plist_path.open("rb") as fh:
                plist_data = plistlib.load(fh)
        except Exception as exc:
            # One-line diagnostic, not silent; other plists still processed.
            print(f"[find_service_managed_dashboard_pids] skipped unparseable {plist_path}: {exc}")
            continue
        label = plist_data.get("Label")
        if not isinstance(label, str) or not label.startswith("ai.hermes.dashboard"):
            # Unrelated plists (e.g. gateway, unrelated tools) are ignored.
            continue
        # Positive PID from launchd = service-managed; 0/non-positive is skipped.
        domain = _launchd_domain()
        loaded, pid = _launchd_print_service_pid(domain, label)
        if loaded and pid is not None and pid > 0:
            result[pid] = label
    return result


def get_dashboard_launchd_label() -> str:
    """LaunchAgent label for the dashboard, scoped per profile."""
    suffix = _profile_suffix()
    return f"ai.hermes.dashboard-{suffix}" if suffix else "ai.hermes.dashboard"


def get_dashboard_launchd_plist_path() -> Path:
    """`~/Library/LaunchAgents/<label>.plist` under the real account home."""
    import pwd
    suffix = _profile_suffix()
    name = f"ai.hermes.dashboard-{suffix}" if suffix else "ai.hermes.dashboard"
    # Use real account home via pwd (same resolution as find_service_managed_dashboard_pids).
    try:
        home = Path(pwd.getpwuid(os.getuid()).pw_dir)  # windows-footgun: ok — POSIX launchd (macOS) helper, never invoked on Windows
    except KeyError:
        home = Path.home()
    return home / "Library" / "LaunchAgents" / f"{name}.plist"


# ------------------------------------------------------------------
# Shared degradation policy: domain-unsupported (not detached gateway spawn)
# ------------------------------------------------------------------

def _degrade_dashboard_launchctl_error(exc: subprocess.CalledProcessError, what: str) -> None:
    """Domain unsupported (5/125) for dashboard launchctl: clear message + manual nohup hint + exit 1.

    WHY no detached spawn here: gateway's degrade calls `_spawn_detached_gateway()`
    (gateway.py:3582) which launches a GATEWAY process, not the dashboard. Reusing it
    for the dashboard would silently start the wrong supervised service.
    """
    if not _launchctl_domain_unsupported(exc.returncode):
        raise exc
    plist_path = get_dashboard_launchd_plist_path()
    label = get_dashboard_launchd_label()
    print(
        f"launchd cannot manage the dashboard service on this macOS version "
        f"({what} exit {exc.returncode}); the domain does not support bootstrapping."
    )
    # Manual workaround: use the plist's own ProgramArguments with nohup.
    log_dir = get_hermes_home() / "logs"
    if plist_path.exists():
        try:
            plist_data = plistlib.loads(plist_path.read_bytes())
            args = plist_data.get("ProgramArguments", [])
            if args:
                cmd_hint = " ".join(f'"{a}"' for a in args)
                print(f"  Manual workaround: nohup {cmd_hint} > {log_dir}/dashboard.log 2>&1 &")
        except Exception as exc:
            print(f"Could not read installed plist for manual workaround: {exc}")
    else:
        print(f"  Manual workaround: nohup python -m hermes_cli.main dashboard --host <host> --port <port> > {log_dir}/dashboard.log 2>&1 &")
    sys.exit(1)


def generate_dashboard_launchd_plist(
    host: str, port: int, *, extra_args: list[str] | None = None
) -> str:
    """Generate launchd plist XML for the dashboard service.

    Profile policy (per controller design):
    - Default profile: pins `-p default` before subcommand so sticky active_profile
      can't reroute the supervised server.
    - Named profile: `--profile <name> --isolated` so the service owns a dedicated
      per-profile server.
    """
    working_dir = _stable_service_working_dir()
    hermes_home = str(get_hermes_home().resolve())
    log_dir = get_hermes_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    label = get_dashboard_launchd_label()
    venv_dir = _service_venv_dir()

    priority_dirs = _build_service_path_dirs()
    _append_node_dir_for_service(priority_dirs)
    sane_path = ":".join(dict.fromkeys(priority_dirs + [p for p in os.environ.get("PATH", "").split(":") if p]))

    profile_parts: list[str] = []
    suffix = _profile_suffix()
    if suffix:
        profile_parts.extend(["--profile", suffix, "--isolated"])
    else:
        profile_parts.extend(["-p", "default"])

    args: list[str] = [
        get_python_path(),
        "-m", "hermes_cli.main",
        *profile_parts,
        "dashboard",
        "--host", host,
        "--port", str(port),
        "--no-open",
        *(extra_args or []),
    ]

    prog_args_xml = "\n        ".join(
        f"<string>{escape(str(part))}</string>" for part in args
    )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{escape(label)}</string>

    <key>ProgramArguments</key>
    <array>
        {prog_args_xml}
    </array>

    <key>WorkingDirectory</key>
    <string>{escape(working_dir)}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{escape(sane_path)}</string>
        <key>VIRTUAL_ENV</key>
        <string>{escape(venv_dir)}</string>
        <key>HERMES_HOME</key>
        <string>{escape(hermes_home)}</string>
        <key>HERMES_SUPERVISED_CHILD</key>
        <string>1</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <!-- ThrottleInterval: 30s minimum between respawns prevents crash-loop storms. -->
    <key>ThrottleInterval</key>
    <integer>30</integer>

    <!-- ExitTimeOut: 25s graceful-drain headroom before SIGTERM -> SIGKILL. -->
    <key>ExitTimeOut</key>
    <integer>25</integer>

    <key>StandardOutPath</key>
    <string>{escape(str(log_dir))}/dashboard.log</string>

    <key>StandardErrorPath</key>
    <string>{escape(str(log_dir))}/dashboard.error.log</string>
</dict>
</plist>
"""


# ------------------------------------------------------------------
# Service lifecycle (macOS-gated; Linux prints a clear hint and exits)
# ------------------------------------------------------------------

def dashboard_service_install(host: str, port: int, extra_args: list[str] | None = None, *, force: bool = False) -> None:
    """Write the dashboard LaunchAgent plist and bootstrap it (mirrors gateway install)."""
    if not is_macos():
        print("Dashboard launchd service is only supported on macOS; on Linux use a systemd unit.")
        sys.exit(1)
    plist_path = get_dashboard_launchd_plist_path()
    if plist_path.exists() and not force:
        print(f"Dashboard service already installed at: {plist_path}")
        print("Use --force to reinstall.")
        return
    # Simpler flow: gateway install (gateway.py:3868-3896) writes, then bootstraps.
    # Only boot out before write if the plist existed (refresh semantics); here we
    # only reach write when missing or force=True, so no pre-bootout needed for missing.
    if plist_path.exists() and force:
        label = get_dashboard_launchd_label()
        domain = _launchd_domain()
        subprocess.run(["launchctl", "bootout", f"{domain}/{label}"], check=False, timeout=30)
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(generate_dashboard_launchd_plist(host, port, extra_args=extra_args), encoding="utf-8")
    label = get_dashboard_launchd_label()
    domain = _launchd_domain()
    try:
        _launchctl_bootstrap(domain, plist_path, label, timeout=30)
    except subprocess.CalledProcessError as e:
        _degrade_dashboard_launchctl_error(e, "launchctl bootstrap")
        return
    print(f"Dashboard service installed at: {plist_path}")


def dashboard_service_start(host: str, port: int, extra_args: list[str] | None = None) -> None:
    """Kickstart the dashboard service; self-heal if plist is missing (mirrors gateway start)."""
    if not is_macos():
        print("Dashboard launchd service is only supported on macOS; on Linux use a systemd unit.")
        sys.exit(1)
    plist_path = get_dashboard_launchd_plist_path()
    label = get_dashboard_launchd_label()

    # Self-heal when plist is missing (gateway.py:3916-3927).
    if not plist_path.exists():
        print("↻ Dashboard launchd plist missing; regenerating service definition")
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_path.write_text(generate_dashboard_launchd_plist(host, port, extra_args=extra_args), encoding="utf-8")
        # After regeneration, bootstrap then kickstart (gateway.py:3925-3926).
        domain = _launchd_domain()
        try:
            _launchctl_bootstrap(domain, plist_path, label, timeout=30)
            _launchctl_kickstart_current(label)
        except subprocess.CalledProcessError as e:
            if _launchd_error_indicates_unloaded(e):
                # Unloaded after regeneration: retry bootstrap then kickstart.
                try:
                    _launchctl_bootstrap(domain, plist_path, label, timeout=30)
                    _launchctl_kickstart_current(label)
                except subprocess.CalledProcessError as e2:
                    _degrade_dashboard_launchctl_error(e2, "launchctl bootstrap after self-heal")
                    return
            else:
                _degrade_dashboard_launchctl_error(e, "launchctl bootstrap (self-heal)")
                return
        print("Dashboard service started (regenerated).")
        return

    # Plist exists: kickstart first (gateway.py:3929-3931). Never bootstrap a running service.
    domain = _launchd_domain()
    try:
        _launchctl_kickstart_current(label)
    except subprocess.CalledProcessError as e:
        if _launchd_error_indicates_unloaded(e):
            # Job not loaded: re-bootstrap then re-kickstart (gateway.py:3936-3938).
            print("↻ Dashboard launchd job was unloaded; reloading service definition")
            try:
                _launchctl_bootstrap(domain, plist_path, label, timeout=30)
            except subprocess.CalledProcessError as e_boot:
                _degrade_dashboard_launchctl_error(e_boot, "launchctl bootstrap")
                return
            try:
                _launchctl_kickstart_current(label)
            except subprocess.CalledProcessError as e_kick:
                raise e_kick
        else:
            raise
    print("Dashboard service started.")


def dashboard_service_stop() -> None:
    if not is_macos():
        print("Dashboard launchd service is only supported on macOS; on Linux use a systemd unit.")
        sys.exit(1)
    label = get_dashboard_launchd_label()
    domain = _launchd_domain()
    subprocess.run(["launchctl", "bootout", f"{domain}/{label}"], check=False, timeout=30)
    print("Dashboard service stopped.")


def dashboard_service_restart() -> None:
    if not is_macos():
        print("Dashboard launchd service is only supported on macOS; on Linux use a systemd unit.")
        sys.exit(1)
    label = get_dashboard_launchd_label()
    domain = _launchd_domain()
    try:
        subprocess.run(["launchctl", "kickstart", "-k", f"{domain}/{label}"], check=True, timeout=90)
        print("Dashboard service restarted.")
    except subprocess.CalledProcessError:
        print(f"Failed to restart; try: launchctl kickstart -k {domain}/{label}")
        raise


def dashboard_service_uninstall() -> None:
    if not is_macos():
        print("Dashboard launchd service is only supported on macOS; on Linux use a systemd unit.")
        sys.exit(1)
    plist_path = get_dashboard_launchd_plist_path()
    label = get_dashboard_launchd_label()
    domain = _launchd_domain()
    subprocess.run(["launchctl", "bootout", f"{domain}/{label}"], check=False, timeout=30)
    if plist_path.exists():
        # Only delete if the parsed Label equals the current label (namespace guard).
        try:
            parsed_label = plistlib.loads(plist_path.read_bytes()).get("Label")
        except Exception as exc:
            print(f"Parse error reading {plist_path}: {exc}; skipping unlink.")
            sys.exit(1)
        if parsed_label == label:
            plist_path.unlink()
            print(f"Removed {plist_path}")
        else:
            print(f"Namespace guard: plist label '{parsed_label}' does not match '{label}'; skipping unlink.")
    print("Dashboard service uninstalled.")


def dashboard_service_status() -> None:
    """Print launchd state + HTTP /api/status probe result."""
    if not is_macos():
        print("Dashboard launchd service is only supported on macOS; on Linux use a systemd unit.")
        sys.exit(1)
    plist_path = get_dashboard_launchd_plist_path()
    label = get_dashboard_launchd_label()

    host: str | None = None
    port: int | None = None
    if plist_path.exists():
        try:
            plist_data = plistlib.loads(plist_path.read_bytes())
            prog_args = plist_data.get("ProgramArguments", [])
            for i, arg in enumerate(prog_args):
                if arg == "--host" and i + 1 < len(prog_args):
                    host = prog_args[i + 1]
                elif arg == "--port" and i + 1 < len(prog_args):
                    try:
                        port = int(prog_args[i + 1])
                    except ValueError as exc:
                        print(f"Could not parse port from plist argument '{prog_args[i + 1]}': {exc}")
        except Exception as exc:
            print(f"Could not read installed plist: {exc}")
    else:
        print("Dashboard service is not installed (no plist found).")
        sys.exit(1)

    domain = _launchd_domain()
    # Reuse gateway parser directly (F3).
    loaded, pid = _launchd_print_service_pid(domain, label)
    if loaded and pid is not None and pid > 0:
        print(f"Dashboard service registered with launchd: {label}")
        print(f"Supervising PID: {pid}")
    elif loaded:
        print(f"Dashboard service registered with launchd: {label} (not running)")
    else:
        print(f"Dashboard service not registered with launchd: {label}")

    # HTTP probe with normalized loopback host (F6 / F3).
    if host is not None and port is not None:
        probe_host = _dashboard_probe_host(host)
        url = f"http://{probe_host}:{port}/api/status"
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                print(f"Dashboard HTTP up ({resp.status}) at {url}")
        except urllib.error.HTTPError as exc:
            # Any HTTP response (including 401) = server is up.
            print(f"Dashboard HTTP responding ({exc.code}) at {url}")
        except Exception as exc:
            print(f"Dashboard HTTP down (connection error): {exc}")
    else:
        print("Could not determine host/port from installed plist for HTTP probe.")


# ------------------------------------------------------------------
# CLI dispatcher for `hermes dashboard service <verb>` (issue #44106)
# ------------------------------------------------------------------

_SERVICE_VERBS: dict[str, Callable[[argparse.Namespace], None]] = {
    "install": lambda a: dashboard_service_install(
        a.host, a.port,
        extra_args=(["--skip-build"] if getattr(a, "skip_build", False) else None),
        force=getattr(a, "force", False),
    ),
    "start": lambda a: dashboard_service_start(
        a.host, a.port,
        extra_args=(["--skip-build"] if getattr(a, "skip_build", False) else None),
    ),
    "stop": lambda a: dashboard_service_stop(),
    "restart": lambda a: dashboard_service_restart(),
    "status": lambda a: dashboard_service_status(),
    "uninstall": lambda a: dashboard_service_uninstall(),
}


def dashboard_service_command(args) -> None:
    """Table-driven dispatcher: 6 verbs, zero elif ladders (issue #44106)."""
    verb = getattr(args, "dashboard_service_command", None)
    if verb is None:
        raise SystemExit("dashboard service requires a verb (install/start/stop/restart/status/uninstall)")
    handler = _SERVICE_VERBS.get(verb)
    if handler is None:
        raise SystemExit(f"Unknown dashboard service command: {verb}")
    handler(args)
