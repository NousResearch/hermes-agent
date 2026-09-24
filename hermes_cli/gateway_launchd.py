"""Gateway launchd (macOS LaunchAgent) backend: plist generation/refresh, bootstrap, start/stop/restart/status.

Extracted from ``hermes_cli/gateway.py``. Bodies read facade helpers through ``_gw()`` (late
binding on ``hermes_cli.gateway``) so the seams tests and callers patch on the facade keep
intercepting the moved code.
"""
from __future__ import annotations

from pathlib import Path
import contextlib
import json
import os
import shlex
import subprocess
import sys
import time
from xml.sax.saxutils import escape

from gateway.restart import LAUNCHD_GUI_EXIT_TIMEOUT_CLAMP_S


def _gw():
    from hermes_cli import gateway  # late: the facade imports this module
    return gateway


def get_launchd_label() -> str:
    """Return the launchd service label, scoped per profile."""
    suffix = _gw()._profile_suffix()
    return f"ai.hermes.gateway-{suffix}" if suffix else "ai.hermes.gateway"


def _probe_launchd_domain_for_label(label: str) -> str:
    """Launchd domain managing ``label`` (uncached): ``gui/<uid>`` (Aqua), then ``user/<uid>``
    (Background/SSH), else the ``launchctl managername`` heuristic. Sibling profiles may live in
    different domains, so never reuse the cached ``_launchd_domain()`` for another label."""
    uid = os.getuid()  # windows-footgun: ok — POSIX launchd (macOS) helper, never invoked on Windows
    gui_domain, user_domain = f"gui/{uid}", f"user/{uid}"

    launchctl_errors = (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError)
    for domain in (gui_domain, user_domain):
        try:
            subprocess.run(["launchctl", "print", f"{domain}/{label}"], check=True, timeout=5, capture_output=True)
            return domain
        except launchctl_errors:
            pass

    # Not loaded anywhere: Aqua → gui/<uid>; anything else (Background, loginwindow) → user/<uid>,
    # the pre-probing default and the recommended domain on macOS 26+.
    try:
        result = subprocess.run(["launchctl", "managername"], timeout=5, **_gw()._CAPTURE_TEXT)
        if "Aqua" in (result.stdout or ""):
            return gui_domain
    except launchctl_errors:
        pass
    return user_domain


def _launchd_domain() -> str:
    """Domain managing the current profile's gateway; cached per process so start/stop/restart agree.

    See #40831, #23387.
    """
    # The cache lives on the facade: tests and callers reset ``hermes_cli.gateway._resolved_launchd_domain``.
    gw = _gw()
    if gw._resolved_launchd_domain is None:
        gw._resolved_launchd_domain = _probe_launchd_domain_for_label(gw.get_launchd_label())
    return gw._resolved_launchd_domain


# 125 ("Domain does not support specified action") and 3/113 ("Could not find service") all mean
# the job isn't loaded in the target domain: re-bootstrap the plist and retry.
_LAUNCHD_JOB_UNLOADED_EXIT_CODES = frozenset({3, 113, 125})


# 5 (EIO) / persistent 125 mean either a stale still-registered label (recoverable: bootout +
# bootstrap, which `_launchctl_bootstrap()` tries first) or a domain that genuinely can't manage
# services (macOS 26+). Only when the retry ALSO fails do callers degrade to a detached process.
# launchctl returns 5 ("Input/output error") or a persistent 125 in two very different situations, so exit 5
# is NOT on its own proof the domain is broken: 1. See #42914. 2. Here launchd cannot supervise the gateway
# at all and we degrade to a detached background process (the `nohup hermes gateway run` workaround). See
# #23387.
_LAUNCHCTL_DOMAIN_UNSUPPORTED_CODES = frozenset({5, 125})


def _launchd_error_indicates_unloaded(exc: subprocess.CalledProcessError) -> bool:
    """True when launchctl failed because the job isn't loaded (retry bootstrap)."""
    return exc.returncode in _LAUNCHD_JOB_UNLOADED_EXIT_CODES


def _launchctl_domain_unsupported(returncode: int) -> bool:
    """True when launchctl can't manage the domain even after a fresh bootstrap (macOS 26+) — degrade to detached."""
    return returncode in _LAUNCHCTL_DOMAIN_UNSUPPORTED_CODES


# EIO from `launchctl bootstrap` = label *already* registered (stale load); recoverable, not an unmanageable domain.
_LAUNCHCTL_BOOTSTRAP_EIO = 5


def _launchctl_bootstrap(domain: str, plist_path, label: str, *, timeout: int = 30) -> None:
    """Bootstrap a launchd job, recovering from a stale still-registered label (EIO 5). Without the
    bootout + retry that case is misread as an unmanageable domain and degrades to detached, silently
    losing auto-start and crash-restart."""
    bootstrap = ["launchctl", "bootstrap", domain, str(plist_path)]
    try:
        subprocess.run(bootstrap, check=True, timeout=timeout)
    except subprocess.CalledProcessError as exc:
        if exc.returncode != _LAUNCHCTL_BOOTSTRAP_EIO:
            raise
        # Stale registration — bootout the leftover label and bootstrap once more.
        # Captured: the bootout is best-effort (a drained job may already be
        # unloaded), so its expected 3/113/125 stderr must not leak to the terminal.
        subprocess.run(
            ["launchctl", "bootout", f"{domain}/{label}"],
            check=False, timeout=timeout, **_gw()._CAPTURE_TEXT)
        subprocess.run(bootstrap, check=True, timeout=timeout)


def _launchd_reload_log_path() -> Path:
    """Path the launchd reload watchdog tails for persistent-orphan detection."""
    return _gw().get_hermes_home() / "logs" / "launchd-reload.log"


def _append_launchd_reload_log(message: str) -> None:
    """Append a timestamped line to the launchd reload log (best-effort)."""
    path = _launchd_reload_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime as _dt
        stamp = _dt.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] {message}\n")
    except OSError:
        pass


def _launchd_reload_budget() -> float:
    """Bootstrap retry window for a plist reload: the failure happens while the old gateway is still
    draining (default 180s), so size it to the drain timeout with a 30s floor."""
    return max(30.0, _gw()._get_restart_drain_timeout())


def _launchctl_supervised_pid(label: str) -> int | None:
    """PID launchd currently runs for ``label``, or None when it runs none. ``launchctl list`` exits 0 for
    a mere registered definition (``state = not running`` on macOS 26+), so a PID — not the exit code — is
    the answer. Domain-agnostic on purpose: ``launchctl print`` domain probes fail on macOS-26 per-user
    domains, which is why the invoking profile verifies through this and not ``_launchd_print_service_pid``."""
    try:
        result = subprocess.run(["launchctl", "list", label], check=False, timeout=10, **_gw()._CAPTURE_TEXT)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return _gw()._parse_launchd_pid_from_list_output(result.stdout)


def _launchctl_label_supervising_process(label: str) -> bool:
    """True when launchd knows ``label`` AND runs a process for it."""
    return _gw()._launchctl_supervised_pid(label) is not None


def _retry_launchctl_bootstrap_until_registered(
    domain: str, plist_path, label: str, *, deadline: float
) -> bool:
    """Retry ``_launchctl_bootstrap`` until the label supervises a process or ``deadline`` passes. Under
    load bootstrap can fail even after bootout, during a drain (default 180s) — ~10s is too short."""
    attempt = 0
    while True:
        attempt += 1
        try:
            _gw()._launchctl_bootstrap(domain, plist_path, label, timeout=30)
            if _gw()._launchctl_label_supervising_process(label):
                return True
            outcome = f"exited 0 but {domain}/{label} has no supervised process (launchctl list)"
        except subprocess.CalledProcessError as exc:
            outcome = f"failed (rc={exc.returncode}) for {domain}/{label}"
        except subprocess.TimeoutExpired:
            outcome = f"timed out for {domain}/{label}"
        _gw()._append_launchd_reload_log(f"bootstrap attempt {attempt} {outcome} — retrying")
        if time.monotonic() >= deadline:
            return False
        time.sleep(2)


# launchd-unsupported marker: written when the domain can't be managed (exit 5/125, macOS 26+) so
# `launchd_status()` can explain missing supervision; cleared on successful bootstrap/kickstart.
def _launchd_unsupported_marker_path() -> Path:
    return _gw().get_hermes_home() / ".gateway-launchd-unsupported"


def _write_launchd_unsupported_marker() -> None:
    """Persist that launchd cannot supervise the gateway on this host."""
    from datetime import datetime, timezone
    payload = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "reason": "launchd domain unsupported (exit 5/125)",
    }
    with contextlib.suppress(OSError):
        _launchd_unsupported_marker_path().write_text(json.dumps(payload), encoding="utf-8")


def _clear_launchd_unsupported_marker() -> None:
    """Clear the unsupported marker when launchd bootstrap succeeds."""
    with contextlib.suppress(OSError):
        _launchd_unsupported_marker_path().unlink(missing_ok=True)


def _launchd_unsupported_marker_exists() -> bool:
    return _launchd_unsupported_marker_path().exists()


def _gateway_run_command() -> list[str]:
    """Build ``python -m hermes_cli.main [--profile X] gateway run --replace``, honoring the active profile."""
    return [_gw().get_python_path(), "-m", "hermes_cli.main", *_gw()._profile_arg().split(), "gateway", "run", "--replace"]


def launchd_program_arguments(command: list[str], stdout_log: Path, stderr_log: Path) -> list[str]:
    """launchd ``ProgramArguments`` that run ``command`` with a Local Network identity macOS accepts (#71206).

    macOS Local Network Privacy attributes a socket to the process launchd spawned for the job. A bare
    venv Python has no application ID and is not platform-entitled, so every LAN connect from the
    launchd gateway dies with ``EHOSTUNREACH`` while the same code works from Terminal (whose grant it
    inherits). An ad-hoc-signed helper .app does not help: nehelper never prompts for it and denies
    (#57812 dead-end table, re-verified live on macOS 26.3). ``/usr/bin/osascript``'s ``do shell script``
    spawns its child as osascript-responsible — an Apple platform binary — so the child is exempt;
    ``/bin/sh -c exec …`` and ``/usr/bin/time`` wrappers are NOT (the launchd job identity is the
    non-entitled first executable). ``do shell script`` buffers the child's stdout/stderr until it exits,
    so the command appends both to the same files the plist's ``StandardOutPath``/``StandardErrorPath``
    name (those keys stay: they are where osascript's own output lands — an empty result line per exit
    and an un-timestamped ``execution error`` line on non-zero exit); ``exec`` keeps the
    gateway a direct child in the job's process group, so ``launchctl bootout`` / ``kickstart -k`` still
    deliver SIGTERM to it and KeepAlive's ``SuccessfulExit`` semantics are preserved (osascript exits 0
    exactly when the shell did).
    """
    shell = f"exec {shlex.join(command)} >> {shlex.quote(str(stdout_log))} 2>> {shlex.quote(str(stderr_log))}"
    applescript = shell.replace("\\", "\\\\").replace('"', '\\"')
    return ["/usr/bin/osascript", "-e", f'do shell script "{applescript}"']


def _timestamped_stderr_gateway_command(error_log: Path, *, external_supervisor: bool = False) -> list[str]:
    """Wrap gateway run so raw stderr lines are timestamped before file write. ``external_supervisor``
    (launchd ProgramArguments only) adds ``--external-supervisor`` so ``hermes update`` hands back to
    launchd, and drops ``--replace``: KeepAlive respawns would re-arm takeover, so two profiles sharing
    a token would kill each other forever.

    ``external_supervisor=True`` is for launchd ProgramArguments only: the inner ``gateway run`` must carry
    ``--external-supervisor`` so ``hermes update`` sees the flag on the live grandchild argv and hands the
    process back to launchd instead of starting a detached watcher (#86893 / #87005). The detached nohup
    fallback stays unmarked.
    Supervised starts also drop ``--replace`` (issue #79048): a launchd service is respawned by KeepAlive,
    so takeover authority would be re-armed on every respawn — two profiles legitimately sharing one
    platform token would each terminate the sibling, and launchd would revive the victim forever. Bounded
    replacement is the lifecycle commands' job (``launchctl kickstart -k``, drain in ``launchd_restart()``,
    bootout+bootstrap in install/refresh), which run before supervision resumes. Mirrors
    ``generate_systemd_unit``, whose ExecStart also runs ``gateway run`` without ``--replace``.
    """
    inner = _gw()._gateway_run_command()
    if external_supervisor:
        inner = [part for part in inner if part != "--replace"]
        if "--external-supervisor" not in inner:
            inner.append("--external-supervisor")
    return [_gw().get_python_path(), "-m", "hermes_cli.stderr_timestamp", "--error-log", str(error_log), "--", *inner]


def _spawn_detached_gateway() -> bool:
    """Launch the gateway detached (launchd fallback for macOS 26+). CLI-managed nohup equivalent:
    stdout → gateway.log, timestamped stderr → gateway.error.log, PID via gateway.pid so stop/status work.

    Used when launchctl can no longer bootstrap/kickstart the gateway on macOS 26+ (issue #23387). Mirrors
    the `nohup hermes gateway run --replace` workaround but keeps it CLI-managed: stdout goes to
    gateway.log, stderr is timestamped into gateway.error.log, and the PID is tracked via the gateway.pid
    file that `run_gateway` writes, so stop/status/restart keep working.
    """
    from hermes_cli._subprocess_compat import windows_detach_popen_kwargs
    log_dir = _gw().get_hermes_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_dir / "gateway.log", "ab") as out:
            subprocess.Popen(
                _timestamped_stderr_gateway_command(log_dir / "gateway.error.log"),
                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.DEVNULL,
                **windows_detach_popen_kwargs(),
            )
    except OSError:
        return False
    return True


def _launchd_fallback_to_detached(reason: str, *, exit_on_failure: bool = True) -> bool:
    """Start the gateway detached when launchd can't manage it; on failure print the manual workaround
    and (by default) exit 1."""
    from hermes_constants import display_hermes_home as _dhh
    _gw()._write_launchd_unsupported_marker()
    print(f"⚠ launchd cannot manage the gateway on this macOS version ({reason}).")
    if _gw()._spawn_detached_gateway():
        print("✓ Started gateway as a background process instead")
        print("  It will NOT auto-start at login or auto-restart on crash.")
        print(f"  Logs: {_dhh()}/logs/gateway.log")
        print("  Stop it with: hermes gateway stop")
        return True
    _gw().print_error("Failed to start the gateway as a background process.")
    print(f"  Try manually: nohup hermes gateway run --replace > {_dhh()}/logs/gateway.log 2>&1 &")
    if exit_on_failure:
        sys.exit(1)
    return False


def _launchd_degrade_or_raise(exc: subprocess.CalledProcessError, what: str) -> None:
    """Shared launchctl failure policy: domain unmanageable (5/125) → detached fallback; else re-raise.

    A 5/125 exit is evidence about the *domain* only when launchd is not already supervising this job.
    EIO (5) is ``launchctl bootstrap``'s answer for a label that is already loaded, so the ordinary
    "reinstall/restart over the live gateway" case lands here with the service up and supervised.
    Degrading there is not a graceful fallback: it writes the permanent launchd-unsupported marker and
    starts a detached gateway *beside* the supervised one, and the marker makes
    :func:`wait_for_launchd_gateway_supervision` answer True unconditionally — so no later
    install/update can tell that nothing ties the gateway to launchd any more. A live supervised PID is
    direct evidence this macOS does manage the job, so surface the failure instead of branding the host.
    """
    if not _launchctl_domain_unsupported(exc.returncode):
        raise exc
    label = _gw().get_launchd_label()
    if _gw()._launchctl_label_supervising_process(label):
        print(f"⚠ {what} failed (exit {exc.returncode}), but launchd still supervises {label}")
        print("  Not switching to the detached fallback — this host manages the job.")
        print("  Apply the definition with: hermes gateway stop && hermes gateway install --force")
        raise exc
    _launchd_fallback_to_detached(f"{what} exit {exc.returncode}")


def generate_launchd_plist() -> str:
    # Stable cwd anchor — never the volatile source checkout (same rot risk as systemd's WorkingDirectory).
    working_dir = _gw()._stable_service_working_dir()
    hermes_home = str(_gw().get_hermes_home().resolve())
    log_dir = _gw().get_hermes_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    label = _gw().get_launchd_label()
    venv_dir = _gw()._service_venv_dir()
    # launchd's default PATH misses Homebrew, nvm, cargo…; prepend venv/bin + node dirs (as in the
    # systemd unit) so node stays resolvable even if the shell PATH changes, then the shell PATH.
    priority_dirs = _gw()._build_service_path_dirs()
    _gw()._append_node_dir_for_service(priority_dirs)
    sane_path = ":".join(dict.fromkeys(priority_dirs + [p for p in os.environ.get("PATH", "").split(":") if p]))

    # ProgramArguments (incl. --profile); the stderr wrapper keeps launchd restart semantics while timestamping
    # stderr; the osascript wrapper gives the job a Local Network identity (see launchd_program_arguments).
    stdout_log, stderr_log = log_dir / "gateway.log", log_dir / "gateway.error.log"
    command = _timestamped_stderr_gateway_command(stderr_log, external_supervisor=True)
    prog_args_xml = "\n        ".join(
        f"<string>{escape(part)}</string>" for part in launchd_program_arguments(command, stdout_log, stderr_log)
    )

    # Persist the configured RLIMIT_NOFILE floor: launchd defaults to soft 256, and every plist
    # rewrite would otherwise strip a manual limit and reintroduce EMFILE crashes.
    nofile_block = ""
    try:
        from hermes_cli.resource_limits import configured_nofile_soft_limit
        nofile_target = configured_nofile_soft_limit()
    except Exception:
        nofile_target = None
    if nofile_target:
        nofile_block = f"""
    <key>SoftResourceLimits</key>
    <dict>
        <key>NumberOfFiles</key>
        <integer>{nofile_target}</integer>
    </dict>
"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
        {prog_args_xml}
    </array>
    
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{sane_path}</string>
        <key>VIRTUAL_ENV</key>
        <string>{venv_dir}</string>
        <key>HERMES_HOME</key>
        <string>{hermes_home}</string>
        <key>HERMES_SUPERVISED_CHILD</key>
        <string>1</string>
    </dict>

    <key>LimitLoadToSessionType</key>
    <array>
        <string>Aqua</string>
        <string>Background</string>
    </array>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <!-- SuccessfulExit=false parks a clean stop (exit 0), including gateway
         EX_CONFIG 78 after stderr_timestamp maps it to 0 — launchd cannot
         honor RestartPreventExitStatus, and KeepAlive=true respawned token
         collisions forever (#89477). Exit 75 and crashes still relaunch.
         ThrottleInterval raises launchd's default 10s minimum respawn interval
         to 30s so a crash-looping gateway can't hammer launchd into a rapid
         respawn storm; ExitTimeOut is the graceful-drain headroom before
         launchd escalates from SIGTERM to SIGKILL on stop. The per-user
         (gui) launchd domain clamps it to 60s regardless of what is written
         here, so 60 is the most a LaunchAgent can get; the gateway reads
         the live value at boot and fits its signal-driven drain inside it
         (gateway.restart.read_launchd_exit_timeout_s). -->
    <key>ThrottleInterval</key>
    <integer>30</integer>

    <key>ExitTimeOut</key>
    <integer>60</integer>
{nofile_block}
    <key>StandardOutPath</key>
    <string>{stdout_log}</string>
    
    <key>StandardErrorPath</key>
    <string>{stderr_log}</string>
</dict>
</plist>
"""


def launchd_plist_is_current() -> bool:
    """Check if the installed launchd plist matches the currently generated one."""
    plist_path = _gw().get_launchd_plist_path()
    if not plist_path.exists():
        return False
    installed = plist_path.read_text(encoding="utf-8")
    norm = _gw()._normalize_launchd_plist_for_comparison
    return norm(installed) == norm(_gw().generate_launchd_plist())


def _spawn_deferred_launchd_reload(
    *, domain: str, label: str, target: str, plist_path: Path, gateway_pid: int
) -> bool:
    """Hand the bootout/bootstrap cycle to a transient ``launchctl submit`` job; True if spawned. The
    helper waits for the OLD gateway to exit (bootstrap during drain fails EIO), then retries bootstrap
    until ``launchctl list`` shows a positive PID or the drain budget elapses."""
    reload_log_path = _launchd_reload_log_path()
    with contextlib.suppress(OSError):
        reload_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Durable pre-bootout marker: distinguishes "helper never started" from "helper ran but failed".
    _gw()._append_launchd_reload_log(f"Launchd reload helper started for {target}")

    _reload_budget = int(_launchd_reload_budget())
    q_target, q_label, q_log = shlex.quote(target), shlex.quote(label), shlex.quote(str(reload_log_path))
    stamp = "$(date '+%Y-%m-%d %H:%M:%S %z')"
    # Require a POSITIVE PID: `launchctl list` also exits 0 for a registered-but-not-running
    # definition, and a crashed job reports `"PID" = -1` (mirrors _parse_launchd_pid_from_list_output).
    listed = f"launchctl list {q_label} 2>/dev/null | grep -qE '\\\"PID\\\" = [0-9]+;'"
    # Unique per reload so concurrent/repeated reloads never collide.
    submit_label = f"{label}.reload.{os.getpid()}.{int(time.time())}"
    reload_script = (
        f"sleep 2; "
        f"launchctl bootout {q_target} 2>/dev/null; "
        # Wait for the OLD gateway to exit: bootout only SIGTERMs and every bootstrap during the drain fails EIO.
        f"_wait_deadline=$(($(date +%s) + {_reload_budget})); "
        f"while kill -0 {gateway_pid} 2>/dev/null; do   if [ $(date +%s) -ge $_wait_deadline ]; then "
        f"    echo \"[{stamp}] old gateway pid {gateway_pid} still alive after {_reload_budget}s drain wait — bootstrapping anyway\" >> {q_log}; "
        f"    break;   fi;   sleep 1; done; "
        # Let launchd finish unregistering the label after the process exits.
        f"sleep 1; _deadline=$(($(date +%s) + {_reload_budget})); while :; do "
        f"  launchctl bootstrap {shlex.quote(domain)} {shlex.quote(str(plist_path))} 2>/dev/null; "
        f"  if {listed}; then break; fi; "
        f"  echo \"[{stamp}] bootstrap not yet registered for {q_target} — retrying\" >> {q_log}; "
        f"  if [ $(date +%s) -ge $_deadline ]; then break; fi;   sleep 2; done; "
        f"if ! {listed}; then "
        f"  echo \"[{stamp}] FAILED launchd reload for {q_target} — service NOT registered after {_reload_budget}s of retries\" >> {q_log}; "
        f"fi; "
        # Submitted jobs stay registered after the script exits; removing our own label ends the one-shot job.
        f"launchctl remove {shlex.quote(submit_label)} 2>/dev/null"
    )
    try:
        # `launchctl submit` rather than setsid: setsid does NOT leave the launchd coalition that bootout kills.
        # Spawn the reload helper via `launchctl submit` (a transient launchd one-shot job) instead of
        # `start_new_session=True`. `start_new_session=True` only calls setsid(2), which creates a new POSIX
        # session but does NOT move the child outside the launchd job's process coalition. When `launchctl
        # bootout` fires on the gateway label, launchd terminates ALL processes in that coalition —
        # including a setsid-detached child (#69098). `launchctl submit` creates a wholly independent
        # transient launchd job that launchd manages separately from the gateway, so bootout of the gateway
        # job cannot reach the helper.
        subprocess.Popen(
            [
                "launchctl", "submit", "-l", submit_label, "-o", str(reload_log_path), "-e", str(reload_log_path),
                "--", "/bin/bash", "-c", reload_script,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        # Fall through to in-process bootout/bootstrap: risky in the coalition, but better than a never-reloaded plist.
        _gw().logger.warning("Deferred launchd reload could not be spawned: %s", e)
        _gw()._append_launchd_reload_log(
            f"FAILED to spawn launchd reload helper for {target}: {e} — falling back to in-process bootout/bootstrap"
        )
        return False
    return True


def refresh_launchd_plist_if_needed() -> bool:
    """Rewrite the installed plist when the generated one differs, then bootout/bootstrap so launchd
    re-reads it immediately."""
    plist_path = _gw().get_launchd_plist_path()
    if not plist_path.exists() or _gw().launchd_plist_is_current():
        return False

    new_plist = _gw().generate_launchd_plist()
    if _gw()._refuse_temp_home_service_write(new_plist, "launchd plist"):
        return False

    plist_path.write_text(new_plist, encoding="utf-8")
    label = _gw().get_launchd_label()
    domain = _gw()._launchd_domain()
    target = f"{domain}/{label}"

    # Inside the gateway's launchd process tree (agent self-update) a direct bootout kills THIS CLI
    # before bootstrap runs, leaving the job unloaded with no KeepAlive.
    try:
        from gateway.status import get_running_pid
        gateway_pid = get_running_pid()
    except Exception:
        gateway_pid = None

    # POSIX ancestry is NOT a reliable "bootout will kill us" test (coalition membership survives
    # reparenting), so always prefer the detached helper; in-process is only the spawn-failure fallback.
    if (
        gateway_pid is not None
        and hasattr(os, "setsid")  # POSIX-only; launchd is macOS so always true here
    ) and _spawn_deferred_launchd_reload(
        domain=domain, label=label, target=target, plist_path=plist_path, gateway_pid=gateway_pid
    ):
        print(
            "↻ Updated gateway launchd service definition; reload deferred to "
            "a transient launchd job (survives the bootout of this process)"
        )
        return True

    # Bootout/bootstrap so launchd reads the new definition; bootstrap can fail silently under load
    # during a drain, and KeepAlive can't revive an unregistered job.
    # Captured: best-effort (the job may already be unloaded), keep expected noise off the terminal.
    subprocess.run(["launchctl", "bootout", target], check=False, timeout=90, **_gw()._CAPTURE_TEXT)
    _reload_budget = _launchd_reload_budget()
    # Wait out the old gateway's drain first so the budget isn't burned on guaranteed EIO ("already loaded").
    if gateway_pid is not None and not _gw()._wait_for_pid_exit(gateway_pid, _reload_budget):
        _gw()._append_launchd_reload_log(
            f"old gateway pid {gateway_pid} still alive after "
            f"{int(_reload_budget)}s drain wait — bootstrapping {target} anyway"
        )
    _deadline = time.monotonic() + _reload_budget
    if not _gw()._retry_launchctl_bootstrap_until_registered(domain, plist_path, label, deadline=_deadline):
        _gw()._append_launchd_reload_log(
            f"FAILED launchd reload of {target} — service NOT registered after "
            f"retrying for {int(_reload_budget)}s (in-process fallback path)"
        )
        _gw().logger.error(
            "launchd reload of %s failed — service not registered after %ds of retries; see %s",
            target, int(_reload_budget), _launchd_reload_log_path(),
        )
        return False
    print("↻ Updated gateway launchd service definition to match the current Hermes install")
    return True


def launchd_install(force: bool = False, *, start_now: bool = True):
    plist_path = _gw().get_launchd_plist_path()
    label = _gw().get_launchd_label()
    # Loading the plist starts the gateway (RunAtLoad), so a no-start install writes it without
    # loading it. A gateway that launchd already runs is still reloaded; this install did not start it.
    load = start_now or _gw()._launchctl_label_supervising_process(label)

    if plist_path.exists() and not force:
        if _gw().launchd_plist_is_current():
            print(f"Service already installed at: {plist_path}")
            print("Use --force to reinstall")
            return
        if load:
            print(f"↻ Repairing outdated launchd service at: {plist_path}")
            if _gw().refresh_launchd_plist_if_needed():
                print("✓ Service definition updated")
            else:
                # The plist was rewritten but launchd never registered it (or the write was refused):
                # a success line here would hide an unloaded service with no KeepAlive.
                from hermes_constants import display_hermes_home
                print(
                    "⚠ Service definition could not be reloaded with launchd. "
                    "Run 'hermes gateway install --force' or check "
                    f"{display_hermes_home()}/logs/launchd-reload.log for details."
                )
            return

    plist_path.parent.mkdir(parents=True, exist_ok=True)
    new_plist = _gw().generate_launchd_plist()
    if _gw()._refuse_temp_home_service_write(new_plist, "launchd plist"):
        return
    print(f"Installing launchd service to: {plist_path}")
    plist_path.write_text(new_plist, encoding="utf-8")

    if not load:
        # A job left loaded but idle (a parked clean exit) keeps its old definition, and that is
        # what `hermes gateway start` would kickstart instead of loading this plist.
        subprocess.run(
            ["launchctl", "bootout", f"{_gw()._launchd_domain()}/{label}"],
            check=False, timeout=90, **_gw()._CAPTURE_TEXT)
        print()
        print("✓ Service installed, not started (launchd starts it at your next login)")
        print()
        print("Next steps:")
        print("  hermes gateway start              # Start it now")
        print("  hermes gateway status             # Check status")
        return

    try:
        _gw()._launchctl_bootstrap(_gw()._launchd_domain(), plist_path, label, timeout=30)
    except subprocess.CalledProcessError as e:
        _gw()._launchd_degrade_or_raise(e, "launchctl bootstrap")
        return

    print()
    print("✓ Service installed and loaded!")
    _gw()._clear_launchd_unsupported_marker()
    print()
    print("Next steps:")
    print("  hermes gateway status             # Check status")
    from hermes_constants import display_hermes_home as _dhh
    print(f"  tail -f {_dhh()}/logs/gateway.log  # View logs")


def launchd_uninstall():
    plist_path = _gw().get_launchd_plist_path()
    # Captured: uninstalling an already-unloaded job is fine — don't print Boot-out failed: 3.
    subprocess.run(
        ["launchctl", "bootout", f"{_launchd_domain()}/{get_launchd_label()}"],
        check=False, timeout=90, **_gw()._CAPTURE_TEXT)
    if plist_path.exists():
        plist_path.unlink()
        print(f"✓ Removed {plist_path}")
    print("✓ Service uninstalled")


def launchd_start():
    plist_path = _gw().get_launchd_plist_path()
    label = _gw().get_launchd_label()

    # Self-heal if the plist is missing entirely (e.g., manual cleanup, failed upgrade)
    if not plist_path.exists():
        new_plist = _gw().generate_launchd_plist()
        if _gw()._refuse_temp_home_service_write(new_plist, "launchd plist"):
            sys.exit(1)
        print("↻ launchd plist missing; regenerating service definition")
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_path.write_text(new_plist, encoding="utf-8")
        if _launchd_bootstrap_and_kickstart(plist_path, label):
            _launchd_ok("✓ Service started")
        return

    _gw().refresh_launchd_plist_if_needed()
    try:
        _launchctl_kickstart_current(label)
    except subprocess.CalledProcessError as e:
        if not _launchd_error_indicates_unloaded(e):
            raise
        # Job not loaded in this domain — re-bootstrap the plist and retry.
        print("↻ launchd job was unloaded; reloading service definition")
        if not _launchd_bootstrap_and_kickstart(plist_path, label):
            return
    _launchd_ok("✓ Service started")


def _launchctl_kickstart_current(label: str) -> None:
    subprocess.run(["launchctl", "kickstart", f"{_launchd_domain()}/{label}"], check=True, timeout=30)


def _launchd_bootstrap_and_kickstart(plist_path: Path, label: str) -> bool:
    """Bootstrap then kickstart; False after degrading to detached (domain unsupported). Other errors propagate."""
    try:
        _gw()._launchctl_bootstrap(_gw()._launchd_domain(), plist_path, label, timeout=30)
        _launchctl_kickstart_current(label)
    except subprocess.CalledProcessError as e:
        _gw()._launchd_degrade_or_raise(e, "launchctl")
        return False
    return True


def _launchd_ok(message: str) -> None:
    """Print a launchd success line and clear the unsupported marker (an OS fix recovers automatically)."""
    print(message)
    _gw()._clear_launchd_unsupported_marker()


def launchd_stop():
    target = f"{_launchd_domain()}/{get_launchd_label()}"
    _gw()._mark_planned_stop()
    # bootout unloads the definition so KeepAlive doesn't respawn; `hermes gateway start` re-bootstraps.
    try:
        # Captured: an already-unloaded job (3/113/125) is handled below, so launchctl's own
        # "Boot-out failed: 3" must not print around the ✓ line; e.stderr stays on the raised error.
        subprocess.run(["launchctl", "bootout", target], check=True, timeout=90, **_gw()._CAPTURE_TEXT)
    except subprocess.CalledProcessError as e:
        # Job already unloaded (3/113/125), or the domain can't be managed at all (5/125, macOS 26+
        # detached-fallback process, issue #23387) — in both cases just fall through to the PID-based kill
        # below.
        if not (_launchd_error_indicates_unloaded(e) or _launchctl_domain_unsupported(e.returncode)):
            raise
    _gw()._wait_for_gateway_exit(timeout=10.0, force_after=5.0)
    print("✓ Service stopped")


def _launchd_kickstart(label: str, domain: str) -> None:
    """``launchctl kickstart -k domain/label``; raises so callers own per-label failure accounting."""
    subprocess.run(["launchctl", "kickstart", "-k", f"{domain}/{label}"], check=True, timeout=90, **_gw()._CAPTURE_TEXT)


def _wait_for_launchd_service_pid(
    label: str, old_pid: int | None, timeout: float = 10.0, *, domain: str
) -> bool:
    """Poll ``domain/label`` (0.5s) until it runs on a fresh PID or ``timeout`` passes — KeepAlive respawn
    isn't instantaneous. launchctl ``TimeoutExpired`` propagates; callers own failure accounting."""
    deadline = time.monotonic() + max(timeout, 0.5)
    while True:
        _loaded, pid = _gw()._launchd_print_service_pid(domain, label)
        if pid is not None and pid > 0 and pid != old_pid:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)


def launchd_restart():
    label = _gw().get_launchd_label()
    domain = _gw()._launchd_domain()
    target = f"{domain}/{label}"
    from gateway.status import get_running_pid
    try:
        pid = get_running_pid()
        if pid is not None and _gw()._request_gateway_self_restart(pid):
            _launchd_ok("✓ Service restart requested")
            return
        if pid is not None and _gw().probe_gateway_loop_liveness(pid) == _gw().GATEWAY_LOOP_WEDGED:
            # Event loop provably dead: it can't process a graceful shutdown, so a full drain wait
            # only stalls the restart (and `hermes update`). Bounded SIGTERM → SIGKILL, ~10s.
            print(f"⚠ Gateway PID {pid} event loop is unresponsive — " "skipping drain and forcing a bounded stop...")
            _gw()._escalate_wedged_gateway(pid)
            pid = None
        if pid is not None:
            # Graceful in-band restart via SIGUSR1 (mirrors systemd); the budget covers both the idle wait
            # and the drain. A bare SIGTERM would lose the resume_pending handoff. Announce BEFORE waiting:
            # surfaces with no other feedback (desktop updater) read silence as "update stuck".
            wait_budget = _gw()._get_restart_exit_wait_budget()
            print(f"→ Stopping gateway (PID {pid}) — draining in-flight runs (up to {wait_budget:.0f}s)...")
            from hermes_cli.update_cmd_drain_report import drain_progress_reporter
            if _gw()._graceful_restart_via_sigusr1(pid, wait_budget, on_progress=drain_progress_reporter(budget_s=wait_budget)):
                # KeepAlive revives a planned exit, so do NOT kickstart (-k would kill the replacement) —
                # but a clean exit doesn't prove supervision, so verify a replacement PID appears first.
                if _gw()._wait_for_launchd_service_pid(label, pid, timeout=15.0, domain=domain):
                    _launchd_ok("✓ Service restart requested")
                    return
                print("⚠ launchd did not revive the gateway after its graceful exit — forcing restart")
            else:
                print(f"⚠ Gateway drain timed out after {wait_budget:.0f}s — forcing launchd restart")
        # Captured: an unloaded job (3/113/125) is the expected case below, which
        # prints its own ↻ line — and e.stderr feeds the update_cmd failure diagnostic.
        _gw()._wait_for_api_server_port_free()
        subprocess.run(["launchctl", "kickstart", "-k", target], check=True, timeout=90, **_gw()._CAPTURE_TEXT)
        _launchd_ok("✓ Service restarted")
    except subprocess.CalledProcessError as e:
        if not _launchd_error_indicates_unloaded(e):
            _gw()._launchd_degrade_or_raise(e, "launchctl kickstart")
            return
        # Job not loaded — bootstrap and start fresh
        print("↻ launchd job was unloaded; reloading")
        try:
            # After a drain the job is usually still registered (bootstrap would hit EIO): boot it out first.
            # Captured: best-effort (the job may already be unloaded after the drain),
            # so an expected Boot-out failed: 3 must not leak past the ↻ line below.
            subprocess.run(["launchctl", "bootout", target], check=False, timeout=90, **_gw()._CAPTURE_TEXT)
            plist_path = str(_gw().get_launchd_plist_path())
            subprocess.run(["launchctl", "bootstrap", _gw()._launchd_domain(), plist_path], check=True, timeout=30)
            subprocess.run(["launchctl", "kickstart", target], check=True, timeout=30)
        except subprocess.CalledProcessError as e2:
            _gw()._launchd_degrade_or_raise(e2, "launchctl")
            return
        _launchd_ok("✓ Service restarted")


# KeepAlive relaunches at most ~once per 10s, so a self-restart leaves the label pid-less that long.
LAUNCHD_SUPERVISION_VERIFY_TIMEOUT = 20.0


def wait_for_launchd_gateway_supervision(
    *,
    timeout: float = LAUNCHD_SUPERVISION_VERIFY_TIMEOUT,
    label: str | None = None,
    poll_interval: float = 0.5,
    old_pid: int | None = None,
    shutdown_timeout: float | None = None,
) -> bool:
    """Poll launchd until it supervises a live gateway; True at once if the detached fallback is active.
    ``launchd_restart`` returns once the restart is *requested* (asynchronous), so it can't see a helper
    dying before bootstrap or a ``launchctl bootstrap`` that exits 0 without registering.

    The ``_request_gateway_self_restart`` branch hands the work to the running gateway and returns
    immediately, and a plist reload is handed to a detached helper. Both are asynchronous, so a caller that
    reads "returned without raising" as "the service is up" cannot see a helper that dies before its first
    bootstrap (#88848) — nor a ``launchctl bootstrap`` that exits 0 without registering, which the reporter
    measured on macOS 26.6.1.
    Judge the outcome the way #80491 taught the helper to judge it: by a live supervised pid, never by an
    exit code.  :func:`_launchctl_supervised_pid` is already that probe, so this only adds the wait.

    ``old_pid`` is the pid launchd ran for the label *before* the restart: a restart that leaves the same
    process running is not a restart, so passing it holds the invoking profile to the same fresh-pid
    contract :func:`_wait_for_launchd_service_pid` enforces for sibling labels. With ``old_pid=None``
    (no pre-restart pid was observable) any supervised pid counts, as before.

    With ``old_pid``, ``timeout`` starts only once launchd stops reporting that pid: the graceful drain
    before it exits is bounded by ``shutdown_timeout`` (default: the gateway's own restart exit budget
    plus launchd's ``ExitTimeOut`` clamp).
    """
    if _gw()._launchd_unsupported_marker_exists():
        return True

    label = label or _gw().get_launchd_label()
    # ``timeout`` measures launchd's respawn, so it starts once the old process is gone. Until then
    # the old gateway is still draining, which is bounded by its own restart budget, not by ours:
    # a busy gateway's graceful shutdown alone can outlast ``timeout`` (a 33s drain failed a 20s
    # window although launchd respawned it at once).
    now = time.monotonic()
    shutdown_deadline = None
    if old_pid is not None:
        if shutdown_timeout is None:
            shutdown_timeout = _gw()._get_restart_exit_wait_budget() + LAUNCHD_GUI_EXIT_TIMEOUT_CLAMP_S
        shutdown_deadline = now + max(shutdown_timeout, 0.0)
    deadline = now + max(timeout, 0.0)
    while True:
        pid = _gw()._launchctl_supervised_pid(label)
        if pid is not None and pid != old_pid:
            return True
        now = time.monotonic()
        if shutdown_deadline is not None:
            if pid == old_pid and now < shutdown_deadline:
                deadline = now + max(timeout, 0.0)  # old gateway still exiting: respawn clock not started
            else:
                shutdown_deadline = None  # old pid gone (or out of budget): the respawn clock runs
        if now >= deadline:
            return False
        time.sleep(max(poll_interval, 0.01))


def launchd_status(deep: bool = False):
    plist_path = _gw().get_launchd_plist_path()
    label = _gw().get_launchd_label()
    try:
        result = subprocess.run(["launchctl", "list", label], timeout=10, **_gw()._CAPTURE_TEXT)
        service_listed = result.returncode == 0
        list_output = result.stdout
    except subprocess.TimeoutExpired:
        service_listed = False
        list_output = ""

    # `launchctl list` exits 0 for any registered definition (even `state = not running`); only a PID proves a process.
    launchd_pid = _gw()._parse_launchd_pid_from_list_output(list_output) if service_listed else None

    # Hermes PID may be a detached fallback process; when launchd IS supervising both PIDs match — don't double-count.
    from gateway.status import get_running_pid
    fallback_pid = get_running_pid(cleanup_stale=False)
    if launchd_pid is not None and fallback_pid == launchd_pid:
        fallback_pid = None

    # Marker from a 5/125 bootstrap/kickstart failure explains *why* launchd can't supervise.
    launchd_unsupported = _gw()._launchd_unsupported_marker_exists()

    print(f"Launchd plist: {plist_path}")
    if _gw().launchd_plist_is_current():
        print("✓ Service definition matches the current Hermes install")
    else:
        print("⚠ Service definition is stale relative to the current Hermes install")
        print("  Run: hermes gateway start")

    if not service_listed:
        print("✗ Gateway service is not loaded")
        print("  Service definition exists locally but launchd has not loaded it.")
        print("  Run: hermes gateway start")
        if fallback_pid:
            print(f"  Note: a detached gateway process is running (PID {fallback_pid})")
    elif launchd_pid is not None:
        print(f"✓ Gateway is supervised by launchd (PID {launchd_pid})")
        print("  Auto-start at login and auto-restart on crash are available.")
        if launchd_unsupported:
            print("  (launchd domain was previously unavailable but is now working)")
    elif launchd_unsupported:
        print("⚠ Gateway service is registered but launchd is not supervising it")
        print("  launchd cannot manage the gateway on this macOS version.")
        if fallback_pid:
            print(f"✓ Detached fallback process is running (PID {fallback_pid})")
            print("  Cron jobs will fire. Stop with: hermes gateway stop")
        else:
            print("✗ No fallback process is running")
            print("  Run: hermes gateway start")
        print("  ⚠ Auto-start at login and auto-restart on crash are NOT available.")
    else:
        print("✓ Gateway service is registered with launchd")
        print(list_output)
        if fallback_pid:
            print(f"  Detached gateway process is running (PID {fallback_pid})")

    if deep:
        log_file = _gw().get_hermes_home() / "logs" / "gateway.log"
        if log_file.exists():
            print()
            print("Recent logs:")
            subprocess.run(["tail", "-20", str(log_file)], timeout=10)
