"""``hermes gateway migrate --backends --apply / --restore / --status``: move this home from the old
``hermes dashboard`` / ``hermes serve`` backend to the gateway's Webapp on the same address.

Shape copied from ``hermes gateway migrate --multiplex`` (``gateway_migrate.py``): plan + preflight that
change nothing, a manifest written before the first change, apply, verify, and a compensator that puts
everything back. The manifest (``backend_migration.json`` in the home) is the single record of the
host's prior state and of where a run got to (``step``); ``--status`` reads it, ``--restore`` undoes a
finished migration from it, and a manifest left in a non-final state by a killed run is rolled back by
the next invocation before anything else happens (never continued forward).

Exit codes: 0 migrated / restored / nothing to do; 2 invalid invocation; 3 blocked by preflight (nothing
changed); 4 failed, rolled back and the old backend verified working; 5 failed and rollback incomplete
(manual steps printed); 6 this kind of backend cannot be migrated automatically (nothing changed);
7 (``--detach`` only) the background run was started but did not confirm it got the run lock; 8 (``--expect``)
the backend found now is not the one the plan was confirmed for, or an earlier run is unfinished
(nothing changed).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import stat
import sys
import time
from pathlib import Path
from typing import Optional

from hermes_cli.gateway_migrate_backends_plan import (
    _SECRET_KEY, APPLY_STEPS, EXIT_BLOCKED, EXIT_OK, EXIT_UNSUPPORTED, EXIT_USAGE, BackendPlan, build_plan,
    format_address, format_plan, manual_steps, parse_address, provider_names,
)
from hermes_cli.gateway_migrate_backends_detach import (
    clear_refusal, detach, read_refusal, record_refusal, refusal_status,
)

logger = logging.getLogger(__name__)

EXIT_ROLLED_BACK, EXIT_ROLLBACK_INCOMPLETE = 4, 5
#: Unit-file states in which systemd starts the old unit at boot: the rollback enables it again.
_ENABLED = frozenset({"enabled", "enabled-runtime", "linked", "linked-runtime", "alias", "indirect"})


def rec_enabled(rec: dict) -> bool:
    return rec.get("supervisor") != "systemd" or (rec.get("unit_file_state") or "") in _ENABLED


def env_writes(plan: BackendPlan) -> dict[str, str]:
    """The ``.env`` keys the apply writes: sign-in settings only the old launcher had, plus a new signing
    key when none exists anywhere (values never leave memory except into ``.env``)."""
    import secrets
    writes = dict(plan.carry)
    if plan.generate_secret:
        writes[_SECRET_KEY] = secrets.token_urlsafe(32)
    return writes

MANIFEST_NAME = "backend_migration.json"
BACKUP_DIR = "backend_migration"
#: Upper bound for the new gateway to answer on every address. A good run passes as soon as it answers
#: (the whole outage was 3.4 to 8.5 s on the Linux and macOS test guests); only a failing run waits the
#: whole budget, with the old address dark, before it rolls back. It stays at 90 s so a slow first start
#: (a small ARM board, a cold disk) is not rolled back as a failure.
VERIFY_SECONDS = 90.0
ROLLBACK_VERIFY_SECONDS = 60.0
FINAL_STATES = frozenset({"done", "rolled_back", "restored"})
RESTORE_STEPS = ("stop_gateway", "restore_files", "restore_gateway_service", "start_old_backend",
                 "verify_old_backend")


class StepFailed(RuntimeError):
    pass


# --------------------------------------------------------------------------- manifest


def manifest_path(home: Path) -> Path:
    return Path(home) / MANIFEST_NAME


def read_manifest(home: Path) -> Optional[dict]:
    try:
        data = json.loads(manifest_path(home).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


_MANIFEST_KEYS = ("state", "steps", "old_backend", "gateway", "backups", "dial")


def unreadable_manifest(home: Path) -> Optional[str]:
    """Why ``backend_migration.json`` exists but cannot be trusted (unreadable, not JSON, a record
    without the fields a rollback needs), else None. Such a home is never treated as un-migrated: a
    new run would overwrite the backups the earlier one may still need."""
    path = manifest_path(home)
    if not path.exists():
        return None
    data = read_manifest(home)
    if data is None:
        return f"{path} exists but cannot be read as a migration record"
    missing = [k for k in _MANIFEST_KEYS if k not in data]
    return f"{path} is missing {', '.join(missing)}" if missing else None


def _refuse_unreadable(home: Path, problem: str, as_json: bool) -> int:
    folder = Path(home) / BACKUP_DIR
    message = (f"{problem}. A backend migration may have stopped half-way and this command cannot tell "
               f"how far it got, so it changes nothing. Check the gateway and the old backend by hand; the "
               f"backups of config.yaml and .env are in {folder}. Once this home is as you want it, move "
               f"{manifest_path(home)} aside and run the command again.")
    if as_json:
        print(json.dumps({"applied": False, "error": "manifest_unreadable", "message": message}))
    else:
        print(f"✗ {message}")
    return EXIT_BLOCKED


def write_manifest(home: Path, data: dict) -> None:
    from utils import atomic_json_write
    # File and directory entry are both on disk before the next step changes anything.
    atomic_json_write(manifest_path(home), data, mode=0o600, fsync_dir=True)


class Run:
    """One run's manifest, persisted on every step transition (what ``--status`` reads)."""

    def __init__(self, host, data: dict):
        self.host, self.data = host, data

    def save(self) -> None:
        write_manifest(self.host.home, self.data)

    def step(self, name: str, state: str = "running", message: Optional[str] = None) -> None:
        for s in self.data["steps"]:
            if s["name"] == name:
                s["state"] = state
        if state == "running":
            self.data["step"] = name
            if self.data.get("kind") == "apply" and name not in self.data.setdefault("entered", []):
                # The journal of apply steps whose changes may have started: rollback and restore undo
                # only these. Saved below, before the step's first change.
                self.data["entered"].append(name)
        if message is not None:
            self.data["message"] = message
        self.save()

    def done_steps(self) -> list[str]:
        return [s["name"] for s in self.data["steps"] if s["state"] in ("done", "running", "failed")]

    def finish(self, state: str, message: str) -> None:
        self.data.update(state=state, message=message, finished_at=self.host.now())
        self.save()


def write_private_file(path: Path, data: bytes) -> None:
    """Replace *path* with *data* through a temp file that is 0600 from its creation (secret bytes are
    never in a file anyone else can read, whatever the umask or the old file's mode), fsynced, then
    renamed over *path* with the directory entry fsynced too."""
    from utils.files import fsync_directory
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    fsync_directory(path.parent)


def _backup_folder(home: Path) -> Path:
    folder = Path(home) / BACKUP_DIR
    folder.mkdir(mode=0o700, exist_ok=True)
    return folder


def _backup(home: Path, name: str, target: str) -> Optional[str]:
    """Copy ``home/name`` into the backup folder (mode 600); None when it does not exist."""
    src = Path(home) / name
    if not src.exists():
        return None
    dst = _backup_folder(home) / target
    write_private_file(dst, src.read_bytes())
    return str(dst)


def _write_private(home: Path, target: str, text: str) -> str:
    dst = _backup_folder(home) / target
    write_private_file(dst, text.encode("utf-8"))
    return str(dst)


def _file_mode(path: Path) -> Optional[int]:
    with contextlib.suppress(OSError):
        return stat.S_IMODE(Path(path).stat().st_mode)
    return None


def new_manifest(plan: BackendPlan, host) -> dict:
    from gateway.status import get_process_start_time
    home = Path(plan.home)
    clear_refusal(home)  # this run's manifest is the newer record from here on
    backups = {"config": _backup(home, "config.yaml", "config.yaml.bak"), "env": _backup(home, ".env", "env.bak"),
               "launcher_env": None, "config_mode": _file_mode(home / "config.yaml"),
               "env_mode": _file_mode(home / ".env")}
    if plan.rec.get("supervisor") == "manual":
        # A hand-started backend is started again from its exact command line and environment: both may
        # carry credentials, so they are kept only in 0600 files (the manifest has the redacted argv).
        backups["launcher_argv"] = _write_private(home, "launcher-argv.json", json.dumps(list(plan.old_argv)))
        if (env := host.process_env(plan.rec["pid"]) if plan.rec.get("pid") else None) is not None:
            backups["launcher_env"] = _write_private(home, "launcher-env.json", json.dumps(env))
    return {
        "version": 1, "kind": "apply", "state": "running", "step": "manifest", "message": "",
        "steps": [{"name": n, "state": "done" if n == "preflight" else "pending"} for n in APPLY_STEPS],
        "entered": [],
        "old_address": format_address(plan.old_host, plan.old_port),
        "new_address": format_address(plan.new_host, plan.new_port),
        # ``old_host``: where the old backend answers (its own bind, never assumed loopback);
        # ``verify_hosts``: every address the gateway must answer on (a specific IP plus its loopback
        # companion). ``host`` is kept for manifests written before both existed.
        "dial": {"host": plan.old_probe_host, "old_host": plan.old_probe_host, "verify_hosts": plan.verify_hosts,
                 "old_port": plan.old_port, "new_port": plan.new_port},
        "started_at": host.now(), "finished_at": None,
        "pid": os.getpid(), "pid_start_time": get_process_start_time(os.getpid()),
        "backups": backups, "old_backend": plan.rec, "old_backend_enabled": rec_enabled(plan.rec),
        "gateway": plan.gateway, "linger": plan.linger, "webapp_before": _webapp_before(host),
        "auth_before": plan.auth_before, "copied_env_keys": sorted(plan.carry),
        "keep_webapp_address": plan.keep_webapp,
    }


# --------------------------------------------------------------------------- apply


def _wait(host, seconds: float, probe) -> bool:
    deadline = time.monotonic() + seconds
    while True:
        if probe():
            return True
        if time.monotonic() >= deadline:
            return False
        host.sleep(0.5)


def _verify_hosts(manifest: dict) -> list[str]:
    dial = manifest["dial"]
    return list(dial.get("verify_hosts") or [dial["host"]])


def _old_host(manifest: dict) -> str:
    dial = manifest["dial"]
    return dial.get("old_host") or dial["host"]


def gateway_serving(host, manifest: dict) -> Optional[str]:
    """None once the gateway answers on every new address exactly as rule 5 requires; else why not."""
    port = manifest["dial"]["new_port"]
    statuses = []
    for dial in _verify_hosts(manifest):
        status = host.gateway_status(dial, port)  # the gateway's /api/status: the plan's probe has no state
        if status is None:
            return f"no HTTP 200 on {format_address(dial, port)}/api/status"
        if status.get("gateway_state") != "running":
            return f"{format_address(dial, port)}/api/status says gateway_state={status.get('gateway_state')!r}"
        statuses.append(status)
    status = statuses[0]
    # /api/status answers even when the bundle is missing or broken; the page itself must be served.
    if (problem := host.webapp_page_problem(_verify_hosts(manifest)[0], port)) is not None:
        return problem
    pid = host.gateway_pid()
    if pid is None or not host.pid_listens_on(pid, port):
        return f"the listener on port {port} is not the gateway's (gateway pid {pid})"
    before = manifest.get("auth_before")
    if before and before["auth_required"] and not manifest.get("keep_webapp_address"):  # same address only
        if not status.get("auth_required"):
            return "sign-in was required before and is not now"
        missing = set(before["providers"]) - set(provider_names(status.get("auth_providers")))
        if missing:
            return f"sign-in provider(s) {', '.join(sorted(missing))} are no longer listed"
    return None


def _wait_port_free(host, manifest: dict) -> None:
    dial, port = _old_host(manifest), manifest["dial"]["old_port"]
    if not _wait(host, 30, lambda: not host.port_in_use(dial, port)):
        raise StepFailed(f"port {port} is still held after the old backend stopped")


def _ensure_gateway_service(host, gw_before: dict) -> None:
    if not gw_before["installed"]:
        host.install_gateway()
    elif not gw_before["enabled"]:
        host.gateway_verb("enable")


def _apply_steps(run: Run, plan: BackendPlan) -> float:
    """Every change, in contract order. Returns the time the old backend stopped answering."""
    host, m = run.host, run.data
    run.step("manifest", "done")
    run.step("webapp_bundle")
    if plan.bundle_missing:
        print("→ Building the Webapp bundle (hermes webapp --build-only)...")
        host.build_bundle()
        if (reason := host.bundle_missing()) is not None:
            raise StepFailed(f"the Webapp bundle is still missing after the build: {reason}")
    run.step("webapp_bundle", "done")
    run.step("write_config")
    if writes := env_writes(plan):
        host.set_env(writes)
        print(f"  ✓ .env: {', '.join(sorted(writes))} (mode 600; values not shown)")
    host.set_config(plan.config_writes())
    print(f"  ✓ config.yaml: {', '.join(plan.config_writes())}")
    run.step("write_config", "done")
    run.step("gateway_service")
    _ensure_gateway_service(host, m["gateway"])
    print("  ✓ the gateway is installed and enabled as a user service")
    run.step("gateway_service", "done")
    run.step("stop_old_backend")
    rec = m["old_backend"]
    stopped_at = host.now()
    host.old_stop(rec)
    host.old_disable(rec)
    if (dropin := host.old_unit_dropin_path(rec)) is not None:
        m["old_unit_dropin"] = dropin  # recorded first: rollback and restore remove it
        run.save()
        host.write_old_unit_dropin(dropin)
    if rec.get("was_active"):
        _wait_port_free(host, m)
    print("  ✓ old backend stopped" + (" and disabled (unit file kept)" if rec["supervisor"] != "manual" else ""))
    if dropin is not None:
        print(f"  ✓ started by hand it exits once and stays down: {dropin}")
    run.step("stop_old_backend", "done")
    run.step("start_gateway")
    host.gateway_verb("restart" if m["gateway"]["active"] else "start")
    run.step("start_gateway", "done")
    run.step("verify")
    why: list[Optional[str]] = [None]

    def ok() -> bool:
        why[0] = gateway_serving(host, m)
        return why[0] is None

    if not _wait(host, VERIFY_SECONDS, ok):
        raise StepFailed(f"verification failed after {VERIFY_SECONDS:.0f}s: {why[0]}")
    run.step("verify", "done")
    return stopped_at


def apply(host, plan: BackendPlan, *, detached: bool = False) -> int:
    m = new_manifest(plan, host)
    if (missing := _snapshot_problem(m)) is not None:
        # Read back before anything changes: a hand-started backend is retired only when it can be
        # started again exactly as it ran.
        for path in (p for p in m["backups"].values() if isinstance(p, str)):
            with contextlib.suppress(OSError):
                Path(path).unlink(missing_ok=True)
        print(f"✗ Nothing changed: {missing} (pid {plan.rec.get('pid')}), so a rollback could not start it "
              f"again as it ran. Check that this user can read the environment and working directory of pid "
              f"{plan.rec.get('pid')} (or start it with an absolute `hermes` command), then re-run.")
        if detached:
            record_refusal(host, mode="apply", reason=missing, exit_code=EXIT_BLOCKED)
        return EXIT_BLOCKED
    run = Run(host, m)
    run.save()  # rule 4: the manifest exists before the first change
    print(f"  ✓ recorded the current state in {manifest_path(host.home)}")
    try:
        stopped_at = _apply_steps(run, plan)
    except (Exception, SystemExit) as exc:  # every failure after the manifest is undone
        failed = run.data["step"]
        logger.warning("backend migration step %s failed; rolling back", failed, exc_info=True)
        run.step(failed, "failed", f"{failed} failed: {exc}")
        print(f"  ✗ {failed} failed: {exc}")
        print("  ↩ Rolling back...")
        return rollback(host, run, reason=f"{failed} failed: {exc}")
    outage = host.now() - stopped_at
    linger = host.linger()
    run.data["linger_after"] = linger
    run.data["outage_seconds"] = round(outage, 2)
    run.finish("done", f"migrated: the gateway serves {run.data['new_address']}")
    gap = (f"old backend retired; {run.data['old_address']} no longer answers" if plan.keep_webapp
           else f"old address unavailable for {outage:.1f}s")
    print(f"\n✓ Migrated: the gateway's Webapp now serves {run.data['new_address']} ({gap}).")
    print(_LAUNCHD_NOTE if run.data["gateway"].get("kind") == "launchd" else _linger_note(linger))
    from hermes_cli.gateway_migrate_backends_plan import hermes_command
    print("  Sign in again once in each browser/app (new cookies). Undo with: "
          f"{hermes_command(host.home, powershell=os.name == 'nt')} gateway migrate --backends --restore")
    return EXIT_OK


_LAUNCHD_NOTE = "  The gateway is a launchd agent of this user: like the old backend, it starts when you log in."


def _linger_note(linger: Optional[bool]) -> str:
    if linger is True:
        return "  Lingering is on: the gateway starts at boot without anyone logging in."
    if linger is False:
        import getpass
        return ("  ⚠ Lingering is OFF: the gateway (like the old backend) starts only when you log in. To serve "
                f"after a reboot without a login run: sudo loginctl enable-linger {getpass.getuser()}")
    return "  Lingering state unknown on this host."


# --------------------------------------------------------------------------- rollback / restore


def _restore_file(home: Path, name: str, backup: Optional[str], mode: Optional[int] = None) -> None:
    """Put ``home/name`` back from *backup* (absent before: removed). The copy is private from its
    creation; the file's recorded mode is set again afterwards."""
    target = Path(home) / name
    if backup is None:
        target.unlink(missing_ok=True)
        return
    write_private_file(target, Path(backup).read_bytes())
    if mode is not None and mode != 0o600:
        os.chmod(target, mode)


def _webapp_before(host) -> Optional[dict]:
    block = host.read_config().get("webapp")
    return block if isinstance(block, dict) else None


def _launcher_argv(m: dict) -> Optional[list]:
    path = (m.get("backups") or {}).get("launcher_argv")
    if not path:
        return None
    with contextlib.suppress(OSError, ValueError):
        argv = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return argv if isinstance(argv, list) and argv else None
    return None


def _launcher_env(m: dict) -> Optional[dict]:
    path = (m.get("backups") or {}).get("launcher_env")
    if not path:
        return None
    with contextlib.suppress(OSError, ValueError):
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return None


def _snapshot_problem(m: dict) -> Optional[str]:
    """Why a hand-started old backend cannot be started again as it ran (its recorded raw command line
    and environment, read back from their private files), else None. Never replaced by a guess: the
    caller's environment would lose the old backend's sign-in settings."""
    if (m.get("old_backend") or {}).get("supervisor") != "manual":
        return None
    if (argv := _launcher_argv(m)) is None:
        return "the old backend's recorded command line is missing or unreadable"
    if "-m" in argv and not (m.get("old_backend") or {}).get("cwd"):
        # `python -m hermes_cli.main` imports from its working directory: started elsewhere it cannot run.
        return "the old backend runs `python -m ...` and its working directory could not be read"
    if not isinstance(_launcher_env(m), dict):
        return "the old backend's recorded environment is missing or unreadable"
    return None


def _discard_launcher_snapshot(m: dict) -> None:
    """Remove the copies of a hand-started backend's command line and environment (they may carry
    credentials) once nothing can need them again: after a verified rollback or restore."""
    for key in ("launcher_argv", "launcher_env"):
        if path := (m.get("backups") or {}).get(key):
            with contextlib.suppress(OSError):
                Path(path).unlink(missing_ok=True)
            m["backups"][key] = None


def _symlinked(home: Path) -> Optional[str]:
    """Why ``config.yaml`` / ``.env`` may not be touched now (a symbolic link: restoring would replace the
    link with a copy, and a mode change would reach its shared target), else None."""
    if links := [n for n in ("config.yaml", ".env") if (Path(home) / n).is_symlink()]:
        return (f"{', '.join(links)} in {home} is a symbolic link now; restoring would replace it with a "
                f"copy. Put the file back in place of the link, then run this again")
    return None


def _entered(m: dict, step: str) -> bool:
    """Whether the apply run may have started *step*'s changes (its journal; no journal: assume so)."""
    entered = m.get("entered")
    return True if entered is None else step in entered


def _undo(run: Run, errors: list[str]) -> None:
    """Put the host back to the manifest's record, undoing only the apply steps that were entered.

    The old backend is started again only once the migrated gateway is proven off its address: when
    stopping that gateway or restoring the files failed, or the gateway still listens there, the old
    backend stays down and the run ends incomplete (exit 5) instead of leaving two session owners."""
    host, m = run.host, run.data
    gw_before, rec = m["gateway"], m["old_backend"]
    backups = m["backups"]
    phase = _phases(run)
    safe = [True]  # may the old backend be started again?

    def attempt(what: str, fn, *, guards: bool = False) -> bool:
        try:
            fn()
            print(f"  ✓ {what}")
            return True
        except (Exception, SystemExit) as exc:  # keep undoing; every failure is reported and logged
            logger.warning("backend migration rollback: %s failed", what, exc_info=True)
            errors.append(f"{what}: {exc}")
            print(f"  ✗ {what}: {exc}")
            if guards:
                safe[0] = False
            return False

    phase("stop_gateway")
    if _entered(m, "start_gateway") and host.gateway_state()["active"]:
        attempt("stopped the gateway", lambda: host.gateway_verb("stop"), guards=True)
    phase("restore_files")
    if _entered(m, "write_config"):
        attempt("restored config.yaml", lambda: _restore_file(host.home, "config.yaml", backups["config"],
                                                              backups.get("config_mode")), guards=True)
        attempt("restored .env", lambda: _restore_file(host.home, ".env", backups["env"], backups.get("env_mode")),
                guards=True)
    phase("restore_gateway_service")
    if _entered(m, "gateway_service"):
        if not gw_before["installed"]:
            if host.gateway_state()["installed"]:
                attempt("removed the gateway service (it was not installed before)", host.uninstall_gateway)
        elif not gw_before["enabled"] and host.gateway_state()["enabled"]:
            attempt("disabled the gateway service again", lambda: host.gateway_verb("disable"))
        if m.get("linger") is False and host.linger() is True:
            attempt("turned lingering off again", lambda: host.set_linger(False))
    if _entered(m, "start_gateway") and gw_before["active"] and safe[0]:
        # It runs the restored config again: not on the old backend's address.
        attempt("started the gateway again (it was running before)", lambda: host.gateway_verb("start"))
    phase("start_old_backend")
    if not _entered(m, "stop_old_backend"):
        return  # never stopped, never disabled: nothing of it to undo
    if m.get("old_unit_dropin"):
        attempt("removed the restart guard of the old unit",
                lambda: host.remove_old_unit_dropin(m["old_unit_dropin"]))
    if safe[0] and (busy := _migrated_listener_up(host, m)) is not None:
        safe[0] = False
        errors.append(busy)
        print(f"  ✗ {busy}")
    if not safe[0]:
        errors.append("did not start the old backend again: the migrated gateway may still own its address")
        print("  ✗ the old backend was not started again (it would run beside the migrated gateway)")
        return
    if (rec.get("was_active") or rec.get("was_loaded")) and (missing := _snapshot_problem(m)) is not None:
        errors.append(f"did not start the old backend again: {missing}")
        print(f"  ✗ the old backend was not started again: {missing}")
        return
    if m.get("old_backend_enabled"):
        attempt("re-enabled the old backend", lambda: host.old_enable(rec))
    if rec.get("was_active") or rec.get("was_loaded"):  # a loaded launchd job was unloaded by the stop step
        attempt("started the old backend", lambda: _start_old(host, run))


def _migrated_listener_up(host, m: dict) -> Optional[str]:
    """Why the old backend may not start yet, else None: only a proven absence of the gateway's listener
    on the old port counts (waits for it briefly). A probe that cannot tell is not proof."""
    port = m["dial"]["old_port"]
    seen: list[Optional[bool]] = [None]
    dials = list(dict.fromkeys([_old_host(m), *_verify_hosts(m), "127.0.0.1"]))

    def gone() -> bool:
        seen[0] = host.gateway_listens(port)
        if seen[0] is None and not any(host.port_in_use(h, port) for h in dials):
            seen[0] = False  # the gateway's identity is unknown, but nothing at all listens on the port
        return seen[0] is False

    if _wait(host, ROLLBACK_VERIFY_SECONDS / 2, gone):
        return None
    if seen[0] is None:
        return (f"cannot tell whether the gateway still listens on port {port} (something listens there and "
                f"its identity cannot be read). If you started the old backend again by hand, stop it, then "
                f"run this command again")
    return f"the gateway still listens on port {port}"


def _phases(run: Run):
    """Step marker for a ``--restore`` run (its steps are the undo phases); a no-op for a rollback."""
    previous: list[str] = []

    def phase(name: str) -> None:
        if run.data.get("kind") != "restore":
            return
        if previous:
            run.step(previous[-1], "done")
        previous.append(name)
        run.step(name)
    return phase


def _start_old(host, run: Run) -> None:
    m = run.data
    rec = m["old_backend"]
    if rec["supervisor"] == "manual":
        again = m.get("old_backend_respawned") or {}
        if (host.same_process(rec.get("pid"), rec.get("start_time"))
                or host.same_process(again.get("pid"), again.get("start_time"))):
            return  # the recorded process, or the one an earlier rollback started, still runs
        if (live := host.live_manual_backend(m["dial"]["old_port"])) is not None:
            # An earlier, interrupted rollback already started one: never a second.
            m["old_backend_respawned"] = live
            run.save()
            return
    pid = host.old_start(rec, _launcher_env(m), _launcher_argv(m))
    if pid is not None:
        m["old_backend_respawned"] = {"pid": pid, "start_time": host.start_time(pid)}
        run.save()  # recorded at once: a rollback interrupted after this sees it


def old_backend_answers(host, m: dict) -> Optional[str]:
    """None once the old backend answers on its address again; else why not."""
    rec = m["old_backend"]
    if not rec.get("was_active"):
        return None  # it was not running before: its enablement is what was restored
    if host.old_unit_active(rec) is False:
        return f"{rec['unit']} is not active"
    dial, port = _old_host(m), m["dial"]["old_port"]
    status = host.http_status(dial, port)
    if status is None:
        return f"no HTTP 200 on {format_address(dial, port)}/api/status"
    gw_pid = host.gateway_pid()
    if gw_pid is not None and host.pid_listens_on(gw_pid, port):
        return f"port {port} is answered by the gateway, not the old backend"
    return None


def _verify_old(host, m: dict) -> Optional[str]:
    why: list[Optional[str]] = [None]

    def ok() -> bool:
        why[0] = old_backend_answers(host, m)
        return why[0] is None

    return None if _wait(host, ROLLBACK_VERIFY_SECONDS, ok) else why[0]


def _manual_rollback_steps(m: dict) -> list[str]:
    """The numbered-by-the-caller steps that finish a rollback by hand, from the recorded state. Every
    ``hermes`` command names this run's profile (``hermes -p NAME``), so it acts on this home."""
    from hermes_cli.gateway_migrate_backends_plan import hermes_command
    rec, b = m["old_backend"], m["backups"]
    home = m.get("home") or "the Hermes home"
    hermes = hermes_command(m.get("home"), powershell=os.name == 'nt')
    steps = [f"{hermes} gateway stop"]
    env_mode = f"{b.get('env_mode') or 0o600:o}"
    steps += [f"cp {b['config']} {home}/config.yaml" if b.get("config") else f"rm {home}/config.yaml",
              f"cp {b['env']} {home}/.env && chmod {env_mode} {home}/.env" if b.get("env") else f"rm {home}/.env"]
    if not m["gateway"]["installed"]:
        steps.append(f"{hermes} gateway uninstall")
    elif m["gateway"]["active"]:
        steps.append(f"{hermes} gateway start")
    if m.get("old_unit_dropin"):
        steps.append(f"rm -f {m['old_unit_dropin']} && systemctl --user daemon-reload")
    if rec["supervisor"] == "systemd":
        systemctl = "systemctl --user" if rec["scope"] == "user" else "systemctl"
        enable = {"enabled": f"{systemctl} enable {rec['unit']}",
                  "enabled-runtime": f"{systemctl} enable --runtime {rec['unit']}"}.get(rec.get("unit_file_state") or "")
        steps += [line for line in (enable, f"{systemctl} start {rec['unit']}" if rec.get("was_active") else None)
                  if line]
    elif rec["supervisor"] == "launchd":
        steps.append(f"mv {rec['unit_path']}.disabled {rec['unit_path']}; launchctl enable {rec['scope']}/{rec['unit']}; "
                     f"launchctl bootstrap {rec['scope']} {rec['unit_path']}")
    else:  # its command line may carry a credential (env NAME=value, a token argument): never printed
        pid = f" (it was pid {rec['pid']})" if rec.get("pid") else ""
        steps.append(f"start the old backend again the way you started it before{pid}, so it serves "
                     f"{m.get('old_address') or 'its address'} again")
    return steps


def rollback(host, run: Run, *, reason: str, final_state: str = "rolled_back") -> int:
    """Compensate in reverse and verify the old backend answers. 4 when verified, 5 otherwise."""
    m = run.data
    m.setdefault("home", str(host.home))
    errors: list[str] = []
    _undo(run, errors)
    if m.get("kind") == "restore":
        run.step("start_old_backend", "done")
        run.step("verify_old_backend")
    why = _verify_old(host, m)
    if errors or why:
        detail = "; ".join(errors + ([f"old backend not verified: {why}"] if why else []))
        run.finish("failed", f"{reason}; rollback incomplete: {detail}")
        print(f"\n✗ Rollback incomplete ({detail}). Do this by hand:")
        print("\n".join(f"  {n}. {s}" for n, s in enumerate(_manual_rollback_steps(m), 1)))
        return EXIT_ROLLBACK_INCOMPLETE
    for s in m["steps"]:
        if m.get("kind") == "restore":
            s["state"] = "done"
        elif s["name"] not in ("preflight", "manifest") and s["state"] in ("done", "running", "failed"):
            s["state"] = "undone"
    done = "restored" if final_state == "restored" else "rolled back"
    _discard_launcher_snapshot(m)  # kept above on an incomplete rollback: the next one needs it
    run.finish(final_state, f"{reason}; {done}, old backend verified on {m['old_address']}")
    print(f"\n↩ {done.capitalize()}: the old backend answers on {m['old_address']} again ({reason}).")
    return EXIT_ROLLED_BACK if final_state == "rolled_back" else EXIT_OK


def restore(host, *, yes: bool, detached: bool = False) -> int:
    m = read_manifest(host.home)
    if m is None or m.get("state") != "done":
        state = "no migration recorded" if m is None else f"the last run is {m.get('state')!r}, not 'done'"
        print(f"✗ Nothing to restore: {state} ({manifest_path(host.home)}).")
        if detached:
            record_refusal(host, mode="restore", reason=f"nothing to restore: {state}", exit_code=EXIT_BLOCKED)
        return EXIT_BLOCKED
    if (reason := _symlinked(host.home)) is not None:
        print(f"✗ Nothing changed: {reason}.")
        if detached:
            record_refusal(host, mode="restore", reason=reason, exit_code=EXIT_BLOCKED)
        return EXIT_BLOCKED
    if (missing := _snapshot_problem(m)) is not None:
        # Checked before the gateway is stopped: without it the old backend could not be started as it ran.
        m.setdefault("home", str(host.home))
        reason = f"{missing}, so it cannot be started again as it ran; the gateway keeps serving"
        print(f"✗ Nothing changed: {reason}. To restore by hand:")
        print("\n".join(f"  {n}. {s}" for n, s in enumerate(_manual_rollback_steps(m), 1)))
        if detached:
            record_refusal(host, mode="restore", reason=reason, exit_code=EXIT_ROLLBACK_INCOMPLETE,
                           old_address=m.get("old_address", ""), new_address=m.get("new_address", ""),
                           manual_steps=_manual_rollback_steps(m))
        return EXIT_ROLLBACK_INCOMPLETE
    clear_refusal(host.home)
    if not _confirm(f"Restore {m['old_address']} to the old backend?", yes):
        print("Aborted; nothing changed.")
        return EXIT_OK
    from gateway.status import get_process_start_time
    # The restore is its own run: UIs tell it from the earlier migration by started_at.
    m.update(kind="restore", state="running", step=RESTORE_STEPS[0], pid=os.getpid(),
             pid_start_time=get_process_start_time(os.getpid()), finished_at=None,
             applied_started_at=m.get("started_at"), applied_finished_at=m.get("finished_at"),
             started_at=host.now(),
             steps=[{"name": n, "state": "pending"} for n in RESTORE_STEPS])
    run = Run(host, m)
    run.save()
    print(f"↩ Restoring the old backend recorded in {manifest_path(host.home)}...")
    code = rollback(host, run, reason="restore requested", final_state="restored")
    if code == EXIT_OK:
        print("✓ Restored.")
    return code


def recover_interrupted(host, m: dict) -> int:
    """Rule 7: an earlier run died mid-way. Roll back (never continue) and report."""
    print(f"↻ An earlier backend migration was interrupted at step '{m.get('step')}' "
          f"(pid {m.get('pid')} is gone; manifest {manifest_path(host.home)}). Rolling it back first...")
    clear_refusal(host.home)
    run = Run(host, m)
    code = rollback(host, run, reason=f"interrupted at {m.get('step')}")
    return EXIT_ROLLBACK_INCOMPLETE if code == EXIT_ROLLBACK_INCOMPLETE else EXIT_ROLLED_BACK


def _in_progress(host, m: dict) -> bool:
    return m.get("state") == "running" and host.same_process(m.get("pid"), m.get("pid_start_time"))


# --------------------------------------------------------------------------- status


def manual_steps_for(m: dict) -> list[str]:
    """The numbered-by-the-caller manual steps of a ``failed`` run (rollback incomplete), else []."""
    if m.get("state") != "failed":
        return []
    try:
        return list(_manual_rollback_steps(m))
    except (KeyError, TypeError, AttributeError):  # a damaged manifest: the message still says why
        return []


def status_payload_from(host, m: dict) -> dict:
    """Rule 13's shape for manifest *m*, plus ``manual_steps``. *host* None skips the liveness check."""
    message = m.get("message") or ""
    if m.get("state") == "running" and host is not None and not _in_progress(host, m):
        message = (message + "; " if message else "") + "interrupted: the next run rolls it back"
    return {"state": m.get("state", "none"), "step": m.get("step", ""),
            "steps": [{"name": s["name"], "state": s["state"]} for s in m.get("steps", [])],
            "message": message, "old_address": m.get("old_address", ""), "new_address": m.get("new_address", ""),
            "started_at": float(m.get("started_at") or 0.0), "finished_at": m.get("finished_at"),
            "manual_steps": manual_steps_for(m)}


def status_payload(host) -> dict:
    if (problem := unreadable_manifest(host.home)) is not None:
        return {"state": "failed", "step": "", "steps": [], "message": f"{problem}; nothing can be undone automatically",
                "old_address": "", "new_address": "", "started_at": 0.0, "finished_at": None, "manual_steps": []}
    m = read_manifest(host.home)
    refusal = read_refusal(host.home)
    if refusal is not None and (refused := refusal_status(refusal, m)) is not None:
        return refused
    if m is None:
        return {"state": "none", "step": "", "steps": [], "message": "", "old_address": "", "new_address": "",
                "started_at": 0.0, "finished_at": None, "manual_steps": []}
    return status_payload_from(host, m)


def print_status(host, as_json: bool) -> int:
    payload = status_payload(host)
    if as_json:
        print(json.dumps(payload, indent=2))
        return EXIT_OK
    print(f"Backend migration: {payload['state']}" + (f" (step {payload['step']})" if payload["step"] else ""))
    for key in ("old_address", "new_address", "message"):
        if payload[key]:
            print(f"  {key.replace('_', ' ')}: {payload[key]}")
    for s in payload["steps"]:
        print(f"  - {s['name']}: {s['state']}")
    if payload["manual_steps"]:
        print("  Do this by hand:")
        print("\n".join(f"    {n}. {s}" for n, s in enumerate(payload["manual_steps"], 1)))
    return EXIT_OK


# --------------------------------------------------------------------------- CLI


def _confirm(question: str, yes: bool) -> bool:
    if yes or not sys.stdin.isatty():
        return True
    from hermes_cli.setup import prompt_yes_no
    return prompt_yes_no(question, True)


def _usage_error(args) -> Optional[str]:
    modes = [m for m in ("apply", "restore", "status") if getattr(args, m, False)]
    if len(modes) > 1:
        return "--apply, --restore and --status are mutually exclusive."
    if getattr(args, "multiplex", False):
        return "--backends cannot be combined with --multiplex."
    if getattr(args, "dry_run", False) and modes and modes != ["apply"]:
        return "--dry-run goes with --apply only."
    apply_only = ("address", "keep_webapp_address", "force_interrupt", "expect")
    if not getattr(args, "apply", False) and any(getattr(args, f, None) for f in apply_only):
        return "--address, --keep-webapp-address, --force-interrupt and --expect go with --apply."
    if getattr(args, "detach", False) and not (getattr(args, "apply", False) or getattr(args, "restore", False)):
        return "--detach goes with --apply or --restore."
    if getattr(args, "address", None) and parse_address(args.address) is None:
        return f"--address must be HOST:PORT, got {args.address!r}."
    if getattr(args, "detach", False) and getattr(args, "dry_run", False):
        return "--detach cannot be combined with --dry-run."
    return None


def _plan_exit(plan: BackendPlan, *, dry_run: bool, as_json: bool) -> Optional[int]:
    """Print the plan; the exit code when the run ends here (dry run, nothing to do, blocked)."""
    if as_json:
        view = plan.to_dict()
        print(json.dumps({"applied": False, "dry_run": dry_run, "steps": view["steps"], "plan": view,
                          "expect_token": plan_token(plan)}, indent=2))
    else:
        print("\n".join(format_plan(plan, dry_run=dry_run)))
    if plan.nothing_to_do:
        return EXIT_OK
    if dry_run or plan.exit_code() != EXIT_OK:
        return plan.exit_code()
    return None


def _unfinished_dry_run(host, m: dict, as_json: bool) -> int:
    from hermes_cli.gateway_migrate_backends_plan import hermes_command
    message = (f"An earlier run is unfinished (state {m.get('state')}, step {m.get('step')}); "
               f"{hermes_command(host.home, powershell=os.name == 'nt')} gateway migrate --backends --apply rolls it back first. Nothing changed.")
    if as_json:
        print(json.dumps({"applied": False, "dry_run": True, "unfinished_run": True, "message": message,
                          "run": status_payload_from(None, m), "plan": None, "steps": []}, indent=2))
    else:
        print(f"↻ {message}")
    return EXIT_BLOCKED


def _refuse_detached(host, plan: BackendPlan, code: int) -> None:
    """A detached run that will not start leaves its reason for ``--status`` (no manifest exists)."""
    if plan.unsupported:
        reason = " ".join(text for _item, text, _steps in plan.unsupported)
        steps = [s for _item, _text, ss in plan.unsupported for s in ss]
    else:
        reason, steps = " ".join(plan.blockers) or "refused", []
    old = format_address(plan.old_host, plan.old_port) if plan.rec and plan.old_host is not None else ""
    new = format_address(plan.new_host, plan.new_port) if plan.rec and plan.new_host is not None else ""
    record_refusal(host, mode="apply", reason=reason, exit_code=code, old_address=old, new_address=new,
                   manual_steps=steps)


#: ``--apply --expect TOKEN``: the backend found now is not the one the caller confirmed (nothing changed).
EXIT_PLAN_CHANGED = 8


def plan_token(plan: BackendPlan) -> Optional[str]:
    """Opaque token for the backend a plan would move: its old and new address, supervisor and unit (the
    pid and start time only for a backend without a unit, so a managed unit's restart does not change it).
    The dry run's JSON carries it as ``expect_token``; ``--apply --expect TOKEN`` (the dashboard's routes)
    refuses to start when the plan found under the run lock has another. None: nothing to move."""
    import hashlib
    rec = plan.rec
    if not rec or plan.old_host is None or plan.new_host is None:
        return None  # nothing to move, or its address is unknown (the plan is blocked)
    managed = rec.get("supervisor") != "manual" and bool(rec.get("unit"))  # a hand-started process has no unit
    identity = {"old": format_address(plan.old_host, plan.old_port), "new": format_address(plan.new_host, plan.new_port),
                "supervisor": rec.get("supervisor"), "unit": rec.get("unit") if managed else None,
                "scope": rec.get("scope") if managed else None, "pid": None if managed else rec.get("pid"),
                "start_time": None if managed else rec.get("start_time")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:32]


def _plan_changed(host, plan: BackendPlan, expected: str, *, as_json: bool, detached: bool) -> int:
    actual = plan_token(plan)
    message = ("The old backend found now is not the one the plan was confirmed for (it was replaced or "
               "changed since); nothing changed. Look at the plan again.")
    if as_json:
        print(json.dumps({"applied": False, "plan_changed": True, "expected": expected, "expect_token": actual,
                          "message": message, "plan": plan.to_dict()}, indent=2))
    else:
        print(f"✗ {message}")
    if detached:
        record_refusal(host, mode="apply", reason="the plan changed since it was confirmed",
                       exit_code=EXIT_PLAN_CHANGED)
    return EXIT_PLAN_CHANGED


def _owner_blockers(host, plan: BackendPlan) -> None:
    """Refusals the apply path adds to the plan before anything changes (shown by the dry run too)."""
    from hermes_cli.gateway_migrate_backends_plan import _NEVER_TOUCHED, _for_this_home
    rec, item = plan.rec, plan.item
    others = [i for i in host.backends()
              if i.supervisor not in _NEVER_TOUCHED and _for_this_home(i, host)
              # a launchd job is listed only while it is loaded: idle or not, launchd may start it
              and (i.running or (i.state or "") in _BOOT_STATES or i.supervisor == "launchd")
              and not (i.pid == item.pid and i.unit == item.unit and i.supervisor == item.supervisor)]
    for other in others:
        what = f"{other.supervisor} {other.unit or ''}".strip() + (f" (pid {other.pid})" if other.pid else "")
        plan.blockers.append(f"Another old backend also serves this home: {what}. Moving one would leave it "
                             f"running beside the gateway as a second session owner; stop and disable it "
                             f"first (or move it instead with --address), then run this again.")
    if rec["supervisor"] == "manual" and rec.get("pid") and rec.get("start_time") is None:
        plan.blockers.append(f"The start time of the hand-started backend (pid {rec['pid']}) cannot be read, so "
                             f"this command cannot prove it would stop that process and no other. Stop it by "
                             f"hand, then run this again.")
    if rec["supervisor"] == "systemd" and (rec.get("unit_file_state") or "") not in host.RESTORABLE_UNIT_STATES:
        plan.blockers.append(f"{rec['unit']} is {rec.get('unit_file_state') or 'in an unknown state'} (systemctl "
                             f"is-enabled); a rollback could not put that back exactly, so nothing is changed. "
                             f"Move it by hand with the steps in the docs.")
    for name in ("config.yaml", ".env"):
        if (Path(host.home) / name).is_symlink():
            plan.blockers.append(f"{Path(host.home) / name} is a symbolic link: the migration would change the "
                                 f"file it points to, which may be shared. Replace the link with a copy of the "
                                 f"file (or move this home off the shared file), then run this again.")
    unit = (plan.gateway or {}).get("unit")
    if unit and host.own_unit() in (unit, f"{unit}.service"):
        plan.blockers.append(f"This command runs inside {host.own_unit()}, which the migration restarts and a "
                             f"rollback stops. Run it from a terminal, or with --detach.")


#: Unit-file states that start a unit at boot: such a unit is a session owner even when it is down now.
_BOOT_STATES = frozenset({"enabled", "enabled-runtime", "linked", "linked-runtime"})


def _in_gateway_unit(host, m: Optional[dict]) -> Optional[str]:
    unit = ((m or {}).get("gateway") or {}).get("unit")
    own = host.own_unit()
    return own if unit and own in (unit, f"{unit}.service") else None


def _refuse_recovery_here(host, m: dict, args, *, as_json: bool, detached: bool) -> Optional[int]:
    """The rollback of an unfinished run is a change too: refuse it where it must not run (exit with
    instructions, nothing changed); None when it may go ahead."""
    from hermes_cli.gateway_migrate_backends_plan import hermes_command
    restore = f"{hermes_command(host.home, powershell=os.name == 'nt')} gateway migrate --backends --restore"
    if (linked := _symlinked(host.home)) is not None:
        reason = f"an earlier run is unfinished and its rollback restores these files, but {linked}"
        code = EXIT_BLOCKED
    elif (inside := _in_gateway_unit(host, m)) is not None:
        reason = (f"an earlier run is unfinished and its rollback stops the gateway, but this run is inside "
                  f"{inside}; roll it back from a terminal with: {restore}")
        code = EXIT_BLOCKED
    elif getattr(args, "expect", None):
        reason = (f"an earlier run is unfinished (step {m.get('step')}), so the confirmed plan no longer "
                  f"applies; roll it back first with: {restore}")
        code = EXIT_PLAN_CHANGED
    else:
        return None
    if as_json:
        print(json.dumps({"applied": False, "unfinished_run": True, "plan_changed": code == EXIT_PLAN_CHANGED,
                          "message": f"Nothing changed: {reason}."}))
    else:
        print(f"✗ Nothing changed: {reason}.")
    if detached:
        record_refusal(host, mode="apply", reason=reason, exit_code=code)
    return code


def _already_migrated(host, m: dict, *, as_json: bool, dry_run: bool) -> int:
    """A finished migration is the record of this home: a second ``--apply`` never plans over it (the
    retired unit may still be found, a ``static`` one stays listed) and never replaces its backups."""
    from hermes_cli.gateway_migrate_backends_plan import hermes_command
    message = (f"Nothing to migrate: already migrated ({m.get('new_address')} is served by the gateway; the old "
               f"backend {(m.get('old_backend') or {}).get('unit') or ''} is retired). To move it back: "
               f"{hermes_command(host.home, powershell=os.name == 'nt')} gateway migrate --backends --restore")
    if as_json:
        print(json.dumps({"applied": False, "dry_run": dry_run, "nothing_to_do": True, "already_migrated": True,
                          "message": message, "plan": None, "steps": [], "expect_token": None}, indent=2))
    else:
        print(f"✓ {' '.join(message.split())}")
    return EXIT_OK


def _unsupported_here(host, args, *, as_json: bool, detached: bool = False) -> Optional[int]:
    """Exit 6 with the manual steps, before ``--detach`` starts anything, when this host's backend can
    only be moved by hand (Windows, a system unit, Nix, ...): the background run could only refuse, and
    its refusal record would outlive a move done by hand. Read-only, like a dry run; leaves no record.
    A confirmed plan's ``--expect`` token is checked first (exit 8), as a foreground run does."""
    address = parse_address(args.address) if getattr(args, "address", None) else None
    plan = build_plan(host, address=address, keep_webapp=bool(getattr(args, "keep_webapp_address", False)),
                      force_interrupt=bool(getattr(args, "force_interrupt", False)))
    if not plan.unsupported:
        return None
    if (expected := getattr(args, "expect", None)) and plan_token(plan) != expected:
        return _plan_changed(host, plan, expected, as_json=as_json, detached=detached)
    return _plan_exit(plan, dry_run=False, as_json=as_json)


def cmd_apply(host, args) -> int:
    detached = bool(getattr(args, "detached_run", False))
    as_json = bool(getattr(args, "json", False))
    if (problem := unreadable_manifest(host.home)) is not None:
        if detached:
            record_refusal(host, mode="apply", reason=problem, exit_code=EXIT_BLOCKED)
        return _refuse_unreadable(host.home, problem, as_json)
    m = read_manifest(host.home)
    if m is not None and m.get("state") in ("running", "failed"):
        if _in_progress(host, m):
            print(f"✗ A backend migration is running (pid {m.get('pid')}); see --status.")
            if detached:
                record_refusal(host, mode="apply", reason=f"another run is live (pid {m.get('pid')})",
                               exit_code=EXIT_BLOCKED)
            return EXIT_BLOCKED
        if getattr(args, "dry_run", False):
            return _unfinished_dry_run(host, m, as_json)
        if not getattr(args, "detach", False):  # detached: the background run rolls it back
            if (code := _refuse_recovery_here(host, m, args, as_json=as_json, detached=detached)) is not None:
                return code
            return recover_interrupted(host, m)
    if m is not None and m.get("state") == "done" and not getattr(args, "detach", False):
        return _already_migrated(host, m, as_json=as_json, dry_run=bool(getattr(args, "dry_run", False)))
    if getattr(args, "detach", False):
        unfinished = m is not None and m.get("state") in ("running", "failed")  # the background run recovers it
        if not unfinished and (code := _unsupported_here(host, args, as_json=as_json, detached=detached)) is not None:
            return code
        return detach(host, args)
    address = parse_address(args.address) if getattr(args, "address", None) else None
    # A background run checks its own cgroup like any other run (build_plan reads /proc/self/cgroup):
    # being started by --detach proves nothing about where it ended up.
    plan = build_plan(host, address=address, keep_webapp=bool(getattr(args, "keep_webapp_address", False)),
                      force_interrupt=bool(getattr(args, "force_interrupt", False)))
    if plan.rec and not plan.unsupported:
        _owner_blockers(host, plan)
    dry_run = bool(getattr(args, "dry_run", False))
    if (expected := getattr(args, "expect", None)) and plan_token(plan) != expected:
        return _plan_changed(host, plan, expected, as_json=as_json, detached=detached)
    if (code := _plan_exit(plan, dry_run=dry_run, as_json=as_json)) is not None:
        if detached and code != EXIT_OK:
            _refuse_detached(host, plan, code)
        return code
    if not _confirm("Migrate this home to the gateway now?", bool(getattr(args, "yes", False))):
        print("Aborted; nothing changed.")
        return EXIT_OK
    print()
    return apply(host, plan, detached=detached)


def cmd_restore(host, args) -> int:
    """``--restore [--detach]``: undo a finished migration (an unfinished run is rolled back instead)."""
    detached = bool(getattr(args, "detached_run", False))
    if (problem := unreadable_manifest(host.home)) is not None:
        if detached:
            record_refusal(host, mode="restore", reason=problem, exit_code=EXIT_BLOCKED)
        return _refuse_unreadable(host.home, problem, bool(getattr(args, "json", False)))
    m = read_manifest(host.home)
    if m is not None and _in_progress(host, m):
        print(f"✗ A backend migration is running (pid {m.get('pid')}); see --status.")
        if detached:
            record_refusal(host, mode="restore", reason=f"another run is live (pid {m.get('pid')})",
                           exit_code=EXIT_BLOCKED)
        return EXIT_BLOCKED
    if getattr(args, "detach", False):
        return detach(host, args, mode="restore")
    if (inside := _in_gateway_unit(host, m)) is not None:
        reason = f"this run is inside {inside}, which the restore stops; run it from a terminal or with --detach"
        print(f"✗ Not restoring: {reason}.")
        if detached:
            record_refusal(host, mode="restore", reason=reason, exit_code=EXIT_BLOCKED)
        return EXIT_BLOCKED
    if m is not None and m.get("state") in ("running", "failed"):
        if (linked := _symlinked(host.home)) is not None:
            print(f"✗ Nothing changed: {linked}.")
            if detached:
                record_refusal(host, mode="restore", reason=linked, exit_code=EXIT_BLOCKED)
            return EXIT_BLOCKED
        return recover_interrupted(host, m)
    return restore(host, yes=bool(getattr(args, "yes", False)), detached=detached)


def cmd_backends(args) -> None:
    """Dispatch for every ``--backends`` mode beyond the read-only report."""
    from hermes_cli.gateway_migrate_backends_host import SystemHost
    from hermes_constants import get_hermes_home
    if (err := _usage_error(args)) is not None:
        print(f"✗ {err}", file=sys.stderr)
        sys.exit(EXIT_USAGE)
    host = SystemHost(get_hermes_home())
    if getattr(args, "status", False):
        sys.exit(print_status(host, bool(getattr(args, "json", False))))
    if (getattr(args, "restore", False) and not getattr(args, "detach", False)
            and not getattr(args, "detached_run", False) and not manifest_path(host.home).exists()):
        sys.exit(restore(host, yes=True))  # nothing to restore: refused without creating the run lock
    if (refused := _take_run_lock(host, args)) is not None:
        sys.exit(refused)
    if getattr(args, "restore", False):
        sys.exit(cmd_restore(host, args))
    sys.exit(cmd_apply(host, args))


def _take_run_lock(host, args) -> Optional[int]:
    """Hold this home's run lock before any run that may change it reads its state (see the lock module).

    ``--dry-run`` and a ``--detach`` caller take none (the background run takes it and reports back)."""
    from hermes_cli import gateway_migrate_backends_lock as run_lock
    if getattr(args, "dry_run", False) or getattr(args, "detach", False):
        return None
    locked = run_lock.acquire(host.home)
    if (token := getattr(args, "detached_ack", None)):
        run_lock.write_ack(host.home, token, locked=locked)
    if locked:
        return None
    # No refusal record here: the run holding the lock owns --status.
    run_lock.print_busy(host.home, as_json=bool(getattr(args, "json", False)),
                        mode="restore" if getattr(args, "restore", False) else "apply")
    return EXIT_BLOCKED


__all__ = ["EXIT_UNSUPPORTED", "cmd_backends", "manual_steps", "status_payload"]
