"""``--detach`` for ``hermes gateway migrate --backends --apply / --restore``, and the record a refused
detached run leaves for ``--status``.

A detached run is started in the background (its own transient systemd unit, or a new session) and its
caller returns at once: the dashboard's button, Desktop, or a terminal inside the old backend's own unit.
When that background run refuses to start (blocked, unsupported, nothing to restore, another run live),
no manifest is written, so nothing would tell the caller why. It writes
``backend_migration/detached-refused.json`` instead, which ``--status`` reports as state ``failed`` with
the reason. It is a separate file on purpose: ``backend_migration.json`` is what ``hermes update`` reads
to keep a retired backend down (``backend_migration_marker``), and a refusal changed nothing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from hermes_cli.gateway_migrate_backends_host import profile_args
from hermes_cli.gateway_migrate_backends_plan import hermes_command

REFUSED_NAME = "detached-refused.json"
#: The background run was started but never said whether it got the run lock: its outcome is unknown.
EXIT_UNCONFIRMED = 7


def log_path(home: Path) -> Path:
    return Path(home) / "logs" / "backend-migration.log"


def detached_flags(args, mode: str, home=None) -> list[str]:
    """The background run's own flags: this home's profile, the same mode, ``--yes``, and the hidden
    ``--detached-run``."""
    if mode == "restore":
        return [*profile_args(home), "gateway", "migrate", "--backends", "--restore", "--yes", "--detached-run"]
    flags = [*profile_args(home), "gateway", "migrate", "--backends", "--apply", "--yes", "--detached-run"]
    if getattr(args, "address", None):
        flags += ["--address", args.address]
    flags += [f for f, on in (("--keep-webapp-address", getattr(args, "keep_webapp_address", False)),
                              ("--force-interrupt", getattr(args, "force_interrupt", False))) if on]
    if getattr(args, "expect", None):
        flags += ["--expect", args.expect]  # checked again by the background run, under the run lock
    return flags


def detach(host, args, mode: str = "apply") -> int:
    """Start the run in the background; return once it reports whether it got this home's run lock.

    Exit 0 when it holds the lock (progress is ``--status``), 3 when another run holds it or it could
    not be started outside this process's unit (nothing changed), 7 when it was started but never
    answered in time (outcome unknown: ``--status`` tells). JSON ``outcome``: ``started``, ``busy``,
    ``refused`` or ``unconfirmed``."""
    from hermes_cli import gateway_migrate_backends_lock as run_lock
    log = log_path(host.home)
    token = run_lock.new_ack_token()
    as_json = bool(getattr(args, "json", False))
    try:
        where = host.spawn_detached([*detached_flags(args, mode, host.home), "--detached-ack", token], log)
    except RuntimeError as exc:
        if as_json:
            print(json.dumps({"detached": False, "mode": mode, "outcome": "refused", "message": str(exc)}))
        else:
            print(f"✗ Not started in the background: {exc}. Nothing changed.")
        return 3
    ack = run_lock.wait_ack(host.home, token)
    if ack is not None and not ack.get("locked"):
        run_lock.print_busy(host.home, as_json=as_json, mode=mode)
        return 3
    what = "restore" if mode == "restore" else "migration"
    status = f"{hermes_command(host.home, powershell=os.name == 'nt')} gateway migrate --backends --status"
    outcome = "started" if ack is not None else "unconfirmed"
    if as_json:
        print(json.dumps({"detached": True, "mode": mode, **where, "confirmed": ack is not None, "outcome": outcome,
                          "log": str(log)}))
    else:
        print(f"→ Backend {what} started in the background ({', '.join(f'{k} {v}' for k, v in where.items())}).")
        if ack is None:
            print("  It has not confirmed that it started: whether it runs is unknown.")
        print(f"  Follow it with: {status}   (log: {log})")
    return 0 if ack is not None else EXIT_UNCONFIRMED


# --------------------------------------------------------------------------- refused detached runs


def _refused_path(home: Path) -> Path:
    return Path(home) / "backend_migration" / REFUSED_NAME


def record_refusal(host, *, mode: str, reason: str, exit_code: int, old_address: str = "",
                   new_address: str = "", manual_steps: Optional[list[str]] = None) -> None:
    """Remember why a detached run refused to start (it changed nothing; no manifest was written)."""
    from utils import atomic_json_write
    path = _refused_path(host.home)
    path.parent.mkdir(mode=0o700, exist_ok=True)
    now = host.now()
    atomic_json_write(path, {"mode": mode, "reason": reason, "exit_code": exit_code, "at": now, "pid": os.getpid(),
                             "old_address": old_address, "new_address": new_address,
                             "manual_steps": list(manual_steps or [])}, mode=0o600)


def clear_refusal(home: Path) -> None:
    _refused_path(home).unlink(missing_ok=True)


def read_refusal(home: Path) -> Optional[dict]:
    try:
        data = json.loads(_refused_path(home).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("at"), (int, float)) else None


def refusal_status(refusal: dict, manifest: Optional[dict]) -> Optional[dict]:
    """The ``--status`` payload for a refused detached run, when it is newer than the manifest's last run.

    A live (``running``) manifest always wins; a run that finished after the refusal wins too."""
    if manifest is not None:
        if manifest.get("state") == "running":
            return None
        last = float(manifest.get("finished_at") or manifest.get("started_at") or 0.0)
        if last > float(refusal["at"]):
            return None
    what = "restore" if refusal.get("mode") == "restore" else "migration"
    return {"state": "failed", "step": "preflight", "steps": [{"name": "preflight", "state": "failed"}],
            "message": f"the background {what} did not start (exit {refusal.get('exit_code')}): "
                       f"{refusal.get('reason')}; nothing changed",
            "old_address": refusal.get("old_address") or "", "new_address": refusal.get("new_address") or "",
            "started_at": float(refusal["at"]), "finished_at": float(refusal["at"]),
            "manual_steps": list(refusal.get("manual_steps") or []), "refused": True}
