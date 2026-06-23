from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_cli import control_db as cp

CONTROL_TO_RUNTIME_PROFILE = {
    "statutepm": "nj-statutes-pm",
}


@dataclass(frozen=True)
class ControlTarget:
    root: Path | None
    db_path: Path
    live: bool


def live_db_path() -> Path:
    return cp.control_db_path().resolve()


def physical_home_live_db_path() -> Path:
    return (Path.home() / ".hermes" / "control-plane" / "control.db").resolve()


def resolve_control_target(*, root: str | Path | None = None, live: bool = False, default_root: Path | None = None) -> ControlTarget:
    explicit_root = root is not None
    env_root = os.getenv("HERMES_CONTROL_ROOT")
    if live:
        target_root = None
    elif explicit_root:
        target_root = Path(root).expanduser()
    elif env_root:
        target_root = Path(env_root).expanduser()
    else:
        target_root = default_root
    db_path = cp.control_db_path(target_root).resolve()
    profile_default_live = db_path == live_db_path() and not explicit_root and not env_root and default_root is None
    physical_live = db_path == physical_home_live_db_path()
    return ControlTarget(root=target_root, db_path=db_path, live=(live or profile_default_live or physical_live))


def require_live_flag_for_mutation(target: ControlTarget, *, live_flag: bool) -> None:
    if target.live and not live_flag:
        raise PermissionError(f"refusing to mutate live control DB without --live: {target.db_path}")


def runtime_profile_for_control_profile(control_profile_id: str, override: str | None = None) -> str:
    if override:
        return override
    return CONTROL_TO_RUNTIME_PROFILE.get(control_profile_id, control_profile_id)


def runtime_profile_presence(profile_name: str) -> str:
    try:
        from hermes_cli.profiles import profile_exists

        return "present" if profile_exists(profile_name) else "missing"
    except Exception as exc:
        return f"error: {exc}"


def validate_pm_runtime_mapping(pm_profile_id: str, pm_runtime_profile: str | None = None) -> str:
    runtime = runtime_profile_for_control_profile(pm_profile_id, pm_runtime_profile)
    if runtime_profile_presence(runtime) != "present":
        raise PermissionError(f"mapped PM runtime profile is not present: {pm_profile_id}->{runtime}")
    return runtime


def help_parse_status(args: list[str], *, env: dict[str, str] | None = None, timeout_s: float = 10.0) -> dict[str, Any]:
    proc = subprocess.run(args, text=True, capture_output=True, env=env, timeout=timeout_s)
    return {
        "command": args,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip()[-500:],
    }


def worker_spawnability_status(worker_profile: str = "statute-worker", *, env: dict[str, str] | None = None) -> dict[str, Any]:
    hermes = shutil.which("hermes")
    if hermes:
        cmd = [hermes, "-p", worker_profile, "control", "worker", "run", "--help"]
    else:
        cmd = [sys.executable, "-m", "hermes_cli.main", "-p", worker_profile, "control", "worker", "run", "--help"]
    try:
        status = help_parse_status(cmd, env=env)
    except Exception as exc:
        return {"status": "dry_run_error", "command": cmd, "error": str(exc)}
    return {
        "status": "dry_run_ok" if status["ok"] else "dry_run_error",
        "command": cmd,
        "returncode": status["returncode"],
        "stderr": status["stderr"],
    }


def profile_id(default: str = "default") -> str:
    if os.getenv("HERMES_PROFILE_ID"):
        return os.environ["HERMES_PROFILE_ID"]
    if os.getenv("HERMES_PROFILE_NAME"):
        return os.environ["HERMES_PROFILE_NAME"]
    if os.getenv("HERMES_PROFILE"):
        return os.environ["HERMES_PROFILE"]
    try:
        from hermes_cli.profiles import get_active_profile_name

        active = get_active_profile_name()
        if active:
            return active
    except Exception:
        pass
    return default


def instance_id(default: str | None = None) -> str | None:
    return os.getenv("HERMES_CONTROL_INSTANCE_ID") or default


def register_runtime_instance(conn, *, profile: str | None = None, instance: str | None = None, lease_ms: int = 120_000) -> str:
    pid = profile or profile_id()
    iid = instance or instance_id() or f"{pid}:{os.getpid()}"
    row = conn.execute("SELECT role FROM cp_profiles WHERE profile_id=?", (pid,)).fetchone()
    role = row["role"] if row else "worker"
    if role != "worker":
        existing = cp.get_instance(conn, iid)
        if not existing or existing["profile_id"] != pid:
            raise PermissionError("pm/admin runtime identities must be bootstrap-created")
        cp.heartbeat_instance(conn, iid, lease_ms=lease_ms)
        return iid
    return cp.register_instance(conn, pid, instance_id=iid, lease_ms=lease_ms)
