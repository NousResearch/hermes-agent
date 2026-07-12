"""MVP method handlers (contract §4)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from gateway.brain_rpc.auth import AuthContext
from gateway.brain_rpc.config import BrainRpcHostConfig
from gateway.brain_rpc.errors import INVALID_ARGUMENT, BrainRpcError
from gateway.brain_rpc.projects_handler import list_projects_snapshot
from gateway.brain_rpc.settings_snapshot import build_settings_snapshot
from gateway.brain_rpc.vault_ops import vault_list, vault_read, vault_stat

logger = logging.getLogger(__name__)

HandlerFn = Callable[[Dict[str, Any], AuthContext, BrainRpcHostConfig], Dict[str, Any]]

BRAIN_RPC_CONTRACT_VERSION = 1


def handle_brain_ping(
    params: Dict[str, Any],
    auth: AuthContext,
    host: BrainRpcHostConfig,
) -> Dict[str, Any]:
    echo = params.get("echo")
    result: Dict[str, Any] = {
        "pong": True,
        "instance_id": auth.instance_id,
        "contract_version": BRAIN_RPC_CONTRACT_VERSION,
    }
    if echo is not None:
        s = str(echo)
        if len(s) > 64:
            raise BrainRpcError(
                INVALID_ARGUMENT,
                "echo exceeds 64 chars",
                details={"limit": 64},
            )
        result["echo"] = s
    return result


def handle_brain_health(
    params: Dict[str, Any],
    auth: AuthContext,
    host: BrainRpcHostConfig,
) -> Dict[str, Any]:
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    vault_ok = host.vault_root.is_dir() and os.access(host.vault_root, os.R_OK)
    hermes_home_present = home.is_dir()
    profiles_present = (
        (host.profiles_dir / "admin.json").is_file()
        or (host.profiles_dir / "contributor.json").is_file()
        or host.profiles_dir.is_dir()
        and any(host.profiles_dir.glob("*.json"))
    )
    projects_db = home / "projects.db"
    projects_db_present = projects_db.is_file()

    checks = {
        "vault_root_readable": bool(vault_ok),
        "hermes_home_present": bool(hermes_home_present),
        "profiles_present": bool(profiles_present),
        "projects_db_present": bool(projects_db_present),
    }

    if not vault_ok:
        status = "unavailable"
    elif not projects_db_present or not profiles_present:
        status = "degraded"
    else:
        status = "ok"

    versions = {
        "brain_rpc": BRAIN_RPC_CONTRACT_VERSION,
        "hermes_artifact": _best_effort_version("HERMES_ARTIFACT_VERSION", "/opt/company-brain/.seed/hermes_artifact"),
        "client_core_seed": _best_effort_version(
            "CLIENT_CORE_SEED_VERSION", "/opt/company-brain/.seed/client_core"
        ),
    }

    return {
        "status": status,
        "checks": checks,
        "versions": versions,
        "profile": auth.subject.hermes_profile,
    }


def _best_effort_version(env_key: str, seed_path: str) -> Optional[str]:
    env = os.environ.get(env_key, "").strip()
    if env:
        return env
    p = Path(seed_path)
    if p.is_file():
        try:
            text = p.read_text(encoding="utf-8").strip()
            return text or None
        except OSError:
            return None
    return None


def handle_settings_snapshot(
    params: Dict[str, Any],
    auth: AuthContext,
    host: BrainRpcHostConfig,
) -> Dict[str, Any]:
    return build_settings_snapshot(params)


def handle_projects_list(
    params: Dict[str, Any],
    auth: AuthContext,
    host: BrainRpcHostConfig,
) -> Dict[str, Any]:
    return list_projects_snapshot(params)


HANDLERS: Dict[str, HandlerFn] = {
    "brain.ping": handle_brain_ping,
    "brain.health": handle_brain_health,
    "vault.list": vault_list,
    "vault.stat": vault_stat,
    "vault.read": vault_read,
    "settings.snapshot": handle_settings_snapshot,
    "projects.list": handle_projects_list,
}
