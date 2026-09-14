"""Harness control-room aggregation and profile-scoped lifecycle API."""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, StrictBool

from hermes_cli.web_routers._common import _spawn_hermes_action

router = APIRouter(prefix="/api")

_ACTION_LOCK = threading.Lock()
_ACTIONS: dict[str, dict[str, Any]] = {}
_ALLOWED_VERBS = {"start", "stop", "restart"}


class HarnessActionRequest(BaseModel):
    confirmed: StrictBool


def _safe_text(value: Any, *, max_len: int = 120) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value[:max_len] if value else None


def _safe_pid(value: Any) -> Optional[int]:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _profile_rows() -> list[tuple[str, Any]]:
    from hermes_cli.profiles import profiles_to_serve
    try:
        return list(profiles_to_serve(True))
    except Exception:
        return []


def _runtime_for(home: Any) -> dict[str, Any]:
    try:
        from gateway.status import read_runtime_status
        value = read_runtime_status(home / "gateway_state.json")
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _gateway_pid(home: Any) -> Optional[int]:
    try:
        from gateway.status import get_running_pid
        return _safe_pid(get_running_pid(home / "gateway.pid"))
    except Exception:
        return None


def _platforms(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    raw = runtime.get("platforms")
    if not isinstance(raw, dict):
        return []
    out = []
    for name, state in raw.items():
        safe_name = _safe_text(name, max_len=64)
        if safe_name is None or not isinstance(state, dict):
            continue
        item = {"name": safe_name}
        safe_state = _safe_text(state.get("state"), max_len=32)
        if safe_state:
            item["state"] = safe_state
        out.append(item)
    return out


def _safe_served_profiles(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        safe = _safe_text(item, max_len=64)
        if safe is not None:
            out.append(safe)
        if len(out) >= 64:
            break
    return out


def _profile_payload(name: str, home: Any) -> dict[str, Any]:
    runtime = _runtime_for(home)
    pid = _gateway_pid(home)
    gateway_state = "running" if pid is not None else "stopped"
    return {
        "name": _safe_text(name, max_len=64) or "unknown",
        "gateway_state": gateway_state,
        "pid": pid,
        "platforms": _platforms(runtime),
        "served_profiles": _safe_served_profiles(runtime.get("served_profiles")),
        "updated_at": _safe_text(runtime.get("updated_at"), max_len=64),
    }


def _find_profile(name: str):
    for profile, home in _profile_rows():
        if profile == name:
            return home
    raise HTTPException(status_code=404, detail="profile_not_found")


def _profile_action_argv(profile: str, verb: str) -> list[str]:
    """Build an explicitly scoped argv, including the default profile.

    ``_profile_cli_args`` intentionally omits ``default`` for the normal web
    routes.  Lifecycle children need an explicit selector so their scrubbed
    environment is pinned to the requested home instead of inheriting the
    dashboard process's profile.
    """
    from hermes_cli import profiles as profiles_mod

    canonical = profiles_mod.normalize_profile_name(profile)
    return ["-p", canonical, "gateway", verb]


@router.get("/harnesses")
def harnesses() -> dict[str, Any]:
    rows = [_profile_payload(name, home) for name, home in _profile_rows()]
    return {"profiles": rows, "count": len(rows)}


@router.post("/harnesses/{profile}/actions/{verb}")
def launch_action(profile: str, verb: str, body: HarnessActionRequest) -> dict[str, Any]:
    if verb not in _ALLOWED_VERBS:
        raise HTTPException(status_code=404, detail="action_not_found")
    if not body.confirmed:
        raise HTTPException(status_code=400, detail="confirmation_required")
    _find_profile(profile)
    from hermes_cli import profiles as profiles_mod
    canonical_profile = profiles_mod.normalize_profile_name(profile)
    from hermes_cli.web_server_gateway import multiplexed_profile_refusal
    if multiplexed_profile_refusal(profile, verb) is not None:
        raise HTTPException(status_code=409, detail="multiplexed_profile_refusal")
    action_name = f"harness:{canonical_profile}:{verb}"
    with _ACTION_LOCK:
        for current in _ACTIONS.values():
            if current.get("profile") != canonical_profile:
                continue
            process = current.get("process")
            if process is not None and process.poll() is None:
                raise HTTPException(status_code=409, detail="action_in_progress")
        invocation_id = uuid.uuid4().hex
        argv = _profile_action_argv(canonical_profile, verb)
        try:
            proc = _spawn_hermes_action(argv, action_name)
        except Exception:
            raise HTTPException(status_code=500, detail="action_failed")
        _ACTIONS[action_name] = {
            "process": proc,
            "invocation_id": invocation_id,
            "profile": canonical_profile,
            "verb": verb,
            "started_at": time.time(),
        }
    return {"ok": True, "profile": profile, "verb": verb, "invocation_id": invocation_id}


@router.get("/harnesses/{profile}/actions/{verb}/status")
def action_status(profile: str, verb: str, invocation_id: str = "") -> dict[str, Any]:
    if verb not in _ALLOWED_VERBS:
        raise HTTPException(status_code=404, detail="action_not_found")
    from hermes_cli import profiles as profiles_mod
    canonical_profile = profiles_mod.normalize_profile_name(profile)
    action_name = f"harness:{canonical_profile}:{verb}"
    with _ACTION_LOCK:
        entry = _ACTIONS.get(action_name)
        if not entry or entry.get("invocation_id") != invocation_id:
            return {"status": "unknown", "invocation_id": invocation_id}
        proc = entry["process"]
        code = proc.poll()
        if code is None:
            return {"status": "running", "invocation_id": invocation_id}
        return {
            "status": "completed" if code == 0 else "failed",
            "invocation_id": invocation_id,
            "exit_code": code if isinstance(code, int) else None,
        }
