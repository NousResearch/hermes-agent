"""Fail-closed resource/power admission for Kanban dispatcher claims."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Optional

_log = logging.getLogger(__name__)
SCHEMA_VERSION = 1
MATERIAL_CLASSES = frozenset({"material", "local_intensive", "dangerous", "production", "integrator", "release", "migration", "schema", "routing", "lockfile", "manifest", "control_plane_exclusive"})
READONLY_CLASSES = frozenset({"readonly_qa"})
CLASS_ALIASES = {"normal": "material", "read_only": "readonly_qa", "read-only": "readonly_qa", "control_plane": "control_plane_exclusive"}

class ResourcePolicyError(ValueError):
    """Policy cannot safely evaluate an admission decision."""


def _safe_reason(value: Any) -> str:
    text = " ".join(str(value).split())
    return text[:240] or "unspecified policy error"


def load_policy(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ResourcePolicyError(f"resource policy load failed: {_safe_reason(exc)}") from exc
    if not isinstance(value, dict):
        raise ResourcePolicyError("resource policy must be an object")
    schema = value.get("schema_version", value.get("version"))
    if schema != SCHEMA_VERSION:
        raise ResourcePolicyError("unsupported resource policy schema")
    required = ("aggregate_material_lanes", "max_read_only_qa_lanes", "per_profile_material", "power")
    if any(key not in value for key in required):
        raise ResourcePolicyError("resource policy missing required fields")
    for key in required[:3]:
        if not isinstance(value[key], int) or value[key] < 0:
            raise ResourcePolicyError(f"resource policy field {key} is invalid")
    if not isinstance(value["power"], dict):
        raise ResourcePolicyError("resource policy power section is invalid")
    classes = value.get("resource_classes")
    if not isinstance(classes, list) or "material" not in classes or "readonly_qa" not in classes:
        raise ResourcePolicyError("resource policy resource_classes are invalid")
    return value


def _rule(policy: Mapping[str, Any], task_id: str, assignee: Optional[str], board: str) -> dict[str, Any]:
    overrides = policy.get("task_overrides") or {}
    profiles = policy.get("profile_defaults") or {}
    boards = policy.get("board_defaults") or {}
    raw = overrides.get(task_id)
    if raw is None:
        raw = profiles.get(assignee or "")
    if raw is None:
        raw = boards.get(board, boards.get("*"))
    if raw is None:
        raw = policy.get("defaults") or {"resource_class": "material"}
    if isinstance(raw, str):
        raw = {"resource_class": raw}
    if not isinstance(raw, dict):
        raise ResourcePolicyError("resource classification rule is invalid")
    return dict(raw)


def classify(policy: Mapping[str, Any], task_id: str, assignee: Optional[str], board: str) -> tuple[str, dict[str, Any]]:
    rule = _rule(policy, task_id, assignee, board)
    cls = CLASS_ALIASES.get(str(rule.get("resource_class", "material")).strip().lower(), str(rule.get("resource_class", "material")).strip().lower())
    if cls not in MATERIAL_CLASSES | READONLY_CLASSES:
        raise ResourcePolicyError("resource classification is invalid")
    rule["resource_class"] = cls
    rule["components"] = sorted({str(v).strip() for v in (rule.get("components") or []) if str(v).strip()})
    rule["mutable_paths"] = sorted({str(v).strip() for v in (rule.get("mutable_paths") or []) if str(v).strip()})
    rule["ports"] = sorted({str(v).strip() for v in (rule.get("ports") or []) if str(v).strip()})
    rule["external_accounts"] = sorted({str(v).strip() for v in (rule.get("external_accounts") or []) if str(v).strip()})
    rule["release_authority"] = str(rule.get("release_authority") or "").strip()
    return cls, rule


def normalize_workspace(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        p = Path(str(path)).expanduser()
        if not p.is_absolute():
            return None
        return str(p.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return None


def git_common_origin(path: Optional[str]) -> Optional[str]:
    normalized = normalize_workspace(path)
    if not normalized:
        return None
    probe = Path(normalized)
    if not probe.exists():
        probe = probe.parent
    try:
        result = subprocess.run(["git", "-C", str(probe), "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True, timeout=3, check=False)
        if result.returncode != 0:
            return None
        common = Path((result.stdout or "").strip())
        if not common.is_absolute():
            common = (probe / common)
        return str(common.resolve(strict=False))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def candidate(*, task_id: str, assignee: Optional[str], board: str, status: str, workspace: Optional[str], git_origin: Optional[str], policy: Mapping[str, Any]) -> dict[str, Any]:
    cls, rule = classify(policy, task_id, assignee, board)
    return {"task_id": task_id, "profile": assignee, "board": board, "status": status, "workspace": normalize_workspace(workspace), "git_origin": git_origin or git_common_origin(workspace), "resource_class": cls, "components": rule["components"], "mutable_paths": rule["mutable_paths"], "ports": rule["ports"], "external_accounts": rule["external_accounts"], "release_authority": rule["release_authority"]}


def _power_state(policy: Mapping[str, Any], now: Optional[float] = None) -> tuple[str, str]:
    power = policy["power"]
    path = Path(str(power.get("state_path", ""))).expanduser()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        state = str(raw.get("state", "")).strip()
        event = raw.get("last_event") or {}
        telemetry = event.get("telemetry") or {}
        timestamp = max(float(raw.get("updated_at", 0) or 0), float(event.get("timestamp", 0) or 0), float(telemetry.get("timestamp", 0) or 0))
        max_age = float(power.get("max_age_seconds", 300))
        age = (time.time() if now is None else float(now)) - timestamp
        if not state or age < 0 or age > max_age:
            return "STALE", "power telemetry stale or missing"
        if state in set(power.get("normal_states") or []):
            return state, "AC_OK"
        if state in set(power.get("ac_loss_states") or []):
            return state, "fresh AC-loss state"
        return "UNKNOWN", "unknown power state"
    except Exception as exc:
        return "UNKNOWN", f"power state unavailable: {_safe_reason(exc)}"


def _overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> Optional[str]:
    for key, label in (("mutable_paths", "mutable path"), ("ports", "port"), ("external_accounts", "external account")):
        if set(left.get(key) or []).intersection(right.get(key) or []):
            return f"{label} overlap"
    if left.get("release_authority") and left.get("release_authority") == right.get("release_authority"):
        return "release authority overlap"
    if left.get("workspace") and left.get("workspace") == right.get("workspace"):
        return "workspace overlap"
    if left.get("git_origin") and left.get("git_origin") == right.get("git_origin"):
        if left.get("components") and right.get("components") and set(left["components"]).isdisjoint(right["components"]):
            return None
        return "git common origin overlap"
    if set(left.get("components") or []).intersection(right.get("components") or []):
        return "declared component overlap"
    return None


def admit(candidate_row: Mapping[str, Any], active: list[Mapping[str, Any]], policy: Mapping[str, Any], *, now: Optional[float] = None) -> tuple[bool, str]:
    cls = str(candidate_row.get("resource_class", "material"))
    if candidate_row.get("status") == "triage":
        return False, "triage is not dispatchable; no portfolio admission effect"
    if cls in MATERIAL_CLASSES and not candidate_row.get("workspace"):
        return False, "material workspace unresolved"
    power_state, power_reason = _power_state(policy, now=now)
    power = policy["power"]
    if power_state in set(power.get("ac_loss_states") or []) and cls in set(power.get("ac_loss_denies") or []):
        return False, "fresh AC-loss denies classified intensive/dangerous work"
    if power_state in {"UNKNOWN", "STALE"} and cls in {"local_intensive", "dangerous"}:
        return False, f"{power_reason}; intensive/dangerous admission is fail-closed"
    # A tick can reconcile a crashed task and immediately retry it. Ignore a
    # stale row for the same task id defensively so it cannot deny itself.
    live = [
        row for row in active
        if row.get("status") == "running"
        and row.get("status") != "triage"
        and row.get("task_id") != candidate_row.get("task_id")
    ]
    for row in live:
        conflict = _overlap(candidate_row, row)
        if conflict:
            return False, conflict
    if cls in set(policy.get("exclusive_resource_classes") or []) or cls == "control_plane_exclusive":
        if any(str(row.get("resource_class")) in set(policy.get("single_flight_classes") or policy.get("exclusive_resource_classes") or []) or str(row.get("resource_class")) == "control_plane_exclusive" for row in live):
            return False, "exclusive resource class is single-flight"
    if cls in MATERIAL_CLASSES:
        if sum(1 for row in live if row.get("resource_class") in MATERIAL_CLASSES) >= int(policy["aggregate_material_lanes"]):
            return False, "aggregate material lane cap"
        if sum(1 for row in live if row.get("resource_class") in MATERIAL_CLASSES and row.get("profile") == candidate_row.get("profile")) >= int(policy["per_profile_material"]):
            return False, "same profile material cap"
    else:
        if sum(1 for row in live if row.get("resource_class") in READONLY_CLASSES) >= int(policy["max_read_only_qa_lanes"]):
            return False, "read-only/QA lane cap"
    return True, "admitted"


def decision(*, task_id: str, assignee: Optional[str], board: str, status: str, workspace: Optional[str], active: list[Mapping[str, Any]], policy: Mapping[str, Any], git_origin: Optional[str] = None, now: Optional[float] = None) -> tuple[dict[str, Any], bool, str]:
    row = candidate(task_id=task_id, assignee=assignee, board=board, status=status, workspace=workspace, git_origin=git_origin, policy=policy)
    allowed, reason = admit(row, active, policy, now=now)
    return row, allowed, reason
