"""Workflow profile loading and validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class WorkflowProfileError(ValueError):
    """Raised when a workflow profile is missing or invalid."""


@dataclass(frozen=True)
class WorkflowRole:
    role_id: str
    name: str = ""
    aliases: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    profile: str = ""
    can_review: bool = False
    can_finalize: bool = False


@dataclass(frozen=True)
class WorkflowProfile:
    profile_id: str
    name: str
    enabled: bool
    dispatcher_role: str
    roles: tuple[WorkflowRole, ...]
    message_rules: Dict[str, Any] = field(default_factory=dict)

    def role_ids(self) -> set[str]:
        return {role.role_id for role in self.roles}

    def get_role(self, role_id: str) -> Optional[WorkflowRole]:
        normalized = _normalize_role_id(role_id)
        for role in self.roles:
            if role.role_id == normalized:
                return role
        return None

    def find_role_by_capability(self, capability: str) -> Optional[WorkflowRole]:
        capability = str(capability or "").strip()
        for role in self.roles:
            if capability in role.capabilities:
                return role
        return None

    def find_roles_in_text(self, text: str) -> List[WorkflowRole]:
        lowered = str(text or "").casefold()
        found: List[WorkflowRole] = []
        for role in self.roles:
            candidates = [role.role_id, role.name, *role.aliases]
            if any(_role_token_present(lowered, candidate) for candidate in candidates if candidate):
                found.append(role)
        return found


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_profile_path(profile_id: str) -> Path:
    configured_root = os.getenv("HERMES_WORKFLOW_PROFILE_DIR", "").strip()
    root = Path(configured_root).expanduser() if configured_root else _repo_root() / "workflow_profiles"
    return root / profile_id / "current.json"


def _normalize_role_id(value: str) -> str:
    return str(value or "").strip().lower()


def _as_string_list(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise WorkflowProfileError(f"{field_name} must be a list")
    items = tuple(str(item).strip() for item in value if str(item or "").strip())
    return items


def _role_token_present(text: str, token: str) -> bool:
    token = str(token or "").strip().casefold()
    if not token:
        return False
    if any("\u4e00" <= ch <= "\u9fff" for ch in token):
        return token in text
    separators = " \t\r\n:：,，;；/\\|-()[]{}"
    start = 0
    while True:
        index = text.find(token, start)
        if index < 0:
            return False
        before = text[index - 1] if index > 0 else " "
        after_index = index + len(token)
        after = text[after_index] if after_index < len(text) else " "
        if before in separators and after in separators:
            return True
        start = index + len(token)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise WorkflowProfileError(f"Workflow profile not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkflowProfileError(f"Workflow profile JSON is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WorkflowProfileError(f"Workflow profile must be a JSON object: {path}")
    return payload


def _parse_roles(raw_roles: Any) -> tuple[WorkflowRole, ...]:
    if not isinstance(raw_roles, list) or not raw_roles:
        raise WorkflowProfileError("roles must be a non-empty list")

    roles: List[WorkflowRole] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_roles):
        if not isinstance(raw, dict):
            raise WorkflowProfileError(f"roles[{index}] must be an object")
        role_id = _normalize_role_id(raw.get("role_id"))
        if not role_id:
            raise WorkflowProfileError(f"roles[{index}].role_id is required")
        if role_id in seen:
            raise WorkflowProfileError(f"duplicate role_id: {role_id}")
        seen.add(role_id)
        capabilities = _as_string_list(raw.get("capabilities"), field_name=f"roles[{index}].capabilities")
        roles.append(
            WorkflowRole(
                role_id=role_id,
                name=str(raw.get("name") or role_id).strip(),
                aliases=_as_string_list(raw.get("aliases"), field_name=f"roles[{index}].aliases"),
                capabilities=capabilities,
                profile=str(raw.get("profile") or role_id).strip().lower(),
                can_review=bool(raw.get("can_review", False)),
                can_finalize=bool(raw.get("can_finalize", False)),
            )
        )
    return tuple(roles)


def validate_workflow_profile(payload: Dict[str, Any]) -> WorkflowProfile:
    profile_id = str(payload.get("profile_id") or "").strip()
    if not profile_id:
        raise WorkflowProfileError("profile_id is required")
    name = str(payload.get("name") or profile_id).strip()
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        raise WorkflowProfileError("enabled must be a boolean")
    dispatcher_role = _normalize_role_id(payload.get("dispatcher_role"))
    if not dispatcher_role:
        raise WorkflowProfileError("dispatcher_role is required")
    roles = _parse_roles(payload.get("roles"))
    if dispatcher_role not in {role.role_id for role in roles}:
        raise WorkflowProfileError(f"dispatcher_role is not defined in roles: {dispatcher_role}")
    message_rules = payload.get("message_rules") or {}
    if not isinstance(message_rules, dict):
        raise WorkflowProfileError("message_rules must be an object")
    return WorkflowProfile(
        profile_id=profile_id,
        name=name,
        enabled=enabled,
        dispatcher_role=dispatcher_role,
        roles=roles,
        message_rules=dict(message_rules),
    )


def load_workflow_profile(profile_id: str, path: Optional[Path] = None) -> WorkflowProfile:
    profile_path = path or _default_profile_path(profile_id)
    return validate_workflow_profile(_load_json(profile_path))


def roles_for_capability(profile: WorkflowProfile, capability: str) -> Iterable[WorkflowRole]:
    return (role for role in profile.roles if capability in role.capabilities)
