"""Small cross-profile relay for Feishu role dispatch.

Feishu does not reliably deliver bot-originated @mentions to another bot app.
When one Hermes profile sends a group message addressed to a role, we mirror
that instruction into the target profile's local inbox so its gateway can
process it as a synthetic inbound message.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_RELAY_BASE_DIR = Path(os.path.expanduser("~/.hermes/cross_bot_relay"))
_PROFILE_REGISTRY_PATH = _RELAY_BASE_DIR / "_registry.json"
_PROFILE_REGISTRY_LOCK_PATH = _RELAY_BASE_DIR / "_registry.lock"
_ROLE_REGISTRY_PATH = _RELAY_BASE_DIR / "_role_registry.json"
_GROUP_PROFILES_PATH = _RELAY_BASE_DIR / "group_profiles.json"
_OUTBOUND_ROLE_TOKEN_RE = re.compile(r"[A-Za-z\u4e00-\u9fff][A-Za-z0-9_\-\u4e00-\u9fff]{0,31}")
_ADDRESS_SEPARATOR_RE = re.compile(r"^[\s:：,，;；、/\\-]+")


def _relay_inbox_dir(profile_name: str) -> Path:
    return _RELAY_BASE_DIR / str(profile_name or "").strip().lower()


def _now_iso() -> str:
    return datetime.now().isoformat()


def _normalize_alias(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().casefold()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


@contextlib.contextmanager
def _profile_registry_lock() -> Any:
    _PROFILE_REGISTRY_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _PROFILE_REGISTRY_LOCK_PATH.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _load_registry() -> Dict[str, Dict[str, Any]]:
    payload = _read_json(_PROFILE_REGISTRY_PATH)
    return {str(k).strip().lower(): dict(v) for k, v in payload.items() if isinstance(v, dict)}


def register_profile(
    profile_name: str,
    *,
    bot_user_id: str = "",
    bot_open_id: str = "",
    bot_name: str = "",
    bot_name_aliases: Optional[List[str]] = None,
) -> None:
    profile = str(profile_name or "").strip().lower()
    if not profile:
        return
    with _profile_registry_lock():
        registry = _load_registry()
        existing = registry.get(profile, {})
        aliases = [str(a or "").strip() for a in (bot_name_aliases or []) if str(a or "").strip()]
        if not aliases and isinstance(existing.get("bot_name_aliases"), list):
            aliases = [str(a or "").strip() for a in existing.get("bot_name_aliases", []) if str(a or "").strip()]
        registry[profile] = {
            "profile": profile,
            "bot_user_id": str(bot_user_id or "").strip() or str(existing.get("bot_user_id", "") or "").strip(),
            "bot_open_id": str(bot_open_id or "").strip() or str(existing.get("bot_open_id", "") or "").strip(),
            "bot_name": str(bot_name or "").strip() or str(existing.get("bot_name", "") or "").strip(),
            "bot_name_aliases": aliases,
            "updated_at": _now_iso(),
        }
        _write_json_atomic(_PROFILE_REGISTRY_PATH, registry)


def _default_group_profile() -> Dict[str, Any]:
    return {"group_name": "default", "chat_ids": [], "roles": {}}


def _load_group_profile(chat_id: str = "") -> Dict[str, Any]:
    payload = _read_json(_GROUP_PROFILES_PATH)
    groups = payload.get("groups", {}) if isinstance(payload, dict) else {}
    if not isinstance(groups, dict):
        return _default_group_profile()
    chat = str(chat_id or "").strip()
    if chat:
        for group_id, group in groups.items():
            if not isinstance(group, dict):
                continue
            chat_ids = group.get("chat_ids", [])
            if isinstance(chat_ids, list) and chat in {str(item or "").strip() for item in chat_ids}:
                result = dict(group)
                result["group_profile_id"] = str(group_id or "")
                return result
    default = groups.get("default") if isinstance(groups.get("default"), dict) else None
    result = dict(default or _default_group_profile())
    result["group_profile_id"] = "default"
    return result


def _normalize_role_registry(raw_roles: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    for role_code, info in raw_roles.items():
        role_key = str(role_code or "").strip().upper()
        if not role_key or not isinstance(info, dict):
            continue
        aliases = info.get("aliases", [])
        fallback_profiles = info.get("fallback_profiles", [])
        if not isinstance(aliases, list):
            aliases = []
        if not isinstance(fallback_profiles, list):
            fallback_profiles = []
        registry[role_key] = {
            "profile": str(info.get("profile", "") or "").strip().lower(),
            "role": str(info.get("role", "") or "").strip(),
            "aliases": [str(a or "").strip() for a in aliases if str(a or "").strip()],
            "fallback_profiles": [
                str(p or "").strip().lower() for p in fallback_profiles if str(p or "").strip()
            ],
        }
    return registry


def _role_registry_for_chat(chat_id: str = "") -> Dict[str, Dict[str, Any]]:
    group_profile = _load_group_profile(chat_id)
    roles = group_profile.get("roles", {})
    if isinstance(roles, dict) and roles:
        return _normalize_role_registry(roles)
    return _normalize_role_registry(_read_json(_ROLE_REGISTRY_PATH))


def _resolve_routable_profile(primary_profile: str, fallback_profiles: List[str]) -> str:
    candidates: List[str] = []
    primary = str(primary_profile or "").strip().lower()
    if primary:
        candidates.append(primary)
    for fallback in fallback_profiles or []:
        candidate = str(fallback or "").strip().lower()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    if not candidates:
        return ""
    registered = _load_registry()
    for candidate in candidates:
        if candidate in registered:
            return candidate
    return candidates[0]


def _build_alias_map(chat_id: str = "") -> Dict[str, tuple[str, str, List[str]]]:
    alias_map: Dict[str, tuple[str, str, List[str]]] = {}
    for role_code, info in _role_registry_for_chat(chat_id).items():
        profile_name = str(info.get("profile", "") or "").strip().lower()
        fallback_profiles = [
            str(p or "").strip().lower()
            for p in info.get("fallback_profiles", [])
            if str(p or "").strip()
        ]
        alias_map[_normalize_alias(role_code)] = (role_code, profile_name, fallback_profiles)
        for alias in info.get("aliases", []):
            normalized = _normalize_alias(alias)
            if normalized and normalized not in alias_map:
                alias_map[normalized] = (role_code, profile_name, fallback_profiles)
    return alias_map


def _first_non_empty_line(text: str) -> tuple[str, List[str]]:
    lines = str(text or "").splitlines()
    for index, line in enumerate(lines):
        if line.strip():
            return line, lines[index + 1 :]
    return "", []


def _match_leading_role_address(text: str, alias_map: Dict[str, tuple[str, str, List[str]]]) -> Optional[Dict[str, str]]:
    first_line, remaining_lines = _first_non_empty_line(text)
    if not first_line:
        return None
    stripped = first_line.lstrip()
    if stripped.startswith("@"):
        stripped = stripped[1:].lstrip()

    best: Optional[re.Match[str]] = None
    best_alias = ""
    for match in _OUTBOUND_ROLE_TOKEN_RE.finditer(stripped):
        if match.start() != 0:
            break
        token = match.group(0)
        normalized = _normalize_alias(token)
        if normalized in alias_map and (best is None or len(token) > len(best_alias)):
            best = match
            best_alias = token
        break
    if best is None:
        return None

    remainder = stripped[best.end():]
    if remainder and not _ADDRESS_SEPARATOR_RE.match(remainder):
        return None
    remainder = _ADDRESS_SEPARATOR_RE.sub("", remainder, count=1).strip()
    tail_parts = []
    if remainder:
        tail_parts.append(remainder)
    tail_parts.extend(line for line in remaining_lines if line.strip())
    instruction = "\n".join(tail_parts).strip() or "请按上一条消息执行。"
    role_code, primary_profile, fallback_profiles = alias_map[_normalize_alias(best_alias)]
    return {
        "role_code": role_code,
        "display_name": best_alias,
        "profile_name": _resolve_routable_profile(primary_profile, fallback_profiles),
        "instruction": instruction,
    }


def compile_outbound_role_mentions(
    text: str,
    *,
    chat_id: str = "",
    exclude_profile: str = "",
) -> List[Dict[str, str]]:
    """Return target profile(s) when outbound text is addressed to a role.

    This intentionally only dispatches command-shaped messages whose first
    meaningful token is a configured role code or alias. It does not route every
    incidental role name found inside normal prose.
    """
    alias_map = _build_alias_map(chat_id)
    if not alias_map:
        return []
    target = _match_leading_role_address(text, alias_map)
    if not target:
        return []
    profile_name = str(target.get("profile_name", "") or "").strip().lower()
    exclude = str(exclude_profile or "").strip().lower()
    if not profile_name or profile_name == exclude:
        return []
    identity = _load_registry().get(profile_name, {})
    return [
        {
            "mention_id": str(identity.get("bot_user_id", "") or identity.get("bot_open_id", "") or "").strip(),
            "display_name": str(target.get("display_name", "") or "").strip(),
            "profile_name": profile_name,
            "role_code": str(target.get("role_code", "") or "").strip().upper(),
            "instruction": str(target.get("instruction", "") or "").strip(),
        }
    ]


def enqueue_relay(
    *,
    target_profile: str,
    chat_id: str,
    text: str,
    sender_profile: str,
    sender_display_name: str = "",
    message_id: str = "",
    workflow_id: str = "",
    task_id: str = "",
    outbound_id: str = "",
    relay_type: str = "text_relay",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    target = str(target_profile or "").strip().lower()
    if not target or not chat_id or not text:
        return False
    inbox = _relay_inbox_dir(target)
    inbox.mkdir(parents=True, exist_ok=True)
    envelope = {
        "kind": "text_relay",
        "relay_type": str(relay_type or "text_relay").strip(),
        "chat_id": str(chat_id or "").strip(),
        "text": str(text or "").strip(),
        "sender_profile": str(sender_profile or "").strip().lower(),
        "sender_display_name": str(sender_display_name or sender_profile or "").strip(),
        "message_id": str(message_id or "").strip(),
        "workflow_id": str(workflow_id or "").strip(),
        "task_id": str(task_id or "").strip(),
        "outbound_id": str(outbound_id or "").strip(),
        "created_at": _now_iso(),
    }
    if isinstance(metadata, dict):
        envelope.update({str(k): v for k, v in metadata.items() if str(k or "").strip()})
    path = inbox / f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.json"
    try:
        path.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
        logger.info(
            "[CrossBotRelay] Enqueued relay %s -> %s chat=%s text=%r",
            envelope["sender_profile"],
            target,
            chat_id,
            envelope["text"][:80],
        )
        return True
    except Exception:
        logger.warning("[CrossBotRelay] Failed to enqueue relay to %s", target, exc_info=True)
        return False


def dequeue_relays(profile_name: str, max_items: int = 10) -> List[Dict[str, Any]]:
    profile = str(profile_name or "").strip().lower()
    if not profile:
        return []
    inbox = _relay_inbox_dir(profile)
    if not inbox.is_dir():
        return []
    envelopes: List[Dict[str, Any]] = []
    for path in sorted(inbox.glob("*.json"))[:max(1, int(max_items))]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                envelopes.append(payload)
        except Exception:
            logger.warning("[CrossBotRelay] Dropping unreadable relay %s", path, exc_info=True)
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    if envelopes:
        logger.info("[CrossBotRelay] Dequeued %d relay(s) for profile %s", len(envelopes), profile)
    return envelopes
