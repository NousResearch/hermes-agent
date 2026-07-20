"""Weixin group policy facade over the generic group policy store."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from pathlib import Path
from typing import Any

from gateway.group_policy_store import (
    GroupPolicyStore,
    default_scope_policy,
    group_policy_store_path,
    normalize_group_policy_mode,
    normalize_group_scope_key,
    split_group_scope_key,
)


WEIXIN_GROUP_PLATFORM = "weixin"


def _normalize_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _weixin_scope_key(chat_id: str) -> str:
    normalized = str(chat_id or "").strip()
    if not normalized:
        raise ValueError("chat_id is required")
    if ":" in normalized:
        platform, normalized_chat_id = split_group_scope_key(normalized)
        if platform != WEIXIN_GROUP_PLATFORM:
            raise ValueError("Weixin group policy only supports weixin scope keys")
        return normalize_group_scope_key(platform, normalized_chat_id)
    return normalize_group_scope_key(WEIXIN_GROUP_PLATFORM, normalized)


def _scope_policy_to_weixin_policy(policy: dict[str, Any]) -> dict[str, Any]:
    scope_key = policy.get("scope_key") or _weixin_scope_key(str(policy.get("chat_id") or ""))
    _, chat_id = split_group_scope_key(scope_key)
    return {
        "scope_key": scope_key,
        "platform": policy.get("platform") or WEIXIN_GROUP_PLATFORM,
        "chat_id": policy.get("chat_id") or chat_id,
        "group_id": chat_id,
        "mode": normalize_group_policy_mode(policy.get("mode")),
        "archive_enabled": bool(policy.get("archive_enabled")),
        "daily_report_enabled": bool(policy.get("daily_report_enabled")),
        "daily_report_target": _normalize_optional_text(policy.get("daily_report_target")),
        "manual_report_target": _normalize_optional_text(policy.get("manual_report_target")),
        "purge_raw_after_rollup": bool(policy.get("purge_raw_after_rollup", True)),
        "group_name": str(policy.get("group_name") or policy.get("chat_name") or "").strip(),
        "notes": str(policy.get("notes") or "").strip(),
        "updated_at": policy.get("updated_at"),
        "updated_by": policy.get("updated_by"),
    }


def default_group_policy(chat_id: str) -> dict[str, Any]:
    if chat_id:
        normalized_scope_key = _weixin_scope_key(chat_id)
    else:
        normalized_scope_key = normalize_group_scope_key(WEIXIN_GROUP_PLATFORM, "__default__")
    _, normalized_chat_id = split_group_scope_key(normalized_scope_key)
    policy = default_scope_policy(normalized_scope_key)
    policy["group_name"] = ""
    weixin_policy = _scope_policy_to_weixin_policy(policy)
    weixin_policy["chat_id"] = normalized_chat_id if chat_id else ""
    weixin_policy["group_id"] = normalized_chat_id if chat_id else ""
    weixin_policy["scope_key"] = normalized_scope_key if chat_id else None
    return weixin_policy


def policy_reply_behavior(policy: dict[str, Any]) -> str:
    mode = normalize_group_policy_mode(policy.get("mode"))
    if mode == "collect_only":
        return "no_reply"
    if mode == "disabled":
        return "disabled"
    return "respond"


def summarize_group_policy_state(policy: dict[str, Any]) -> dict[str, Any]:
    mode = normalize_group_policy_mode(policy.get("mode"))
    daily_report_target = _normalize_optional_text(policy.get("daily_report_target"))
    manual_report_target = _normalize_optional_text(policy.get("manual_report_target"))
    return {
        "chat_id": str(policy.get("chat_id") or "").strip(),
        "group_name": str(policy.get("group_name") or "").strip() or None,
        "mode": mode,
        "collect_only": mode == "collect_only",
        "replies_disabled": mode in {"collect_only", "disabled"},
        "reply_behavior": policy_reply_behavior(policy),
        "archive_enabled": bool(policy.get("archive_enabled")),
        "daily_report_enabled": bool(policy.get("daily_report_enabled")),
        "daily_report_target": daily_report_target,
        "manual_report_target": manual_report_target,
        "has_daily_report_target": bool(daily_report_target),
        "has_manual_report_target": bool(manual_report_target),
        "purge_raw_after_rollup": bool(policy.get("purge_raw_after_rollup", True)),
        "notes": str(policy.get("notes") or "").strip() or None,
        "updated_at": policy.get("updated_at"),
        "updated_by": policy.get("updated_by"),
    }


@dataclass
class WeixinGroupPolicyStore:
    path: Path | None = None

    def __post_init__(self) -> None:
        self._store = GroupPolicyStore(path=self.path)
        self.path = self._store.path

    def get_policy(self, chat_id: str) -> dict[str, Any]:
        return _scope_policy_to_weixin_policy(self._store.get_policy(_weixin_scope_key(chat_id)))

    def list_policies(self) -> list[dict[str, Any]]:
        policies: list[dict[str, Any]] = []
        for policy in self._store.list_policies():
            if policy.get("platform") != WEIXIN_GROUP_PLATFORM:
                continue
            policies.append(_scope_policy_to_weixin_policy(policy))
        return sorted(policies, key=lambda item: str(item.get("chat_id") or ""))

    def has_policy(self, chat_id: str) -> bool:
        return self._store.has_policy(_weixin_scope_key(chat_id))

    def set_policy(
        self,
        chat_id: str,
        *,
        mode: str,
        archive_enabled: bool | None = None,
        daily_report_enabled: bool | None = None,
        daily_report_target: str | None = None,
        manual_report_target: str | None = None,
        purge_raw_after_rollup: bool | None = None,
        group_name: str | None = None,
        notes: str | None = None,
        updated_by: str | None = None,
    ) -> dict[str, Any]:
        scope_policy = self._store.set_policy(
            _weixin_scope_key(chat_id),
            mode=mode,
            archive_enabled=archive_enabled,
            daily_report_enabled=daily_report_enabled,
            daily_report_target=daily_report_target,
            manual_report_target=manual_report_target,
            purge_raw_after_rollup=purge_raw_after_rollup,
            chat_name=group_name,
            notes=notes,
            updated_by=updated_by,
        )
        return _scope_policy_to_weixin_policy(scope_policy)

    def clear_policy(self, chat_id: str) -> dict[str, Any]:
        self._store.clear_policy(_weixin_scope_key(chat_id))
        return self.get_policy(chat_id)


_policy_store_singleton: WeixinGroupPolicyStore | None = None
_policy_store_singleton_lock = threading.Lock()


def get_policy_store() -> WeixinGroupPolicyStore:
    global _policy_store_singleton
    path = group_policy_store_path()
    with _policy_store_singleton_lock:
        if _policy_store_singleton is None or _policy_store_singleton.path != path:
            _policy_store_singleton = WeixinGroupPolicyStore(path=path)
    return _policy_store_singleton


def get_group_policy(chat_id: str) -> dict[str, Any]:
    return get_policy_store().get_policy(chat_id)


def list_group_policies() -> list[dict[str, Any]]:
    return get_policy_store().list_policies()


def has_group_policy(chat_id: str) -> bool:
    return get_policy_store().has_policy(chat_id)


def set_group_policy(chat_id: str, **kwargs: Any) -> dict[str, Any]:
    return get_policy_store().set_policy(chat_id, **kwargs)


def clear_group_policy(chat_id: str) -> dict[str, Any]:
    return get_policy_store().clear_policy(chat_id)
