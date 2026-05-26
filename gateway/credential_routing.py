"""Gateway-scoped credential routing helpers.

This module intentionally keeps routing decisions close to the gateway
source (platform/chat/thread) instead of changing global provider selection.
Routes can allow/deny credential labels and can mark some labels as
conditional on a fresh quota snapshot.  Conditional labels fail closed when
quota data is absent, stale, invalid, or over threshold.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _platform_value(source: Any) -> str:
    platform = getattr(source, "platform", None)
    return str(getattr(platform, "value", platform) or "").strip().lower()


def _matches_any(value: Any, candidates: Any) -> bool:
    allowed = {str(item) for item in _as_list(candidates)}
    if not allowed:
        return True
    return str(value or "") in allowed


def _route_matches(rule: dict[str, Any], source: Any, provider: str) -> bool:
    rule_provider = str(rule.get("provider") or "").strip().lower()
    if rule_provider and rule_provider != str(provider or "").strip().lower():
        return False

    rule_platform = str(rule.get("platform") or "").strip().lower()
    if rule_platform and rule_platform != _platform_value(source):
        return False

    if not _matches_any(getattr(source, "chat_id", None), rule.get("chat_ids") or rule.get("chat_id")):
        return False
    if not _matches_any(getattr(source, "thread_id", None), rule.get("thread_ids") or rule.get("thread_id")):
        return False
    if not _matches_any(getattr(source, "chat_name", None), rule.get("chat_names") or rule.get("chat_name")):
        return False
    if not _matches_any(getattr(source, "chat_type", None), rule.get("chat_types") or rule.get("chat_type")):
        return False
    return True


def resolve_gateway_credential_route(
    *,
    user_config: Optional[dict[str, Any]],
    source: Any,
    provider: str,
) -> Optional[dict[str, Any]]:
    """Return the first matching gateway credential route for a source."""
    cfg = user_config or {}
    routing = cfg.get("gateway_credential_routing")
    if not isinstance(routing, dict):
        return None
    rules = routing.get("rules")
    if not isinstance(rules, list):
        return None
    for rule in rules:
        if isinstance(rule, dict) and _route_matches(rule, source, provider):
            return rule
    return None


def _entry_label(entry: Any) -> str:
    return str(getattr(entry, "label", "") or getattr(entry, "id", "") or "")


def _quota_value(quota: dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = quota.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _timestamp(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
        return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
    except (TypeError, ValueError):
        pass
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def conditional_quota_allows(entry: Any, rule: dict[str, Any], *, now: Optional[float] = None) -> bool:
    """Fail-closed quota gate for labels that require live/fresh quota proof."""
    label = _entry_label(entry)
    conditional = set(_as_list(rule.get("conditional_labels") or rule.get("quota_checked_labels")))
    if label not in conditional:
        return True

    extra = getattr(entry, "extra", None)
    quota = extra.get("quota") if isinstance(extra, dict) else None
    if not isinstance(quota, dict):
        logger.warning("gateway credential route: label=%s denied; quota missing", label)
        return False

    now = time.time() if now is None else now
    ttl = float(rule.get("quota_refresh_ttl_seconds") or rule.get("quota_ttl_seconds") or 300)
    observed = _timestamp(quota.get("observed_at"))
    if observed is None or now - observed > ttl:
        logger.warning("gateway credential route: label=%s denied; quota stale", label)
        return False

    primary = _quota_value(quota, "primary_percent", "primary_used_percent", "primary_usedPercent")
    secondary = _quota_value(quota, "secondary_percent", "secondary_used_percent", "secondary_usedPercent")
    max_primary = float(rule.get("max_primary_percent") or rule.get("primary_max_percent") or 80)
    max_secondary = float(rule.get("max_secondary_percent") or rule.get("secondary_max_percent") or 80)

    if primary is None or secondary is None:
        logger.warning("gateway credential route: label=%s denied; incomplete quota", label)
        return False
    if primary >= max_primary or secondary >= max_secondary:
        logger.info(
            "gateway credential route: label=%s denied; quota primary=%.1f secondary=%.1f max_primary=%.1f max_secondary=%.1f",
            label,
            primary,
            secondary,
            max_primary,
            max_secondary,
        )
        return False
    return True


def select_entry_for_gateway_route(pool: Any, rule: dict[str, Any]) -> Any:
    """Select a credential entry for a matched route without mutating pool contents.

    The selection is deliberately conservative: allow/deny filters first,
    conditional quota labels fail closed, then pick the lowest priority/request
    count among available entries.  This avoids constructing a temporary pool
    whose persistence path could overwrite the full credential pool.
    """
    allow = set(_as_list(rule.get("allow_labels")))
    deny = set(_as_list(rule.get("deny_labels")))

    try:
        entries = list(pool._available_entries(clear_expired=True, refresh=True))  # noqa: SLF001 - intentional internal gateway integration
    except Exception:
        entries = list(pool.entries()) if hasattr(pool, "entries") else []

    candidates = []
    for entry in entries:
        label = _entry_label(entry)
        if allow and label not in allow:
            continue
        if label in deny:
            continue
        if not conditional_quota_allows(entry, rule):
            continue
        candidates.append(entry)

    if not candidates:
        return None

    selected = min(
        candidates,
        key=lambda entry: (
            int(getattr(entry, "request_count", 0) or 0),
            int(getattr(entry, "priority", 0) or 0),
            _entry_label(entry),
        ),
    )
    try:
        from dataclasses import replace

        updated = replace(selected, request_count=int(getattr(selected, "request_count", 0) or 0) + 1)
        if hasattr(pool, "_replace_entry"):
            pool._replace_entry(selected, updated)  # noqa: SLF001 - preserve full pool while recording route use
        if hasattr(pool, "_persist"):
            pool._persist()  # noqa: SLF001
        return updated
    except Exception:
        return selected


def route_runtime_kwargs(
    runtime: dict[str, Any],
    *,
    source: Any,
    user_config: Optional[dict[str, Any]],
    pool_loader: Callable[[str], Any],
) -> dict[str, Any]:
    """Apply a matched gateway credential route to runtime kwargs."""
    provider = str(runtime.get("provider") or "").strip().lower()
    if not provider:
        return runtime
    rule = resolve_gateway_credential_route(user_config=user_config, source=source, provider=provider)
    if not rule:
        return runtime

    pool = runtime.get("credential_pool")
    if pool is None:
        try:
            pool = pool_loader(provider)
        except Exception as exc:
            logger.warning("gateway credential route: failed to load pool for provider=%s: %s", provider, exc)
            return runtime

    selected = select_entry_for_gateway_route(pool, rule)
    if selected is None:
        if str(rule.get("fallback_policy") or "").strip().lower() == "fail_closed":
            raise RuntimeError(f"No gateway credential route candidates available for provider {provider}")
        return runtime

    routed = dict(runtime)
    routed["api_key"] = getattr(selected, "runtime_api_key", None) or getattr(selected, "access_token", "")
    routed["base_url"] = getattr(selected, "runtime_base_url", None) or getattr(selected, "base_url", None) or routed.get("base_url")
    routed["credential_pool"] = None
    routed["credential_label"] = _entry_label(selected)
    logger.info(
        "Gateway credential route applied: platform=%s chat_id=%s provider=%s label=%s",
        _platform_value(source),
        getattr(source, "chat_id", None),
        provider,
        routed["credential_label"],
    )
    return routed
