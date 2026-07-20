"""Application-level admin/user tier for agent tool surface (Phase 1 of #20744).

Complements ``gateway.slash_access`` (slash commands only). When operators list
``allow_admin_from`` for a platform scope, non-admin chat users can still talk
to the agent, but their **tool schema** and optional **iteration budget** are
clamped via config — without a policy engine or DB.

Config (per-platform ``extra`` / bridged platform cfg, or top-level ``gateway``):

- ``allow_admin_from`` / ``group_allow_admin_from`` — same meaning as slash access
  (tier gating is **on** only when the relevant admin list is non-empty).
- ``user_toolsets`` — optional list of toolset names non-admins may use. When
  tier gating is on and this list is empty/absent, non-admins get a small safe
  default (web + no memory / skill_manage / cron / messaging interface).
- ``user_max_iterations`` — optional int clamp for non-admin turns.

Admin users (or gated-off / empty admin list) keep full platform toolsets and
the normal ``agent.max_turns`` budget.

Orthogonal to open PR #3995 (generic multi-role ``user_roles``) and draft
#22509 (daimon): this is the tiny two-tier whitelist path from RFC #20744.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Iterable, List, Mapping, Optional, Sequence

from gateway.slash_access import (
    SlashAccessPolicy,
    policy_for_source,
    policy_from_extra,
)

# Default non-admin toolsets when tier is on and user_toolsets is unset.
# Intentionally excludes memory, skills, cronjob, delegation, messaging,
# terminal/code backends — operators expand via user_toolsets.
_DEFAULT_USER_TOOLSETS: FrozenSet[str] = frozenset({
    "web",
    "browser",
    "vision",
    "image_gen",
    "tts",
    "todo",
})

# Toolsets never offered to non-admins unless explicitly listed in user_toolsets.
# Defense-in-depth denylist applied after intersection.
_SENSITIVE_TOOLSETS: FrozenSet[str] = frozenset({
    "memory",
    "skills",
    "skill_manage",
    "cronjob",
    "cron",
    "delegation",
    "code_execution",
    "terminal",
    "file",
    "process",
    "session_search",
    "send_message",
    "messaging",
    "homeassistant",
    "kanban",
})


@dataclass(frozen=True)
class UserTierDecision:
    """Result of resolving tier for one gateway turn."""

    is_admin: bool
    tier_gating_enabled: bool
    enabled_toolsets: List[str]
    max_iterations: int
    reason: str


def _coerce_str_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(";", ",").split(",")]
        return [p for p in parts if p]
    if isinstance(raw, (list, tuple, set, frozenset)):
        out: List[str] = []
        for item in raw:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    return [str(raw).strip()] if str(raw).strip() else []


def _coerce_optional_int(raw: Any) -> Optional[int]:
    if raw is None or raw is False:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value < 1:
        return None
    return value


def _platform_extra_from_config(user_config: Mapping[str, Any], platform_key: str) -> dict:
    """Best-effort platform extra dict (same shape slash_access expects)."""
    platforms = user_config.get("platforms") or user_config.get("platform") or {}
    if not isinstance(platforms, dict):
        platforms = {}
    plat = platforms.get(platform_key) or {}
    if not isinstance(plat, dict):
        plat = {}
    # Prefer nested extra; else the platform dict itself (bridged keys live here).
    extra = plat.get("extra") if isinstance(plat.get("extra"), dict) else plat
    if not isinstance(extra, dict):
        extra = {}
    # Merge gateway-level defaults for user_toolsets / user_max_iterations.
    gateway = user_config.get("gateway") or {}
    if not isinstance(gateway, dict):
        gateway = {}
    merged = dict(extra)
    if "user_toolsets" not in merged and "user_toolsets" in gateway:
        merged["user_toolsets"] = gateway["user_toolsets"]
    if "user_tools" not in merged and "user_tools" in gateway:
        # RFC alias — treat as toolset names for Phase 1.
        merged.setdefault("user_toolsets", gateway["user_tools"])
    if "user_max_iterations" not in merged and "user_max_iterations" in gateway:
        merged["user_max_iterations"] = gateway["user_max_iterations"]
    user_limits = merged.get("user_limits") or gateway.get("user_limits")
    if isinstance(user_limits, dict):
        if "user_max_iterations" not in merged and "max_iterations" in user_limits:
            merged["user_max_iterations"] = user_limits["max_iterations"]
    return merged


def policy_for_tier(
    *,
    source: Any,
    user_config: Optional[Mapping[str, Any]] = None,
    platform_config: Any = None,
) -> SlashAccessPolicy:
    """Resolve slash/admin policy for tier decisions.

    Prefers live ``SessionSource`` + ``PlatformConfig`` when available (gateway);
    falls back to synthetic resolution from ``user_config`` for unit tests.
    """
    if platform_config is not None and source is not None:
        try:
            return policy_for_source(source, platform_config)
        except Exception:
            pass
    if user_config is None or source is None:
        return policy_from_extra({}, "dm")
    platform_key = getattr(getattr(source, "platform", None), "value", None) or str(
        getattr(source, "platform", "") or ""
    )
    if hasattr(source, "platform") and hasattr(source.platform, "name"):
        # Platform enum — config keys are usually lowercase names
        platform_key = source.platform.value if hasattr(source.platform, "value") else source.platform.name.lower()
    extra = _platform_extra_from_config(user_config, platform_key)
    chat_type = (getattr(source, "chat_type", None) or "dm") or "dm"
    scope = "group" if str(chat_type).lower() not in ("dm", "private", "direct") else "dm"
    return policy_from_extra(extra, scope)


def filter_toolsets_for_user(
    enabled_toolsets: Sequence[str],
    *,
    is_admin: bool,
    tier_gating_enabled: bool,
    user_toolsets_config: Any = None,
) -> List[str]:
    """Intersect platform toolsets with the non-admin whitelist."""
    base = [str(t).strip() for t in enabled_toolsets if str(t).strip()]
    if is_admin or not tier_gating_enabled:
        return list(base)

    configured = _coerce_str_list(user_toolsets_config)
    if configured:
        allow = {t.lower() for t in configured}
        # Explicit listing wins even for otherwise-sensitive toolsets.
        filtered = [t for t in base if t.lower() in allow]
    else:
        allow = {t.lower() for t in _DEFAULT_USER_TOOLSETS}
        filtered = [t for t in base if t.lower() in allow]
        filtered = [t for t in filtered if t.lower() not in _SENSITIVE_TOOLSETS]

    # Preserve order, drop dups
    seen = set()
    out: List[str] = []
    for t in filtered:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def resolve_user_tier(
    *,
    source: Any,
    enabled_toolsets: Sequence[str],
    max_iterations: int,
    user_config: Optional[Mapping[str, Any]] = None,
    platform_config: Any = None,
    user_id: Optional[str] = None,
) -> UserTierDecision:
    """Resolve admin/user tier and return adjusted toolsets + iteration budget."""
    policy = policy_for_tier(
        source=source,
        user_config=user_config,
        platform_config=platform_config,
    )
    uid = user_id if user_id is not None else getattr(source, "user_id", None)
    is_admin = policy.is_admin(uid)
    gating = bool(policy.enabled)

    platform_key = ""
    if source is not None:
        plat = getattr(source, "platform", None)
        if plat is not None and hasattr(plat, "value"):
            platform_key = str(plat.value)
        elif plat is not None:
            platform_key = str(plat)
    extra = _platform_extra_from_config(user_config or {}, platform_key) if user_config else {}
    user_toolsets_cfg = extra.get("user_toolsets", extra.get("user_tools"))
    user_max = _coerce_optional_int(extra.get("user_max_iterations"))

    tools = filter_toolsets_for_user(
        enabled_toolsets,
        is_admin=is_admin,
        tier_gating_enabled=gating,
        user_toolsets_config=user_toolsets_cfg,
    )

    iters = int(max_iterations)
    reason = "admin" if is_admin else ("ungated" if not gating else "user")
    if gating and not is_admin and user_max is not None:
        iters = min(iters, user_max)
        reason = f"user_clamped_max_iterations={user_max}"

    return UserTierDecision(
        is_admin=is_admin,
        tier_gating_enabled=gating,
        enabled_toolsets=tools,
        max_iterations=iters,
        reason=reason,
    )


def apply_user_tier_to_agent_kwargs(
    *,
    source: Any,
    enabled_toolsets: Sequence[str],
    max_iterations: int,
    user_config: Optional[Mapping[str, Any]] = None,
    platform_config: Any = None,
) -> tuple[List[str], int, UserTierDecision]:
    """Convenience wrapper used by gateway/run.py construction sites."""
    decision = resolve_user_tier(
        source=source,
        enabled_toolsets=enabled_toolsets,
        max_iterations=max_iterations,
        user_config=user_config,
        platform_config=platform_config,
    )
    return decision.enabled_toolsets, decision.max_iterations, decision
