"""Deterministic per-turn request routing for Codex providers.

The router is intentionally local and cheap: no LLM call, no prompt mutation,
and no tool/schema/system-prompt changes. It returns request-time knobs that
callers apply to the current turn only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from hermes_constants import parse_reasoning_effort


@dataclass(frozen=True)
class RouterPreset:
    """Concrete request settings for one routing preset."""

    name: str
    model: str
    effort: str
    service_tier: str = "standard"


@dataclass(frozen=True)
class RouterDecision:
    """Result of routing a single user turn."""

    enabled: bool
    source: str
    preset: str | None
    model: str | None
    reasoning_config: dict[str, Any] | None
    service_tier: str | None
    request_overrides: dict[str, Any]
    message_override: str | None = None
    reason: str = ""


DEFAULT_PRESETS: dict[str, RouterPreset] = {
    "low_standard": RouterPreset("low_standard", "gpt-5.5", "low", "standard"),
    "high_standard": RouterPreset("high_standard", "gpt-5.5", "high", "standard"),
    "xhigh_standard": RouterPreset("xhigh_standard", "gpt-5.5", "xhigh", "standard"),
    "xhigh_fast": RouterPreset("xhigh_fast", "gpt-5.4", "xhigh", "standard"),
    "high_priority": RouterPreset("high_priority", "gpt-5.5", "high", "priority"),
}

_PREFIX_PRESETS: dict[str, str] = {
    "!low": "low_standard",
    "!cheap": "low_standard",
    "!standard": "high_standard",
    "!deep": "xhigh_standard",
    "!xhigh": "xhigh_standard",
    "!fast": "xhigh_fast",
    "!priority": "high_priority",
}

_LOW_RE = re.compile(r"\b(grammar|gramática|translate|tradu(?:z|ção)|format(?:ting)?|resum[eo]|summari[sz]e|typo|ortografia)\b", re.I)
_FAST_RE = re.compile(r"\b(quick|rápido|rapido|barato|cheap|fast|low latency|baixo custo)\b", re.I)
_XHIGH_RE = re.compile(r"\b(architecture|arquitetura|migration|migração|migracao|refactor|multi[- ]?file|múltiplos arquivos|multiplos arquivos|cached agent|gateway parity|prompt cache|race condition|root cause|obscur[eo]|complex[oa])\b", re.I)
_HIGH_RE = re.compile(r"\b(debug|bug|fix|corrig|sql|join|python|test|integration|integração|dataverse|gateway|cli|api)\b", re.I)
_EMERGENCY_RE = re.compile(r"\b(emergency|urgente|produção|producao|prod|incident|incidente|sev[ -]?1|hotfix)\b", re.I)
_SLASH_RE = re.compile(r"^\s*/")


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(p).strip() for p in value if str(p).strip()]
    return []


def _load_presets(router_cfg: Mapping[str, Any]) -> dict[str, RouterPreset]:
    presets = dict(DEFAULT_PRESETS)
    raw_presets = router_cfg.get("presets") if isinstance(router_cfg, Mapping) else None
    if not isinstance(raw_presets, Mapping):
        return presets
    for name, raw in raw_presets.items():
        if not isinstance(raw, Mapping):
            continue
        key = str(name).strip()
        model = str(raw.get("model") or "").strip()
        effort = str(raw.get("effort") or raw.get("reasoning_effort") or "").strip().lower()
        service_tier = str(raw.get("service_tier") or "standard").strip().lower()
        if not key or not model or parse_reasoning_effort(effort) is None:
            continue
        if service_tier not in {"standard", "priority"}:
            service_tier = "standard"
        presets[key] = RouterPreset(key, model, effort, service_tier)
    return presets


def _provider_allowed(provider: str | None, allowlist: list[str]) -> bool:
    # V1 is Codex-only. Missing/empty allowlist fails closed to the default
    # Codex provider instead of becoming a wildcard for every provider.
    effective_allowlist = allowlist or ["openai-codex"]
    provider_text = (provider or "").strip().lower()
    return any(item.lower() == provider_text for item in effective_allowlist)


def _request_overrides_for_service_tier(service_tier: str | None) -> dict[str, Any]:
    if service_tier == "priority":
        return {"service_tier": "priority"}
    return {}


def _choose_preset(message: str, router_cfg: Mapping[str, Any]) -> tuple[str, str, str | None]:
    stripped = message.lstrip()
    first = stripped.split(maxsplit=1)[0].lower() if stripped else ""
    if first in _PREFIX_PRESETS:
        remaining = stripped[len(first):].lstrip()
        return _PREFIX_PRESETS[first], "prefix", remaining

    if _EMERGENCY_RE.search(message):
        if _as_bool(router_cfg.get("allow_auto_priority"), default=False):
            return "high_priority", "auto", None
        return "xhigh_standard", "auto", None
    if _FAST_RE.search(message):
        return "xhigh_fast", "auto", None
    if _XHIGH_RE.search(message):
        return "xhigh_standard", "auto", None
    if _LOW_RE.search(message):
        return "low_standard", "auto", None
    if _HIGH_RE.search(message):
        return "high_standard", "auto", None
    return str(router_cfg.get("default_preset") or "high_standard"), "default", None


def route_turn(
    user_message: Any,
    *,
    router_config: Mapping[str, Any] | None,
    current_model: str,
    provider: str | None,
    explicit_reasoning_config: Mapping[str, Any] | None = None,
    explicit_service_tier: str | None = None,
    explicit_request_overrides: Mapping[str, Any] | None = None,
    explicit_model_override: bool = False,
) -> RouterDecision:
    """Return per-turn routing knobs for ``user_message``.

    Explicit reasoning/service/request overrides win. Slash commands and non-text
    messages are no-ops. Disabled or non-allowlisted providers are also no-ops.
    """

    cfg: Mapping[str, Any] = router_config or {}
    if not _as_bool(cfg.get("enabled"), default=False):
        return RouterDecision(False, "disabled", None, None, None, None, {}, reason="router disabled")
    if not isinstance(user_message, str) or not user_message.strip():
        return RouterDecision(False, "no-op", None, None, None, None, {}, reason="non-text or empty message")
    if _SLASH_RE.match(user_message):
        return RouterDecision(False, "no-op", None, None, None, None, {}, reason="slash command")
    if explicit_model_override or explicit_reasoning_config or explicit_service_tier or explicit_request_overrides:
        return RouterDecision(False, "explicit", None, None, None, None, {}, reason="explicit override present")

    allowlist = _str_list(cfg.get("providers_allowlist") or cfg.get("provider_allowlist"))
    if not _provider_allowed(provider, allowlist):
        return RouterDecision(False, "no-op", None, None, None, None, {}, reason="provider not allowlisted")

    presets = _load_presets(cfg)
    preset_name, source, message_override = _choose_preset(user_message, cfg)
    preset = presets.get(preset_name) or presets["high_standard"]
    if preset.service_tier == "priority" and not _as_bool(cfg.get("allow_auto_priority"), default=False) and source != "prefix":
        preset = presets["xhigh_standard"]
        preset_name = preset.name
    reasoning_config = parse_reasoning_effort(preset.effort)
    request_overrides = _request_overrides_for_service_tier(preset.service_tier)
    service_tier = "priority" if preset.service_tier == "priority" else None
    reason = f"{source}:{preset_name}"
    return RouterDecision(
        True,
        source,
        preset.name,
        preset.model or current_model,
        reasoning_config,
        service_tier,
        request_overrides,
        message_override=message_override,
        reason=reason,
    )
