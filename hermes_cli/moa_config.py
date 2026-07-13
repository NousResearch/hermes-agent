"""Mixture-of-Agents configuration and slash-command helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from copy import deepcopy
from typing import Any

MOA_MARKER_PREFIX = "__HERMES_MOA_TURN_V1__"
DEFAULT_MOA_PRESET_NAME = "default"

DEFAULT_MOA_REFERENCE_MODELS: list[dict[str, str]] = [
    {"provider": "openai-codex", "model": "gpt-5.5"},
    {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"},
]

DEFAULT_MOA_AGGREGATOR: dict[str, str] = {
    "provider": "openrouter",
    "model": "anthropic/claude-opus-4.8",
}


def _coerce_float_or_none(value: Any) -> float | None:
    """Coerce to a float, or None when unset/blank/invalid.

    Used for optional sampling params (reference_temperature /
    aggregator_temperature) where None means 'don't send the parameter —
    provider default applies', matching how a single-model Hermes agent
    never sends temperature unless explicitly configured.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _coerce_int_or_none(value: Any) -> int | None:
    """Coerce to a positive int, or None when unset/blank/invalid/non-positive.

    Used for optional caps (e.g. reference_max_tokens) where None means
    'no cap' — the safe default that preserves prior uncapped behavior.
    """
    if value is None or value == "":
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        try:
            n = int(float(value))
        except (TypeError, ValueError):
            return None
    return n if n > 0 else None


def _coerce_positive_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _coerce_fanout(value: Any) -> str:
    """Normalize the fan-out cadence; unknown values fall back to default."""
    mode = str(value or "").strip().lower()
    return mode if mode in {"per_iteration", "user_turn"} else "per_iteration"


def _clean_slot(slot: Any) -> dict[str, str] | None:
    if not isinstance(slot, dict):
        return None
    provider = str(slot.get("provider") or "").strip()
    model = str(slot.get("model") or "").strip()
    if not provider or not model:
        return None
    # MoA is a virtual provider whose presets are themselves MoA runs. Allowing
    # one as a reference or aggregator slot would create a recursive MoA tree
    # (the runtime guards in moa_loop.py skip references / raise on aggregators,
    # but that surfaces only mid-turn). Reject it here so it can never be saved:
    # an invalid slot is dropped, falling back to the preset's defaults.
    if provider.lower() == "moa":
        return None
    return {"provider": provider, "model": model}


def _clean_provider_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        name = str(item or "").strip().lower()
        if name and name not in result:
            result.append(name)
    return result


def _clean_slot_list(value: Any, excluded: set[str]) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for item in value:
        slot = _clean_slot(item)
        if slot is None or slot["provider"].lower() in excluded:
            continue
        if slot not in result:
            result.append(slot)
    return result


def _default_preset() -> dict[str, Any]:
    return {
        "reference_models": deepcopy(DEFAULT_MOA_REFERENCE_MODELS),
        "aggregator": deepcopy(DEFAULT_MOA_AGGREGATOR),
        "fallback_aggregators": [],
        # None = temperature omitted from API calls (provider default),
        # matching single-model agent behavior.
        "reference_temperature": None,
        "aggregator_temperature": None,
        "max_tokens": 4096,
        "reference_max_tokens": None,
        "context_length": None,
        "max_advisors": None,
        "max_reference_cost_usd": None,
        "max_fanout_latency_seconds": None,
        "circuit_breaker_seconds": 60.0,
        "quota_cooldown_seconds": 900.0,
        "fanout": "per_iteration",
        "auto_routes": {},
        "enabled": True,
    }


def _clean_auto_routes(value: Any) -> dict[str, str]:
    """Нормализовать необязательную таблицу маршрутов auto-пресета."""
    if not isinstance(value, dict):
        return {}
    routes: dict[str, str] = {}
    for key in ("fast", "balanced", "research", "code_heavy", "max"):
        target = str(value.get(key) or "").strip()
        if target:
            routes[key] = target
    return routes


def _normalize_preset(raw: Any, excluded_providers: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    excluded = excluded_providers or set()

    refs_explicitly_empty = raw.get("reference_models") == []
    raw_refs = raw.get("reference_models")
    if not isinstance(raw_refs, list):
        # A hand-edited scalar / single mapping (or a bad type) must degrade to
        # defaults instead of crashing the iteration, mirroring the tolerance
        # for the scalar fields below (reference_temperature / max_tokens).
        raw_refs = [raw_refs] if isinstance(raw_refs, dict) else []
    cleaned_refs = [_clean_slot(item) for item in raw_refs]
    cleaned_refs = [item for item in cleaned_refs if item is not None]
    refs = [item for item in cleaned_refs if item["provider"].lower() not in excluded]
    # Явный пустой список означает одиночный маршрут: агрегатор работает без
    # fan-out. Отсутствующее/повреждённое значение сохраняет старый безопасный
    # fallback на стандартных советников.
    if not refs and not refs_explicitly_empty and not cleaned_refs:
        refs = [
            item for item in deepcopy(DEFAULT_MOA_REFERENCE_MODELS)
            if item["provider"].lower() not in excluded
        ]

    aggregator = _clean_slot(raw.get("aggregator"))
    if aggregator and aggregator["provider"].lower() in excluded:
        aggregator = None
    if aggregator is None:
        aggregator = deepcopy(DEFAULT_MOA_AGGREGATOR)
    fallback_aggregators = _clean_slot_list(raw.get("fallback_aggregators"), excluded)

    return {
        "enabled": bool(raw.get("enabled", True)),
        "reference_models": refs,
        "aggregator": aggregator,
        "fallback_aggregators": fallback_aggregators,
        "reference_temperature": _coerce_float_or_none(raw.get("reference_temperature")),
        "aggregator_temperature": _coerce_float_or_none(raw.get("aggregator_temperature")),
        "max_tokens": _coerce_int(raw.get("max_tokens"), 4096),
        # Optional cap on how much each reference ADVISOR may generate per turn.
        # None (default) = uncapped: advisors write full-length advice, matching
        # prior behavior so existing presets are unchanged. Set a value (e.g.
        # 600) to make advisors give concise advice — the dominant MoA latency
        # is advisor generation (turn latency correlates ~0.88 with output
        # tokens), and the aggregator only needs the gist of each advisor's
        # judgement, so capping roughly halves per-turn wall time. Does NOT cap
        # the acting aggregator (its output is the user-visible answer).
        "reference_max_tokens": _coerce_int_or_none(raw.get("reference_max_tokens")),
        # Полный рабочий бюджет хода (input + output). Для MoA он не должен
        # превышать окно самого узкого постоянного советника.
        "context_length": _coerce_int_or_none(raw.get("context_length")),
        "max_advisors": _coerce_int_or_none(raw.get("max_advisors")),
        "max_reference_cost_usd": _coerce_positive_float_or_none(
            raw.get("max_reference_cost_usd")
        ),
        "max_fanout_latency_seconds": _coerce_positive_float_or_none(
            raw.get("max_fanout_latency_seconds")
        ),
        "circuit_breaker_seconds": (
            _coerce_positive_float_or_none(raw.get("circuit_breaker_seconds")) or 60.0
        ),
        "quota_cooldown_seconds": (
            _coerce_positive_float_or_none(raw.get("quota_cooldown_seconds")) or 900.0
        ),
        # When the reference fan-out runs. "per_iteration" (default) re-runs
        # the advisors whenever the advisory view changes — i.e. every tool
        # iteration, so advice tracks live task state. "user_turn" runs the
        # advisors ONCE per user turn (the original MoA shape): the
        # aggregator gets their upfront plan-level advice, then acts alone
        # for the rest of the tool loop.
        "fanout": _coerce_fanout(raw.get("fanout")),
        "auto_routes": _clean_auto_routes(raw.get("auto_routes")),
    }


def normalize_moa_config(raw: Any) -> dict[str, Any]:
    """Return validated MoA config with named presets.

    Backward compatible with the first PR shape where ``moa`` itself contained
    ``reference_models`` and ``aggregator`` directly.
    """
    if not isinstance(raw, dict):
        raw = {}

    excluded_providers = _clean_provider_names(raw.get("excluded_providers"))
    excluded_set = set(excluded_providers)

    presets_raw = raw.get("presets")
    presets: dict[str, dict[str, Any]] = {}
    if isinstance(presets_raw, dict):
        for name, preset in presets_raw.items():
            clean_name = str(name or "").strip()
            if clean_name:
                presets[clean_name] = _normalize_preset(preset, excluded_set)

    # Legacy flat config becomes the default preset.
    if not presets:
        presets[DEFAULT_MOA_PRESET_NAME] = _normalize_preset(raw, excluded_set)

    default_name = str(raw.get("default_preset") or "").strip()
    if not default_name or default_name not in presets:
        default_name = next(iter(presets), DEFAULT_MOA_PRESET_NAME)
    if default_name not in presets:
        presets[default_name] = _default_preset()

    active_name = str(raw.get("active_preset") or "").strip()
    if active_name not in presets:
        active_name = ""

    active = presets[default_name]
    return {
        "default_preset": default_name,
        "active_preset": active_name,
        "excluded_providers": excluded_providers,
        "presets": presets,
        # Compatibility/flattened view for existing dashboard/desktop callers.
        "reference_models": deepcopy(active["reference_models"]),
        "aggregator": deepcopy(active["aggregator"]),
        "fallback_aggregators": deepcopy(active.get("fallback_aggregators") or []),
        "reference_temperature": active["reference_temperature"],
        "aggregator_temperature": active["aggregator_temperature"],
        "max_tokens": active["max_tokens"],
        "reference_max_tokens": active.get("reference_max_tokens"),
        "context_length": active.get("context_length"),
        "max_advisors": active.get("max_advisors"),
        "max_reference_cost_usd": active.get("max_reference_cost_usd"),
        "max_fanout_latency_seconds": active.get("max_fanout_latency_seconds"),
        "circuit_breaker_seconds": active.get("circuit_breaker_seconds"),
        "quota_cooldown_seconds": active.get("quota_cooldown_seconds"),
        "fanout": active.get("fanout", "per_iteration"),
        "enabled": active["enabled"],
    }


def validate_moa_config(raw: Any) -> list[dict[str, str]]:
    """Проверить связи пресетов без сетевых вызовов и раскрытия credentials."""
    issues: list[dict[str, str]] = []
    if not isinstance(raw, dict):
        return [{"severity": "error", "code": "moa_not_mapping", "message": "moa must be a mapping"}]

    excluded = set(_clean_provider_names(raw.get("excluded_providers")))
    presets = raw.get("presets")
    if not isinstance(presets, dict) or not presets:
        return issues  # legacy flat config remains supported

    for name, value in presets.items():
        if not isinstance(value, dict):
            issues.append({
                "severity": "error", "code": "preset_not_mapping",
                "message": f"preset {name!r} must be a mapping",
            })
            continue
        refs = value.get("reference_models", [])
        if not isinstance(refs, list):
            issues.append({
                "severity": "error", "code": "reference_models_not_list",
                "message": f"preset {name!r} reference_models must be a list",
            })
            refs = []
        seen: set[tuple[str, str]] = set()
        for item in refs:
            slot = _clean_slot(item)
            if slot is None:
                issues.append({
                    "severity": "error", "code": "invalid_reference_slot",
                    "message": f"preset {name!r} contains an invalid reference slot",
                })
                continue
            key = (slot["provider"].lower(), slot["model"].lower())
            if key in seen:
                issues.append({
                    "severity": "warning", "code": "duplicate_reference_slot",
                    "message": f"preset {name!r} contains duplicate reference {slot['provider']}:{slot['model']}",
                })
            seen.add(key)
            if key[0] in excluded:
                issues.append({
                    "severity": "warning", "code": "excluded_reference_provider",
                    "message": f"preset {name!r} reference provider {slot['provider']} is excluded",
                })

        aggregator = _clean_slot(value.get("aggregator"))
        if aggregator is None:
            issues.append({
                "severity": "error", "code": "invalid_aggregator_slot",
                "message": f"preset {name!r} requires a valid aggregator",
            })
        elif aggregator["provider"].lower() in excluded:
            issues.append({
                "severity": "error", "code": "excluded_aggregator_provider",
                "message": f"preset {name!r} aggregator provider {aggregator['provider']} is excluded",
            })

        for route, target in _clean_auto_routes(value.get("auto_routes")).items():
            target_cfg = presets.get(target)
            if not isinstance(target_cfg, dict) or not bool(target_cfg.get("enabled", True)):
                issues.append({
                    "severity": "error", "code": "invalid_auto_route",
                    "message": f"preset {name!r} route {route!r} targets unavailable preset {target!r}",
                })
        for field in (
            "max_advisors",
            "max_reference_cost_usd",
            "max_fanout_latency_seconds",
            "circuit_breaker_seconds",
            "quota_cooldown_seconds",
        ):
            if field in value and value.get(field) not in {None, ""}:
                try:
                    valid = float(value[field]) > 0
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    issues.append({
                        "severity": "error",
                        "code": "invalid_budget_value",
                        "message": f"preset {name!r} field {field!r} must be positive",
                    })
    return issues


def moa_config_revision(raw: Any) -> str:
    """Вернуть стабильную revision для optimistic concurrency MoA-конфига."""
    normalized = normalize_moa_config(raw)
    payload = json.dumps(
        normalized,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _slot_runtime_available(slot: dict[str, str]) -> bool:
    """Проверить локальную runtime-конфигурацию, не выполняя модельный запрос."""
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(
            requested=slot.get("provider"), target_model=slot.get("model")
        )
        # Одного синтаксически корректного endpoint недостаточно: OpenRouter
        # может разрешиться в публичный base URL с пустым API key. Локальные
        # no-auth providers уже получают от resolver SDK-заглушку
        # ``no-key-required``, поэтому проверка auth carrier их не отключает.
        has_auth = bool(runtime.get("api_key") or runtime.get("credential_pool"))
        return bool(runtime.get("provider") and runtime.get("base_url") and has_auth)
    except Exception:
        return False


def resolve_available_moa_preset(
    preset: dict[str, Any], availability_check: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Отфильтровать недоступных советников и выбрать безопасный агрегатор."""
    check = availability_check or _slot_runtime_available
    resolved = deepcopy(preset)
    unavailable: list[str] = []
    cooling_down: list[dict[str, Any]] = []

    def slot_ready(slot: dict[str, str]) -> bool:
        try:
            from agent.moa_runtime import circuit_status

            circuit = circuit_status(slot)
        except Exception:
            circuit = {"active": False}
        if circuit.get("active"):
            cooling_down.append({
                "slot": f"{slot.get('provider')}:{slot.get('model')}",
                "retry_after_seconds": int(circuit.get("retry_after_seconds") or 0),
                "reason": str(circuit.get("reason") or "transient"),
            })
            return False
        return bool(check(slot))

    refs: list[dict[str, str]] = []
    for slot in resolved.get("reference_models") or []:
        if slot_ready(slot):
            refs.append(slot)
        else:
            unavailable.append(f"reference:{slot['provider']}:{slot['model']}")
    resolved["reference_models"] = refs

    primary = resolved.get("aggregator") or {}
    candidates = [primary, *(resolved.get("fallback_aggregators") or [])]
    selected = next((slot for slot in candidates if slot and slot_ready(slot)), None)
    if selected is None:
        unavailable.append("aggregator:no_available_candidate")
    elif selected != primary:
        unavailable.append(f"aggregator:{primary.get('provider', '')}:{primary.get('model', '')}")
        resolved["aggregator"] = selected
        refs = [slot for slot in refs if slot != selected]
        resolved["reference_models"] = refs

    status = {
        "degraded": bool(unavailable),
        "unavailable": unavailable,
        "reference_count": len(refs),
        "aggregator_available": selected is not None,
        "aggregator": (
            f"{selected.get('provider')}:{selected.get('model')}" if selected else ""
        ),
        "configured_aggregator": (
            f"{primary.get('provider')}:{primary.get('model')}" if primary else ""
        ),
        "fallback_used": bool(selected is not None and selected != primary),
        "cooling_down": cooling_down,
    }
    return resolved, status


def evaluate_moa_runtime_config(raw: Any) -> dict[str, Any]:
    """Сформировать очищенный startup/status отчёт по всем включённым пресетам."""
    cfg = normalize_moa_config(raw)
    cache: dict[tuple[str, str], bool] = {}

    def cached_check(slot: dict[str, str]) -> bool:
        key = (slot.get("provider", ""), slot.get("model", ""))
        if key not in cache:
            cache[key] = _slot_runtime_available(slot)
        return cache[key]

    presets: dict[str, Any] = {}
    for name, preset in cfg["presets"].items():
        if not preset.get("enabled", True):
            continue
        _resolved, status = resolve_available_moa_preset(preset, cached_check)
        presets[name] = status
    return {
        "degraded": any(item["degraded"] for item in presets.values()),
        "presets": presets,
        "validation_issues": validate_moa_config(raw),
        "telemetry": _safe_moa_telemetry(),
        "router_feedback": _safe_route_feedback(),
    }


def _safe_moa_telemetry() -> dict[str, Any]:
    try:
        from agent.moa_runtime import read_moa_telemetry

        return read_moa_telemetry()
    except Exception:
        return {"events": 0, "presets": {}}


def _safe_route_feedback() -> dict[str, Any]:
    try:
        from agent.moa_runtime import route_feedback_summary

        return route_feedback_summary()
    except Exception:
        return {}


def list_moa_presets(config: Any) -> list[str]:
    cfg = normalize_moa_config(config)
    return list(cfg["presets"].keys())


def resolve_moa_preset(config: Any, name: str | None = None) -> dict[str, Any]:
    cfg = normalize_moa_config(config)
    preset_name = str(name or cfg.get("default_preset") or DEFAULT_MOA_PRESET_NAME).strip()
    preset = cfg["presets"].get(preset_name)
    if preset is None:
        raise KeyError(preset_name)
    return deepcopy(preset)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def classify_moa_auto_route(messages: Any) -> str:
    """Выбрать семантический маршрут без дополнительного LLM-вызова.

    Классификатор намеренно консервативен: специализированные дорогие маршруты
    включаются только по ясным признакам, всё неоднозначное остаётся balanced.
    """
    user_texts: list[str] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            text = _message_text(message.get("content")).strip()
            if text and not text.startswith("[Mixture of Agents"):
                user_texts.append(text)
    if not user_texts:
        return "balanced"

    text = user_texts[-1]
    if (
        len(user_texts) > 1
        and re.fullmatch(
            r"\s*(продолжай|дальше|давай дальше|дожми|continue|go on|proceed)\s*[.!]?\s*",
            text,
            flags=re.IGNORECASE,
        )
    ):
        text = user_texts[-2]
    lowered = text.casefold()

    if re.search(
        r"(?:\b(?:preset|пресет|режим)\s*[:=-]?\s*max\b|"
        r"максимальн(?:ое качество|ая глубина|о тщательно)|"
        r"maximum quality|deepest possible)",
        lowered,
    ):
        return "max"

    if re.search(
        r"(?:https?://|\b(?:web|internet|online|latest|current|news)\b|"
        r"в интернете|актуальн|свеж(?:ие|ая|ую)|новост|"
        r"с источниками|первичн(?:ые|ых) источник|официальн(?:ая|ую|ые) документац)",
        lowered,
    ):
        return "research"

    if re.search(
        r"(?:\b(?:repo|repository|code|bug|fix|refactor|implement|pytest|test|"
        r"lint|build|stack trace|traceback|api)\b|"
        r"репозитор|код|баг|ошибк|исправ|рефактор|реализ|тест|сборк|архитектур|"
        r"\.(?:py|ts|tsx|js|jsx|rs|go|java|kt|cs|cpp|c|yaml|yml|json|toml)\b)",
        lowered,
    ):
        return "code_heavy"

    if len(text) <= 220 and re.search(
        r"(?:перевед|переформулир|сократи|исправь текст|форматир|одной строкой|"
        r"кратко объясни|что значит|translate|rewrite|rephrase|format|one line|"
        r"briefly explain)",
        lowered,
    ):
        return "fast"
    return "balanced"


def resolve_moa_preset_for_messages(
    config: Any, name: str | None, messages: Any
) -> tuple[str, dict[str, Any]]:
    """Разрешить обычный или настраиваемый auto-пресет для текущего хода."""
    cfg = normalize_moa_config(config)
    selected_name = str(name or cfg.get("default_preset") or DEFAULT_MOA_PRESET_NAME).strip()
    selected = cfg["presets"].get(selected_name)
    if selected is None:
        raise KeyError(selected_name)

    routes = selected.get("auto_routes") or {}
    if not routes:
        return selected_name, deepcopy(selected)

    route_key = classify_moa_auto_route(messages)
    try:
        from agent.moa_runtime import apply_route_feedback

        route_key = apply_route_feedback(route_key)
    except Exception:
        pass
    target_name = routes.get(route_key) or routes.get("balanced")
    target = cfg["presets"].get(target_name)
    if not target or not target.get("enabled", True):
        return selected_name, deepcopy(selected)
    return target_name, deepcopy(target)


def exact_moa_preset_name(config: Any, text: str) -> str | None:
    """Return the preset name iff ``text`` exactly matches an *enabled* preset.

    Used by the no-explicit-provider switch path (PATH B in
    ``hermes_cli/model_switch.py``) to recognize a bare ``/model <preset>``
    that the user typed without the ``moa:`` prefix. This is an *implicit*
    match, so it must honor the per-preset ``enabled`` opt-out: a user who set
    ``enabled: false`` to disable a preset must not have a plain model switch
    whose name happens to collide with that preset key silently pivot the
    session onto the MoA virtual provider (issue #55187). Explicit selection
    via ``--provider moa`` / the model picker does not go through here, so a
    disabled preset is still reachable when the user explicitly asks for it.
    """
    wanted = str(text or "").strip()
    if not wanted:
        return None
    cfg = normalize_moa_config(config)
    preset = cfg["presets"].get(wanted)
    if preset is None or not preset.get("enabled", True):
        return None
    return wanted


def set_active_moa_preset(config: Any, name: str | None) -> dict[str, Any]:
    cfg = normalize_moa_config(config)
    clean = str(name or "").strip()
    if clean and clean not in cfg["presets"]:
        raise KeyError(clean)
    cfg["active_preset"] = clean
    return cfg


def encode_moa_turn(prompt: str, config: Any = None, preset: str | None = None) -> str:
    """Encode a /moa one-shot turn for frontends that can only send text."""
    payload = {
        "prompt": str(prompt or ""),
        "config": resolve_moa_preset(config or {}, preset),
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    return f"{MOA_MARKER_PREFIX}{encoded}"


def decode_moa_turn(message: Any) -> tuple[str, dict[str, Any] | None]:
    """Decode a hidden /moa one-shot marker."""
    if not isinstance(message, str) or not message.startswith(MOA_MARKER_PREFIX):
        return message, None
    encoded = message[len(MOA_MARKER_PREFIX):].strip()
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8"))
    except Exception:
        return message, None
    prompt = str(payload.get("prompt") or "")
    return prompt, _normalize_preset(payload.get("config") or {})


def build_moa_turn_prompt(user_prompt: str, config: Any = None, preset: str | None = None) -> str:
    """Build the hidden one-shot payload used by TUI/gateway routing."""
    return encode_moa_turn(user_prompt, config, preset=preset)


def moa_usage() -> str:
    return "Usage: /moa <prompt>  (runs one prompt through the default MoA preset, then restores your model; pick a preset from the model picker to switch for the session)"
