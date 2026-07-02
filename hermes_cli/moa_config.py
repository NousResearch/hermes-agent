"""Mixture-of-Agents configuration and slash-command helpers."""

from __future__ import annotations

import base64
import json
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


def _coerce_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _default_preset() -> dict[str, Any]:
    return {
        "reference_models": deepcopy(DEFAULT_MOA_REFERENCE_MODELS),
        "aggregator": deepcopy(DEFAULT_MOA_AGGREGATOR),
        "reference_temperature": 0.6,
        "aggregator_temperature": 0.4,
        "max_tokens": 4096,
        "reference_max_tokens": None,
        "enabled": True,
    }


def _normalize_preset(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}

    raw_refs = raw.get("reference_models")
    if not isinstance(raw_refs, list):
        # A hand-edited scalar / single mapping (or a bad type) must degrade to
        # defaults instead of crashing the iteration, mirroring the tolerance
        # for the scalar fields below (reference_temperature / max_tokens).
        raw_refs = [raw_refs] if isinstance(raw_refs, dict) else []
    refs = [_clean_slot(item) for item in raw_refs]
    refs = [item for item in refs if item is not None]
    if not refs:
        refs = deepcopy(DEFAULT_MOA_REFERENCE_MODELS)

    aggregator = _clean_slot(raw.get("aggregator")) or deepcopy(DEFAULT_MOA_AGGREGATOR)

    return {
        "enabled": bool(raw.get("enabled", True)),
        "reference_models": refs,
        "aggregator": aggregator,
        "reference_temperature": _coerce_float(raw.get("reference_temperature"), 0.6),
        "aggregator_temperature": _coerce_float(raw.get("aggregator_temperature"), 0.4),
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
    }


def normalize_moa_config(raw: Any) -> dict[str, Any]:
    """Return validated MoA config with named presets.

    Backward compatible with the first PR shape where ``moa`` itself contained
    ``reference_models`` and ``aggregator`` directly.
    """
    if not isinstance(raw, dict):
        raw = {}

    presets_raw = raw.get("presets")
    presets: dict[str, dict[str, Any]] = {}
    if isinstance(presets_raw, dict):
        for name, preset in presets_raw.items():
            clean_name = str(name or "").strip()
            if clean_name:
                presets[clean_name] = _normalize_preset(preset)

    # Legacy flat config becomes the default preset.
    if not presets:
        presets[DEFAULT_MOA_PRESET_NAME] = _normalize_preset(raw)

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
        "presets": presets,
        # Compatibility/flattened view for existing dashboard/desktop callers.
        "reference_models": deepcopy(active["reference_models"]),
        "aggregator": deepcopy(active["aggregator"]),
        "reference_temperature": active["reference_temperature"],
        "aggregator_temperature": active["aggregator_temperature"],
        "max_tokens": active["max_tokens"],
        "reference_max_tokens": active.get("reference_max_tokens"),
        "enabled": active["enabled"],
    }


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
