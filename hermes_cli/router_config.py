"""Model Router configuration and slash-command helpers.

The Model Router is the inverse of Mixture-of-Agents: instead of fanning a
prompt out to several reference models and aggregating, a small *classifier*
call decides which execution tier the prompt needs ("simple" or "complex"),
and the whole turn then runs on that tier's model. Failures walk a
configurable fallback chain.

Schema (per preset)::

    router:
      default_preset: default
      active_preset: ""
      save_traces: false
      trace_dir: ""
      presets:
        default:
          enabled: true
          classifier: {provider: openai-codex, model: gpt-5.5}
          classifier_max_tokens: 16
          classifier_context_messages: 4
          default_route: simple
          routes:
            simple:  {provider: lmstudio, model: google/gemma-4-e4b}
            complex: {provider: openai-codex, model: gpt-5.5}
          fallbacks:
            - {provider: lmstudio, model: qwen/qwen3-4b-thinking-2507}
            - {provider: openai-codex, model: gpt-5.5}
          channel_hints:
            whatsapp: simple
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from hermes_cli.moa_config import _coerce_int, _coerce_int_or_none

DEFAULT_ROUTER_PRESET_NAME = "default"

ROUTER_TIERS = ("simple", "complex")

# Default slots mirror the MoA default conventions (moa_config.py): a cheap,
# widely available model for the simple tier and the strongest mainstream
# slot for the complex tier. These are placeholders until the user runs
# `hermes router configure` — the router only activates when explicitly
# selected as the provider, so defaults never route anything on their own.
DEFAULT_ROUTER_CLASSIFIER: dict[str, str] = {
    "provider": "openrouter",
    "model": "deepseek/deepseek-v4-flash",
}

DEFAULT_ROUTER_ROUTES: dict[str, dict[str, str]] = {
    "simple": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
    "complex": {"provider": "openai-codex", "model": "gpt-5.5"},
}

DEFAULT_ROUTER_FALLBACKS: list[dict[str, str]] = [
    {"provider": "openai-codex", "model": "gpt-5.5"},
]


def _clean_slot(slot: Any) -> dict[str, str] | None:
    if not isinstance(slot, dict):
        return None
    provider = str(slot.get("provider") or "").strip()
    model = str(slot.get("model") or "").strip()
    if not provider or not model:
        return None
    # Router and MoA are virtual providers whose "models" are preset names.
    # Allowing either as a classifier/route/fallback slot would nest one
    # virtual run inside another (router→router recursion, or a router tier
    # that spins up a full MoA fan-out mid-turn). Reject at validation time so
    # such a slot can never be saved; an invalid slot is dropped, falling back
    # to the preset's defaults. moa_config._clean_slot carries the symmetric
    # guard for MoA slots.
    if provider.lower() in {"router", "moa"}:
        return None
    return {"provider": provider, "model": model}


def _coerce_route_name(value: Any, default: str = "simple") -> str:
    name = str(value or "").strip().lower()
    return name if name in ROUTER_TIERS else default


def _default_preset() -> dict[str, Any]:
    return {
        "enabled": True,
        "classifier": deepcopy(DEFAULT_ROUTER_CLASSIFIER),
        "classifier_max_tokens": 16,
        "classifier_context_messages": 4,
        "default_route": "simple",
        "routes": deepcopy(DEFAULT_ROUTER_ROUTES),
        "fallbacks": deepcopy(DEFAULT_ROUTER_FALLBACKS),
        "channel_hints": {"whatsapp": "simple"},
        # Experimental: when > 0, user messages shorter than this (and without
        # code fences/URLs) skip the classifier call and route "simple".
        "short_circuit_chars": 0,
    }


def _normalize_preset(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}

    classifier = _clean_slot(raw.get("classifier")) or deepcopy(DEFAULT_ROUTER_CLASSIFIER)

    # Routes: guarantee every tier exists so the runtime never KeyErrors on a
    # verdict. Hand-edited partial/broken tiers degrade to defaults, mirroring
    # moa_config's tolerance for scalar fields.
    raw_routes = raw.get("routes")
    if not isinstance(raw_routes, dict):
        raw_routes = {}
    routes: dict[str, dict[str, str]] = {}
    for tier in ROUTER_TIERS:
        routes[tier] = _clean_slot(raw_routes.get(tier)) or deepcopy(DEFAULT_ROUTER_ROUTES[tier])

    raw_fallbacks = raw.get("fallbacks")
    if not isinstance(raw_fallbacks, list):
        raw_fallbacks = [raw_fallbacks] if isinstance(raw_fallbacks, dict) else []
    fallbacks = [_clean_slot(item) for item in raw_fallbacks]
    fallbacks = [item for item in fallbacks if item is not None]
    # No implicit default fallbacks when the user explicitly configured an
    # empty list — an empty chain means "no fallbacks", unlike routes which
    # must always exist. Only a *missing* key gets defaults.
    if "fallbacks" not in raw:
        fallbacks = deepcopy(DEFAULT_ROUTER_FALLBACKS)

    raw_hints = raw.get("channel_hints")
    channel_hints: dict[str, str] = {}
    if isinstance(raw_hints, dict):
        for platform, tier in raw_hints.items():
            clean_platform = str(platform or "").strip().lower()
            clean_tier = str(tier or "").strip().lower()
            if clean_platform and clean_tier in ROUTER_TIERS:
                channel_hints[clean_platform] = clean_tier

    return {
        "enabled": bool(raw.get("enabled", True)),
        "classifier": classifier,
        # Verdict is a single word; 16 tokens is generous headroom while
        # keeping per-turn classifier spend negligible.
        "classifier_max_tokens": _coerce_int(raw.get("classifier_max_tokens"), 16),
        "classifier_context_messages": _coerce_int(raw.get("classifier_context_messages"), 4),
        # Fail-open target: used when the classifier errors, times out, or
        # returns garbage — and when enabled is false. Defaults to "simple"
        # (local model) so chat keeps working when the classifier provider
        # is down.
        "default_route": _coerce_route_name(raw.get("default_route")),
        "routes": routes,
        "fallbacks": fallbacks,
        "channel_hints": channel_hints,
        "short_circuit_chars": _coerce_int_or_none(raw.get("short_circuit_chars")) or 0,
    }


def normalize_router_config(raw: Any) -> dict[str, Any]:
    """Return validated Router config with named presets.

    Mirrors ``normalize_moa_config``: unknown/broken values degrade to
    defaults, a flat legacy shape (fields directly under ``router:``) becomes
    the default preset, and a flattened view of the default preset is exposed
    for dashboard/desktop callers.
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

    # Flat config (no presets key) becomes the default preset.
    if not presets:
        presets[DEFAULT_ROUTER_PRESET_NAME] = _normalize_preset(raw)

    default_name = str(raw.get("default_preset") or "").strip()
    if not default_name or default_name not in presets:
        default_name = next(iter(presets), DEFAULT_ROUTER_PRESET_NAME)
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
        "save_traces": bool(raw.get("save_traces", False)),
        "trace_dir": str(raw.get("trace_dir") or "").strip(),
        # Compatibility/flattened view for dashboard/desktop callers.
        "classifier": deepcopy(active["classifier"]),
        "classifier_max_tokens": active["classifier_max_tokens"],
        "classifier_context_messages": active["classifier_context_messages"],
        "short_circuit_chars": active["short_circuit_chars"],
        "default_route": active["default_route"],
        "routes": deepcopy(active["routes"]),
        "fallbacks": deepcopy(active["fallbacks"]),
        "channel_hints": deepcopy(active["channel_hints"]),
        "enabled": active["enabled"],
    }


def list_router_presets(config: Any) -> list[str]:
    cfg = normalize_router_config(config)
    return list(cfg["presets"].keys())


def resolve_router_preset(config: Any, name: str | None = None) -> dict[str, Any]:
    cfg = normalize_router_config(config)
    preset_name = str(name or cfg.get("default_preset") or DEFAULT_ROUTER_PRESET_NAME).strip()
    preset = cfg["presets"].get(preset_name)
    if preset is None:
        raise KeyError(preset_name)
    return deepcopy(preset)


def exact_router_preset_name(config: Any, text: str) -> str | None:
    """Return the preset name iff ``text`` exactly matches an *enabled* preset.

    Same contract as ``exact_moa_preset_name``: this backs the implicit bare
    ``/model <preset>`` match (PATH B in ``hermes_cli/model_switch.py``), so it
    honors the per-preset ``enabled`` opt-out — a disabled preset must not
    capture a plain model switch whose name happens to collide. Explicit
    selection via ``--provider router`` / the model picker bypasses this.
    """
    wanted = str(text or "").strip()
    if not wanted:
        return None
    cfg = normalize_router_config(config)
    preset = cfg["presets"].get(wanted)
    if preset is None or not preset.get("enabled", True):
        return None
    return wanted


def set_active_router_preset(config: Any, name: str | None) -> dict[str, Any]:
    cfg = normalize_router_config(config)
    clean = str(name or "").strip()
    if clean and clean not in cfg["presets"]:
        raise KeyError(clean)
    cfg["active_preset"] = clean
    return cfg


def router_usage() -> str:
    return (
        "Usage: /router  (shows the Model Router preset for this session: "
        "classifier, tiers, fallbacks, and the last routing decision; pick "
        "'Model Router' from the model picker to switch the session onto it)"
    )
