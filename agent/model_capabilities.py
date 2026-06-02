"""Model capability metadata resolution for degraded-mode policy.

The capability layer is intentionally conservative: missing metadata should
never imply a strong model.  Configured provider/model metadata can opt a local
or custom model into specific task classes without changing provider routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


_DEFAULT_SCORES = {
    "general": 0.35,
    "coding": 0.25,
    "system_debugging": 0.30,
    "long_context": 0.10,
    "autonomous_ops": 0.25,
}

_BUILTIN_MODEL_SCORES = {
    # Brian's baseline: GPT 5.5 is sufficient for all existing Hermes tasks.
    # Human-facing scores may be written on a 1-100 scale (55) and are
    # normalized internally to 0-1 (0.55).  Keep the explicit normalized values
    # here so GPT 5.5 is not treated as an unknown/degraded model when provider
    # metadata is absent.
    "gpt-5.5": {
        "general": 0.55,
        "coding": 0.55,
        "system_debugging": 0.55,
        "long_context": 0.55,
        "autonomous_ops": 0.55,
    },
}


@dataclass(frozen=True)
class ModelCapability:
    provider: str
    model: str
    scores: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_SCORES))
    context_tokens: int | None = None
    supports_tools: bool | None = None
    supports_vision: bool | None = None
    max_safe_tool_iterations: int | None = None
    degraded_label: str = "unknown"
    raw: dict[str, Any] = field(default_factory=dict)

    def score(self, key: str = "general") -> float:
        key = (key or "general").strip()
        if key in self.scores:
            return self.scores[key]
        return self.scores.get("general", _DEFAULT_SCORES["general"])


def _to_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < 0:
        return 0.0
    if parsed > 1 and parsed <= 100:
        parsed = parsed / 100.0
    if parsed > 1:
        return 1.0
    return parsed


def _builtin_scores_for_model(model: str) -> dict[str, float]:
    model_norm = (model or "").strip().lower()
    if not model_norm:
        return {}
    for key, scores in _BUILTIN_MODEL_SCORES.items():
        if model_norm == key or model_norm.startswith(f"{key}-"):
            return dict(scores)
    return {}


def _to_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def _provider_entry(config: Mapping[str, Any], provider: str) -> dict[str, Any]:
    providers = config.get("providers") if isinstance(config, Mapping) else None
    if not isinstance(providers, Mapping):
        return {}
    provider_norm = (provider or "").strip().lower()
    for key, value in providers.items():
        if not isinstance(value, Mapping):
            continue
        candidates = {
            str(key).strip().lower(),
            str(value.get("name") or "").strip().lower(),
            str(value.get("provider_key") or "").strip().lower(),
            f"custom:{str(key).strip().lower()}",
        }
        if provider_norm in candidates:
            return dict(value)
    return {}


def _model_entry(provider_cfg: Mapping[str, Any], model: str) -> dict[str, Any]:
    models = provider_cfg.get("models") if isinstance(provider_cfg, Mapping) else None
    if not isinstance(models, Mapping):
        return {}
    model_norm = (model or "").strip().lower()
    for key, value in models.items():
        if str(key).strip().lower() == model_norm and isinstance(value, Mapping):
            return dict(value)
    return {}


def resolve_model_capability(
    *,
    provider: str,
    model: str,
    config: Mapping[str, Any] | None = None,
) -> ModelCapability:
    """Resolve capability metadata for ``provider``/``model``.

    Metadata is read from ``providers.<provider>.models.<model>`` first and
    falls back to provider-level fields.  Missing values use conservative
    defaults so unknown models are treated as weak until explicitly described.
    """

    config = config or {}
    provider_cfg = _provider_entry(config, provider)
    model_cfg = _model_entry(provider_cfg, model)

    scores = dict(_DEFAULT_SCORES)
    scores.update(_builtin_scores_for_model(model))
    for source in (provider_cfg.get("scores"), model_cfg.get("scores")):
        if isinstance(source, Mapping):
            for key, value in source.items():
                scores[str(key)] = _to_float(value, scores.get(str(key), _DEFAULT_SCORES["general"]))

    context_tokens = (
        _to_positive_int(model_cfg.get("context_length"))
        or _to_positive_int(model_cfg.get("context_tokens"))
        or _to_positive_int(provider_cfg.get("context_length"))
    )

    raw_limits = model_cfg.get("limits")
    limits: Mapping[str, Any] = raw_limits if isinstance(raw_limits, Mapping) else {}
    raw_provider_limits = provider_cfg.get("limits")
    provider_limits: Mapping[str, Any] = raw_provider_limits if isinstance(raw_provider_limits, Mapping) else {}
    max_iters = (
        _to_positive_int(limits.get("max_safe_tool_iterations"))
        or _to_positive_int(provider_limits.get("max_safe_tool_iterations"))
    )

    supports_tools = _coerce_bool(model_cfg.get("supports_tools"))
    if supports_tools is None:
        supports_tools = _coerce_bool(provider_cfg.get("supports_tools"))
    supports_vision = _coerce_bool(model_cfg.get("supports_vision"))
    if supports_vision is None:
        supports_vision = _coerce_bool(provider_cfg.get("supports_vision"))

    degraded_label = str(
        model_cfg.get("degraded_label")
        or provider_cfg.get("degraded_label")
        or "unknown"
    ).strip() or "unknown"

    return ModelCapability(
        provider=(provider or "").strip(),
        model=(model or "").strip(),
        scores=scores,
        context_tokens=context_tokens,
        supports_tools=supports_tools,
        supports_vision=supports_vision,
        max_safe_tool_iterations=max_iters,
        degraded_label=degraded_label,
        raw={"provider": provider_cfg, "model": model_cfg},
    )
