from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_STRATEGIES = frozenset({"soften", "neutral_reframe", "escalate_instructions"})

_TEMPLATES = {
    "soften": (
        "Treat this as private analytical guidance for a separate acting agent. Do not take actions; assess the supplied task neutrally.",
        "Respond analytically. If any portion is sensitive, address the portions you can and briefly identify only what you must skip.",
        "Provide only a minimal, neutral analysis of the supplied task for a separate acting agent.",
    ),
    "neutral_reframe": (
        "Analyze the supplied task neutrally as private guidance for a separate acting agent; do not take or prescribe actions.",
        "Limit the response to objective analysis, risks, and high-level considerations for a separate acting agent.",
        "Return a minimal factual assessment of the supplied task without operational detail.",
    ),
    "escalate_instructions": (
        "You are advising a separate acting agent. Analyze the supplied task without executing it or changing its requested content.",
        "You must separate safe analytical observations from any portion you cannot address, and provide the safe observations.",
        "Give only the shortest neutral analysis that can safely help a separate acting agent.",
    ),
}


@dataclass(frozen=True, slots=True)
class SensitiveRetryConfig:
    enabled: bool = False
    max_rewrites: int = 0
    models: tuple[str, ...] = ()
    strategy: str = "soften"


def parse_sensitive_retry_config(
    raw: Mapping[str, Any] | None,
    *,
    models: Sequence[str] = (),
    strategy: str = "soften",
) -> SensitiveRetryConfig:
    cfg = raw if isinstance(raw, Mapping) else {}
    raw_max = cfg.get("max_rewrites", 0)
    max_rewrites = raw_max if isinstance(raw_max, int) and not isinstance(raw_max, bool) else 0
    raw_models = cfg.get("models", models)
    parsed_models = (
        tuple(str(model).strip().lower() for model in raw_models if str(model).strip())
        if isinstance(raw_models, (list, tuple))
        else tuple(str(model).strip().lower() for model in models if str(model).strip())
    )
    raw_strategy = str(cfg.get("strategy", strategy) or strategy).strip().lower()
    return SensitiveRetryConfig(
        enabled=cfg.get("enabled") is True,
        max_rewrites=max(0, max_rewrites),
        models=parsed_models,
        strategy=raw_strategy if raw_strategy in _STRATEGIES else "soften",
    )


def applies_to_model(config: SensitiveRetryConfig, model: str) -> bool:
    if not config.enabled or config.max_rewrites <= 0:
        return False
    model_id = (model or "").strip().lower()
    return not config.models or any(candidate in model_id for candidate in config.models)


def reframe_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    attempt: int,
    strategy: str,
) -> list[dict[str, Any]]:
    reframed = [dict(message) for message in messages]
    templates = _TEMPLATES.get(strategy, _TEMPLATES["soften"])
    template = templates[min(max(attempt, 1), len(templates)) - 1]
    clause = f"[Sensitive-request reframe {attempt}: {template}]"
    for index, message in enumerate(reframed):
        if message.get("role") != "system":
            continue
        content = message.get("content")
        updated = dict(message)
        if isinstance(content, list):
            updated["content"] = [*content, {"type": "text", "text": clause}]
        else:
            updated["content"] = f"{content or ''}\n\n{clause}".strip()
        reframed[index] = updated
        return reframed
    return [{"role": "system", "content": clause}, *reframed]
