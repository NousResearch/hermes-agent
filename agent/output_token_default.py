"""Context-aware output-token default for custom providers (PR 1 of the starvation fix set).

When a custom provider omits ``max_tokens`` and no ephemeral boost applies, the
request goes out with NO output cap. Two failure classes follow:

- Hosted gateways with large contexts may apply their own small internal cap
  (observed: ~1,000 tokens, thinking-inclusive) — reasoning starves the answer
  (see the output-cap starvation breaker for the reactive half of this fix).
- Context-tight local servers (vLLM/llama.cpp/MLX) fill to end-of-context or
  400 when ``prompt + max_tokens`` exceeds it, so a blind large default is wrong.

The consensus design (3-reviewer design review): compute an explicit default from
the endpoint's own metadata when the cap is omitted::

    effective = min(TARGET_CAP, max(FLOOR, context_length - prompt_estimate - BUFFER))

- Hosted gateway, 270K context, 50K prompt → clamped to TARGET_CAP (65,535):
  thinking never starves the answer.
- Local 16K model, 12K prompt → headroom ≈ 3.5K: vLLM accepts the request.

The computed value is quantized down to a 4,096 boundary: generation parameters
do not affect KV prefix caching, but some proxy cache buckets key on the raw
request body, and a cap that changes by a few tokens every turn would churn
those. Only applies when the endpoint's context length is actually known
(``get_model_context_length`` resolution; 0/unknown → no default, legacy
behavior preserved). Native OpenAI/Anthropic/Google providers and any explicit
``max_tokens``/ephemeral boost are never touched.

Compression-threshold interaction (reviewed): the compressor's
``_compute_threshold_tokens`` derives from ``agent.max_tokens`` (user config) —
NOT from the wire default computed here — and its degenerate-window guards
(``effective_window <= 0`` fallback, 85% trigger cap) already handle small
windows, so no compressor change is needed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Published Gemini ceiling: matches the native-Gemini auto-raise default.
OUTPUT_DEFAULT_TARGET_CAP = 65_535
# Never send a cap below this — too small to complete any real turn.
OUTPUT_DEFAULT_FLOOR = 1_024
# Headroom reserved for the response side of the context split.
OUTPUT_DEFAULT_HEADROOM_BUFFER = 512
# Quantization bucket: stable wire values across small prompt drift.
OUTPUT_DEFAULT_QUANTUM = 4_096


def estimate_prompt_tokens(messages: Any) -> int:
    """JSON-serialized size estimate for headroom math (per review: no vendored
    tokenizers for arbitrary custom models). Serializing the WHOLE payload counts
    everything that will ride the wire — tool-call JSON arguments, base64 image
    parts, tool results — so the estimate errs high. Over-estimation is safe
    (only shrinks the output default); under-estimation risks endpoint 400s."""
    if not messages:
        return 0
    try:
        import json

        return len(json.dumps(messages, default=str)) // 4
    except Exception:  # noqa: BLE001 — estimate only; never break the request
        return 0


def context_aware_output_default(
    *,
    model: str,
    base_url: str,
    provider: str,
    api_key: str | None,
    custom_providers: list | None,
    config_context_length: int | None,
    prompt_token_estimate: int,
) -> int | None:
    """Output-cap default from endpoint metadata, or None (send no cap / legacy).

    Returns None when the context length is unknown — a guessed default on an
    unknown endpoint risks the exact 400s this default exists to avoid.
    """
    from agent.model_metadata import get_model_context_length

    try:
        context_length = get_model_context_length(
            model, base_url=base_url or "", api_key=api_key or "",
            config_context_length=config_context_length, provider=provider or "",
            custom_providers=custom_providers,
        )
    except Exception:  # noqa: BLE001 — a metadata failure must never break the request
        logger.debug("output default: context length lookup failed", exc_info=True)
        return None
    if not isinstance(context_length, int) or context_length <= 0:
        return None
    headroom = context_length - int(prompt_token_estimate or 0) - OUTPUT_DEFAULT_HEADROOM_BUFFER
    if headroom < OUTPUT_DEFAULT_FLOOR:
        # Prompt already fills the window: no safe output default exists; sending
        # one would guarantee truncation or a 400. Legacy behavior (no cap) is the
        # fail-open choice — the starvation breaker handles the aftermath.
        return None
    effective = min(OUTPUT_DEFAULT_TARGET_CAP, headroom)
    # Quantize down so consecutive turns with slowly-growing prompts share one
    # wire value (proxy request-body cache buckets).
    quantized = max(
        OUTPUT_DEFAULT_FLOOR,
        (effective // OUTPUT_DEFAULT_QUANTUM) * OUTPUT_DEFAULT_QUANTUM,
    )
    return quantized


_MANAGED_CAP_HOSTS = (
    # Routes that manage their own caps (or reject max_tokens outright); the
    # omission there is honored and must stay untouched.
    "api.openai.com", "openai.azure.com", "api.anthropic.com", "anthropic.com",
    "generativelanguage.googleapis.com", "openrouter.ai", "api.nous.ai",
    "bedrock",
)


def context_aware_output_default_for_params(model: str, params: dict[str, Any]) -> int | None:
    """Param-dict form of ``context_aware_output_default``; None when the route
    manages its own caps (native bases), has no base URL, or the cap is unset."""
    base_url = str(params.get("base_url") or "")
    if not base_url:
        return None
    lowered = base_url.lower()
    if any(host in lowered for host in _MANAGED_CAP_HOSTS):
        return None
    return context_aware_output_default(
        model=model,
        base_url=base_url,
        provider=str(params.get("provider") or ""),
        api_key=params.get("api_key"),
        custom_providers=params.get("custom_providers"),
        config_context_length=params.get("config_context_length"),
        prompt_token_estimate=estimate_prompt_tokens(params.get("messages")),
    )
