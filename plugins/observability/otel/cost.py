# SPDX-License-Identifier: MIT
"""Local-model cost estimation for the observability/otel plugin.

Hermes runs against local backends (Ollama, HF transformers, vLLM) where the
provider returns **no price**. The TraceVerse ``genai-otel-instrument`` library
nevertheless populates cost for local models from a pricing DB plus a
**model-size → price-tier** estimate, so a local run shows real (non-zero) cost
in the dashboard rather than ``$0``.

This module mirrors that methodology so the plugin's spans **and** dashboard log
records carry the same ``cost_usd`` the rest of the platform expects:

* parse the parameter size from the model name (``llama3.1:8b`` → 8.0 B,
  ``smollm2:360m`` → 0.36 B, ``qwen3:0.6b`` → 0.6 B);
* map the size to a per-1K-token prompt/completion price tier;
* ``cost = prompt_tokens/1000 * promptPrice + completion_tokens/1000 * completionPrice``.

Numbers are kept identical to ``genai_otel.cost_calculator`` (the canonical
source) so Hermes cost lines up with every other agent on the platform. When the
size can't be determined, cost is ``None`` (the caller then omits it, exactly as
the library returns ``0.0``/unknown).
"""

from __future__ import annotations

import re
from typing import Optional

# Per-1,000-token prices by parameter-count tier — verbatim from
# genai_otel.cost_calculator._get_local_model_price_tier. (prompt, completion)
_SIZE_TIERS = (
    (1.0, 0.0001, 0.0002),    # < 1B
    (10.0, 0.0003, 0.0006),   # 1–10B
    (20.0, 0.0005, 0.0010),   # 10–20B
    (80.0, 0.0008, 0.0008),   # 20–80B
)
_XLARGE = (0.0012, 0.0012)    # 80B+

# HuggingFace model-name → size (billions) fallback for names without an
# explicit Nb/Nm suffix. Mirrors the library's hardcoded map (subset).
_HF_SIZE_MAP = {
    "t5-small": 0.06,
    "gpt2": 0.124,
    "gpt2-medium": 0.355,
    "gpt2-large": 0.774,
    "gpt2-xl": 1.5,
    "bert-base": 0.11,
    "bert-large": 0.34,
    "distilbert": 0.066,
}

# (\d+(?:\.\d+)?)(m|b) followed by a boundary — matches "8b", "0.6b", "360m".
_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)(m|b)(?:\s|:|$|-)")


def extract_param_count_billions(model: str) -> Optional[float]:
    """Return the model's parameter count in billions, or ``None`` if unknown.

    ``llama3.1:8b`` → 8.0 · ``qwen3:0.6b`` → 0.6 · ``smollm2:360m`` → 0.36.
    Falls back to the HF name map for suffix-less names (``gpt2`` → 0.124).
    """
    if not model:
        return None
    m = model.lower()
    match = _PARAM_RE.search(m)
    if match:
        value = float(match.group(1))
        return value / 1000.0 if match.group(2) == "m" else value
    for name, size in _HF_SIZE_MAP.items():
        if name in m:
            return size
    return None


def _price_tier(param_billions: float) -> tuple[float, float]:
    for ceiling, prompt_price, completion_price in _SIZE_TIERS:
        if param_billions < ceiling:
            return prompt_price, completion_price
    return _XLARGE


def estimate_cost_usd(
    model: Optional[str],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> Optional[float]:
    """Estimate USD cost for a local model call from its size + token usage.

    Returns ``None`` when the model size can't be inferred or there are no
    tokens — callers omit cost in that case (no fake ``$0`` rows). The formula
    and tier numbers match ``genai-otel-instrument`` so platform cost is
    consistent across agents.
    """
    param_billions = extract_param_count_billions(model or "")
    if param_billions is None:
        return None
    in_tok = int(input_tokens) if input_tokens else 0
    out_tok = int(output_tokens) if output_tokens else 0
    if in_tok == 0 and out_tok == 0:
        return None
    prompt_price, completion_price = _price_tier(param_billions)
    cost = (in_tok / 1000.0) * prompt_price + (out_tok / 1000.0) * completion_price
    return round(cost, 8)
