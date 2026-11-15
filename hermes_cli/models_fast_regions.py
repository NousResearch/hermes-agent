"""OpenAI data-residency hosts that may receive a fast ``service_tier``.

Regional support is published per region and per model, so ``*.api.openai.com`` is not a license:
au/jp/in serve "Standard pay-as-you-go only", ca/sg/kr/gb have no regional processing, and EU Fast
excludes GPT-6 Astra and GPT-6 Sol. Ultrafast needs no second table: only Ultrafast models get that
tier, and GPT-6.1 Sol (US+EU) / GPT-6 Astra (US only) line up with this matrix.
https://developers.openai.com/api/docs/guides/data-residency
https://developers.openai.com/api/docs/guides/fast-mode#is-fast-mode-compatible-with-data-residency-zero-data-retention-and-a-baa
"""

from typing import Optional

# Host -> documented Fast models; None = every model the global endpoint accepts.
_FAST_REGION_MODELS: dict[str, Optional[frozenset[str]]] = {
    "us.api.openai.com": None,
    "eu.api.openai.com": frozenset({"gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-luna"}),
}


def openai_fast_region_allows(base_url: Optional[str], model_id: Optional[str]) -> bool:
    """True when a regional OpenAI host documents fast tiers for this model."""
    from agent.model_metadata import strip_codex_context_variant_suffix
    from hermes_cli.models import _strip_vendor_prefix
    from utils import base_url_hostname

    host = base_url_hostname(str(base_url or ""))
    if host not in _FAST_REGION_MODELS:
        return False
    models = _FAST_REGION_MODELS[host]
    base = _strip_vendor_prefix(strip_codex_context_variant_suffix(str(model_id or ""))).split(":")[0]
    return models is None or base in models
