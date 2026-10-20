"""Third-party catalogue entries that lag an alias the provider has moved server-side.

The DeepSeek API serves ``deepseek-chat`` / ``deepseek-reasoner`` from V4 Flash (1M), but
OpenRouter still reports 163,840 for ``deepseek-chat`` (live-checked) and the models.dev entry
is 128K. Only the alias *identity* identifies these as stale: the number alone cannot, because
128K is also the legitimate window of the shorter ``deepseek`` catch-all family.
"""
import logging
from typing import Optional

from hermes_constants import openrouter_variant_base

logger = logging.getLogger("agent.model_metadata")

_DEEPSEEK_V4_ALIAS_SLUGS = frozenset({"deepseek-chat", "deepseek-reasoner"})


def deepseek_v4_alias_context_length(model: str) -> Optional[int]:
    """Catalog window for the V4 alias slugs, else None. Routing variants (``:nitro``,
    ``:batch``) and vendor prefixes resolve to the same alias."""
    from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS

    base = (openrouter_variant_base(model) or model).lower()
    slug = base.rsplit("/", 1)[-1]
    return DEFAULT_CONTEXT_LENGTHS.get(slug) if slug in _DEEPSEEK_V4_ALIAS_SLUGS else None


def stale_alias_underreport(model: str, ctx: int) -> Optional[int]:
    """Hardcoded catalog window when the provider value underreports an alias, else None.

    Sibling of the MiniMax-M3 models.dev guard: the provider catalog lags a server-side alias
    move, so the hardcoded catalog wins. Versioned ids are self-describing and never match.
    """
    catalog = deepseek_v4_alias_context_length(model)
    return catalog if catalog and ctx < catalog else None


def rejects_openrouter_entry(model: str, or_ctx: int, stale_32k: bool) -> bool:
    """True, and logged, when OpenRouter's provider-unaware window must not be trusted:
    the known 32K underreport (*stale_32k* says the model is in that family) or an alias
    entry that lags its server-side move."""
    if or_ctx == 32768 and stale_32k:
        logger.info("Rejecting OpenRouter metadata context=%s for %r (known 32K underreport); falling through to hardcoded defaults", or_ctx, model)
        return True
    stale_alias = stale_alias_underreport(model, or_ctx)
    if stale_alias:
        logger.info("Rejecting OpenRouter metadata context=%s for %r (stale alias entry); using hardcoded default %s", or_ctx, model, f"{stale_alias:,}")
        return True
    return False
