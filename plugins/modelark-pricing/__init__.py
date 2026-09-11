"""ModelArk Coding Plan notional pricing — keeps the $1.00 card cap live (fleet, 2026-09-11).

The Coding Plan is a flat subscription: no call is invoiced, and ``agent.usage_pricing`` has no entry
for the Ark model ids, so without this every ModelArk call records ``estimated_cost_usd = 0`` /
``cost_status = unknown``. ``hermes_cli.kanban_db._cumulative_session_cost`` sums exactly that column,
so the per-card cap would compare $0.00 against every cap — the 2026-09-08 failure that forced the
first ModelArk rollback. We price the Ark ids at DeepSeek's own list rate (the bundled ``deepseek``
snapshot, so an upstream refresh of DeepSeek's list flows through) and mark the source
``modelark-proxy``. Reports treat that source as "modelark subscription", never as money spent.

How it hooks in: rows are added to ``usage_pricing._OFFICIAL_DOCS_PRICING`` keyed by the provider
strings a named custom provider can arrive with (``custom``, ``modelark``, ``custom:modelark``).
``get_pricing_entry`` consults that dict through a module global on every call, so every caller —
turn usage, auxiliary accounting, insights — sees the entry without any function being patched.
``setdefault`` means a future upstream entry for the same key wins.
"""
from __future__ import annotations

import dataclasses
import logging

logger = logging.getLogger(__name__)

SOURCE = "modelark-proxy"
LABEL = "modelark subscription"
ARK_HOST = "ark.ap-southeast.bytepluses.com"
PRICING_VERSION = "modelark-subscription-proxy-2026-09b"
SOURCE_URL = ("ModelArk Coding Plan (flat subscription): notional DeepSeek list price for cap accounting "
              "only, not invoiced")
# Ark model id -> DeepSeek family key in the bundled "deepseek" snapshot.
PROXY = {
    "deepseek-v4-flash-ga-260731": "deepseek-v4-flash",
    "deepseek-v4-flash-260425": "deepseek-v4-flash",
    "deepseek-v4-pro-ga-260813": "deepseek-v4-pro",
    "deepseek-v4-pro-260425": "deepseek-v4-pro",
    # The Coding Plan answers with the SERVED name ("deepseek-v4-flash"), and auxiliary accounting
    # prices on response.model — so title/compression calls arrive under the bare name. Found on the
    # wire 2026-09-11: 4 title_generation rows recorded $0 until these were added.
    "deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek-v4-pro": "deepseek-v4-pro",
}
PROVIDER_KEYS = ("custom", "modelark", "custom:modelark")


def install() -> int:
    """Add the proxy rows. Returns how many (provider, model) keys now carry a modelark-proxy entry."""
    from agent import usage_pricing as up

    table = up._OFFICIAL_DOCS_PRICING
    n = 0
    for ark_id, family in PROXY.items():
        base = table.get(("deepseek", family))
        if base is None:
            logger.warning("modelark-pricing: no bundled ('deepseek', %r) price — %s stays unpriced", family, ark_id)
            continue
        entry = dataclasses.replace(
            base,
            # Ark has no prefix cache for these ids today; if a cache-write count ever appears, price it
            # at the input rate rather than letting a None rate turn the whole call "unknown" ($0).
            cache_write_cost_per_million=(base.cache_write_cost_per_million
                                          if base.cache_write_cost_per_million is not None
                                          else base.input_cost_per_million),
            source=SOURCE, source_url=SOURCE_URL, pricing_version=PRICING_VERSION,
        )
        for provider in PROVIDER_KEYS:
            if table.setdefault((provider, ark_id), entry).source == SOURCE:
                n += 1
    return n


def register(ctx) -> None:  # noqa: ARG001 — plugin loader entry point
    n = install()
    logger.info("modelark-pricing: %d notional price rows active (%s)", n, PRICING_VERSION)
