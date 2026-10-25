"""xAI (Grok) provider profile."""

from decimal import Decimal
from typing import Any

from hermes_cli.version_info import get_version_info
from providers import register_provider
from providers.base import ProviderProfile

_USD_TICKS = Decimal(10**10)  # https://docs.x.ai/developers/cost-tracking


class XaiProfile(ProviderProfile):
    def get_usage_cost(self, model: str, usage: Any, *, base_url: str | None = None) -> Any | None:
        """Billed ``usage.cost_in_usd_ticks`` on a direct api.x.ai origin; else None (list-price path)."""
        from agent.usage_pricing import CostResult, direct_first_party_origin, format_cost_label

        ticks = (getattr(usage, "raw_usage", None) or {}).get("cost_in_usd_ticks")
        if type(ticks) is not int or ticks < 0 or not direct_first_party_origin("xai", base_url):
            return None
        amount = Decimal(ticks) / _USD_TICKS
        return CostResult(
            amount_usd=amount, status="actual", source="provider_cost_api",
            label=format_cost_label(amount).removeprefix("~"), pricing_version="xai-cost-in-usd-ticks",
        )


xai = XaiProfile(
    name="xai", aliases=("grok", "x-ai", "x.ai"), api_mode="codex_responses", env_vars=("XAI_API_KEY",),
    base_url="https://api.x.ai/v1", auth_type="api_key",
    default_headers={"User-Agent": f"Hermes-Agent/{get_version_info().base_version}"},
)

register_provider(xai)
