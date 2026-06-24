from __future__ import annotations

from typing import Any

from .models import MarketMetadata, SelectedMarket


def _get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def enrich_market_metadata(client: Any, selected_market: SelectedMarket, *, token_id: str) -> MarketMetadata:
    market_info = client.get_clob_market_info(selected_market.condition_id) if selected_market.condition_id else {}
    fee_rate_bps = client.get_fee_rate_bps(token_id)
    fee_exponent = client.get_fee_exponent(token_id)
    return MarketMetadata(
        condition_id=selected_market.condition_id,
        token_id=token_id,
        tick_size=float(_get(market_info, "tick_size", "tickSize", default=0.01)),
        neg_risk=bool(_get(market_info, "neg_risk", "negRisk", default=False)),
        fee_rate_bps=int(fee_rate_bps),
        fee_exponent=int(fee_exponent),
    )
