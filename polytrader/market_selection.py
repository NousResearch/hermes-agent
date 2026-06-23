from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable

from .models import SelectedMarket

_CRYPTO_NAMES = {
    "BTC": ("btc", "bitcoin"),
    "ETH": ("eth", "ethereum"),
    "SOL": ("sol", "solana"),
    "XRP": ("xrp", "ripple"),
}


def _decode(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _event_market(event: dict[str, Any]) -> dict[str, Any] | None:
    markets = event.get("markets") or []
    if not markets:
        return None
    return markets[0]


def _active_not_closed(event: dict[str, Any], market: dict[str, Any]) -> bool:
    if event.get("closed") or market.get("closed"):
        return False
    if event.get("active") is False or market.get("active") is False:
        return False
    if market.get("enableOrderBook") is False:
        return False
    end = event.get("endDate") or market.get("endDate") or event.get("end_date") or market.get("end_date")
    if not end:
        return True
    try:
        parsed = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
    except ValueError:
        return True
    return parsed > datetime.now(timezone.utc)


def _mentions_5m(text: str, slug: str) -> bool:
    haystack = f"{slug} {text}".lower()
    five_markers = ("5m", "5 min", "5-minute", "5 minute", "5min")
    fifteen_markers = ("15m", "15 min", "15-minute", "15 minute", "15min")
    return any(marker in haystack for marker in five_markers) and not any(marker in haystack for marker in fifteen_markers)


def _matches_symbol(text: str, slug: str, crypto_symbol: str) -> bool:
    names = _CRYPTO_NAMES.get(crypto_symbol.upper(), (crypto_symbol.lower(),))
    haystack = f"{slug} {text}".lower()
    return any(name in haystack for name in names)


def _selected_from_event(event: dict[str, Any]) -> SelectedMarket | None:
    market = _event_market(event)
    if not market:
        return None
    outcomes = _decode(market.get("outcomes")) or []
    token_ids = _decode(market.get("clobTokenIds")) or _decode(market.get("clob_token_ids")) or []
    side_to_token = {str(outcome).strip().lower(): str(token) for outcome, token in zip(outcomes, token_ids)}
    if "up" not in side_to_token or "down" not in side_to_token:
        return None
    return SelectedMarket(
        slug=str(event.get("slug") or market.get("slug") or ""),
        title=str(event.get("title") or market.get("question") or market.get("title") or ""),
        condition_id=market.get("conditionId") or market.get("condition_id"),
        up_token_id=side_to_token["up"],
        down_token_id=side_to_token["down"],
        end_date_iso=event.get("endDate") or market.get("endDate"),
    )


def select_5m_updown_market(
    events: Iterable[dict[str, Any]],
    *,
    crypto_symbol: str,
    market_slug: str | None = None,
) -> SelectedMarket:
    candidates = list(events)
    if market_slug:
        for event in candidates:
            if event.get("slug") == market_slug or (_event_market(event) or {}).get("slug") == market_slug:
                selected = _selected_from_event(event)
                if selected is None:
                    raise ValueError(f"pinned market {market_slug!r} does not expose Up/Down CLOB tokens")
                return selected
        raise ValueError(f"pinned market {market_slug!r} not found")

    for event in candidates:
        market = _event_market(event)
        if not market or not _active_not_closed(event, market):
            continue
        slug = str(event.get("slug") or market.get("slug") or "")
        title = str(event.get("title") or market.get("question") or market.get("title") or "")
        text = f"{title} {market.get('question', '')}"
        if not _matches_symbol(text, slug, crypto_symbol):
            continue
        if "up" not in text.lower() or "down" not in text.lower():
            continue
        if not _mentions_5m(text, slug):
            continue
        selected = _selected_from_event(event)
        if selected:
            return selected
    raise ValueError(f"no active 5-minute {crypto_symbol.upper()} up/down market found")
