from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterable

from institutional_btc_vol.historical_schema import HistoricalOptionQuote, HistoricalSpreadSnapshot


def _parse_ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _quote_key(quote: HistoricalOptionQuote) -> str:
    # Coarse point-in-time bucket. v1 aligns same-session observations by UTC date;
    # the resulting decision time is still the max available_ts across matched legs,
    # so the strategy cannot see either side before both are available.
    return _parse_ts(quote.available_ts).date().isoformat()


def _atm_distance(quote: HistoricalOptionQuote) -> float:
    strike = quote.strike_btc_equivalent if quote.strike_btc_equivalent is not None else quote.strike_native
    underlying = quote.underlying_btc_equivalent if quote.underlying_btc_equivalent is not None else quote.underlying_price
    return abs(float(strike) - float(underlying))


def _select_nearest_atm(quotes: list[HistoricalOptionQuote], target_tenor: int, max_dte_gap: float) -> HistoricalOptionQuote | None:
    eligible = [quote for quote in quotes if abs(float(quote.dte) - float(target_tenor)) <= max_dte_gap]
    if not eligible:
        return None
    return sorted(eligible, key=lambda quote: (abs(float(quote.dte) - float(target_tenor)), _atm_distance(quote), quote.native_symbol))[0]


def build_venue_pair_spread_snapshots(
    quotes: Iterable[HistoricalOptionQuote],
    *,
    left_venue: str,
    right_venue: str,
    venue_pair: str,
    tenors: tuple[int, ...] = (1, 7, 30),
    replay_id: str,
    max_dte_gap: float = 3.0,
) -> list[HistoricalSpreadSnapshot]:
    grouped: dict[str, dict[str, list[HistoricalOptionQuote]]] = defaultdict(lambda: defaultdict(list))
    for quote in quotes:
        if quote.venue in {left_venue, right_venue}:
            grouped[_quote_key(quote)][quote.venue].append(quote)

    snapshots: list[HistoricalSpreadSnapshot] = []
    for bucket in sorted(grouped):
        left_quotes = grouped[bucket].get(left_venue, [])
        right_quotes = grouped[bucket].get(right_venue, [])
        if not left_quotes or not right_quotes:
            continue
        for tenor in tenors:
            left = _select_nearest_atm(left_quotes, tenor, max_dte_gap)
            right = _select_nearest_atm(right_quotes, tenor, max_dte_gap)
            if left is None or right is None:
                continue
            left_avail = _parse_ts(left.available_ts)
            right_avail = _parse_ts(right.available_ts)
            decision = max(left_avail, right_avail)
            spread_vol_pts = round((float(left.mid_iv) - float(right.mid_iv)) * 100.0, 2)
            snapshots.append(
                HistoricalSpreadSnapshot(
                    replay_id=replay_id,
                    decision_ts=_iso_z(decision),
                    available_ts=_iso_z(decision),
                    tenor=f"{tenor}d",
                    venue_pair=venue_pair,
                    left_mid_iv=float(left.mid_iv),
                    right_mid_iv=float(right.mid_iv),
                    spread_vol_pts=spread_vol_pts,
                    left_instrument=left.native_symbol,
                    right_instrument=right.native_symbol,
                    source_ids=[left.source_id, right.source_id],
                    source_hashes=[left.source_sha256, right.source_sha256],
                    normalization={
                        "atm_rule": "nearest_strike",
                        "tenor_bucket_rule": "nearest_dte_with_max_gap",
                        "max_dte_gap": max_dte_gap,
                        "left_venue": left_venue,
                        "right_venue": right_venue,
                    },
                )
            )
    return snapshots
