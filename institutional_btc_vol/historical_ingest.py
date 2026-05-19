from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from institutional_btc_vol.historical_schema import HistoricalOptionQuote, HistoricalSchemaError
from institutional_btc_vol.historical_sources import source_file_sha256


class HistoricalIngestError(ValueError):
    pass


def _parse_ts(value: Any) -> datetime:
    text = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _float(value: Any) -> float:
    return float(str(value).strip().replace("%", ""))


def _option_iv(value: Any) -> float:
    parsed = _float(value)
    return parsed / 100.0 if parsed > 2 else parsed


def _dte(event_ts: datetime, expiry: str) -> float:
    expiry_date = datetime.fromisoformat(str(expiry)).date()
    return max(float((expiry_date - event_ts.date()).days), 0.0)


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_generic_ibit_options_csv(
    path: str | Path,
    *,
    source_id: str,
    source_confidence: str,
    available_lag_seconds: int = 0,
) -> list[HistoricalOptionQuote]:
    raw_path = Path(path)
    digest = source_file_sha256(raw_path)
    quotes: list[HistoricalOptionQuote] = []
    for row in _read_csv_dicts(raw_path):
        try:
            event = _parse_ts(row.get("ts") or row.get("event_ts") or row.get("timestamp"))
            available = event + timedelta(seconds=available_lag_seconds)
            bid = _float(row["bid"])
            ask = _float(row["ask"])
            btc_per_share = _float(row["btc_per_share"])
            strike = _float(row["strike"])
            underlying = _float(row["underlying_price"])
            mid = _float(row.get("mid") or ((bid + ask) / 2))
            quote = HistoricalOptionQuote(
                as_of_utc=_iso_z(event),
                event_ts=_iso_z(event),
                available_ts=_iso_z(available),
                venue="OPRA",
                underlying_symbol="IBIT",
                native_symbol=str(row.get("symbol") or row.get("native_symbol") or "").strip(),
                instrument_id=str(row.get("instrument_id") or row.get("symbol") or "").strip(),
                expiry=str(row["expiry"]).strip(),
                dte=_dte(event, str(row["expiry"]).strip()),
                option_type=str(row["option_type"]).strip().lower(),
                strike_native=strike,
                strike_btc_equivalent=strike / btc_per_share,
                underlying_price=underlying,
                underlying_btc_equivalent=underlying / btc_per_share,
                btc_per_share=btc_per_share,
                bid=bid,
                ask=ask,
                mid=mid,
                bid_iv=_option_iv(row["bid_iv"]) if row.get("bid_iv") else None,
                ask_iv=_option_iv(row["ask_iv"]) if row.get("ask_iv") else None,
                mid_iv=_option_iv(row.get("mid_iv") or row.get("iv") or row.get("model_iv")),
                iv_source="model_estimated_from_historical_bid_ask_mid",
                source_id=source_id,
                source_sha256=digest,
                source_confidence=source_confidence,
            )
            quote.to_dict()
            quotes.append(quote)
        except (KeyError, ValueError, HistoricalSchemaError):
            continue
    if not quotes:
        raise HistoricalIngestError(f"no valid historical option rows parsed from {raw_path}")
    return quotes


def parse_generic_deribit_options_jsonl(
    path: str | Path,
    *,
    source_id: str,
    source_confidence: str,
    available_lag_seconds: int = 0,
) -> list[HistoricalOptionQuote]:
    raw_path = Path(path)
    digest = source_file_sha256(raw_path)
    quotes: list[HistoricalOptionQuote] = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            event = _parse_ts(row.get("timestamp") or row.get("event_ts") or row.get("ts"))
            available = event + timedelta(seconds=available_lag_seconds)
            bid = _float(row.get("bid") or row.get("best_bid_price"))
            ask = _float(row.get("ask") or row.get("best_ask_price"))
            mid = _float(row.get("mid") or ((bid + ask) / 2))
            strike = _float(row.get("strike") or row.get("strike_native"))
            underlying = _float(row.get("underlying_price") or row.get("index_price"))
            mark_iv_value = row.get("mark_iv") or row.get("mid_iv") or row.get("iv")
            quote = HistoricalOptionQuote(
                as_of_utc=_iso_z(event),
                event_ts=_iso_z(event),
                available_ts=_iso_z(available),
                venue="Deribit",
                underlying_symbol="BTC",
                native_symbol=str(row.get("instrument_name") or row.get("native_symbol") or "").strip(),
                instrument_id=str(row.get("instrument_id") or row.get("instrument_name") or "").strip(),
                expiry=str(row["expiry"]).strip(),
                dte=_dte(event, str(row["expiry"]).strip()),
                option_type=str(row["option_type"]).strip().lower(),
                strike_native=strike,
                strike_btc_equivalent=strike,
                underlying_price=underlying,
                underlying_btc_equivalent=underlying,
                btc_per_share=None,
                bid=bid,
                ask=ask,
                mid=mid,
                bid_iv=_option_iv(row["bid_iv"]) if row.get("bid_iv") else None,
                ask_iv=_option_iv(row["ask_iv"]) if row.get("ask_iv") else None,
                mid_iv=_option_iv(mark_iv_value),
                iv_source="vendor_mark_iv",
                source_id=source_id,
                source_sha256=digest,
                source_confidence=source_confidence,
            )
            quote.to_dict()
            quotes.append(quote)
        except (json.JSONDecodeError, KeyError, ValueError, HistoricalSchemaError):
            continue
    if not quotes:
        raise HistoricalIngestError(f"no valid historical option rows parsed from {raw_path}")
    return quotes
