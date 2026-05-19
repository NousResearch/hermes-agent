from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Protocol

ALLOWED_SOURCE_CONFIDENCE = {
    "licensed_vendor_api",
    "licensed_vendor_api_databento",
    "public_screen_reference",
    "manual_fixture",
    "derived_internal",
}

EVIDENCE_STATUS = "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE"


class HistoricalSchemaError(ValueError):
    pass


class JsonRecord(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


def _parse_ts(value: str, field_name: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise HistoricalSchemaError(f"invalid timestamp for {field_name}: {value}") from exc


def _validate_source_confidence(value: str) -> None:
    if value not in ALLOWED_SOURCE_CONFIDENCE:
        raise HistoricalSchemaError(f"invalid source_confidence: {value}")


def _validate_event_available(event_ts: str, available_ts: str) -> None:
    if _parse_ts(available_ts, "available_ts") < _parse_ts(event_ts, "event_ts"):
        raise HistoricalSchemaError("available_ts cannot precede event_ts")


def _validate_hash(value: str, field_name: str = "source_sha256") -> None:
    if len(str(value)) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in str(value)):
        raise HistoricalSchemaError(f"{field_name} must be a 64 character SHA-256 hex digest")


@dataclass(frozen=True)
class HistoricalOptionQuote:
    as_of_utc: str
    event_ts: str
    available_ts: str
    venue: str
    underlying_symbol: str
    native_symbol: str
    instrument_id: str
    expiry: str
    dte: float
    option_type: str
    strike_native: float
    strike_btc_equivalent: float | None
    underlying_price: float
    underlying_btc_equivalent: float | None
    btc_per_share: float | None
    bid: float
    ask: float
    mid: float
    bid_iv: float | None
    ask_iv: float | None
    mid_iv: float
    iv_source: str
    source_id: str
    source_sha256: str
    source_confidence: str
    size_bid: float | None = None
    size_ask: float | None = None
    quality_flags: list[str] = field(default_factory=lambda: ["ok"])

    def with_updates(self, **updates: Any) -> "HistoricalOptionQuote":
        return replace(self, **updates)

    def to_dict(self) -> dict[str, Any]:
        _validate_source_confidence(self.source_confidence)
        _validate_hash(self.source_sha256)
        _validate_event_available(self.event_ts, self.available_ts)
        if self.ask < self.bid:
            raise HistoricalSchemaError("ask must be >= bid")
        if not (self.bid <= self.mid <= self.ask):
            raise HistoricalSchemaError("mid must be inside bid/ask")
        if self.dte < 0:
            raise HistoricalSchemaError("dte must be non-negative")
        if self.option_type not in {"call", "put"}:
            raise HistoricalSchemaError("option_type must be call or put")
        return {
            "record_type": "option_quote",
            **asdict(self),
            "execution_confidence": "screen_only_not_executable",
            "evidence_status": EVIDENCE_STATUS,
        }


@dataclass(frozen=True)
class HistoricalUnderlyingMark:
    as_of_utc: str
    event_ts: str
    available_ts: str
    symbol: str
    price: float
    source_id: str
    source_sha256: str
    source_confidence: str
    quality_flags: list[str] = field(default_factory=lambda: ["ok"])

    def to_dict(self) -> dict[str, Any]:
        _validate_source_confidence(self.source_confidence)
        _validate_hash(self.source_sha256)
        _validate_event_available(self.event_ts, self.available_ts)
        if self.price <= 0:
            raise HistoricalSchemaError("underlying price must be positive")
        return {
            "record_type": "underlying_mark",
            **asdict(self),
            "execution_confidence": "screen_only_not_executable",
            "evidence_status": EVIDENCE_STATUS,
        }


@dataclass(frozen=True)
class HistoricalEtfHolding:
    as_of_utc: str
    event_ts: str
    available_ts: str
    etf_symbol: str
    btc_per_share: float
    shares_per_btc: float
    source_id: str
    source_sha256: str
    source_confidence: str
    quality_flags: list[str] = field(default_factory=lambda: ["ok"])

    def to_dict(self) -> dict[str, Any]:
        _validate_source_confidence(self.source_confidence)
        _validate_hash(self.source_sha256)
        _validate_event_available(self.event_ts, self.available_ts)
        if self.btc_per_share <= 0 or self.shares_per_btc <= 0:
            raise HistoricalSchemaError("ETF holding conversion values must be positive")
        return {
            "record_type": "etf_holding",
            **asdict(self),
            "execution_confidence": "screen_only_not_executable",
            "evidence_status": EVIDENCE_STATUS,
        }


@dataclass(frozen=True)
class HistoricalSpreadSnapshot:
    replay_id: str
    decision_ts: str
    available_ts: str
    tenor: str
    venue_pair: str
    left_mid_iv: float
    right_mid_iv: float
    spread_vol_pts: float
    left_instrument: str
    right_instrument: str
    source_ids: list[str]
    source_hashes: list[str]
    normalization: dict[str, Any] = field(default_factory=lambda: {"atm_rule": "nearest_strike", "tenor_bucket_rule": "nearest_dte_with_max_gap"})
    quality_flags: list[str] = field(default_factory=lambda: ["ok"])

    def with_updates(self, **updates: Any) -> "HistoricalSpreadSnapshot":
        return replace(self, **updates)

    def to_dict(self) -> dict[str, Any]:
        if _parse_ts(self.available_ts, "available_ts") > _parse_ts(self.decision_ts, "decision_ts"):
            raise HistoricalSchemaError("available_ts cannot be after decision_ts")
        for digest in self.source_hashes:
            _validate_hash(digest, "source_hashes")
        if not self.source_ids or len(self.source_ids) != len(self.source_hashes):
            raise HistoricalSchemaError("source_ids and source_hashes must be non-empty and aligned")
        return {
            "record_type": "spread_snapshot",
            **asdict(self),
            "execution_confidence": "screen_only_not_executable",
            "evidence_status": EVIDENCE_STATUS,
        }


def write_jsonl_records(records: Iterable[JsonRecord | dict[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for record in records:
        if isinstance(record, dict):
            row = record
        else:
            row = record.to_dict()
        lines.append(json.dumps(row, sort_keys=True))
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
