from __future__ import annotations

import base64
import csv
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

DATABENTO_HIST_BASE = "https://hist.databento.com/v0"
DATABENTO_DATASET = "GLBX.MDP3"
BTC_OPTIONS_PARENT = "BTC.OPT"
BTC_FUTURES_PARENT = "BTC.FUT"
SENTINEL_INT = 9223372036854775807


DEFAULT_DATABENTO_ENV_FILES = (
    Path(".env"),
    Path.home() / ".hermes" / ".env",
    Path.home() / "oildesk" / "backend" / ".env",
)


def load_databento_api_key(
    env_file: Path | None = None,
    *,
    default_env_files: list[Path] | tuple[Path, ...] | None = None,
) -> str | None:
    """Load Databento API key without logging or persisting it.

    Primary source is this process environment. Then an explicit env_file, then
    configured default env files are checked. The value is never printed or
    written by this module.
    """
    for name in ("DATABENTO_API_KEY", "DATABENTO_KEY"):
        value = os.environ.get(name)
        if value:
            return value.strip().strip('"').strip("'")

    candidates: list[Path] = []
    env_file_from_env = os.environ.get("DATABENTO_ENV_FILE")
    if env_file_from_env:
        candidates.append(Path(env_file_from_env).expanduser())
    if env_file is not None:
        candidates.append(env_file.expanduser())
    candidates.extend(default_env_files if default_env_files is not None else DEFAULT_DATABENTO_ENV_FILES)

    for candidate in candidates:
        key = _load_databento_api_key_from_env_file(candidate.expanduser())
        if key:
            return key
    return None


def _load_databento_api_key_from_env_file(candidate: Path) -> str | None:
    if not candidate.exists():
        return None
    for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() in {"DATABENTO_API_KEY", "DATABENTO_KEY"}:
            return value.strip().strip('"').strip("'") or None
    return None


def _auth_header(api_key: str) -> str:
    token = base64.b64encode(f"{api_key}:".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def databento_get_text(endpoint: str, params: dict[str, Any], *, api_key: str, timeout: int = 60) -> str:
    query = urllib.parse.urlencode({key: value for key, value in params.items() if value is not None})
    url = f"{DATABENTO_HIST_BASE}/{endpoint}?{query}"
    req = urllib.request.Request(url, headers={"Authorization": _auth_header(api_key), "Accept": "text/csv,*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def get_dataset_end(api_key: str, dataset: str = DATABENTO_DATASET) -> datetime | None:
    text = databento_get_text("metadata.get_dataset_range", {"dataset": dataset}, api_key=api_key, timeout=30)
    payload = json.loads(text)
    end = payload.get("end")
    if not end:
        return None
    return _parse_iso_datetime(str(end))


def _parse_iso_datetime(value: str) -> datetime:
    """Parse ISO timestamps from Databento, truncating nanoseconds to Python microseconds."""
    text = str(value).replace("Z", "+00:00")
    tz_pos = max(text.rfind("+"), text.rfind("-", 10))
    main = text[:tz_pos] if tz_pos > 10 else text
    tz = text[tz_pos:] if tz_pos > 10 else ""
    if "." in main:
        head, frac = main.split(".", 1)
        main = f"{head}.{frac[:6].ljust(6, '0')}"
    return datetime.fromisoformat(main + tz)


def _parse_ns_datetime(value: Any) -> datetime | None:
    try:
        raw = int(str(value))
    except (TypeError, ValueError):
        return None
    if raw <= 0 or raw >= SENTINEL_INT:
        return None
    return datetime.fromtimestamp(raw / 1_000_000_000, tz=timezone.utc)


def _scaled_price(value: Any, display_factor: Any = 1_000_000_000) -> float | None:
    try:
        raw = int(str(value))
        factor = float(display_factor or 1_000_000_000)
    except (TypeError, ValueError):
        return None
    if raw <= -SENTINEL_INT or raw >= SENTINEL_INT or factor <= 0:
        return None
    return raw / factor


def _read_csv_rows(csv_text: str) -> list[dict[str, str]]:
    if not csv_text.strip():
        return []
    return list(csv.DictReader(StringIO(csv_text)))


def parse_cme_definition_rows(csv_text: str) -> dict[str, dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {}
    for row in _read_csv_rows(csv_text):
        instrument_id = str(row.get("instrument_id") or "").strip()
        raw_symbol = str(row.get("raw_symbol") or "").strip()
        if not instrument_id or not raw_symbol:
            continue
        display_factor = row.get("display_factor") or 1_000_000_000
        expiration_dt = _parse_ns_datetime(row.get("expiration"))
        option_type = {"C": "call", "P": "put"}.get(str(row.get("instrument_class") or "").upper())
        definitions[instrument_id] = {
            "instrument_id": instrument_id,
            "raw_symbol": raw_symbol,
            "option_type": option_type,
            "strike_native": _scaled_price(row.get("strike_price"), display_factor),
            "expiry": expiration_dt.date().isoformat() if expiration_dt else None,
            "expiration_ts": expiration_dt.isoformat() if expiration_dt else None,
            "underlying": str(row.get("underlying") or "").strip() or None,
            "display_factor": float(display_factor),
            "security_type": row.get("security_type"),
            "source_confidence": "licensed_vendor_api_databento",
            "execution_confidence": "screen_only_not_executable",
        }
    return definitions


def parse_cme_bbo_rows(csv_text: str, definitions: dict[str, dict[str, Any]], *, as_of: datetime) -> list[dict[str, Any]]:
    latest_by_instrument: dict[str, dict[str, str]] = {}
    for row in _read_csv_rows(csv_text):
        instrument_id = str(row.get("instrument_id") or "").strip()
        if instrument_id in definitions:
            latest_by_instrument[instrument_id] = row

    normalized: list[dict[str, Any]] = []
    for instrument_id, bbo in latest_by_instrument.items():
        definition = definitions[instrument_id]
        bid = _scaled_price(bbo.get("bid_px_00"), definition.get("display_factor"))
        ask = _scaled_price(bbo.get("ask_px_00"), definition.get("display_factor"))
        if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
            continue
        expiry = definition.get("expiry")
        dte = None
        if expiry:
            dte = max((datetime.fromisoformat(expiry).date() - as_of.date()).days, 0)
        normalized.append(
            {
                "native_symbol": definition["raw_symbol"],
                "instrument_id": instrument_id,
                "expiry": expiry,
                "dte": dte,
                "option_type": definition.get("option_type"),
                "strike_native": definition.get("strike_native"),
                "underlying": definition.get("underlying"),
                "price_bid": bid,
                "price_ask": ask,
                "price_mid": (bid + ask) / 2,
                "bid_size": _safe_int(bbo.get("bid_sz_00")),
                "ask_size": _safe_int(bbo.get("ask_sz_00")),
                "source_confidence": "licensed_vendor_api_databento",
                "execution_confidence": "screen_only_not_executable",
                "timestamp_source": as_of.isoformat(),
                "quality_flags": ["ok"],
            }
        )
    return sorted(normalized, key=lambda row: (row.get("dte") is None, row.get("dte") or 99999, abs(float(row.get("strike_native") or 0))))


def _safe_int(value: Any) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    if parsed >= SENTINEL_INT:
        return None
    return parsed


def fetch_cme_btc_option_snapshot(
    *,
    api_key: str,
    raw_dir: Path,
    as_of: datetime,
    lookback_days: int = 3,
    limit: int = 5000,
) -> dict[str, Any]:
    """Fetch a Databento CME BTC option BBO snapshot.

    Returns screen-only vendor marks. These are not executable quotes and should
    only be used as internal evidence until quote/trade records are attached.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_end = get_dataset_end(api_key)
    end = min(as_of.astimezone(timezone.utc), dataset_end) if dataset_end else as_of.astimezone(timezone.utc)
    start = end - timedelta(days=lookback_days)
    definition_start = (end - timedelta(days=lookback_days)).replace(hour=0, minute=0, second=0, microsecond=0)
    definition_end = end

    definition_csv = databento_get_text(
        "timeseries.get_range",
        {
            "dataset": DATABENTO_DATASET,
            "symbols": BTC_OPTIONS_PARENT,
            "schema": "definition",
            "stype_in": "parent",
            "start": definition_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": definition_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "encoding": "csv",
        },
        api_key=api_key,
        timeout=90,
    )
    bbo_csv = databento_get_text(
        "timeseries.get_range",
        {
            "dataset": DATABENTO_DATASET,
            "symbols": BTC_OPTIONS_PARENT,
            "schema": "bbo-1m",
            "stype_in": "parent",
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "encoding": "csv",
            "limit": str(limit),
        },
        api_key=api_key,
        timeout=90,
    )

    (raw_dir / "databento_cme_btc_options_definition.csv").write_text(definition_csv, encoding="utf-8")
    (raw_dir / "databento_cme_btc_options_bbo_1m.csv").write_text(bbo_csv, encoding="utf-8")

    definitions = parse_cme_definition_rows(definition_csv)
    rows = parse_cme_bbo_rows(bbo_csv, definitions, as_of=end)
    return {
        "dataset": DATABENTO_DATASET,
        "symbol": BTC_OPTIONS_PARENT,
        "schema": "bbo-1m",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "definition_rows": len(definitions),
        "bbo_rows": max(0, len(bbo_csv.splitlines()) - 1),
        "normalized_rows": rows,
        "source_confidence": "licensed_vendor_api_databento",
        "execution_confidence": "screen_only_not_executable",
    }
