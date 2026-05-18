from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from hermes_constants import get_hermes_home

GOVERNOR_SCHEMA_VERSION = 2


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_governor_db_path() -> Path:
    return get_hermes_home() / "state" / "governor.db"


def _to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_reset(value: Any) -> Optional[datetime]:
    if value in {None, ""}:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(text)
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                return None
    # x-ratelimit-reset values are normally epoch seconds. If a provider ever
    # sends a small relative delta, interpreting it as epoch would point at 1970;
    # treat small values as seconds from now instead.
    if number < 10_000_000:
        return datetime.fromtimestamp(_utc_now().timestamp() + number, tz=timezone.utc)
    return datetime.fromtimestamp(number, tz=timezone.utc)


def compute_governor_band(daily_used_pct: Optional[float], weekly_used_pct: Optional[float]) -> str:
    pressure = max(float(daily_used_pct or 0.0), float(weekly_used_pct or 0.0))
    if pressure >= 98.0:
        return "post-reserve"
    if pressure >= 95.0:
        return "black"
    if pressure >= 85.0:
        return "red"
    if pressure >= 70.0:
        return "amber"
    return "green"


def ensure_governor_schema(db_path: Optional[Path | str] = None) -> Path:
    path = Path(db_path) if db_path is not None else get_governor_db_path()
    if not path.exists():
        raise FileNotFoundError(
            f"governor.db does not exist at {path}; G2 must migrate the canonical G1 database, not recreate it"
        )
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS xai_buckets (
              bucket_key TEXT PRIMARY KEY,
              bucket_scope TEXT NOT NULL,
              window_label TEXT NOT NULL,
              limit_value REAL NOT NULL,
              remaining_value REAL NOT NULL,
              used_value REAL NOT NULL,
              used_pct REAL NOT NULL,
              reset_at TEXT,
              observed_at TEXT NOT NULL,
              raw_headers_json TEXT NOT NULL
            )
            """
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_xai_buckets_observed_at ON xai_buckets(observed_at)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_xai_buckets_scope ON xai_buckets(bucket_scope, window_label)"
        )
        db.execute(f"PRAGMA user_version = {GOVERNOR_SCHEMA_VERSION}")
    return path


def _normalize_headers(headers: Mapping[str, Any]) -> dict[str, str]:
    return {str(k).lower(): str(v).strip() for k, v in headers.items() if v not in {None, ""}}


def _float_header(headers: Mapping[str, str], key: str) -> Optional[float]:
    value = headers.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _window_label_from_reset(reset_at: Optional[datetime]) -> str:
    if reset_at is None:
        return "observed"
    seconds = max(0, int((reset_at - _utc_now()).total_seconds()))
    if seconds <= 90:
        return "minute"
    if seconds <= 3900:
        return "hour"
    if seconds <= 93_600:
        return "day"
    if seconds <= 691_200:
        return "week"
    return "observed"


def _iter_xai_bucket_observations(headers: Mapping[str, str]) -> list[dict[str, Any]]:
    scopes = set()
    for key in headers:
        match = re.match(r"x-ratelimit-limit-(.+)", key)
        if match:
            scopes.add(match.group(1))
    observations: list[dict[str, Any]] = []
    for scope in sorted(scopes):
        limit_value = _float_header(headers, f"x-ratelimit-limit-{scope}")
        remaining_value = _float_header(headers, f"x-ratelimit-remaining-{scope}")
        if limit_value is None or remaining_value is None or limit_value <= 0:
            continue
        remaining_value = max(0.0, min(remaining_value, limit_value))
        used_value = limit_value - remaining_value
        used_pct = max(0.0, min(100.0, (used_value / limit_value) * 100.0))
        reset_at = _parse_reset(headers.get(f"x-ratelimit-reset-{scope}"))
        explicit_window = headers.get(f"x-ratelimit-window-{scope}")
        window_label = explicit_window or _window_label_from_reset(reset_at)
        observations.append(
            {
                "bucket_scope": scope,
                "window_label": window_label,
                "limit_value": limit_value,
                "remaining_value": remaining_value,
                "used_value": used_value,
                "used_pct": used_pct,
                "reset_at": reset_at,
            }
        )
    return observations


def record_xai_rate_limit_headers(headers: Mapping[str, Any], *, db_path: Optional[Path | str] = None) -> bool:
    normalized = _normalize_headers(headers)
    observations = _iter_xai_bucket_observations(normalized)
    if not observations:
        return False
    path = ensure_governor_schema(db_path)
    observed_at = _to_iso(_utc_now()) or ""
    raw = json.dumps(normalized, sort_keys=True)
    highest_used = max(obs["used_pct"] for obs in observations)
    band = compute_governor_band(highest_used, highest_used)
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA foreign_keys = ON")
        for obs in observations:
            bucket_key = f"{obs['bucket_scope']}:{obs['window_label']}"
            db.execute(
                """
                INSERT INTO xai_buckets(
                  bucket_key, bucket_scope, window_label, limit_value, remaining_value,
                  used_value, used_pct, reset_at, observed_at, raw_headers_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(bucket_key) DO UPDATE SET
                  bucket_scope=excluded.bucket_scope,
                  window_label=excluded.window_label,
                  limit_value=excluded.limit_value,
                  remaining_value=excluded.remaining_value,
                  used_value=excluded.used_value,
                  used_pct=excluded.used_pct,
                  reset_at=excluded.reset_at,
                  observed_at=excluded.observed_at,
                  raw_headers_json=excluded.raw_headers_json
                """,
                (
                    bucket_key,
                    obs["bucket_scope"],
                    obs["window_label"],
                    obs["limit_value"],
                    obs["remaining_value"],
                    obs["used_value"],
                    obs["used_pct"],
                    _to_iso(obs["reset_at"]),
                    observed_at,
                    raw,
                ),
            )
        db.execute(
            """
            UPDATE provider_state
               SET band=?, daily_used_pct=?, weekly_used_pct=?, last_polled_at=?, source='observed'
             WHERE provider='xai'
            """,
            (band, highest_used, highest_used, observed_at),
        )
    return True


def get_xai_bucket_rows(*, db_path: Optional[Path | str] = None) -> list[dict[str, Any]]:
    path = ensure_governor_schema(db_path)
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute("SELECT * FROM xai_buckets ORDER BY used_pct DESC, bucket_key")]


def get_provider_state(provider: str, *, db_path: Optional[Path | str] = None) -> Optional[dict[str, Any]]:
    path = ensure_governor_schema(db_path)
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM provider_state WHERE provider=?", (provider,)).fetchone()
        return dict(row) if row else None


def get_transition_rate(transition_key: str, *, db_path: Optional[Path | str] = None) -> Optional[int]:
    path = ensure_governor_schema(db_path)
    key = str(transition_key or "").strip()
    if not key:
        return None
    with sqlite3.connect(path) as db:
        row = db.execute(
            "SELECT current_rate_per_hour FROM transition_rates WHERE transition_key=?",
            (key,),
        ).fetchone()
        if row is not None:
            return int(row[0])
        if key.startswith("PlatformOps.PlatformChange->"):
            row = db.execute(
                "SELECT current_rate_per_hour FROM transition_rates WHERE transition_key=?",
                ("PlatformOps.PlatformChange->*",),
            ).fetchone()
            if row is not None:
                return int(row[0])
    return None
