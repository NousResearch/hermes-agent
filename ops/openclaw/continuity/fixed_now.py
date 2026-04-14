from __future__ import annotations

import datetime as dt
import os
from typing import Mapping

FIXED_NOW_ENV_KEY = "OPENCLAW_AUTOPILOT_FIXED_NOW_TS"


def parse_fixed_now_ts(raw: object) -> int | None:
    txt = str(raw or "").strip()
    if not txt:
        return None
    try:
        parsed = int(txt)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def fixed_now_ts(env: Mapping[str, str] | None = None, *, env_key: str = FIXED_NOW_ENV_KEY) -> int | None:
    source = os.environ if env is None else env
    return parse_fixed_now_ts(source.get(env_key))


def now_ts(env: Mapping[str, str] | None = None, *, fallback_ts: int | None = None) -> int:
    fixed = fixed_now_ts(env)
    if fixed is not None:
        return fixed
    if fallback_ts is not None:
        return int(fallback_ts)
    return int(dt.datetime.now(dt.timezone.utc).timestamp())


def ts_to_iso_utc(ts: int | float) -> str:
    return (
        dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def now_iso_utc(env: Mapping[str, str] | None = None, *, fallback_ts: int | None = None) -> str:
    return ts_to_iso_utc(now_ts(env, fallback_ts=fallback_ts))
