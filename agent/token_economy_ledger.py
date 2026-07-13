"""Metadata-only token economy ledger and context guards.

Never stores prompt text, response text, headers, cookies, tokens, or API keys.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

SOFT_CONTEXT_WARN_TOKENS = 80_000
HARD_CONTEXT_GUARD_TOKENS = 120_000
DUPLICATE_HIGH_CONTEXT_TTL_SECONDS = 15 * 60
LEDGER_FIELDS = (
    "ts",
    "session_id_hash",
    "provider",
    "model",
    "lane",
    "approx_input_tokens",
    "max_tokens",
    "status",
    "error_class",
    "api_calls",
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def hermes_home() -> Path:
    raw = os.environ.get("HERMES_HOME")
    if raw:
        return Path(raw).expanduser()
    runtime_home = Path(__file__).resolve().parents[3] / "home"
    if runtime_home.exists():
        return runtime_home
    return Path.home() / ".hermes"


def ledger_path() -> Path:
    return hermes_home() / "logs" / "token_economy_ledger.jsonl"


def duplicate_guard_path() -> Path:
    return hermes_home() / "state" / "token_economy_duplicate_guard.json"


def hash_value(value: str | None) -> str:
    return hashlib.sha256((value or "").encode("utf-8", "ignore")).hexdigest()[:16]


def evaluate_context_guard(approx_tokens: int) -> str:
    if approx_tokens >= HARD_CONTEXT_GUARD_TOKENS:
        return "hard"
    if approx_tokens >= SOFT_CONTEXT_WARN_TOKENS:
        return "warning"
    return "ok"


def append_ledger_event(event: Dict[str, Any], path: Path | None = None) -> Dict[str, Any]:
    target = path or ledger_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    clean: Dict[str, Any] = {}
    for field in LEDGER_FIELDS:
        value = event.get(field)
        if field == "ts" and not value:
            value = _utcnow().isoformat()
        if field == "session_id_hash" and not value:
            value = hash_value(str(event.get("session_id", "")))
        clean[field] = value
    target.open("a", encoding="utf-8").write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n")
    return clean


def _load_duplicate_state(path: Path | None = None) -> Dict[str, Any]:
    target = path or duplicate_guard_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_duplicate_state(data: Dict[str, Any], path: Path | None = None) -> None:
    target = path or duplicate_guard_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def high_context_fingerprint(provider: str, model: str, session_id: str, user_message: str, approx_tokens: int) -> str:
    bucket = int(max(0, approx_tokens) / 10_000) * 10_000
    raw = "|".join([provider or "", model or "", session_id or "", str(bucket), user_message or ""])
    return hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest()


def check_and_record_duplicate_high_context(
    provider: str,
    model: str,
    session_id: str,
    user_message: str,
    approx_tokens: int,
    *,
    now: datetime | None = None,
    path: Path | None = None,
) -> Optional[Dict[str, Any]]:
    if approx_tokens < SOFT_CONTEXT_WARN_TOKENS:
        return None
    current = now or _utcnow()
    state = _load_duplicate_state(path)
    fp = high_context_fingerprint(provider, model, session_id, user_message, approx_tokens)
    previous = state.get(fp)
    if isinstance(previous, dict):
        try:
            last_seen = datetime.fromisoformat(str(previous.get("last_seen")).replace("Z", "+00:00"))
            if last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=timezone.utc)
            age = (current - last_seen.astimezone(timezone.utc)).total_seconds()
            if age < DUPLICATE_HIGH_CONTEXT_TTL_SECONDS:
                previous["age_seconds"] = int(age)
                return previous
        except Exception:
            pass
    state[fp] = {
        "last_seen": current.isoformat(),
        "provider": provider or "unknown",
        "model": model or "unknown",
        "approx_input_tokens_bucket": int(max(0, approx_tokens) / 10_000) * 10_000,
    }
    # Keep bounded; oldest order is not critical for this small guard.
    if len(state) > 200:
        state = dict(list(state.items())[-200:])
    _save_duplicate_state(state, path)
    return None
