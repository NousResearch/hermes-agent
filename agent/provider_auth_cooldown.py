"""Provider/model auth-error cooldown helpers.

Metadata-only state: no prompts, responses, headers, cookies, tokens, or API keys.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

AUTH_COOLDOWN_HOURS = 6
AUTH_COOLDOWN_SECONDS = AUTH_COOLDOWN_HOURS * 60 * 60
_AUTH_ERROR_MARKERS = (
    "token_expired",
    "invalid_api_key",
    "authentication_error",
    "unauthorized",
    "401",
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def hermes_home() -> Path:
    raw = os.environ.get("HERMES_HOME")
    if raw:
        return Path(raw).expanduser()
    runtime_home = Path(__file__).resolve().parents[3] / "home"
    if runtime_home.exists():
        return runtime_home
    return Path.home() / ".hermes"


def cooldown_state_path() -> Path:
    return hermes_home() / "state" / "provider_auth_cooldowns.json"


def _key(provider: str, model: str) -> str:
    return f"{provider or 'unknown'}:{model or 'unknown'}"


def load_cooldowns(path: Path | None = None) -> Dict[str, Any]:
    target = path or cooldown_state_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_cooldowns(data: Dict[str, Any], path: Path | None = None) -> None:
    target = path or cooldown_state_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def classify_auth_error(error: BaseException | str | None, status_code: int | None = None) -> Optional[str]:
    parts = [str(status_code or "")]
    if error is not None:
        for attr in ("body", "message", "response", "code"):
            try:
                val = getattr(error, attr, None)
                if val:
                    parts.append(str(val))
            except Exception:
                pass
        try:
            parts.append(str(error))
        except Exception:
            pass
    text = " ".join(parts).lower()
    if status_code == 401 or "token_expired" in text:
        if "token_expired" in text:
            return "token_expired"
        return "auth_401"
    for marker in _AUTH_ERROR_MARKERS:
        if marker in text:
            return marker
    return None


def set_auth_cooldown(
    provider: str,
    model: str,
    error_class: str,
    *,
    now: datetime | None = None,
    path: Path | None = None,
) -> Dict[str, Any]:
    current = now or _utcnow()
    blocked_until = current + timedelta(seconds=AUTH_COOLDOWN_SECONDS)
    data = load_cooldowns(path)
    key = _key(provider, model)
    previous = data.get(key, {}) if isinstance(data.get(key), dict) else {}
    data[key] = {
        "provider": provider or "unknown",
        "model": model or "unknown",
        "error_class": error_class,
        "last_seen": current.isoformat(),
        "blocked_until": blocked_until.isoformat(),
        "count": int(previous.get("count", 0)) + 1,
    }
    save_cooldowns(data, path)
    return data[key]


def get_active_auth_cooldown(
    provider: str,
    model: str,
    *,
    now: datetime | None = None,
    path: Path | None = None,
) -> Optional[Dict[str, Any]]:
    current = now or _utcnow()
    data = load_cooldowns(path)
    record = data.get(_key(provider, model))
    if not isinstance(record, dict):
        return None
    until = _parse_ts(record.get("blocked_until"))
    if until and until > current:
        return record
    return None
