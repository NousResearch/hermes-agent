"""Local suggestion history: dismissals, mutes, and suggested-at records.

Persisted as JSON under ``$HERMES_HOME/wisdom/agent_led_history.json``. The
Gateway is the authority for cross-device state once it exposes a
suggestion-history/mute endpoint; until then, calls are recorded locally and
flagged ``pending_gateway`` so a later sync can replay them.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
FILE_NAME = "agent_led_history.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def history_path() -> Path:
    try:
        from hermes_constants import get_hermes_home

        root = Path(get_hermes_home())
    except Exception:
        root = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    return root / "wisdom" / FILE_NAME


class SuggestionHistory:
    """Small JSON-backed ledger; every mutation is atomic on disk."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or history_path()
        self._data = self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        raw.setdefault("version", 1)
        raw.setdefault("dismissals", {})
        raw.setdefault("suggested", {})
        raw.setdefault("handled", {})
        raw.setdefault("mutes", {})
        raw.setdefault("pending_gateway", [])
        return raw

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def _queue_gateway(self, kind: str, payload: dict[str, Any]) -> None:
        self._data["pending_gateway"].append(
            {"kind": kind, "payload": payload, "queued_at": _iso(_now())}
        )

    # -- dismissals ------------------------------------------------------
    def record_dismissal(
        self,
        skill_name: str,
        content_hash: str,
        *,
        suppression_days: int = 30,
        at: datetime | None = None,
        client: Any = None,
    ) -> dict[str, Any]:
        when = at or _now()
        record = {
            "content_hash": content_hash,
            "dismissed_at": _iso(when),
            "suppressed_until": _iso(when + timedelta(days=suppression_days)),
        }
        with _LOCK:
            self._data["dismissals"][skill_name] = record
            synced = self._push_gateway(client, "record_suggestion_dismissal", {
                "skill_name": skill_name, **record,
            })
            if not synced:
                self._queue_gateway("dismissal", {"skill_name": skill_name, **record})
            self._save()
        return {**record, "gateway": "synced" if synced else "pending_gateway"}

    def is_suppressed(self, skill_name: str, content_hash: str, *, at: datetime | None = None) -> bool:
        """True while a dismissal for this exact content hash is still in force."""
        when = at or _now()
        record = self._data["dismissals"].get(skill_name)
        if not isinstance(record, dict):
            return False
        if record.get("content_hash") != content_hash:
            return False  # content changed materially; eligible again
        until = _parse(record.get("suppressed_until"))
        return until is not None and when < until

    # -- suggested / handled --------------------------------------------
    def record_suggested(self, skill_name: str, content_hash: str, *, at: datetime | None = None) -> None:
        with _LOCK:
            self._data["suggested"][skill_name] = {
                "content_hash": content_hash,
                "suggested_at": _iso(at or _now()),
            }
            self._save()

    def recently_suggested(
        self, skill_name: str, content_hash: str, *, cooldown_days: int, at: datetime | None = None
    ) -> bool:
        record = self._data["suggested"].get(skill_name)
        if not isinstance(record, dict) or record.get("content_hash") != content_hash:
            return False
        when = _parse(record.get("suggested_at"))
        return when is not None and (at or _now()) - when < timedelta(days=cooldown_days)

    def record_handled(self, skill_name: str, content_hash: str, outcome: str, *, at: datetime | None = None) -> None:
        """Outcome is one of accepted / declined / published."""
        with _LOCK:
            self._data["handled"][skill_name] = {
                "content_hash": content_hash,
                "outcome": outcome,
                "handled_at": _iso(at or _now()),
            }
            self._save()

    def previously_handled(self, skill_name: str, content_hash: str) -> bool:
        record = self._data["handled"].get(skill_name)
        return isinstance(record, dict) and record.get("content_hash") == content_hash

    # -- mutes -----------------------------------------------------------
    def record_mute(
        self,
        scope: str,
        *,
        days: int | None,
        at: datetime | None = None,
        client: Any = None,
    ) -> dict[str, Any]:
        """Mute proactive messages for ``scope`` (``'*'`` or a skill id)."""
        when = at or _now()
        record = {
            "muted_at": _iso(when),
            "muted_until": _iso(when + timedelta(days=days)) if days is not None else None,
        }
        with _LOCK:
            self._data["mutes"][scope] = record
            synced = self._push_gateway(client, "set_suggestion_mute", {"scope": scope, **record})
            if not synced:
                self._queue_gateway("mute", {"scope": scope, **record})
            self._save()
        return {**record, "gateway": "synced" if synced else "pending_gateway"}

    def is_muted(self, scope: str = "*", *, at: datetime | None = None) -> bool:
        when = at or _now()
        for key in ("*", scope):
            record = self._data["mutes"].get(key)
            if not isinstance(record, dict):
                continue
            until = record.get("muted_until")
            if until is None:
                return True
            parsed = _parse(until)
            if parsed is not None and when < parsed:
                return True
        return False

    # -- gateway ---------------------------------------------------------
    @staticmethod
    def _push_gateway(client: Any, method: str, payload: dict[str, Any]) -> bool:
        fn = getattr(client, method, None) if client is not None else None
        if not callable(fn):
            return False
        try:
            fn(**payload)
            return True
        except Exception as exc:
            logger.debug("Suggestion history: %s not synced (%s)", method, type(exc).__name__)
            return False

    def pending_gateway(self) -> list[dict[str, Any]]:
        return list(self._data["pending_gateway"])

    def snapshot(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._data))
