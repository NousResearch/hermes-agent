"""Loopback browser-extension pairing: one-click scoped bearer tokens.

Flow (matching the Hermes Browser extension's pairing client):
  1. Extension POSTs /api/browser-extension/pair/start (loopback only).
  2. The user approves on a gateway-served page (approve -> grant).
  3. The extension polls /api/browser-extension/pair/status/<id> and
     receives a scoped bearer token once approved.

Tokens are persisted under HERMES_HOME/state/browser_pairing.json so they
survive gateway restarts. Pairings are short-lived; tokens are long-lived
until the browser clears them (or their max age passes).
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from pathlib import Path

PAIRING_TTL_SECONDS = 180
TOKEN_MAX_AGE_SECONDS = 365 * 24 * 3600
PAIRING_ID_BYTES = 16
TOKEN_BYTES = 32


class BrowserPairingStore:
    """Persistent store for pending pairings and issued scoped tokens."""

    def __init__(self, state_path: str | os.PathLike | None = None, now=None):
        self._state_path = Path(state_path or self._default_state_path())
        self._now = now or time.time
        self._lock = threading.Lock()
        self._pairings: dict[str, dict] = {}
        self._tokens: dict[str, dict] = {}
        self._load()

    @staticmethod
    def _default_state_path() -> Path:
        home = os.environ.get("HERMES_HOME") or ""
        base = Path(home) if home else Path.home() / ".hermes"
        return base / "state" / "browser_pairing.json"

    def _load(self) -> None:
        try:
            with open(self._state_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self._pairings = {
                key: value
                for key, value in data.get("pairings", {}).items()
                if value.get("status") in ("pending", "approved")
            }
            self._tokens = data.get("tokens", {})
        except (OSError, ValueError):
            self._pairings = {}
            self._tokens = {}
        self._prune()

    def _save(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            payload = {"pairings": self._pairings, "tokens": self._tokens}
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(tmp, self._state_path)
        except OSError:
            # State persistence is best-effort: pairing still works in memory.
            pass

    def _prune(self) -> None:
        now = self._now()
        self._pairings = {
            key: value
            for key, value in self._pairings.items()
            if value.get("status") == "approved"
            or float(value.get("expires_at", 0)) > now
        }
        self._tokens = {
            key: value
            for key, value in self._tokens.items()
            if not value.get("revoked")
            and float(value.get("created_at", 0)) > now - TOKEN_MAX_AGE_SECONDS
        }

    def create_pairing(self, name: str = "", extension_id: str = "") -> dict:
        pairing_id = secrets.token_urlsafe(PAIRING_ID_BYTES)
        now = self._now()
        record = {
            "name": str(name or "Hermes Browser Extension")[:80],
            "extension_id": str(extension_id or "")[:200],
            "created_at": now,
            "expires_at": now + PAIRING_TTL_SECONDS,
            "status": "pending",
            "token": None,
        }
        with self._lock:
            self._prune()
            self._pairings[pairing_id] = record
            self._save()
        return {"pairing_id": pairing_id, **record}

    def get_pairing(self, pairing_id: str) -> dict | None:
        with self._lock:
            record = self._pairings.get(pairing_id)
            if not record:
                return None
            if record["status"] == "pending" and float(record["expires_at"]) <= self._now():
                record["status"] = "expired"
                self._save()
            return dict(record)

    def grant_pairing(self, pairing_id: str) -> dict | None:
        with self._lock:
            record = self._pairings.get(pairing_id)
            if not record or record["status"] != "pending":
                return None
            now = self._now()
            if float(record["expires_at"]) <= now:
                record["status"] = "expired"
                self._save()
                return None
            token = secrets.token_hex(TOKEN_BYTES)
            record["status"] = "approved"
            record["token"] = token
            self._tokens[token] = {
                "extension_id": record["extension_id"],
                "created_at": now,
                "revoked": False,
            }
            self._save()
            return dict(record)

    def deny_pairing(self, pairing_id: str) -> dict | None:
        with self._lock:
            record = self._pairings.get(pairing_id)
            if not record:
                return None
            record["status"] = "denied"
            self._save()
            return dict(record)

    def is_valid_token(self, token: str) -> bool:
        if not token or len(token) < 32:
            return False
        with self._lock:
            record = self._tokens.get(token)
            if not record or record.get("revoked"):
                return False
            if float(record.get("created_at", 0)) <= self._now() - TOKEN_MAX_AGE_SECONDS:
                return False
            return True

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "pairings": dict(self._pairings),
                "tokens": dict(self._tokens),
            }
