"""Per-tenant secret storage + one-time secret-entry links.

Secrets are entered out-of-band (a one-time HTTPS form link), never typed into
chat. They live in the tenant's OWN confined home (0600) — the sandbox already
stops siblings/other tenants from reading them.

Storage is behind a small interface so the basic LocalStore can later be swapped
for a VaultStore (when we move to S3 / Vault) without touching callers.

Values are stored plaintext-at-rest for the basic config (protected by the 0600
+ confinement boundary). `_seal`/`_unseal` are the single hook to add real
at-rest encryption (Fernet with ORCHARD_SECRET_KEY, or KMS/Vault) later.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .config import Settings
from .security import write_secret


class LocalStore:
    """One secrets.json per tenant home. Keys are env-var names; values are the
    raw secret. Never logs or returns values except via get()/all() (used only
    to inject into the worker, never to display)."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def _file(self, tenant_id: str) -> Path:
        return self.settings.paths.home_for(tenant_id) / "secrets.json"

    def _load(self, tenant_id: str) -> dict[str, str]:
        f = self._file(tenant_id)
        if not f.exists():
            return {}
        try:
            return {k: self._unseal(v) for k, v in json.loads(f.read_text()).items()}
        except Exception:
            return {}

    def _save(self, tenant_id: str, data: dict[str, str]) -> None:
        sealed = {k: self._seal(v) for k, v in data.items()}
        write_secret(
            self._file(tenant_id),
            json.dumps(sealed, indent=2),
            self.settings.security.secret_mode_int,
        )

    # --- at-rest sealing hook (identity for basic; Fernet/KMS later) ---------
    def _seal(self, value: str) -> str:
        return value

    def _unseal(self, value: str) -> str:
        return value

    # --- API -----------------------------------------------------------------
    def set(self, tenant_id: str, name: str, value: str) -> None:
        data = self._load(tenant_id)
        data[name] = value
        self._save(tenant_id, data)

    def get(self, tenant_id: str, name: str) -> str | None:
        return self._load(tenant_id).get(name)

    def all(self, tenant_id: str) -> dict[str, str]:
        return self._load(tenant_id)

    def names(self, tenant_id: str) -> list[str]:
        return sorted(self._load(tenant_id).keys())

    def delete(self, tenant_id: str, name: str) -> bool:
        data = self._load(tenant_id)
        if name in data:
            del data[name]
            self._save(tenant_id, data)
            return True
        return False


def make_store(settings: Settings) -> LocalStore:
    if settings.secrets.store == "local":
        return LocalStore(settings)
    raise ValueError(f"unknown secret store {settings.secrets.store!r} (only 'local' for now)")


class LinkStore:
    """One-time, short-lived links for secret entry. A link authorizes setting
    exactly ONE named secret for ONE tenant, once, before it expires."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS links (
                token       TEXT PRIMARY KEY,
                tenant_id   TEXT NOT NULL,
                target      TEXT NOT NULL,   -- "integration:<id>" or "secret:<ENV>"
                label       TEXT,
                created_at  REAL NOT NULL,
                ttl         REAL NOT NULL,
                used        INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._db.commit()

    def mint(self, tenant_id: str, target: str, label: str, ttl: float, now: float) -> str:
        import secrets as _secrets
        token = _secrets.token_urlsafe(32)
        self._db.execute(
            "INSERT INTO links VALUES (?,?,?,?,?,?,0)",
            (token, tenant_id, target, label, now, ttl),
        )
        self._db.commit()
        return token

    def _valid_row(self, token: str, now: float) -> sqlite3.Row | None:
        row = self._db.execute("SELECT * FROM links WHERE token=?", (token,)).fetchone()
        if not row or row["used"] or (now - row["created_at"]) > row["ttl"]:
            return None
        return row

    def peek(self, token: str, now: float) -> sqlite3.Row | None:
        """Return the link if still valid, without consuming it (for the GET form)."""
        return self._valid_row(token, now)

    def consume(self, token: str, now: float) -> sqlite3.Row | None:
        """Atomically mark a valid link used; return it, or None if invalid."""
        row = self._valid_row(token, now)
        if not row:
            return None
        self._db.execute("UPDATE links SET used=1 WHERE token=?", (token,))
        self._db.commit()
        return row

    def purge_expired(self, now: float) -> None:
        self._db.execute("DELETE FROM links WHERE used=1 OR (? - created_at) > ttl", (now,))
        self._db.commit()

    def close(self) -> None:
        self._db.close()
