"""Secure, short-lived human control handoffs for Browser Use sessions.

The public handoff token is a bearer credential.  Only its SHA-256 digest is
stored; the Browser Use live URL is released solely while the row is pending
and unexpired.  Completion and wake delivery are separate, idempotent state
transitions so repeated clicks cannot start duplicate agent turns.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote, urlparse

import requests

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_TTL_MINUTES = 30
MAX_TTL_MINUTES = 30
_TOKEN_BYTES = 32
_TOKEN_HEX_LENGTH = _TOKEN_BYTES * 2
# Wake turns may legitimately use tools for several minutes. Reclaim only
# after the existing wake self-post's 10-minute ceiling plus safety margin.
_WAKE_RECLAIM_SECONDS = 900
WAKE_MESSAGE = (
    "The human browser handoff is complete. Resume this exact task now. "
    "First inspect the current page with browser_exec and verify the requested "
    "login, CAPTCHA, or other blocking step succeeded; then continue from "
    "where you paused."
)
EXPIRED_WAKE_MESSAGE = (
    "The browser handoff expired before the owner clicked Done. Inspect the "
    "current browser if it is still available, then explain the blocker or "
    "request a new handoff instead of assuming the human step succeeded."
)


class BrowserHandoffError(RuntimeError):
    """A safe, user-facing browser handoff failure."""


@dataclass(frozen=True)
class BrowserHandoffConfig:
    enabled: bool
    public_base_url: str
    ttl_minutes: int
    discord_user_id: str


@dataclass(frozen=True)
class BrowserHandoff:
    id: int
    token_digest: str
    status: str
    wake_status: str
    session_id: str
    browser_key: str
    browser_id: str
    live_url: str
    instruction: str
    source: dict[str, Any]
    created_at: float
    expires_at: float


def load_browser_handoff_config() -> BrowserHandoffConfig:
    from hermes_cli.config import cfg_get, read_raw_config

    raw = read_raw_config()
    section = cfg_get(raw, "browser", "handoff", default={})
    if not isinstance(section, dict):
        section = {}
    enabled = bool(section.get("enabled", False))
    base = str(section.get("public_base_url") or "").strip().rstrip("/")
    try:
        ttl = int(section.get("ttl_minutes", DEFAULT_TTL_MINUTES))
    except (TypeError, ValueError):
        ttl = DEFAULT_TTL_MINUTES
    ttl = max(1, min(ttl, MAX_TTL_MINUTES))
    user_id = str(section.get("discord_user_id") or "").strip()
    return BrowserHandoffConfig(enabled, base, ttl, user_id)


def _validate_public_base_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise BrowserHandoffError(
            "browser.handoff.public_base_url must be a public HTTPS URL"
        )
    if parsed.query or parsed.fragment:
        raise BrowserHandoffError(
            "browser.handoff.public_base_url cannot contain a query or fragment"
        )
    return value.rstrip("/")


def _validate_live_url(value: str) -> str:
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        host == "browser-use.com" or host.endswith(".browser-use.com")
    ):
        raise BrowserHandoffError(
            "The active browser does not expose a trusted Browser Use live-control URL"
        )
    return value


def _db_path() -> Path:
    path = get_hermes_home() / "state" / "browser-handoffs.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class BrowserHandoffStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else _db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.path), timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=10000")
        return db

    def _init_db(self) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS browser_handoffs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_digest TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    wake_status TEXT NOT NULL DEFAULT 'pending',
                    wake_claimed_at REAL,
                    session_id TEXT NOT NULL,
                    browser_key TEXT NOT NULL,
                    browser_id TEXT NOT NULL,
                    live_url TEXT NOT NULL,
                    instruction TEXT NOT NULL,
                    source_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    completed_at REAL
                )
                """
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_browser_handoffs_browser "
                "ON browser_handoffs(browser_id, status)"
            )
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_browser_handoffs_one_pending "
                "ON browser_handoffs(browser_id) WHERE status='pending'"
            )
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass

    @staticmethod
    def _record(row: sqlite3.Row) -> BrowserHandoff:
        try:
            source = json.loads(row["source_json"] or "{}")
        except (TypeError, ValueError):
            source = {}
        return BrowserHandoff(
            id=int(row["id"]),
            token_digest=str(row["token_digest"]),
            status=str(row["status"]),
            wake_status=str(row["wake_status"]),
            session_id=str(row["session_id"]),
            browser_key=str(row["browser_key"]),
            browser_id=str(row["browser_id"]),
            live_url=str(row["live_url"]),
            instruction=str(row["instruction"]),
            source=source if isinstance(source, dict) else {},
            created_at=float(row["created_at"]),
            expires_at=float(row["expires_at"]),
        )

    def create(
        self,
        *,
        token_digest: str,
        session_id: str,
        browser_key: str,
        browser_id: str,
        live_url: str,
        instruction: str,
        source: dict[str, Any],
        ttl_seconds: int,
    ) -> BrowserHandoff:
        now = time.time()
        expires = now + ttl_seconds
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE browser_handoffs SET status='expired' "
                "WHERE browser_id=? AND status='pending' AND expires_at<=?",
                (browser_id, now),
            )
            existing = db.execute(
                "SELECT id FROM browser_handoffs "
                "WHERE browser_id=? AND status='pending'",
                (browser_id,),
            ).fetchone()
            if existing is not None:
                raise BrowserHandoffError(
                    "A human handoff is already pending for this browser; "
                    "use the link already sent by Discord"
                )
            try:
                cur = db.execute(
                    """
                    INSERT INTO browser_handoffs (
                        token_digest, status, wake_status, session_id, browser_key,
                        browser_id, live_url, instruction, source_json,
                        created_at, expires_at
                    ) VALUES (?, 'pending', 'pending', ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        token_digest,
                        session_id,
                        browser_key,
                        browser_id,
                        live_url,
                        instruction,
                        json.dumps(source, separators=(",", ":"), sort_keys=True),
                        now,
                        expires,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise BrowserHandoffError(
                    "A human handoff is already pending for this browser; "
                    "use the link already sent by Discord"
                ) from exc
            row = db.execute(
                "SELECT * FROM browser_handoffs WHERE id=?", (cur.lastrowid,)
            ).fetchone()
        return self._record(row)

    def lookup(
        self, token: str, *, now: Optional[float] = None
    ) -> Optional[BrowserHandoff]:
        if len(token) != _TOKEN_HEX_LENGTH:
            return None
        try:
            bytes.fromhex(token)
        except ValueError:
            return None
        digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        at = time.time() if now is None else now
        with self._lock, self._connect() as db:
            row = db.execute(
                "SELECT * FROM browser_handoffs WHERE token_digest=?", (digest,)
            ).fetchone()
            if row is None:
                return None
            if row["status"] == "pending" and float(row["expires_at"]) <= at:
                db.execute(
                    "UPDATE browser_handoffs SET status='expired' "
                    "WHERE id=? AND status='pending'",
                    (row["id"],),
                )
                row = db.execute(
                    "SELECT * FROM browser_handoffs WHERE id=?", (row["id"],)
                ).fetchone()
        return self._record(row)

    def complete(self, token: str) -> Optional[BrowserHandoff]:
        record = self.lookup(token)
        if record is None:
            return None
        if record.status == "pending":
            with self._lock, self._connect() as db:
                db.execute(
                    "UPDATE browser_handoffs SET status='completed', completed_at=? "
                    "WHERE id=? AND status='pending'",
                    (time.time(), record.id),
                )
                row = db.execute(
                    "SELECT * FROM browser_handoffs WHERE id=?", (record.id,)
                ).fetchone()
            return self._record(row)
        return record

    def claim_wake(self, record_id: int) -> Optional[BrowserHandoff]:
        now = time.time()
        reclaim_before = now - _WAKE_RECLAIM_SECONDS
        with self._lock, self._connect() as db:
            cur = db.execute(
                """
                UPDATE browser_handoffs
                SET wake_status='delivering', wake_claimed_at=?
                WHERE id=? AND status IN ('completed', 'expired') AND (
                    wake_status='pending' OR
                    (wake_status='delivering' AND COALESCE(wake_claimed_at, 0) < ?)
                )
                """,
                (now, record_id, reclaim_before),
            )
            if cur.rowcount != 1:
                return None
            row = db.execute(
                "SELECT * FROM browser_handoffs WHERE id=?", (record_id,)
            ).fetchone()
        return self._record(row)

    def finish_wake(self, record_id: int, *, delivered: bool) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE browser_handoffs SET wake_status=?, wake_claimed_at=NULL "
                "WHERE id=? AND wake_status='delivering'",
                ("delivered" if delivered else "pending", record_id),
            )

    def pending_wake_ids(self, limit: int = 20) -> list[int]:
        now = time.time()
        reclaim_before = now - _WAKE_RECLAIM_SECONDS
        with self._lock, self._connect() as db:
            # Expiry is a wake event. The agent already ended its turn at the
            # handoff boundary and must not remain dormant forever.
            db.execute(
                "UPDATE browser_handoffs SET status='expired' "
                "WHERE status='pending' AND expires_at<=?",
                (now,),
            )
            rows = db.execute(
                """
                SELECT id FROM browser_handoffs
                WHERE status IN ('completed', 'expired') AND (
                    wake_status='pending' OR
                    (wake_status='delivering' AND COALESCE(wake_claimed_at, 0) < ?)
                ) ORDER BY id LIMIT ?
                """,
                (reclaim_before, int(limit)),
            ).fetchall()
        return [int(row["id"]) for row in rows]

    def cancel(self, record_id: int) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE browser_handoffs SET status='cancelled' "
                "WHERE id=? AND status='pending'",
                (record_id,),
            )

    def cancel_for_browser(self, browser_id: str) -> int:
        if not browser_id:
            return 0
        with self._lock, self._connect() as db:
            cur = db.execute(
                "UPDATE browser_handoffs SET status='cancelled' "
                "WHERE browser_id=? AND status='pending'",
                (browser_id,),
            )
        return int(cur.rowcount)

    def retains_browser(self, browser_id: str) -> bool:
        if not browser_id:
            return False
        now = time.time()
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE browser_handoffs SET status='expired' "
                "WHERE browser_id=? AND status='pending' AND expires_at<=?",
                (browser_id, now),
            )
            row = db.execute(
                "SELECT 1 FROM browser_handoffs "
                "WHERE browser_id=? AND (status='pending' OR ("
                "status IN ('completed', 'expired') AND wake_status!='delivered'"
                ")) LIMIT 1",
                (browser_id,),
            ).fetchone()
        return row is not None


def browser_handoff_retains_browser(browser_id: str) -> bool:
    try:
        return BrowserHandoffStore().retains_browser(browser_id)
    except Exception:
        logger.debug(
            "Failed to check pending browser handoff for %s",
            browser_id,
            exc_info=True,
        )
        # Fail closed for resource retention: a database failure must not pin
        # a paid browser indefinitely.
        return False


def _source_from_context() -> dict[str, Any]:
    from gateway.session_context import get_session_env

    platform = get_session_env("HERMES_SESSION_PLATFORM", "")
    if not platform:
        return {}
    source: dict[str, Any] = {
        "platform": platform,
        "chat_id": get_session_env("HERMES_SESSION_CHAT_ID", ""),
        "chat_type": get_session_env("HERMES_SESSION_CHAT_TYPE", "") or "dm",
    }
    names = {
        "chat_name": "HERMES_SESSION_CHAT_NAME",
        "thread_id": "HERMES_SESSION_THREAD_ID",
        "user_id": "HERMES_SESSION_USER_ID",
        "user_id_alt": "HERMES_SESSION_USER_ID_ALT",
        "user_name": "HERMES_SESSION_USER_NAME",
        "scope_id": "HERMES_SESSION_SCOPE_ID",
        "message_id": "HERMES_SESSION_MESSAGE_ID",
        "profile": "HERMES_SESSION_PROFILE",
    }
    for key, env_name in names.items():
        value = get_session_env(env_name, "")
        if value:
            source[key] = value
    return source


def _discord_bot_token() -> str:
    try:
        from agent.secret_scope import get_secret

        return str(get_secret("DISCORD_BOT_TOKEN", "") or "")
    except Exception:
        return str(os.getenv("DISCORD_BOT_TOKEN", "") or "")


def send_discord_handoff_dm(user_id: str, message: str) -> None:
    """Create a Discord DM channel and send the handoff notification."""
    token = _discord_bot_token()
    if not token:
        raise BrowserHandoffError("DISCORD_BOT_TOKEN is not configured")
    headers = {"Authorization": f"Bot {token}", "Content-Type": "application/json"}
    try:
        opened = requests.post(
            "https://discord.com/api/v10/users/@me/channels",
            headers=headers,
            json={"recipient_id": user_id},
            timeout=15,
        )
        opened.raise_for_status()
        channel_id = str(opened.json().get("id") or "")
        if not channel_id:
            raise RuntimeError("Discord returned no DM channel ID")
        sent = requests.post(
            f"https://discord.com/api/v10/channels/{channel_id}/messages",
            headers=headers,
            json={"content": message},
            timeout=15,
        )
        sent.raise_for_status()
    except requests.RequestException as exc:
        raise BrowserHandoffError(f"Discord DM delivery failed: {exc}") from exc


def create_browser_handoff(
    *,
    instruction: str,
    task_id: str,
    session_name: str = "",
    store: Optional[BrowserHandoffStore] = None,
    notifier: Callable[[str, str], None] = send_discord_handoff_dm,
) -> dict[str, Any]:
    """Persist a handoff and DM its public URL to the configured owner."""
    clean_instruction = " ".join(str(instruction or "").split()).strip()
    if not clean_instruction:
        raise BrowserHandoffError(
            "A short description of what the human must do is required"
        )
    if len(clean_instruction) > 500:
        raise BrowserHandoffError("The handoff instruction must be 500 characters or fewer")

    cfg = load_browser_handoff_config()
    if not cfg.enabled:
        raise BrowserHandoffError("Browser handoff is disabled in browser.handoff.enabled")
    base_url = _validate_public_base_url(cfg.public_base_url)
    if not cfg.discord_user_id.isdigit():
        raise BrowserHandoffError("browser.handoff.discord_user_id must be a Discord user ID")

    from gateway.session_context import get_session_env
    from tools.browser_tool import _get_session_info
    from tools.browser_use_cli import _provider_session_cache_key

    browser_key = _provider_session_cache_key(task_id, session_name)
    info = _get_session_info(browser_key) or {}
    if isinstance(info, dict):
        info["owner_task_id"] = str(task_id or "browser-exec-default")
    if not bool((info.get("features") or {}).get("browser_use")):
        raise BrowserHandoffError(
            "Human handoff is currently supported only for Browser Use cloud sessions"
        )
    browser_id = str(info.get("bb_session_id") or "")
    live_url = _validate_live_url(str(info.get("live_url") or ""))
    if not browser_id:
        raise BrowserHandoffError("The active Browser Use session has no browser ID")

    session_id = get_session_env("HERMES_SESSION_ID", "") or str(task_id or "")
    if not session_id:
        raise BrowserHandoffError("Hermes could not identify the paused session to resume")

    token = secrets.token_hex(_TOKEN_BYTES)
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    ttl_seconds = cfg.ttl_minutes * 60
    provider_expiry = str(info.get("expires_at") or "").strip()
    if provider_expiry:
        try:
            parsed_expiry = datetime.fromisoformat(
                provider_expiry.replace("Z", "+00:00")
            )
            if parsed_expiry.tzinfo is None:
                parsed_expiry = parsed_expiry.replace(tzinfo=timezone.utc)
            provider_seconds = int(parsed_expiry.timestamp() - time.time())
            if provider_seconds <= 0:
                raise BrowserHandoffError("The Browser Use session has already expired")
            ttl_seconds = min(ttl_seconds, provider_seconds)
        except BrowserHandoffError:
            raise
        except (TypeError, ValueError):
            logger.warning("Ignoring malformed Browser Use expires_at value")
    handoff_store = store or BrowserHandoffStore()
    source = _source_from_context()
    record = handoff_store.create(
        token_digest=digest,
        session_id=session_id,
        browser_key=browser_key,
        browser_id=browser_id,
        live_url=live_url,
        instruction=clean_instruction,
        source=source,
        ttl_seconds=ttl_seconds,
    )
    profile = str(source.get("profile") or "").strip()
    profile_prefix = (
        f"/p/{quote(profile, safe='')}"
        if profile and profile not in {"default", "custom"}
        else ""
    )
    link = f"{base_url}{profile_prefix}/browser-handoff/{token}"
    effective_minutes = max(1, int((record.expires_at - time.time() + 59) // 60))
    message = (
        f"yo i need u to do this: {clean_instruction}\n\n"
        f"Take control here (no login required): {link}\n\n"
        f"This link expires in {effective_minutes} minute"
        f"{'s' if effective_minutes != 1 else ''}. "
        "Click Done on the page when finished. Do not forward this link."
    )
    try:
        notifier(cfg.discord_user_id, message)
    except Exception:
        handoff_store.cancel(record.id)
        raise
    return {
        "handoff_id": record.id,
        "expires_at": record.expires_at,
        "message": (
            "I sent Alex a Discord DM with a browser-control link valid for "
            f"about {effective_minutes} minute"
            f"{'s' if effective_minutes != 1 else ''}. "
            "End this turn now and wait. The same Hermes session will be woken "
            "when Alex clicks Done; then inspect the current page before continuing."
        ),
    }


def cancel_browser_handoffs(browser_id: str) -> None:
    try:
        BrowserHandoffStore().cancel_for_browser(browser_id)
    except Exception:
        logger.debug("Failed to revoke browser handoffs for %s", browser_id, exc_info=True)


def security_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-store, max-age=0",
        "Pragma": "no-cache",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Content-Security-Policy": (
            "default-src 'none'; frame-src https://browser-use.com "
            "https://*.browser-use.com; style-src 'unsafe-inline'; "
            "form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
        ),
    }


def render_handoff_page(
    record: Optional[BrowserHandoff], *, complete_path: str = ""
) -> tuple[str, int]:
    if record is None or record.status in {"expired", "cancelled"}:
        title = "Link unavailable"
        body = "This handoff link is invalid or has expired."
        status = 410
        frame = ""
    elif record.status == "completed":
        title = "Done"
        body = "Hermes has been notified and will continue the task."
        status = 200
        frame = ""
    else:
        title = "Take over the browser"
        body = html.escape(record.instruction)
        status = 200
        live = html.escape(record.live_url, quote=True)
        action = html.escape(complete_path, quote=True)
        frame = (
            f'<iframe title="Remote browser" src="{live}" allow="clipboard-read; clipboard-write"></iframe>'
            f'<p class="fallback"><a href="{live}" target="_blank" '
            'rel="noreferrer noopener">Open the live browser in a new tab</a></p>'
            f'<form method="post" action="{action}"><button type="submit">Done</button></form>'
        )
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
html,body{{margin:0;background:#0b0d12;color:#f4f5f7;font:16px system-ui,sans-serif}}
main{{max-width:1200px;margin:auto;padding:20px}}h1{{margin:.2em 0}}p{{color:#c9ccd3}}
iframe{{width:100%;height:min(72vh,820px);border:1px solid #343947;border-radius:12px;background:white}}
button{{background:#6d5dfc;color:white;border:0;border-radius:10px;padding:13px 28px;
font-size:17px;font-weight:700;cursor:pointer}}
a{{color:#a9c7ff}}.fallback{{margin:.75em 0}}form{{margin-top:16px}}
</style></head><body><main><h1>{html.escape(title)}</h1><p>{body}</p>{frame}</main></body></html>"""
    return doc, status
