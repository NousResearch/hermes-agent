"""Profile-, account- and conversation-local recovery of iLink ID-only quotes."""

from __future__ import annotations

import hashlib
import logging
import math
import shutil
import sqlite3
import time
import uuid
from contextlib import closing, suppress
from pathlib import Path

from hermes_constants import mkdir_under_hermes_home

logger = logging.getLogger(__name__)
DEFAULT_QUOTE_CACHE = {
    "enabled": True, "retention_days": 30, "max_messages_per_account": 10_000,
    "media_retention_days": 7, "max_media_bytes_per_account": 256 * 1024 * 1024,
    "max_single_media_bytes": 25 * 1024 * 1024,
}


def _positive(value, default):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0 else default


def message_id(message: dict) -> str:
    value = message.get("message_id")
    if value is None or str(value).strip() == "":
        value = next((item.get("msg_id") for item in message.get("item_list") or [] if item.get("msg_id")), "")
    return str(value).strip()


def _nth_index(text: str, needle: str, occurrence, offset: int = 0) -> int:
    if not needle or not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 0:
        return -1
    # A malformed wire index must not turn a short quote into an unbounded loop.
    if occurrence > len(text):
        return -1
    position = offset
    for _ in range(occurrence + 1):
        position = text.find(needle, position)
        if position < 0:
            return -1
        offset = position
        position += len(needle)
    return offset


def partial_quote(text: str, partial: dict) -> str | None:
    start_text, end_text = partial.get("start"), partial.get("end")
    if not isinstance(start_text, str) or not isinstance(end_text, str):
        return None
    start = _nth_index(text, start_text, partial.get("startindex"))
    if start < 0:
        return None
    for offset in (0, start + len(start_text)):
        end = _nth_index(text, end_text, partial.get("endindex"), offset)
        if end < start:
            continue
        quote = text[start:end + len(end_text)]
        expected = str(partial.get("quotemd5") or "").lower()
        if not expected or hashlib.md5(quote.encode()).hexdigest() == expected:
            return quote
    return None


class WeixinQuoteStore:
    """SQLite sidecar; cache failures leave delivery intact and disable further cache I/O."""

    def __init__(self, home: str, account_id: str, policy: dict | None = None):
        from gateway.platforms.weixin import _coerce_bool

        policy = policy or {}
        self.policy = {key: _positive(policy.get(key), value) for key, value in DEFAULT_QUOTE_CACHE.items() if key != "enabled"}
        self.policy.update({key: max(1, int(value)) for key, value in self.policy.items() if key.startswith("max_")})
        self.enabled = _coerce_bool(policy.get("enabled"), default=True)
        self.account_id = account_id
        self.root = Path(home) / "weixin"
        self.media_root = self.root / "quote-media" / hashlib.sha256(account_id.encode()).hexdigest()[:24]
        self.db_path = self.root / "quotes.sqlite"

    def _connect(self):
        mkdir_under_hermes_home(self.root)
        db = sqlite3.connect(self.db_path, timeout=5)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS quotes (
                account TEXT NOT NULL, peer TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL,
                media_path TEXT, media_type TEXT, media_name TEXT, media_size INTEGER, created REAL NOT NULL,
                PRIMARY KEY(account, peer, id))""")
            db.execute("CREATE INDEX IF NOT EXISTS quotes_account_created ON quotes(account, created)")
        except sqlite3.Error:
            db.close()
            raise
        return db

    def _disable(self, exc):
        self.enabled = False
        logger.warning("Weixin quote cache unavailable: %s", exc)

    def _copy_media(self, source: str | None):
        if not source:
            return None, 0
        path = Path(source)
        if not path.is_file() or path.is_symlink() or path.stat().st_size > self.policy["max_single_media_bytes"]:
            return None, 0
        mkdir_under_hermes_home(self.media_root)
        destination = self.media_root / f"{uuid.uuid4().hex}{path.suffix}"
        shutil.copyfile(path, destination)
        return str(destination), destination.stat().st_size

    def _unlink(self, path: str | None):
        if not path:
            return
        candidate = Path(path)
        if candidate.resolve().is_relative_to(self.media_root.resolve()):
            candidate.unlink(missing_ok=True)

    def _prune(self, db, now: float):
        cutoff = now - self.policy["retention_days"] * 86400
        rows = db.execute("SELECT * FROM quotes WHERE account=? ORDER BY created DESC", (self.account_id,)).fetchall()
        media_bytes = 0
        for index, row in enumerate(rows):
            expired = row["created"] < cutoff or index >= self.policy["max_messages_per_account"]
            key = (self.account_id, row["peer"], row["id"])
            if expired:
                db.execute("DELETE FROM quotes WHERE account=? AND peer=? AND id=?", key)
                self._unlink(row["media_path"])
            elif row["media_path"]:
                media_bytes += row["media_size"] or 0
                if row["created"] < now - self.policy["media_retention_days"] * 86400 or media_bytes > self.policy["max_media_bytes_per_account"]:
                    db.execute("UPDATE quotes SET media_path=NULL,media_size=0 WHERE account=? AND peer=? AND id=?", key)
                    self._unlink(row["media_path"])
        db.commit()
        self._prune_orphans(db, now)

    def _prune_orphans(self, db, now):
        if not self.media_root.exists():
            return
        retained = {Path(row[0]).resolve() for row in db.execute(
            "SELECT media_path FROM quotes WHERE account=? AND media_path IS NOT NULL", (self.account_id,))}
        for path in self.media_root.iterdir():
            # A concurrent cache writer may have copied a file before committing its row.
            if path.is_file() and path.resolve() not in retained and path.stat().st_mtime < now - 300:
                self._unlink(str(path))

    def put(self, peer: str, identifier: str, body: str, source: str | None = None, mime: str | None = None, name: str | None = None):
        if not self.enabled or not identifier or not (body or source):
            return
        owned_path = None
        try:
            with closing(self._connect()) as db:
                owned_path, size = self._copy_media(source)
                old = db.execute("SELECT media_path FROM quotes WHERE account=? AND peer=? AND id=?", (self.account_id, peer, identifier)).fetchone()
                db.execute("""INSERT INTO quotes VALUES (?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(account,peer,id) DO UPDATE SET body=excluded.body,
                    media_path=COALESCE(excluded.media_path,quotes.media_path),
                    media_type=COALESCE(excluded.media_type,quotes.media_type),
                    media_name=COALESCE(excluded.media_name,quotes.media_name),
                    media_size=CASE WHEN excluded.media_path IS NULL THEN quotes.media_size ELSE excluded.media_size END,
                    created=excluded.created""",
                           (self.account_id, peer, identifier, body, owned_path, mime, name, size, time.time()))
                db.commit()
                if old and owned_path and old["media_path"] != owned_path:
                    self._unlink(old["media_path"])
                self._prune(db, time.time())
        except (OSError, sqlite3.Error) as exc:
            if owned_path:
                with suppress(OSError):
                    self._unlink(owned_path)
            self._disable(exc)

    def find(self, peer: str, identifier: str) -> dict | None:
        if not self.enabled or not identifier or not self.db_path.exists():
            return None
        try:
            with closing(self._connect()) as db:
                self._prune(db, time.time())
                row = db.execute("SELECT * FROM quotes WHERE account=? AND peer=? AND id=?", (self.account_id, peer, identifier)).fetchone()
                return dict(row) if row else None
        except (OSError, sqlite3.Error) as exc:
            self._disable(exc)
            return None

    def sweep(self):
        if not self.enabled or not self.db_path.exists():
            return
        try:
            with closing(self._connect()) as db:
                self._prune(db, time.time())
        except (OSError, sqlite3.Error) as exc:
            self._disable(exc)


    def remove_account(self):
        """Remove this identity's cached rows and owned copies, including orphaned copies."""
        try:
            if self.db_path.exists():
                with closing(self._connect()) as db:
                    db.execute("DELETE FROM quotes WHERE account=?", (self.account_id,))
                    db.commit()
            if self.media_root.exists() and not self.media_root.is_symlink():
                for path in self.media_root.iterdir():
                    if path.is_file() or path.is_symlink():
                        self._unlink(str(path))
        except (OSError, sqlite3.Error) as exc:
            self._disable(exc)


async def resolve_quote(adapter, items: list[dict], peer: str, text: str, paths: list[str], types: list[str]):
    import asyncio

    reference = next((item.get("ref_msg") for item in items if item.get("ref_msg")), None)
    if not reference:
        return text, None
    identifier = str(reference.get("svr_id") or (reference.get("message_item") or {}).get("msg_id") or "")
    if not identifier or reference.get("title") or reference.get("message_item"):
        return text, identifier or None
    record = await asyncio.to_thread(adapter._quote_store.find, peer, identifier)
    if not record:
        return f"[引用消息内容未缓存]\n{text}".strip(), identifier
    body = record["body"]
    if reference.get("partial_text") and body:
        body = partial_quote(body, reference["partial_text"]) or body
    media_path, mime = record["media_path"], record["media_type"]
    if media_path and await asyncio.to_thread(Path(media_path).is_file):
        paths.append(media_path)
        types.append(mime or "application/octet-stream")
        body = body or f"[引用附件: {record['media_name'] or Path(media_path).name}]"
    elif mime:
        body = f"[引用附件已过期: {record['media_name'] or mime}]"
    return f"[引用: {body}]\n{text}".strip(), identifier
