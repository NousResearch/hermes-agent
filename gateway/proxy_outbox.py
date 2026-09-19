"""Durable text delivery from a remote agent host to a proxy gateway.

The API-server host enqueues proactive output. A thin gateway that owns the
native messaging adapter leases, sends, and acknowledges it over authenticated
HTTP. Missing acknowledgements fail closed because the platform-side outcome is
ambiguous and retrying could duplicate a message.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from hermes_constants import get_hermes_home

MAX_ITEMS = 500
MAX_ATTEMPTS = 1
STALE_AFTER_SECONDS = 24 * 60 * 60
LEASE_SECONDS = 10 * 60
NATIVE_SEND_TIMEOUT_SECONDS = 9 * 60
_DB_LOCK = threading.Lock()
_DELIVERY_ID_RE = re.compile(r"[0-9a-f]{32}")


def _platform_name(platform: Any) -> str:
    return str(getattr(platform, "value", platform)).strip().lower()


def enabled_platforms() -> set[str]:
    """Return valid logical platforms explicitly fronted by this host."""
    from gateway.config import Platform
    from hermes_cli.config import load_config

    raw = (load_config().get("gateway") or {}).get("proxy_outbox_platforms", [])
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, list):
        return set()

    excluded = {Platform.LOCAL.value, Platform.API_SERVER.value, Platform.RELAY.value}
    enabled: set[str] = set()
    for item in raw:
        name = _platform_name(item)
        if not name or name in excluded:
            continue
        try:
            enabled.add(Platform(name).value)
        except ValueError:
            continue
    return enabled


def fronts_platform(platform: Any) -> bool:
    return _platform_name(platform) in enabled_platforms()


def _db_path() -> Path:
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    from hermes_state import apply_wal_with_fallback

    apply_wal_with_fallback(conn, db_label="state.db (proxy_outbox)")
    # Existing experimental installations may have additional media columns;
    # selecting this common text-only subset remains backwards compatible.
    conn.execute(
        """CREATE TABLE IF NOT EXISTS proxy_outbox (
            delivery_id TEXT PRIMARY KEY,
            platform TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'text',
            content TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            lease_until REAL,
            last_error TEXT
        )"""
    )
    return conn


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _clean_metadata(metadata: Optional[dict[str, Any]]) -> dict[str, str]:
    clean: dict[str, str] = {}
    for key in ("thread_id", "reply_to", "job_id"):
        value = (metadata or {}).get(key)
        if value is None:
            continue
        text = str(value)
        if len(text) > 256 or any(ch in text for ch in "\r\n\x00"):
            raise ValueError(f"invalid proxy outbox metadata: {key}")
        clean[key] = text
    return clean


def enqueue(
    *,
    platform: Any,
    chat_id: str,
    content: str,
    metadata: Optional[dict[str, Any]] = None,
    kind: str = "text",
) -> str:
    """Queue one bounded text delivery and return its opaque identifier."""
    platform_name = _platform_name(platform)
    if not fronts_platform(platform_name):
        raise ValueError(f"proxy outbox does not front platform '{platform_name}'")
    if kind != "text":
        raise ValueError("proxy outbox supports text delivery only")
    chat_id = str(chat_id).strip()
    if not chat_id or len(chat_id) > 512 or any(ch in chat_id for ch in "\r\n\x00"):
        raise ValueError("invalid proxy outbox chat_id")
    if not isinstance(content, str) or len(content.encode("utf-8")) > 256 * 1024:
        raise ValueError("proxy outbox text exceeds 256 KiB")

    delivery_id = uuid.uuid4().hex
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        _prune_locked(conn, now)
        active_count = conn.execute(
            "SELECT COUNT(*) FROM proxy_outbox WHERE state IN ('pending', 'leased')"
        ).fetchone()[0]
        if int(active_count) >= MAX_ITEMS:
            raise RuntimeError("proxy outbox is full")
        conn.execute(
            """INSERT INTO proxy_outbox
               (delivery_id, platform, chat_id, kind, content, metadata_json,
                state, attempts, created_at, updated_at)
               VALUES (?, ?, ?, 'text', ?, ?, 'pending', 0, ?, ?)""",
            (
                delivery_id,
                platform_name,
                chat_id,
                content,
                json.dumps(_clean_metadata(metadata), separators=(",", ":")),
                now,
                now,
            ),
        )
    return delivery_id


def lease(
    *,
    platforms: set[str],
    limit: int = 4,
    lease_seconds: int = LEASE_SECONDS,
) -> list[dict[str, Any]]:
    """Lease pending text for platforms owned by the requesting gateway."""
    requested = {_platform_name(item) for item in platforms} & enabled_platforms()
    if not requested:
        return []
    limit = max(1, min(int(limit), 8))
    lease_seconds = max(15, min(int(lease_seconds), 15 * 60))
    now = time.time()
    placeholders = ",".join("?" for _ in requested)
    leased: list[dict[str, Any]] = []

    with _DB_LOCK, _transaction() as conn:
        _prune_locked(conn, now)
        rows = conn.execute(
            f"""SELECT delivery_id, platform, chat_id, content,
                       metadata_json, attempts
                FROM proxy_outbox
                WHERE state='pending' AND platform IN ({placeholders})
                  AND attempts < ?
                ORDER BY created_at ASC LIMIT ?""",
            (*sorted(requested), MAX_ATTEMPTS, limit),
        ).fetchall()
        for delivery_id, platform, chat_id, content, metadata_json, attempts in rows:
            changed = conn.execute(
                """UPDATE proxy_outbox
                   SET state='leased', attempts=attempts+1, updated_at=?, lease_until=?
                   WHERE delivery_id=? AND state='pending'""",
                (now, now + lease_seconds, delivery_id),
            ).rowcount
            if changed != 1:
                continue
            leased.append(
                {
                    "delivery_id": delivery_id,
                    "platform": platform,
                    "chat_id": chat_id,
                    "content": content,
                    "metadata": json.loads(metadata_json or "{}"),
                    "attempt": int(attempts) + 1,
                }
            )
    return leased


def acknowledge(
    delivery_id: str,
    *,
    attempt: int,
    success: bool,
    error: str = "",
) -> bool:
    """Record the exact leased attempt without retrying ambiguous sends."""
    now = time.time()
    state = "delivered" if success else "failed"
    with _DB_LOCK, _transaction() as conn:
        changed = conn.execute(
            """UPDATE proxy_outbox
               SET state=?, updated_at=?, lease_until=NULL, last_error=?
               WHERE delivery_id=? AND state='leased' AND attempts=?""",
            (
                state,
                now,
                (error or "")[:500] or None,
                delivery_id,
                int(attempt),
            ),
        ).rowcount
    return changed == 1


def delivery_result(delivery_id: str) -> Optional[tuple[bool, Optional[str]]]:
    """Return a terminal result, or ``None`` while work is active."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        _prune_locked(conn, now)
        row = conn.execute(
            "SELECT state, last_error FROM proxy_outbox WHERE delivery_id=?",
            (delivery_id,),
        ).fetchone()
    if not row:
        return False, "delivery not found"
    if row[0] == "delivered":
        return True, None
    if row[0] == "failed":
        return False, row[1] or "delivery failed"
    return None


def fail_pending(delivery_id: str, error: str) -> bool:
    """Cancel work that no consumer has leased before the caller times out."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        changed = conn.execute(
            """UPDATE proxy_outbox SET state='failed', updated_at=?, last_error=?
               WHERE delivery_id=? AND state='pending'""",
            (now, error[:500], delivery_id),
        ).rowcount
    return changed == 1


def _prune_locked(conn: sqlite3.Connection, now: float) -> None:
    conn.execute(
        """UPDATE proxy_outbox SET state='failed', lease_until=NULL, updated_at=?,
           last_error='delivery outcome unknown after consumer lease expired'
           WHERE state='leased' AND lease_until < ?""",
        (now, now),
    )
    cutoff = now - STALE_AFTER_SECONDS
    conn.execute(
        """UPDATE proxy_outbox SET state='failed', updated_at=?,
           last_error='delivery expired before confirmation'
           WHERE state='pending' AND updated_at < ?""",
        (now, cutoff),
    )
    conn.execute(
        """DELETE FROM proxy_outbox
           WHERE state IN ('delivered', 'failed') AND updated_at < ?""",
        (cutoff,),
    )
    total = int(conn.execute("SELECT COUNT(*) FROM proxy_outbox").fetchone()[0])
    overflow = max(0, total - MAX_ITEMS)
    if overflow:
        conn.execute(
            """DELETE FROM proxy_outbox WHERE delivery_id IN (
                   SELECT delivery_id FROM proxy_outbox
                   WHERE state IN ('delivered', 'failed')
                   ORDER BY updated_at ASC LIMIT ?
               )""",
            (overflow,),
        )


async def deliver_once(
    proxy_url: str,
    proxy_key: str,
    adapters: Mapping[Any, Any],
    *,
    session: Any = None,
) -> int:
    """Lease and deliver one batch through this gateway's native adapters."""
    from aiohttp import ClientSession, ClientTimeout

    by_name = {_platform_name(platform): adapter for platform, adapter in adapters.items()}
    if not by_name:
        return 0
    headers = {"Authorization": f"Bearer {proxy_key}"}
    own_session = session is None
    if own_session:
        session = ClientSession(timeout=ClientTimeout(total=30))

    try:
        async with session.get(
            f"{proxy_url.rstrip('/')}/v1/proxy/outbox",
            # Proactive traffic is low-volume. One lease at a time keeps an
            # interrupted consumer from stranding an already-leased batch.
            params={"platforms": ",".join(sorted(by_name)), "limit": "1"},
            headers=headers,
        ) as response:
            response.raise_for_status()
            payload = await response.json()
        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise ValueError("invalid proxy outbox response")

        delivered = 0
        first_ack_error: Optional[Exception] = None
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("invalid proxy outbox item")
            delivery_id = item.get("delivery_id")
            platform = item.get("platform")
            attempt = item.get("attempt")
            chat_id = item.get("chat_id")
            content = item.get("content")
            metadata = item.get("metadata", {})
            if (
                not isinstance(delivery_id, str)
                or not _DELIVERY_ID_RE.fullmatch(delivery_id)
                or platform not in by_name
                or type(attempt) is not int
                or attempt < 1
                or not isinstance(chat_id, str)
                or not isinstance(content, str)
                or not isinstance(metadata, dict)
            ):
                raise ValueError("invalid proxy outbox item")

            error = ""
            try:
                result = await asyncio.wait_for(
                    by_name[platform].send(chat_id, content, metadata=metadata),
                    timeout=NATIVE_SEND_TIMEOUT_SECONDS,
                )
                if isinstance(result, dict):
                    success = result.get("success") is True
                else:
                    success = getattr(result, "success", False) is True
                if not success:
                    error = "native adapter reported delivery failure"
            except Exception as exc:  # noqa: BLE001 - failure must be ACKed
                success = False
                error = f"native adapter raised {type(exc).__name__}"

            try:
                async with session.post(
                    f"{proxy_url.rstrip('/')}/v1/proxy/outbox/{delivery_id}/ack",
                    json={"attempt": attempt, "success": success, "error": error},
                    headers=headers,
                ) as response:
                    response.raise_for_status()
            except Exception as exc:  # keep ACKing the rest of an already-leased batch
                if first_ack_error is None:
                    first_ack_error = exc
            if success:
                delivered += 1
        if first_ack_error is not None:
            raise first_ack_error
        return delivered
    finally:
        if own_session:
            await session.close()
