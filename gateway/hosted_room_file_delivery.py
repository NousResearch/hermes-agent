"""Bounded native document delivery with fail-closed, non-replaying receipts."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import os
import re
import shutil
import sqlite3
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from gateway.native_document_guard import (
    require_native_document,
)
from gateway.platforms.base import BasePlatformAdapter

MAX_BYTES = 15_000_000
MAX_ACTIVE_SENDS = 4
MAX_RECEIPTS = 4096
RECEIPT_SECONDS = 30 * 24 * 60 * 60
FETCH_SECONDS = 20.0
SEND_SECONDS = 60.0
logger = logging.getLogger(__name__)


class FileDeliveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class Document:
    name: str
    data: bytes


def _native_document_sender(adapter):
    sender = getattr(adapter, "send_document", None)
    implementation = getattr(sender, "__func__", sender)
    if sender is None or implementation is BasePlatformAdapter.send_document:
        raise FileDeliveryError("unsupported")
    if not getattr(implementation, "strict_native_document_guard", False):
        raise FileDeliveryError("unsupported")
    return sender


def native_document_limit(adapter, source):
    _native_document_sender(adapter)
    platform = str(getattr(source.platform, "value", source.platform))
    if platform not in {
        "telegram",
        "discord",
        "matrix",
        "signal",
        "slack",
        "whatsapp",
        "whatsapp_cloud",
    }:
        raise FileDeliveryError("unsupported")
    maximum = MAX_BYTES
    if platform == "discord":
        maximum = 10_000_000
        try:
            channel = adapter._client.get_channel(int(source.chat_id))
            live_limit = channel.guild.filesize_limit
            if type(live_limit) is int and live_limit > 0:
                maximum = min(MAX_BYTES, live_limit)
        except (AttributeError, TypeError, ValueError):
            pass
    if platform == "matrix":
        value = getattr(adapter, "_max_media_bytes", None)
        if type(value) is int and value > 0:
            maximum = min(maximum, value)
    return maximum


def _receipt_connection(db_path):
    from hermes_state_wal import apply_wal_with_fallback

    conn = sqlite3.connect(db_path, timeout=10)
    try:
        apply_wal_with_fallback(conn, db_label="state.db (Group Chat file delivery)")
        conn.execute("""CREATE TABLE IF NOT EXISTS hosted_room_file_deliveries (
            delivery_key TEXT PRIMARY KEY, delivery_scope TEXT NOT NULL,
            state TEXT NOT NULL, attempts INTEGER NOT NULL, created_at REAL NOT NULL,
            updated_at REAL NOT NULL)""")
        conn.execute("""CREATE INDEX IF NOT EXISTS idx_group_file_delivery_scope
            ON hosted_room_file_deliveries(delivery_scope, state, updated_at)""")
        conn.commit()
        return conn
    except Exception:
        conn.close()
        raise


def reserve_delivery(db_path, key, scope):
    now = time.time()
    conn = _receipt_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM hosted_room_file_deliveries WHERE updated_at<?",
            (now - RECEIPT_SECONDS,),
        )
        conn.execute(
            """UPDATE hosted_room_file_deliveries SET state='unknown', updated_at=?
            WHERE state IN ('fetching','sending') AND updated_at<?""",
            (now, now - 120),
        )
        row = conn.execute(
            "SELECT state, attempts, delivery_scope FROM hosted_room_file_deliveries WHERE delivery_key=?",
            (key,),
        ).fetchone()
        if row:
            state, attempts, stored_scope = row
            if stored_scope != scope:
                raise FileDeliveryError("receipt_conflict")
            if state != "failed" or attempts >= 3:
                conn.commit()
                return "busy" if state in {"fetching", "sending"} else state
        else:
            attempts = 0
        busy = conn.execute(
            "SELECT COUNT(*) FROM hosted_room_file_deliveries WHERE state IN ('fetching','sending')"
        ).fetchone()[0]
        same_file = conn.execute(
            "SELECT 1 FROM hosted_room_file_deliveries WHERE delivery_scope=? AND state IN ('fetching','sending') LIMIT 1",
            (scope,),
        ).fetchone()
        if busy >= MAX_ACTIVE_SENDS or same_file:
            conn.commit()
            return "busy"
        count = conn.execute(
            "SELECT COUNT(*) FROM hosted_room_file_deliveries"
        ).fetchone()[0]
        if row is None and count >= MAX_RECEIPTS:
            raise FileDeliveryError("receipt_full")
        conn.execute(
            """INSERT INTO hosted_room_file_deliveries VALUES (?,?,'fetching',?,?,?)
            ON CONFLICT(delivery_key) DO UPDATE SET state='fetching', attempts=excluded.attempts,
            updated_at=excluded.updated_at""",
            (key, scope, attempts + 1, now, now),
        )
        conn.commit()
        return "new"
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def mark_delivery(db_path, key, state):
    if state not in {"sending", "delivered", "failed", "unknown"}:
        raise ValueError("invalid file delivery state")
    conn = _receipt_connection(db_path)
    try:
        with conn:
            changed = conn.execute(
                """UPDATE hosted_room_file_deliveries SET state=?, updated_at=?
                WHERE delivery_key=? AND state IN ('fetching','sending')""",
                (state, time.time(), key),
            )
            if changed.rowcount != 1:
                raise FileDeliveryError("receipt_unavailable")
    finally:
        conn.close()


def delivery_identity(source_key, request_id, selection_key):
    scope = hashlib.sha256(f"{source_key}\0{selection_key}".encode()).hexdigest()
    key = hashlib.sha256(f"{scope}\0{request_id}".encode()).hexdigest()
    return key, scope


def _temporary_document(db_path, document):
    root = Path(db_path).parent / "group-file-delivery-tmp"
    root.mkdir(mode=0o700, exist_ok=True)
    if root.is_symlink():
        raise FileDeliveryError("temporary_unavailable")
    root.chmod(0o700)
    with os.scandir(root) as entries:
        for index, entry in enumerate(entries):
            if index >= 64:
                break
            if (
                re.fullmatch(r"send-[a-z0-9_]+", entry.name)
                and entry.is_dir(follow_symlinks=False)
                and entry.stat(follow_symlinks=False).st_mtime < time.time() - 3600
            ):
                shutil.rmtree(entry.path)
    folder = tempfile.TemporaryDirectory(prefix="send-", dir=root)
    try:
        name = "".join(
            char
            for char in document.name
            if unicodedata.category(char) not in {"Cc", "Cf"}
        )
        if not name or name in {".", ".."} or any(char in name for char in "/\\"):
            raise FileDeliveryError("invalid_filename")
        name = re.sub(r'[<>:"|?*]', "_", name)
        if len(name.encode("utf-8")) > 220:
            suffix = Path(name).suffix.encode("utf-8")[:20].decode("utf-8", "ignore")
            prefix = name.encode("utf-8")[:180].decode("utf-8", "ignore")
            name = prefix + "-" + hashlib.sha256(name.encode()).hexdigest()[:8] + suffix
        path = Path(folder.name) / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as output:
            output.write(document.data)
        return folder, path
    except BaseException:
        folder.cleanup()
        raise


async def deliver_document(
    *, db_path, key, scope, adapter, source, load, recheck, metadata, reply_to
):
    maximum = native_document_limit(adapter, source)
    reservation = asyncio.create_task(
        asyncio.to_thread(reserve_delivery, db_path, key, scope)
    )
    try:
        state = await asyncio.shield(reservation)
    except asyncio.CancelledError:
        try:
            state = await _settle_owned_work(reservation)
            if state == "new":
                await _settle_owned_work(
                    asyncio.create_task(
                        asyncio.to_thread(mark_delivery, db_path, key, "failed")
                    )
                )
        except Exception as exc:
            logger.warning(
                "Cancelled file reservation cleanup failed: %s", type(exc).__name__
            )
        raise
    except Exception:
        raise FileDeliveryError("receipt_unavailable") from None
    if state != "new":
        return state
    folder = None
    sending = False
    try:
        document = await asyncio.wait_for(
            asyncio.to_thread(load, maximum), FETCH_SECONDS
        )
        if (
            not isinstance(document, Document)
            or not isinstance(document.data, bytes)
            or not 0 < len(document.data) <= maximum
        ):
            raise FileDeliveryError("too_large")
        await recheck()
        preparation = asyncio.create_task(
            asyncio.to_thread(_temporary_document, db_path, document)
        )
        try:
            folder, path = await asyncio.shield(preparation)
        except asyncio.CancelledError:
            folder, path = await _settle_owned_work(preparation)
            raise
        await recheck()
        if native_document_limit(adapter, source) < len(document.data):
            raise FileDeliveryError("too_large")
        sender = _native_document_sender(adapter)
        await asyncio.to_thread(mark_delivery, db_path, key, "sending")
        await recheck()
        sending = True
        kwargs = dict(
            chat_id=source.chat_id,
            file_path=str(path),
            reply_to=reply_to,
            metadata={**metadata, "group_file_delivery_id": key},
        )
        parameters = inspect.signature(sender).parameters
        kwargs[
            "filename"
            if "filename" in parameters and "file_name" not in parameters
            else "file_name"
        ] = path.name
        with require_native_document():
            result = await asyncio.wait_for(
                sender(**kwargs), SEND_SECONDS
            )
        if getattr(result, "success", False) is not True:
            await asyncio.to_thread(mark_delivery, db_path, key, "unknown")
            return "unknown"
        await asyncio.to_thread(mark_delivery, db_path, key, "delivered")
        return "delivered"
    except asyncio.CancelledError:
        await _settle_owned_work(
            asyncio.create_task(
                asyncio.to_thread(
                    mark_delivery, db_path, key, "unknown" if sending else "failed"
                )
            )
        )
        raise
    except Exception:
        try:
            await asyncio.to_thread(
                mark_delivery, db_path, key, "unknown" if sending else "failed"
            )
        except Exception:
            pass
        if sending:
            return "unknown"
        raise
    finally:
        if folder is not None:
            await _settle_owned_work(
                asyncio.create_task(asyncio.to_thread(folder.cleanup))
            )


async def _settle_owned_work(task):
    """An executor operation cannot be cancelled once it starts; observe its result."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()
