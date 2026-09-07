"""Telegram transport for one gateway-owned Group Chat.

Durable input receipts become canonical room messages; committed room events become attributed
Telegram messages sent by each member's own bot. This is a transport, not a second authority: the
room log decides what happened, this queue only records what was delivered, and a delivery whose
outcome is unknown is never blindly repeated.

It sends only. Inbound updates arrive through the platform's own polling connection
(``hosted_room_ingress``); no second token consumer is created here.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCK_SCOPE = "hosted-room-telegram"

SCHEMA = """
CREATE TABLE IF NOT EXISTS inbox (
 event_id TEXT PRIMARY KEY, chat_id INTEGER NOT NULL, message_id INTEGER NOT NULL,
 user_id INTEGER NOT NULL, text TEXT NOT NULL, reply_to INTEGER,
 topic_id INTEGER, received_at REAL NOT NULL,
 state TEXT NOT NULL DEFAULT 'pending', result TEXT,
 UNIQUE(chat_id,message_id));
CREATE TABLE IF NOT EXISTS deliveries (
 event_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, profile TEXT NOT NULL,
 thread_id TEXT NOT NULL, status TEXT NOT NULL, message_id INTEGER,
 PRIMARY KEY(event_id,chunk_index));
CREATE TABLE IF NOT EXISTS cursor (singleton INTEGER PRIMARY KEY CHECK(singleton=1), seq INTEGER NOT NULL);
INSERT OR IGNORE INTO cursor VALUES(1,0);
"""

_BINDING_FIELDS = ("room_id", "queue_db", "chat_id", "owner_id", "control_profile", "bots")


def hosted_room_binding_path():
    """The private binding document this gateway serves, or ``None`` when none is configured.

    A missing setting is off. A configured value that is not a usable path is a configuration
    error rather than an implicit off switch: a gateway told to serve a room must not quietly
    serve nothing. A relative path resolves against the Hermes root, never the working directory.
    """
    from pathlib import Path

    from hermes_cli.config import load_config_readonly
    from hermes_constants import get_default_hermes_root

    node = load_config_readonly()
    for key in ("gateway", "hosted_rooms", "telegram", "binding_file"):
        if node is None:
            return None
        if not isinstance(node, dict):
            # A configured section of the wrong shape is a broken document, not an absent one.
            raise ValueError(f"gateway.hosted_rooms.telegram.binding_file: {key} is not a section")
        node = node.get(key)
    if node is None:
        return None
    if not isinstance(node, str) or not node.strip():
        raise ValueError("gateway.hosted_rooms.telegram.binding_file must be a path")
    path = Path(node.strip()).expanduser()
    return path if path.is_absolute() else get_default_hermes_root() / path


def load_binding(path: Path, *, include_disabled: bool = False) -> dict[str, Any] | None:
    """Validate one binding document; ``None`` when it is present but disabled.

    The same document every participating gateway reads. A malformed one is an error: a gateway
    told to serve a room must not fall back to serving nothing.
    """
    binding = json.loads(Path(path).read_text())
    if not isinstance(binding, dict):
        raise ValueError("hosted room binding must be an object")
    if type(binding.get("enabled")) is not bool:
        raise ValueError("binding enabled must be a boolean")
    if not binding["enabled"] and not include_disabled:
        return None
    missing = [field for field in _BINDING_FIELDS if field not in binding]
    if missing:
        raise ValueError(f"hosted room binding is missing {', '.join(missing)}")
    chat_id, owner_id = binding["chat_id"], binding["owner_id"]
    if type(chat_id) is not int or type(owner_id) is not int:
        raise ValueError("chat and owner IDs must be integers")
    if not isinstance(binding["queue_db"], str) or not binding["queue_db"]:
        raise ValueError("queue must be an absolute path")
    if not isinstance(binding["room_id"], str) or not binding["room_id"].strip():
        raise ValueError("room_id must be a nonempty string")
    queue_db = Path(binding["queue_db"])
    if chat_id >= 0 or owner_id <= 0 or not queue_db.is_absolute():
        raise ValueError("hosted room binding needs a private group, its owner and an absolute queue")
    if not isinstance(binding.get("bots"), dict) or not binding["bots"]:
        raise ValueError("hosted room binding must name each member's bot")
    from hermes_cli.profiles import validate_profile_name

    seen_ids, seen_names = set(), set()
    for profile, identity in binding["bots"].items():
        if not isinstance(profile, str) or (profile != "default" and validate_profile_name(profile)):
            raise ValueError("binding contains an invalid profile")
        if (not isinstance(identity, dict) or type(identity.get("id")) is not int
                or identity["id"] <= 0 or not isinstance(identity.get("username"), str)
                or not re.fullmatch(r"[A-Za-z0-9_]{5,32}", identity["username"])):
            raise ValueError("binding contains an invalid bot identity")
        if identity["id"] in seen_ids or identity["username"].lower() in seen_names:
            raise ValueError("members need distinct bot identities")
        seen_ids.add(identity["id"])
        seen_names.add(identity["username"].lower())
    if not isinstance(binding["control_profile"], str) or binding["control_profile"] not in binding["bots"]:
        raise ValueError("control_profile must be a bound member")
    return binding


def initialize_queue(path: Path, *, recover: bool = False) -> None:
    """Share additive migration; only the exclusive consumer recovers interrupted sends."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with closing(sqlite3.connect(path, timeout=10)) as db, db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(SCHEMA)
        db.execute("BEGIN IMMEDIATE")
        if not any(row[1] == "media_json" for row in db.execute("PRAGMA table_info(inbox)")):
            db.execute("ALTER TABLE inbox ADD COLUMN media_json TEXT")
        if recover:
            db.execute("UPDATE deliveries SET status='uncertain' WHERE status='sending'")
    path.chmod(0o600)


def _member_token(profile: str) -> str:
    """This member profile's own bot token, from its own secret scope."""
    from agent.secret_scope import build_profile_secret_scope
    from hermes_cli.profiles import get_profile_dir

    from hermes_constants import get_default_hermes_root

    home = get_default_hermes_root() if profile == "default" else Path(get_profile_dir(profile))
    if not home.is_dir():
        raise ValueError("hosted room member profile does not exist")
    token = build_profile_secret_scope(home).get("TELEGRAM_BOT_TOKEN")
    if not isinstance(token, str) or not token.strip():
        raise ValueError(f"member profile {profile} has no Telegram bot token")
    return token.strip()


class Transport:
    """One binding's durable input and delivery queue."""

    def __init__(self, service, binding: dict[str, Any]) -> None:
        self.service, self.config = service, binding
        self.path, self.room = Path(binding["queue_db"]), binding["room_id"]
        self.halt = threading.Event()
        self.ready = threading.Event()
        self.error: str | None = None
        self.thread: threading.Thread | None = None
        self._mutex = None
        self.member_profiles: dict[str, str] = {}
        self.profile_handles: dict[str, str] = {}

    def start(self) -> None:
        """The lifecycle owner retains this handle before any resource is acquired."""
        if self.thread is not None:
            if self.thread.is_alive() and self.ready.is_set() and not self.error and not self.halt.is_set():
                return
            raise RuntimeError("hosted room transport must stop before restarting")
        from gateway.hosted_rooms import room_state
        from gateway.hosted_room_discussion import validate_room
        from gateway.status import _get_scope_lock_path

        room = validate_room(room_state(self.service.db_path, room_id=self.room),
                             local_profiles=self.config["bots"])
        for member in room.members:
            if member.target is None or member.target.get("kind") != "local":
                raise ValueError("Telegram binding requires local member profiles")
            self.member_profiles[member.member_id] = member.profile
            self.profile_handles[member.profile] = "@" + member.handle
        if set(self.profile_handles) != set(self.config["bots"]):
            raise ValueError("Telegram binding must match the canonical local roster")
        identity = f"{self.config['chat_id']}:{self.room}"
        # Native token-owner checks intentionally exclude serve. Reuse the namespace,
        # not that predicate. Never unlink: every contender must open the same inode.
        path = _get_scope_lock_path(_LOCK_SCOPE, identity).with_suffix(".mutex")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if path.stat().st_size == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError("hosted room transport binding is already owned") from exc
        self._mutex = handle
        initialize_queue(self.path, recover=True)
        self.thread = threading.Thread(target=self._run, name="hosted-room-telegram", daemon=True)
        self.thread.start()
        if not self.ready.wait(40):
            self.halt.set()
            raise RuntimeError("hosted room transport initialization did not finish")
        if self.error or not self.thread.is_alive():
            raise RuntimeError(f"hosted room transport initialization failed: {self.error or 'worker exited'}")

    def _run(self) -> None:
        try:
            asyncio.run(self.run())
        except BaseException as exc:
            self.error = type(exc).__name__
            logger.error("hosted room transport stopped: %s", self.error)
        finally:
            self.ready.set()

    @contextmanager
    def db(self):
        # sqlite3's context manager commits/rolls back, but does not close.
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            db.row_factory = sqlite3.Row
            with db:
                yield db

    def thread_for(self, row) -> str:
        if row["reply_to"]:
            with self.db() as db:
                found = db.execute(
                    "SELECT thread_id FROM deliveries WHERE message_id=? AND status=?",
                    (row["reply_to"], "sent")).fetchone()
                if found:
                    return found[0]
                found = db.execute(
                    "SELECT result FROM inbox WHERE message_id=? AND state=?",
                    (row["reply_to"], "accepted")).fetchone()
                if found:
                    return json.loads(found[0])["thread_id"]
            # A reply must not silently become a different conversation.
            raise ValueError("unmapped Telegram reply")
        return row["event_id"]

    def media_for(self, row):
        """Verified ``(reference, bytes)`` for one input's captured media.

        The receipt the ingress wrote is a claim about cached bytes, re-checked here against the
        bytes themselves and the room's own cap: the cache can change between capture and ingest,
        so this read is bounded independently. Media that could not be captured, or a reference
        that no longer holds, is terminal for this input -- admitting its caption alone would
        publish a message that reads as if the file had arrived.
        """
        from gateway.hosted_room_attachments import MAX_ATTACHMENT_BYTES

        raw = row["media_json"] if "media_json" in row.keys() else None
        if not raw:
            return []
        media = json.loads(raw)
        rejected = media.get("rejected")
        if rejected:
            raise ValueError(f"media not captured: {rejected.get('reason', 'unknown')}")
        verified = []
        for ref in media.get("attachments") or []:
            try:
                size, digest = int(ref["size"]), str(ref["sha256"])
                with Path(ref["path"]).open("rb") as handle:
                    data = handle.read(MAX_ATTACHMENT_BYTES + 1)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                raise ValueError(f"unusable media reference: {type(exc).__name__}") from exc
            if (len(data) != size or len(data) > MAX_ATTACHMENT_BYTES
                    or hashlib.sha256(data).hexdigest() != digest):
                raise ValueError("cached media no longer matches its receipt")
            verified.append((ref, data))
        return verified

    def ingest(self) -> None:
        with self.db() as db:
            rows = db.execute(
                "SELECT * FROM inbox WHERE state='pending' ORDER BY message_id LIMIT 25").fetchall()
        for row in rows:
            try:
                if (row["chat_id"] != self.config["chat_id"]
                        or row["user_id"] != self.config["owner_id"]):
                    raise ValueError("input receipt identity mismatch")
                thread_id = self.thread_for(row)
                media = self.media_for(row)
            except ValueError as exc:
                # Quarantine invalid routing or media, never invent a conversation. Only these
                # are terminal here; an uncertain service acceptance stays pending under its
                # original event id.
                with self.db() as db:
                    db.execute(
                        "UPDATE inbox SET state='rejected', result=? WHERE event_id=?",
                        (json.dumps({"error": str(exc)}), row["event_id"]))
                logger.warning("hosted room input quarantined event=%s", row["event_id"])
                continue
            if not media and row["text"].strip().split("@", 1)[0].lower() in {"/stop", "!stop"}:
                count = self.service.stop_room(self.room, cancel_id=row["event_id"])
                result = {"thread_id": thread_id, "control": "stop", "tasks": count}
            else:
                text = row["text"]
                # Public bot handles and native member handles name the same roster.
                for profile, identity in self.config["bots"].items():
                    text = re.sub(
                        "@" + re.escape(identity["username"]) + r"\b", self.profile_handles[profile], text,
                        flags=re.IGNORECASE)
                # The canonical upload is idempotent on the Telegram-derived upload id, so a
                # retried input stages the same bytes instead of a second attachment.
                attachments = [
                    {key: uploaded[key] for key in ("attachment_id", "kind", "name", "size", "mime")}
                    for uploaded in (
                        self.service.put_attachment(
                            room_id=self.room, upload_id=ref["upload_id"], kind=ref["kind"],
                            name=ref["name"], mime=ref["mime"], data=data)
                        for ref, data in media)]
                event = self.service.send(
                    room_id=self.room, event_id=row["event_id"],
                    payload={"text": text, "thread_id": thread_id,
                             **({"attachments": attachments} if attachments else {})},
                    actor_id=f"telegram:{row['user_id']}")
                result = {"thread_id": thread_id, "seq": event["seq"],
                          **({"attachments": [item["attachment_id"] for item in attachments]}
                             if attachments else {})}
            with self.db() as db:
                db.execute(
                    "UPDATE inbox SET state='accepted', result=? WHERE event_id=?",
                    (json.dumps(result), row["event_id"]))

    def events(self, cursor: int = 0) -> list[dict[str, Any]]:
        from gateway.hosted_rooms import MAX_LOG_LIMIT, read_events
        rows = []
        while True:
            page = read_events(
                self.service.db_path, room_id=self.room, since_seq=cursor, limit=MAX_LOG_LIMIT)
            rows.extend(page["events"])
            if not page["has_more"]:
                return rows
            next_cursor = int(page["cursor"])
            if next_cursor <= cursor:
                raise RuntimeError("non-advancing canonical cursor")
            cursor = next_cursor

    def reply_target(self, thread_id: str):
        with self.db() as db:
            rows = db.execute(
                "SELECT message_id,result FROM inbox WHERE state='accepted' ORDER BY message_id DESC"
            ).fetchall()
            for row in rows:
                if json.loads(row["result"])["thread_id"] == thread_id:
                    return row["message_id"]
            row = db.execute(
                "SELECT message_id FROM deliveries WHERE thread_id=? AND status='sent' ORDER BY rowid LIMIT 1",
                (thread_id,)).fetchone()
            return row[0] if row else None

    def published_media(self, event):
        """Committed attachment bytes for one room event, verified against its own manifest."""
        parts = []
        for entry in event["payload"].get("attachments") or []:
            stored = self.service.read_attachment(
                room_id=self.room, attachment_id=entry["attachment_id"],
                recipient_member_id=None, event_id=event["event_id"], viewer=True)
            fields = ("attachment_id", "kind", "name", "size", "mime")
            if ({key: stored.attachment[key] for key in fields}
                    != {key: entry[key] for key in fields}):
                raise ValueError("committed attachment metadata changed")
            if len(stored.data) != int(entry["size"]):
                raise ValueError("committed attachment bytes changed")
            parts.append((stored.attachment, stored.data))
        return parts

    async def send_media(self, bot, attachment, data, reply_id):
        """Deliver one committed attachment as the picture it is, or as the file it is."""
        from telegram import ReplyParameters

        common = dict(
            chat_id=self.config["chat_id"], read_timeout=60, write_timeout=60, connect_timeout=15,
            reply_parameters=ReplyParameters(message_id=reply_id, allow_sending_without_reply=True)
            if reply_id else None)
        if attachment["kind"] == "image":
            return await bot.send_photo(photo=bytes(data), **common)
        return await bot.send_document(document=bytes(data), filename=attachment["name"], **common)

    async def publish(self, bots) -> None:
        from telegram import LinkPreviewOptions, ReplyParameters

        with self.db() as db:
            cursor = db.execute("SELECT seq FROM cursor WHERE singleton=1").fetchone()[0]
        events = self.events(cursor)
        committed = {
            event["payload"].get("message_event_id") for event in events
            if event["kind"] == "turn.settled"}
        for event in events:
            if event["seq"] <= cursor:
                continue
            payload, kind = event["payload"], event["kind"]
            text, profile = None, self.config["control_profile"]
            thread_id = payload.get("thread_id", "")
            if kind == "message.member":
                if event["event_id"] not in committed:
                    return
                profile, text = self.member_profiles[payload["member_id"]], payload["text"]
            elif kind == "message.user" and event["actor"].get("id") != f"telegram:{self.config['owner_id']}":
                # Anything the room accepted from another surface is rendered with its own actor,
                # never re-sent to the chat it came from.
                text = f"{event['actor'].get('id', 'user')}:\n{payload['text']}"
            elif kind == "room.stop_requested":
                text = "Room work was paused."
            # An attachment-only message has no text and still has to reach the group.
            media = self.published_media(event) if text is not None else []
            if text or media:
                if profile not in bots:
                    raise ValueError("unknown canonical speaker")
                reply_id = self.reply_target(thread_id)
                # 1800 code points fit Telegram's UTF-16 limit, including emoji. Media follows the
                # text in the SAME receipt space, so each part is acknowledged and an uncertain
                # one still blocks the rest.
                parts = [("text", (text or "")[start:start + 1800])
                         for start in range(0, len(text or ""), 1800)]
                parts.extend(("media", item) for item in media)
                for index, (part_kind, part) in enumerate(parts):
                    with self.db() as db:
                        existing = db.execute(
                            "SELECT status FROM deliveries WHERE event_id=? AND chunk_index=?",
                            (event["event_id"], index)).fetchone()
                        if existing:
                            if existing[0] == "sent":
                                continue
                            if existing[0] != "retry_authorized":
                                raise RuntimeError("uncertain Telegram delivery requires readback")
                            db.execute(
                                "UPDATE deliveries SET status='sending' WHERE event_id=? AND chunk_index=?",
                                (event["event_id"], index))
                        else:
                            db.execute(
                                "INSERT INTO deliveries(event_id,chunk_index,profile,thread_id,status)"
                                " VALUES(?,?,?,?,?)",
                                (event["event_id"], index, profile, thread_id, "sending"))
                    try:
                        if part_kind == "media":
                            message = await self.send_media(bots[profile], part[0], part[1], reply_id)
                        else:
                            message = await bots[profile].send_message(
                                chat_id=self.config["chat_id"], text=part, parse_mode=None,
                                link_preview_options=LinkPreviewOptions(is_disabled=True),
                                # Telegram hides other bots' messages from the Bot API, so a peer
                                # may not be able to reply to a bot-rendered root; the canonical
                                # thread mapping is persisted for every returned message id.
                                reply_parameters=ReplyParameters(
                                    message_id=reply_id, allow_sending_without_reply=True)
                                if reply_id else None,
                                read_timeout=30, write_timeout=30, connect_timeout=15)
                    except BaseException:
                        with self.db() as db:
                            db.execute(
                                "UPDATE deliveries SET status='uncertain' WHERE event_id=? AND chunk_index=?",
                                (event["event_id"], index))
                        raise
                    with self.db() as db:
                        db.execute(
                            "UPDATE deliveries SET status='sent',message_id=? WHERE event_id=? AND chunk_index=?",
                            (message.message_id, event["event_id"], index))
            with self.db() as db:
                db.execute("UPDATE cursor SET seq=? WHERE singleton=1", (event["seq"],))

    async def run(self) -> None:
        from telegram import Bot

        bots: dict[str, Any] = {}
        try:
            async def initialize():
                for profile, identity in self.config["bots"].items():
                    bot = Bot(_member_token(profile))
                    bots[profile] = bot
                    await bot.initialize()
                    me = await bot.get_me()
                    if me.id != identity["id"] or me.username != identity["username"]:
                        raise ValueError("member bot identity mismatch")
            await asyncio.wait_for(initialize(), timeout=30)
            self.ready.set()
            logger.warning("hosted room transport ready chat=%s", self.config["chat_id"])
            while not self.halt.is_set():
                try:
                    self.ingest()
                    await self.publish(bots)
                except Exception as exc:
                    logger.error("hosted room transport iteration failed: %s", type(exc).__name__)
                await asyncio.sleep(0.25)
        finally:
            for bot in bots.values():
                try:
                    await asyncio.wait_for(bot.shutdown(), timeout=5)
                except Exception as exc:
                    logger.error("hosted room bot cleanup failed: %s", type(exc).__name__)

    def stop(self, *, timeout: float = 5.0) -> bool:
        """Retain ownership until the actual worker and its HTTP clients have ceased."""
        self.halt.set()
        thread = self.thread
        if thread is not None and thread.ident is not None:
            thread.join(timeout)
            if thread.is_alive():
                return False
        self.thread = None
        if self._mutex is not None:
            self._mutex.close()
            self._mutex = None
        return True

