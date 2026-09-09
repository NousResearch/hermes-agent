"""Reply-specific native input receipts; never a next-message capture registry."""

from __future__ import annotations

import secrets
import sqlite3
import time
import os
from contextlib import closing, contextmanager
from pathlib import Path

from agent.i18n import t

MAX_PENDING = 128
INPUT_TIMEOUT = 300


def text(key, **values):
    return t("gateway.group_compose." + key, **values)


class ReplyInput:
    def __init__(self, adapter, on_reply):
        self.adapter, self.on_reply = adapter, on_reply
        self.token = secrets.token_hex(8)
        self.deadline = time.monotonic() + INPUT_TIMEOUT
        self.prompt_id = None
        self.chat_id = None
        self.bot_id = None
        self.pending = True
        self.path = Path(adapter._session_store.sessions_dir) / "native_reply_inputs.sqlite3"

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        except FileExistsError:
            pass
        with closing(sqlite3.connect(self.path, timeout=5)) as conn, conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS native_reply_inputs (
                token TEXT PRIMARY KEY, state TEXT NOT NULL, event_id TEXT, result TEXT,
                bot_id TEXT, chat_id TEXT, prompt_id TEXT
            )""")
            conn.execute("CREATE INDEX IF NOT EXISTS native_reply_prompt ON native_reply_inputs(bot_id, chat_id, prompt_id)")
            yield conn

    def register(self):
        pending = getattr(self.adapter, "_native_reply_inputs", None)
        if not isinstance(pending, dict):
            pending = self.adapter._native_reply_inputs = {}
        for token, request in list(pending.items()):
            if request.deadline <= time.monotonic():
                pending.pop(token, None)
        if len(pending) >= MAX_PENDING:
            raise RuntimeError("native input limit")
        # Retain ID-only tombstones: old replies must not turn into ordinary agent turns.
        with self._connect() as conn:
            conn.execute("INSERT INTO native_reply_inputs(token, state) VALUES (?, 'pending')", (self.token,))
        pending[self.token] = self

    def bind_prompt(self, prompt_id):
        with self._connect() as conn:
            conn.execute(
                "UPDATE native_reply_inputs SET bot_id=?, chat_id=?, prompt_id=? WHERE token=?",
                (self.bot_id, self.chat_id, str(prompt_id), self.token),
            )
        self.prompt_id = str(prompt_id)

    def claim(self, event_id):
        with self._connect() as conn:
            updated = conn.execute(
                "UPDATE native_reply_inputs SET state='claimed', event_id=? WHERE token=? AND state='pending'",
                (event_id, self.token),
            ).rowcount
            row = conn.execute(
                "SELECT state, event_id, result FROM native_reply_inputs WHERE token=?", (self.token,)
            ).fetchone()
        if updated:
            self.pending = False
            return None
        if row and row[0] == "done" and row[1] == event_id:
            return row[2] or text("closed")
        return text("closed")

    def finish(self, event_id, result):
        with self._connect() as conn:
            conn.execute(
                "UPDATE native_reply_inputs SET state='done', result=? WHERE token=? AND event_id=?",
                (result, self.token, event_id),
            )

    def cancel(self):
        self.deadline = 0
        self.pending = False
        with self._connect() as conn:
            return bool(conn.execute("UPDATE native_reply_inputs SET state='closed' WHERE token=? AND state='pending'", (self.token,)).rowcount)

    async def respond(self, runner, event):
        if (
            self.deadline <= time.monotonic()
            or not self.prompt_id
            or str(event.reply_to_message_id) != self.prompt_id
            or str(event.source.chat_id) != self.chat_id
            or runner._adapter_for_source(event.source) is not self.adapter
            or not event.allow_gateway_control
        ):
            return text("closed")
        return await self.on_reply(event, self)


class NativeReplySubmission:
    """In-process ingress marker, not serializable event metadata or authority."""

    def __init__(self, adapter, token, valid):
        self.adapter, self.token, self.valid = adapter, token, valid

    async def respond(self, runner, event):
        request = getattr(self.adapter, "_native_reply_inputs", {}).get(self.token)
        if not self.valid or request is None:
            return text("closed")
        return await request.respond(runner, event)


def prompt_token(adapter, bot_id, chat_id, prompt_id):
    store = getattr(adapter, "_session_store", None)
    if store is None:
        return None
    path = Path(store.sessions_dir) / "native_reply_inputs.sqlite3"
    if not path.exists():
        return None
    with closing(sqlite3.connect(path, timeout=5)) as conn:
        row = conn.execute(
            "SELECT token FROM native_reply_inputs WHERE bot_id=? AND chat_id=? AND prompt_id=?",
            (str(bot_id), str(chat_id), str(prompt_id)),
        ).fetchone()
    return row[0] if row else None


async def handle_native_reply(runner, event, submission):
    if not isinstance(submission, NativeReplySubmission):
        return None
    try:
        return await submission.respond(runner, event)
    except Exception:
        # A consumed native reply must NEVER fall through to the conversational agent.
        return text("unknown")
