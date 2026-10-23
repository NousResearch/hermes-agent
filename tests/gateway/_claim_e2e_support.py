"""Shared support for the channel-claim e2e replay (t_24b1959a).

Replays the 2026-10-23 incident pattern on channel 8grzznya3trpbfxjwf5goe778h:
a real GatewayRunner + MattermostAdapter + SessionAuthority over real SQLite,
with only the model transport stubbed (deterministic answers, wall-clock delays
for heartbeat coverage). Helpers here are deliberately free of imports from the
claim-lock surface so the same harness runs RED on pre-claim-lock trees.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

# ── Incident constants ────────────────────────────────────────────────────────
CHANNEL = "8grzznya3trpbfxjwf5goe778h"          # the DM channel from the incident
USER = "gweslien"                                 # incident sender id
INCIDENT_ROOT = "mircy79x43fmudxbtdjz4kknjw"     # incident thread root post id
SLOW_TOKEN = " SLOW"                             # triggers the stub's delay


def answer_for(text: str) -> str:
    """Deterministic assistant answer for a prompt (the one seam the harness controls).

    The first-ever session message carries an appended onboarding ``[System note: ...]``
    block the harness must not fold into the hash — the answer identifies the USER text.
    """
    core = text.split("[System note:", 1)[0] if "[System note:" in text else text
    last_user = _last_user_line(core) or core
    return "ANSWER::" + hashlib.sha256(last_user.encode()).hexdigest()[:10]


def _last_user_line(text: str) -> str | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def write_home_config(home: Path) -> None:
    """Minimal hermetic profile-home config: stub custom provider, memory off."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: stub-glm-5.2\n"
        "  provider: custom\n"
        "memory:\n"
        "  memory_enabled: false\n",
        encoding="utf-8",
    )


def mm_ws_event(post_id: str, text: str, *, root_id: str | None = None,
                channel_id: str = CHANNEL, user_id: str = USER,
                channel_type: str = "D") -> dict:
    """Raw Mattermost websocket payload exactly as _handle_ws_event expects it."""
    post = {
        "id": post_id,
        "user_id": user_id,
        "channel_id": channel_id,
        "root_id": root_id or "",
        "message": text,
        "type": "",
        "create_at": int(time.time() * 1000),
    }
    return {
        "event": "posted",
        "data": {
            "post": json.dumps(post),
            "channel_type": channel_type,
            "sender_name": user_id,
        },
        "broadcast": {"channel_id": channel_id},
    }


class CaptureHandler(logging.Handler):
    """In-memory journal: every propagated log record, kept in arrival order."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[tuple[str, str, str]] = []   # (logger, levelname, message)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append((record.name, record.levelname, record.getMessage()))
        except Exception:
            pass

    def lines(self, *needles: str) -> list[str]:
        out = []
        for name, level, msg in self.records:
            haystack = f"{name} {msg}"
            if all(n in haystack for n in needles):
                out.append(f"[{level}] {name}: {msg}")
        return out


def install_journal() -> CaptureHandler:
    """Attach the capture handler to root and the gateway/agent logger trees.

    Gateway runtime logging may attach its own handlers; wiring the collector
    to the named loggers as well keeps capture independent of propagation.
    """
    handler = CaptureHandler()
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)
    for name in ("gateway", "agent", "plugins", "tools", "hermes_state_runtime"):
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        if lg.level == logging.NOTSET or lg.level > logging.DEBUG:
            lg.setLevel(logging.DEBUG)
    return handler


def remove_journal(handler: CaptureHandler) -> None:
    for lg in (logging.getLogger(), *(logging.getLogger(n) for n in
                                      ("gateway", "agent", "plugins", "tools", "hermes_state_runtime"))):
        try:
            lg.removeHandler(handler)
        except Exception:
            pass


class SendLog:
    """Recorded outbound surface of one adapter (send + edit_message), ordered."""

    def __init__(self) -> None:
        self.sends: list[dict] = []
        self.edits: list[dict] = []
        self._n = 0

    def next_message_id(self) -> str:
        self._n += 1
        return f"mm-{self._n:04d}"

    def record_send(self, chat_id, content, reply_to=None, metadata=None) -> dict:
        entry = {
            "kind": "send",
            "chat_id": chat_id,
            "content": str(content),
            "reply_to": reply_to,
            "meta_thread": (metadata or {}).get("thread_id") if metadata else None,
            "message_id": self.next_message_id(),
        }
        self.sends.append(entry)
        return entry

    def record_edit(self, chat_id, message_id, content, finalize=False) -> dict:
        entry = {
            "kind": "edit",
            "chat_id": chat_id,
            "content": str(content),
            "message_id": message_id,
            "finalize": finalize,
        }
        self.edits.append(entry)
        return entry

    # ── queries ──────────────────────────────────────────────────────────────
    def _bubble_finals(self) -> dict[str, str]:
        """message_id -> latest known content (send first, edits overwrite)."""
        ids: dict[str, str] = {}
        for s in self.sends:
            ids[s["message_id"]] = s["content"]
        for e in self.edits:      # arrival order preserves latest-edit semantics
            ids.setdefault(e["message_id"], e["content"])
            ids[e["message_id"]] = e["content"]
        return ids

    def final_answers(self) -> list[str]:
        return [c for c in self._bubble_finals().values() if "ANSWER::" in c]

    def answers_for(self, text: str) -> list[str]:
        want = answer_for(text)
        return [c for c in self._bubble_finals().values() if want in c]

    def heartbeat_edits(self) -> list[dict]:
        return [e for e in self.edits if "Working" in e["content"] or "iteration" in e["content"].lower()]

    def edits_of(self, message_id: str) -> list[dict]:
        return [e for e in self.edits if e["message_id"] == message_id]


def patch_adapter_recording(monkeypatch, adapter, log: SendLog) -> None:
    """Replace the delivery surface with the recorder; keep SendResult contract."""

    async def send(chat_id, content, reply_to=None, metadata=None):
        entry = log.record_send(chat_id, content, reply_to, metadata)
        from gateway.platforms.base import SendResult

        return SendResult(success=True, message_id=entry["message_id"])

    async def edit_message(chat_id, message_id, content, *, finalize: bool = False):
        log.record_edit(chat_id, message_id, content, finalize)
        from gateway.platforms.base import SendResult

        return SendResult(success=True, message_id=message_id)

    monkeypatch.setattr(adapter, "send", send)
    monkeypatch.setattr(adapter, "edit_message", edit_message)


# ── Model transport stub ─────────────────────────────────────────────────────
def _role_of(m) -> str:
    if isinstance(m, dict):
        return str(m.get("role", ""))
    return str(getattr(m, "role", "") or "")


def _content_to_text(c) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict):
                parts.append(str(p.get("text") or p.get("content") or ""))
            elif isinstance(p, str):
                parts.append(p)
            else:
                parts.append(str(getattr(p, "text", None) or getattr(p, "content", None) or p))
        return "\n".join(t for t in parts if t)
    if isinstance(c, dict):
        return str(c.get("text") or c.get("content") or "")
    return str(c)


def _message_content(m) -> str:
    if isinstance(m, dict):
        return _content_to_text(m.get("content"))
    return _content_to_text(getattr(m, "content", None))


def patch_model_transport(monkeypatch, *, slow_seconds: float = 3.2) -> dict:
    """Stub openai chat.completions.create (async + sync) with deterministic answers.

    A prompt whose LAST user line contains the SLOW marker sleeps (async path)
    so the heartbeat ticker emits multiple ticks inside one turn.
    Returns the call counter dict for assertions.
    """
    counters = {"async": 0, "sync": 0}

    def _last_user_content(kwargs) -> str:
        # #93650 bypass: the wire moves bulk fields into extra_body and leaves an
        # empty-list placeholder in the typed kwargs — fold both views.
        messages = kwargs.get("messages")
        if not messages and isinstance(kwargs.get("extra_body"), dict):
            messages = kwargs["extra_body"].get("messages") or messages
        last = ""
        for m in reversed(list(messages or [])):
            if _role_of(m) == "user":
                last = _message_content(m)
                break
        if not last:
            for m in reversed(list(messages or [])):
                if _role_of(m) != "system":
                    last = _message_content(m)
                    break
        return last

    def _answer(kwargs) -> str:
        return answer_for(_last_user_content(kwargs))

    async def async_create(self, **kwargs):
        counters["async"] += 1
        if SLOW_TOKEN.strip() in (_last_user_content(kwargs) or ""):
            await asyncio.sleep(slow_seconds)  # health: allow HX013 -- the slow model IS the behavior under test: it keeps the turn in flight to expose the double-admit
        # Always return a full completion: hermes detects "stream request returned
        # a final response object" and switches the session to non-streaming
        # (agent/chat_completion_helpers.py); the harness then reads its answers
        # off the delivered surface, exactly like the incident journal did.
        return _completion_payload(_answer(kwargs), kwargs.get("model", ""))

    def sync_create(self, **kwargs):
        counters["sync"] += 1
        if SLOW_TOKEN.strip() in (_last_user_content(kwargs) or ""):
            time.sleep(slow_seconds)  # health: allow HX013 -- the slow model IS the behavior under test: it keeps the turn in flight to expose the double-admit
        return _completion_payload(_answer(kwargs), kwargs.get("model", ""))

    from openai.resources.chat.completions import AsyncCompletions, Completions

    monkeypatch.setattr(AsyncCompletions, "create", async_create)
    monkeypatch.setattr(Completions, "create", sync_create)
    return counters


def _completion_payload(answer: str, model: str = ""):
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate({
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "created": 1,
        "model": model or "stub-glm-5.2",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    })


# ── SQLite inspection (tolerant of pre-claim-lock schemas) ────────────────────
def state_db_path():
    import hermes_state

    return Path(hermes_state.DEFAULT_DB_PATH)


def all_state_dbs() -> list[Path]:
    """Every state.db under the isolated home (root + profile-scoped stores)."""
    root = state_db_path().parent
    found: list[Path] = []
    seen: set[Path] = set()
    for path in [state_db_path(), *sorted(root.glob("**/state.db"))]:
        if path in seen or ".venv" in path.parts:
            continue
        seen.add(path)
        if path.exists():
            found.append(path)
    return found


def db_rows(table: str, where: str = "1=1", params: tuple = ()) -> list[dict]:
    """Read rows from any state store under the isolated home; missing tables read as []."""
    for path in all_state_dbs():
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        try:
            try:
                cur = conn.execute(f"SELECT * FROM {table} WHERE {where}", params)
            except sqlite3.OperationalError:
                continue
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()
    return []


def session_keys_for_channel(channel_id: str = CHANNEL) -> set[str]:
    rows = db_rows("sessions", "session_key LIKE ?", (f"%{channel_id}%",))
    return {r["session_key"] for r in rows}


def channel_claim_rows() -> list[dict]:
    return db_rows("channel_claims")


def compression_block_lines(journal) -> list[str]:
    """Per-session compression-config block lines seen in the journal.

    Accepts a CaptureHandler or a list of (name, level, msg) triples
    (build one from pytest's caplog.records — that capture survives the
    plugin loader's logging reconfiguration; a fresh CaptureHandler does not).
    """
    records = journal.records if hasattr(journal, "records") else journal
    needles = ("ompres",)
    out = []
    for name, level, msg in records:
        if any(n in msg.lower() for n in needles) and (
            "threshold" in msg.lower() or "session=" in msg.lower() or "auto-lowered" in msg.lower()
        ):
            out.append(f"[{level}] {name}: {msg}")
    return out


async def settle(seconds: float = 0.25) -> None:
    """Drain window: the elapsed-quiet delay IS the assertion for callers that
    verify no stray sends/sessions appeared."""
    await asyncio.sleep(seconds)  # health: allow HX013 -- absence window: elapsed-quiet with no new sends IS the assertion of settle() callers
