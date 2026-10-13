"""Session bridge: mirror local (non-Matrix) Hermes sessions into ONE Matrix room.

Desktop/TUI/webui-born sessions never touch Matrix, so FluxChat clients can't see or
steer them. This bridge mirrors ALL of them into one room on the configured homeserver
— one Matrix thread per session (thread root = session announcement) — and routes
thread replies back into the session as a real turn, mirroring what the desktop
sidebar shows (user + final agent replies; no tool noise).

Room resolution NEVER keys off the room title: the session-hosting room is marked with
the ``org.hermes.sessions`` state event (``{"version": 1, "sources": [...]}``). Config
may pin the room by ID + expected marker; when unset, the bridge creates the room, sets
the marker itself, and persists the room id in its state file. Room title is cosmetic.

Wire shape reuses the FluxChat agent protocol (``org.fluxchat.agent`` hidden field,
settle-edit semantics) exactly as the Matrix adapter emits it, so both FluxChat clients
render mirrored sessions with zero client changes.

Design notes:
- The bridge owns its own mautrix client (same access token as the gateway's Matrix
  adapter → same @fbhermes identity). Its outbound events are self-sent, so the
  gateway adapter's ``_is_self_sender`` drop keeps them from re-entering as input; the
  adapter additionally drops foreign room events for the bridge room (see
  ``is_bridge_room``) so the gateway's own Matrix session machinery never double-answers
  there. The bridge room stays unencrypted (the bridge creates it unencrypted).
- INPUT is only foreign plain m.text events inside a KNOWN session thread. Everything
  the bridge itself sends (roots, working bubbles, settle edits, mirrored replies) is
  OUTPUT. Session execution for a remote reply is a one-turn resume subprocess
  (``hermes -p <profile> chat --resume <session> -Q``), the same transport family the
  bot relay uses; the appended transcript rows ride back out through the poller.
- Sessions with zero user messages (drafts) never get threads; matrix-origin sessions
  and internal listing sources are excluded (they already live in Matrix elsewhere).

Heuristics worth knowing (poll-only observation — no desktop turn hooks exist):
- A turn settles DONE when a final assistant row (non-empty content, no tool_calls,
  not a compression summary) appears after a user row; the working bubble edits to the
  reply's first line as the protocol ``result`` headline and the full reply posts as
  its own thread reply.
- A turn settles FAILED when the assistant row carries an error ``finish_reason`` or
  when a bubble goes stale (no settle observed within ``_BUBBLE_STALE_SECONDS``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

AGENT_STATUS_KEY = "org.fluxchat.agent"
MARKER_EVENT_TYPE = "org.hermes.sessions"
MARKER_VERSION = 1
DEFAULT_TITLE = "Hermes Sessions"
DEFAULT_SOURCES = ("desktop", "tui", "webui")
INTERNAL_SOURCES = ("kanban", "tool", "oneshot", "subagent", "cron", "matrix", "unknown")
_POLL_SECONDS = 3.0
_SYNC_TIMEOUT_MS = 30_000
_BUBBLE_STALE_SECONDS = 45 * 60
_TURN_TIMEOUT_SECONDS = 1800
_MAX_QUEUED_TURNS_PER_SESSION = 5
_MAX_CONCURRENT_TURNS = 2
_BODY_CAP = 16_000  # mirror's own body cap, well under Matrix event limits

# ---------------------------------------------------------------------------
# Pure mapping logic (unit-testable, no I/O)
# ---------------------------------------------------------------------------

_RESULT_STRIP_RE = re.compile(r"^[#>\- ]+")


def result_line(response: Any) -> str:
    """First non-empty line of a reply, trimmed to one result line (protocol ``result``).

    Mirrors ``gateway.agent_status._result_line`` so mirrored sessions headline exactly
    like gateway-origin runs; kept local so the bridge imports no gateway runner state.
    """
    text = str(response or "").strip()
    for line in text.splitlines():
        if line := line.strip():
            # Heading/list markers lead; bold markers wrap the lead-in ("**Done.**").
            line = _RESULT_STRIP_RE.sub("", line).strip()
            if line.startswith("**") and line.endswith("**"):
                line = line[2:-2].strip()
            return line[:240]
    return ""


def should_mirror_session(row: dict, sources: tuple[str, ...], *, backfill_days: float,
                          now: float) -> bool:
    """Sidebar-parity gate for a sessions-table row: mirror sources only, real content,
    not archived/hidden, inside the backfill window (fresh rows always pass)."""
    src = str(row.get("source") or "").strip().lower()
    if src not in {s.strip().lower() for s in sources} or src in INTERNAL_SOURCES:
        return False
    if row.get("archived") or row.get("hidden"):
        return False
    if int(row.get("message_count") or 0) <= 0:
        return False
    started = float(row.get("started_at") or 0.0)
    last = float(row.get("last_activity_at") or started or 0.0)
    return now - last <= backfill_days * 86_400


def visible_user_rows(rows: list[dict]) -> list[dict]:
    """Rows a human sees as their own messages: role=user, real content."""
    return [r for r in rows
            if r.get("role") == "user" and str(r.get("content") or "").strip()]


def final_reply_rows(rows: list[dict]) -> list[dict]:
    """Final assistant messages (sidebar/transcript parity: no tool-call carriers, no
    empty content, no compression summaries)."""
    out = []
    for r in rows:
        if r.get("role") != "assistant" or int(r.get("_compressed_summary") or 0):
            continue
        if not str(r.get("content") or "").strip():
            continue
        tcs = str(r.get("tool_calls") or "").strip()
        if tcs and tcs not in ("[]", "null"):
            continue
        out.append(r)
    return out


def error_reply_rows(rows: list[dict]) -> list[dict]:
    """Assistant rows whose finish_reason marks a provider/agent error."""
    return [r for r in rows
            if r.get("role") == "assistant"
            and str(r.get("finish_reason") or "") in ("error", "agent_error", "content_filter")]


@dataclass
class TurnDelta:
    """What the poller observed since its last cursor for one session."""
    new_user_rows: list[dict] = field(default_factory=list)
    final_replies: list[dict] = field(default_factory=list)
    error_replies: list[dict] = field(default_factory=list)


def detect_turn_delta(rows: list[dict], last_msg_id: int) -> TurnDelta:
    """Split rows after ``last_msg_id`` into mirrorable streams (pure)."""
    after = [r for r in rows if int(r["id"]) > last_msg_id]
    return TurnDelta(
        new_user_rows=visible_user_rows(after),
        final_replies=final_reply_rows(after),
        error_replies=error_reply_rows(after),
    )


def thread_reply_content(body: str, root: str, *, agent_status: Optional[dict] = None) -> dict:
    """One m.text event body threaded under ``root`` (adapter's relation shape)."""
    content: dict[str, Any] = {"msgtype": "m.text", "body": body[:_BODY_CAP]}
    if root:
        content["m.relates_to"] = {
            "rel_type": "m.thread", "event_id": root, "is_falling_back": True,
            "m.in_reply_to": {"event_id": root},
        }
    if agent_status:
        content[AGENT_STATUS_KEY] = agent_status
    return content


def status_edit_content(new_body: str, target_event: str, status: dict) -> dict:
    """Settle edit of a tracked bubble (m.replace over m.new_content, adapter shape)."""
    new_content: dict[str, Any] = {"msgtype": "m.text", "body": new_body[:_BODY_CAP],
                                   AGENT_STATUS_KEY: status}
    return {
        "msgtype": "m.text", "body": f"* {new_body[:_BODY_CAP]}",
        "m.new_content": new_content,
        "m.relates_to": {"rel_type": "m.replace", "event_id": target_event},
        AGENT_STATUS_KEY: status,
    }


def working_status(run: str) -> dict:
    return {"v": 1, "state": "working", "run": run}


def settled_status(run: str, reply: str, *, failed: bool = False, model: str = "",
                   cost_usd: Optional[float] = None) -> dict:
    status: dict[str, Any] = {"v": 1, "state": "failed" if failed else "done", "run": run}
    if line := result_line(reply):
        status["result"] = line
    if model:
        status["model"] = model
    if isinstance(cost_usd, (int, float)) and cost_usd >= 0:
        # Canonical JSON (unencrypted rooms) rejects floats outright, and FluxChat
        # desktop's serde needs a JSON number (a string kills the whole protocol parse).
        # Send the protocol field only when it's integral; otherwise fold the exact cost
        # into the ``result`` headline (still ${:.2}-readable, just not in the chip).
        cents = round(float(cost_usd) * 100)
        if cents % 100 == 0:
            status["cost_usd"] = cents // 100
        else:
            exact = f"${cents / 100:.2f}"
            status["result"] = f"{status['result']} ({exact})" if status.get("result") else exact
    return status


def root_announcement(title: str, source: str, started_at: float) -> str:
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(started_at)) if started_at else "unknown time"
    return f"🧵 {title or 'Untitled session'} — source: {source} · started {when}"


def user_message_body(localpart: str, source: str, text: str) -> str:
    return f"**{localpart} ({source}):** {text[:_BODY_CAP]}"


def route_inbound_event(sender: str, own_mxid: str, content: dict,
                        session_roots: dict[str, str],
                        controllers: frozenset[str] = frozenset()) -> Optional[str]:
    """INPUT routing: a controller's plain-text reply inside a KNOWN session thread
    returns that session id; anything else (self-sent, non-controller, edits, notices,
    protocol events, unknown threads, room-level chatter) is not session input → None.
    Loop guard lives here."""
    if not sender or (own_mxid and sender.lower() == own_mxid.lower()):
        return None
    if controllers and sender.lower() not in {c.lower() for c in controllers}:
        return None
    if content.get("msgtype") != "m.text":
        return None
    if AGENT_STATUS_KEY in content:  # protocol events are output vocabulary
        return None
    if (content.get("m.relates_to") or {}).get("rel_type") == "m.replace":
        return None  # edits never steer
    relates = content.get("m.relates_to") or {}
    if relates.get("rel_type") != "m.thread":
        return None
    return session_roots.get(str(relates.get("event_id") or ""))


# ---------------------------------------------------------------------------
# Persisted bridge state
# ---------------------------------------------------------------------------

@dataclass
class SessionThread:
    root: str = ""
    last_msg_id: int = 0
    source: str = ""
    title: str = ""
    profile: str = "default"
    created_at: float = field(default_factory=time.time)
    # open turns: run_id -> {"bubble": event_id, "opened_at": ts, "user_row": message id}
    open: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"root": self.root, "last_msg_id": self.last_msg_id, "source": self.source,
                "title": self.title, "profile": self.profile, "created_at": self.created_at,
                "open": self.open}

    @classmethod
    def from_dict(cls, raw: dict) -> SessionThread:
        return cls(root=str(raw.get("root") or ""), last_msg_id=int(raw.get("last_msg_id") or 0),
                   source=str(raw.get("source") or ""), title=str(raw.get("title") or ""),
                   profile=str(raw.get("profile") or "default"),
                   created_at=float(raw.get("created_at") or 0) or time.time(),
                   open=dict(raw.get("open") or {}))


class BridgeState:
    """JSON state file: room id + per-session thread bookkeeping. Atomic writes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.room_id: str = ""
        self.sessions: dict[str, SessionThread] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.room_id = str(raw.get("room_id") or "")
        self.sessions = {sid: SessionThread.from_dict(t)
                         for sid, t in (raw.get("sessions") or {}).items()
                         if isinstance(t, dict)}

    def save(self) -> None:
        payload = {"room_id": self.room_id,
                   "sessions": {sid: t.to_dict() for sid, t in self.sessions.items()}}
        tmp = self.path.with_suffix(".json.tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.path)


# Module-level room-id cache the Matrix adapter's drop hook consults (mtime-cached).
_room_cache: tuple[float, str] = (-1.0, "")


def _state_path() -> Path:
    from hermes_cli.config import get_hermes_home
    return get_hermes_home() / "platforms" / "matrix" / "store" / "session_bridge.json"


def is_bridge_room(room_id: str) -> bool:
    """True when ``room_id`` is the session-bridge room (Matrix adapter drop hook).
    Cheap: stat + mtime cache; False when the bridge has never run (no state file)."""
    global _room_cache
    path = _state_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return False
    if mtime != _room_cache[0]:
        try:
            room = str(json.loads(path.read_text(encoding="utf-8")).get("room_id") or "")
        except (OSError, ValueError):
            room = ""
        _room_cache = (mtime, room)
    return bool(room_id) and room_id == _room_cache[1]


# ---------------------------------------------------------------------------
# Bridge runtime (own Matrix client; runs inside gateway or standalone)
# ---------------------------------------------------------------------------

class SessionBridge:
    def __init__(self, *, homeserver: str, access_token: str,
                 sources: tuple[str, ...] = DEFAULT_SOURCES, room_id: str = "",
                 invite: tuple[str, ...] = (), controllers: tuple[str, ...] = (),
                 backfill_days: float = 7.0,
                 state_path: Optional[Path] = None, client: Any = None,
                 db_path: Optional[Path] = None) -> None:
        self._homeserver = homeserver
        self._token = access_token
        self.sources = tuple(s.strip().lower() for s in sources if s.strip())
        self.configured_room = room_id.strip()
        self.invite = tuple(i for i in invite if i.strip())
        # Only these accounts may steer sessions from the room (empty → anyone but self).
        self.controllers = frozenset(c for c in (x.strip() for x in controllers) if c)
        self.backfill_days = backfill_days
        self.state = BridgeState(state_path or _state_path())
        self._client = client  # injected in tests; built lazily in production
        self._own_mxid = ""
        self._db_path = db_path
        self._turn_locks: dict[str, asyncio.Lock] = {}
        self._queues: dict[str, asyncio.Queue] = {}
        self._drainers: dict[str, asyncio.Task] = {}
        self._sem = asyncio.Semaphore(_MAX_CONCURRENT_TURNS)
        self._running = True
        self._room_ready = asyncio.Event()

    # -- wiring ---------------------------------------------------------

    def _db(self) -> sqlite3.Connection:
        if self._db_path is None:
            from hermes_cli.config import get_hermes_home
            self._db_path = get_hermes_home() / "state.db"
        conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    async def _ensure_client(self):
        if self._client is None:
            from mautrix.api import HTTPAPI
            from mautrix.client import Client
            from mautrix.client.state_store import MemoryStateStore, MemorySyncStore
            from mautrix.types import UserID
            # Token auth needs the mxid BEFORE Client construction: resolve it via the
            # raw API (whoami) first — the adapter gets it from login; we have a token.
            api = HTTPAPI(base_url=self._homeserver, token=self._token)
            try:
                resp = await api.request("GET", "_matrix/client/v3/account/whoami")
                self._own_mxid = str(resp.get("user_id") or "")
            finally:
                session = getattr(api, "client_session", None)
                if session is not None:
                    await session.close()
            self._client = Client(mxid=UserID(self._own_mxid),
                                  device_id=os.environ.get("MATRIX_DEVICE_ID") or None,
                                  api=HTTPAPI(base_url=self._homeserver, token=self._token),
                                  state_store=MemoryStateStore(), sync_store=MemorySyncStore())
        if not self._own_mxid:
            whoami = await self._client.whoami()
            self._own_mxid = str(whoami.user_id)
        return self._client

    async def _ensure_room(self) -> str:
        """Resolve/create the session room; verify or set the org.hermes.sessions marker.
        Never trust the title — the state event is the identity."""
        from mautrix.types import EventType, RoomCreatePreset, RoomID, UserID
        client = await self._ensure_client()
        # mautrix EventType is a str subclass; the bare string serializes identically in
        # the /state/{type}/{key} path (t_class only matters for typed deserialization).
        marker: Any = MARKER_EVENT_TYPE
        content: dict[str, Any] = {"version": MARKER_VERSION, "sources": list(self.sources)}

        async def marker_present(room: str) -> bool:
            try:
                ev = await client.get_state_event(RoomID(room), marker)
            except Exception:
                return False
            raw = ev if isinstance(ev, dict) else getattr(ev, "content", None)
            return isinstance(raw, dict) and raw.get("version") == MARKER_VERSION

        room = self.configured_room or self.state.room_id
        if room:
            if await marker_present(room):
                self.state.room_id = room
                return room
            # Pinned/persisted room without the marker: set it (we hold power as creator
            # of our own room; a pinned foreign room may legitimately reject) and proceed.
            try:
                await client.send_state_event(RoomID(room), marker, content=content)
                self.state.room_id = room
                return room
            except Exception as exc:
                logger.warning("session-bridge: cannot set %s marker in %s: %s",
                               MARKER_EVENT_TYPE, room, exc)
        new_room = await client.create_room(
            name=DEFAULT_TITLE,
            invitees=[UserID(u) for u in self.invite],
            is_direct=False, preset=RoomCreatePreset.PRIVATE)
        room = str(new_room)
        await client.send_state_event(RoomID(room), marker, content=content)
        self.state.room_id = room
        self.state.save()
        logger.info("session-bridge: created session room %s with %s marker",
                    room, MARKER_EVENT_TYPE)
        return room

    # -- sends ----------------------------------------------------------

    async def _send(self, content: dict) -> Optional[str]:
        from mautrix.types import EventType, RoomID
        try:
            return str(await asyncio.wait_for(
                self._client.send_message_event(RoomID(self.state.room_id),
                                                EventType.ROOM_MESSAGE, content), timeout=45))
        except Exception as exc:
            logger.warning("session-bridge: send failed: %s", exc)
            return None

    async def _edit(self, target: str, content: dict) -> bool:
        from mautrix.types import EventType, RoomID
        try:
            await asyncio.wait_for(
                self._client.send_message_event(RoomID(self.state.room_id),
                                                EventType.ROOM_MESSAGE, content), timeout=45)
            return True
        except Exception as exc:
            logger.warning("session-bridge: edit of %s failed: %s", target, exc)
            return False

    # -- output poll loop -------------------------------------------------

    async def _poll_once(self) -> None:
        conn = self._db()
        try:
            rows = conn.execute(
                "select id, source, title, started_at, last_activity_at, message_count, "
                "archived, hidden, profile_name, model, estimated_cost_usd, actual_cost_usd, "
                "cost_status from sessions").fetchall()
            eligible = {}
            now = time.time()
            for row in rows:
                d = dict(row)
                if should_mirror_session(d, self.sources, backfill_days=self.backfill_days, now=now):
                    eligible[str(d["id"])] = d
            for sid, meta in eligible.items():
                try:
                    await self._sync_session(conn, sid, meta)
                except Exception as exc:
                    logger.debug("session-bridge: poll of %s failed: %s", sid, exc)
        finally:
            conn.close()
        self.state.save()

    async def _sync_session(self, conn: sqlite3.Connection, sid: str, meta: dict) -> None:
        thread = self.state.sessions.get(sid)
        if thread is None or not thread.root:
            first_user = conn.execute(
                "select 1 from messages where session_id = ? and role = 'user' "
                "and content is not null and content != '' limit 1", (sid,)).fetchone()
            if not first_user:  # draft rule: no thread before the first real user message
                return
            root = await self._send({"msgtype": "m.text", "body": root_announcement(
                meta.get("title") or "", meta.get("source") or "",
                float(meta.get("started_at") or 0))})
            if not root:
                return
            thread = SessionThread(root=root, source=str(meta.get("source") or ""),
                                   title=str(meta.get("title") or ""),
                                   profile=str(meta.get("profile_name") or "default"))
            self.state.sessions[sid] = thread
            logger.info("session-bridge: opened thread %s for session %s (%s)",
                        root, sid, thread.source)

        rows = [dict(r) for r in conn.execute(
            "select id, role, content, tool_calls, finish_reason, _compressed_summary, "
            "message_uid, timestamp from messages where session_id = ? order by id", (sid,)).fetchall()]
        delta = detect_turn_delta(rows, thread.last_msg_id)

        # Protocol rule (gateway.agent_status): cost rides only when it isn't unknown.
        session_cost = None
        if meta.get("cost_status") not in (None, "unknown"):
            for key in ("actual_cost_usd", "estimated_cost_usd"):
                if isinstance(meta.get(key), (int, float)):
                    session_cost = meta[key]
                    break
        model = str(meta.get("model") or "")

        for r in delta.new_user_rows:
            run = f"{sid}:{r.get('message_uid') or r['id']}"
            localpart = self._own_mxid.split(":")[0].lstrip("@") or "hermes"
            await self._send(thread_reply_content(
                user_message_body(localpart, thread.source, str(r.get("content") or "")),
                thread.root))
            # Working bubbles are for LIVE turns only: a first sync backfills old history
            # and every stale row would otherwise raise a phantom "working" bubble.
            if float(r.get("timestamp") or 0) >= thread.created_at - 1:
                bubble = await self._send(thread_reply_content(
                    "⏳ Working…", thread.root, agent_status=working_status(run)))
                if bubble:
                    thread.open[run] = {"bubble": bubble, "opened_at": time.time(),
                                        "user_row": int(r["id"])}

        # Settle passes in row order: error rows then final replies close open bubbles.
        for r in delta.error_replies:
            await self._settle_open_bubbles(thread, str(r.get("content") or "error"),
                                            failed=True, model=model, cost=None)
        for r in delta.final_replies:
            text = str(r.get("content") or "")
            await self._settle_open_bubbles(thread, text, failed=False, model=model,
                                            cost=session_cost)
            await self._send(thread_reply_content(text, thread.root))

        thread.last_msg_id = max([thread.last_msg_id] + [int(r["id"]) for r in rows])

        # Stale bubbles: desktop runs can die without ever appending a settle row.
        for run, info in list(thread.open.items()):
            if time.time() - float(info.get("opened_at") or 0) > _BUBBLE_STALE_SECONDS:
                await self._settle_bubble(thread, run, "no reply observed", failed=True,
                                          model=model, cost=None)
                thread.open.pop(run, None)

    async def _settle_open_bubbles(self, thread: SessionThread, reply: str, *, failed: bool,
                                   model: str, cost) -> None:
        for run in list(thread.open):
            await self._settle_bubble(thread, run, reply, failed=failed, model=model, cost=cost)
            thread.open.pop(run, None)

    async def _settle_bubble(self, thread: SessionThread, run: str, reply: str, *,
                             failed: bool, model: str, cost) -> None:
        info = thread.open.get(run)
        if not info:
            return
        status = settled_status(run, reply, failed=failed, model=model, cost_usd=cost)
        headline = status.get("result") or ("failed" if failed else "done")
        await self._edit(str(info["bubble"]),
                         status_edit_content(headline, str(info["bubble"]), status))

    # -- input: sync loop + turn execution --------------------------------

    async def _input_loop(self) -> None:
        client = await self._ensure_client()
        since = None
        try:  # drain backlog; never replay history as new input
            initial = await asyncio.wait_for(client.sync(timeout=0), timeout=60)
            since = initial.get("next_batch") if isinstance(initial, dict) else None
        except Exception as exc:
            logger.debug("session-bridge: initial sync failed: %s", exc)
        while self._running:
            if not self.state.room_id:
                await asyncio.sleep(1)
                continue
            try:
                resp = await asyncio.wait_for(
                    client.sync(since=since, timeout=_SYNC_TIMEOUT_MS),
                    timeout=_SYNC_TIMEOUT_MS / 1000 + 15)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("session-bridge: sync error: %s", exc)
                await asyncio.sleep(2)
                continue
            if not isinstance(resp, dict):
                continue
            since = resp.get("next_batch") or since
            events = (resp.get("rooms", {}).get("join", {}).get(self.state.room_id, {})
                      .get("timeline", {}).get("events", []) or [])
            roots = {t.root: sid for sid, t in self.state.sessions.items() if t.root}
            for ev in events:
                if not isinstance(ev, dict) or ev.get("type") != "m.room.message":
                    continue
                raw_content = ev.get("content")
                content: dict[str, Any] = raw_content if isinstance(raw_content, dict) else {}
                sid = route_inbound_event(str(ev.get("sender") or ""), self._own_mxid,
                                          content, roots, self.controllers)
                body = str(content.get("body") or "").strip()
                if sid and body:
                    await self._enqueue_turn(sid, body)

    async def _enqueue_turn(self, session_id: str, text: str) -> None:
        queue = self._queues.get(session_id)
        if queue is None:
            queue = self._queues[session_id] = asyncio.Queue(maxsize=_MAX_QUEUED_TURNS_PER_SESSION)
            self._drainers[session_id] = asyncio.get_running_loop().create_task(
                self._drain_session(session_id))
        try:
            queue.put_nowait(text)
        except asyncio.QueueFull:
            thread = self.state.sessions.get(session_id)
            if thread:
                await self._send(thread_reply_content(
                    "⚠️ Session is busy — too many queued turns; dropped this one.",
                    thread.root))

    async def _drain_session(self, session_id: str) -> None:
        lock = self._turn_locks.setdefault(session_id, asyncio.Lock())
        queue = self._queues[session_id]
        while self._running:
            text = await queue.get()
            async with self._sem, lock:
                await self._run_resume_turn(session_id, text)

    async def _run_resume_turn(self, session_id: str, text: str) -> None:
        thread = self.state.sessions.get(session_id)
        profile = thread.profile if thread else "default"
        from tools.bot_relay import _hermes_cli
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(text)
            query_file = fh.name
        argv = [_hermes_cli(), "-p", profile, "chat", "--resume", session_id,
                "-Q", "--query-file", query_file]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            try:
                await asyncio.wait_for(proc.wait(), timeout=_TURN_TIMEOUT_SECONDS)
            except TimeoutError:
                proc.kill()
                logger.warning("session-bridge: resume turn for %s timed out", session_id)
        except Exception as exc:
            logger.warning("session-bridge: resume turn for %s failed: %s", session_id, exc)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(query_file)
            # The poller mirrors whatever the subprocess appended; nudge it for snappy UX.
            with contextlib.suppress(Exception):
                await self._poll_once()

    # -- lifecycle --------------------------------------------------------

    async def run(self) -> None:
        room = await self._ensure_room()
        self._room_ready.set()
        logger.info("session-bridge: active in room %s (sources: %s)",
                    room, ",".join(self.sources))
        poll = asyncio.create_task(self._poll_loop())
        inp = asyncio.create_task(self._input_loop())
        try:
            while self._running:
                await asyncio.sleep(30)
        finally:
            self._running = False
            poll.cancel()
            inp.cancel()
            for task in list(self._drainers.values()):
                task.cancel()
            for task in (poll, inp):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._poll_once()
            except Exception as exc:
                logger.debug("session-bridge: poll tick failed: %s", exc)
            await asyncio.sleep(_POLL_SECONDS)

    def stop(self) -> None:
        self._running = False


def bridge_config(platform_config: Any) -> Optional[dict]:
    """Read the ``session_bridge`` settings off the Matrix PlatformConfig extra."""
    extra = getattr(platform_config, "extra", None) or {}
    if not extra.get("session_bridge"):
        return None
    sources = str(extra.get("session_bridge_sources") or ",".join(DEFAULT_SOURCES))
    cfg = {
        "room_id": str(extra.get("session_bridge_room") or ""),
        "sources": tuple(s for s in (x.strip().lower() for x in sources.split(",")) if s),
        "invite": tuple(s for s in (x.strip() for x in str(
            extra.get("session_bridge_invite") or "").split(",")) if s),
        "controllers": tuple(s for s in (x.strip() for x in str(
            extra.get("session_bridge_controllers") or "").split(",")) if s),
        "backfill_days": float(extra.get("session_bridge_backfill_days") or 7),
    }
    return cfg


def make_bridge_from_config(config: Any) -> Optional[SessionBridge]:
    """Build a bridge from a gateway config's Matrix platform block; None when not
    enabled or credentials are missing."""
    try:
        from gateway.config import Platform
        pc = (getattr(config, "platforms", None) or {}).get(Platform.MATRIX)
        if pc is None:
            return None
        cfg = bridge_config(pc)
        if not cfg:
            return None
        homeserver = os.environ.get("MATRIX_HOMESERVER", "") or str(pc.extra.get("homeserver") or "")
        token = os.environ.get("MATRIX_ACCESS_TOKEN", "") or str(pc.token or "")
        if not homeserver or not token:
            logger.warning("session-bridge: enabled but homeserver/token missing; not started")
            return None
        invite = cfg.pop("invite", ())
        if not invite:
            allowed = os.environ.get("MATRIX_ALLOWED_USERS", "") or str(
                pc.extra.get("allowed_users") or "")
            invite = tuple(u.strip() for u in allowed.split(",") if u.strip())
        return SessionBridge(homeserver=homeserver, access_token=token, invite=invite, **cfg)
    except Exception as exc:
        logger.warning("session-bridge: could not initialize: %s", exc)
        return None
