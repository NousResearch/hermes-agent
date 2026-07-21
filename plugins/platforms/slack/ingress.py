"""Slack Socket Mode ingress primitives.

This module owns the small, durable thread-follow index used by the optional
out-of-process Slack ingress.  The actual connector is built on top of these
primitives below; keeping the store independent makes its lifetime and bounds
straightforward to test.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from gateway.platforms.base import MessageEvent, SendResult
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor

try:
    import websockets
except ImportError:  # pragma: no cover - surfaced by start()
    websockets = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class FollowStore:
    """Bounded SQLite index of Slack threads that should auto-forward.

    Rows expire after ``ttl_seconds`` of inactivity.  ``max_threads`` is a hard
    LRU cap so a busy workspace cannot grow this state without bound.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        ttl_seconds: float,
        max_threads: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_threads <= 0:
            raise ValueError("max_threads must be positive")
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = float(ttl_seconds)
        self.max_threads = int(max_threads)
        self._clock = clock
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS followed_threads (
                    team_id TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    thread_ts TEXT NOT NULL,
                    last_seen REAL NOT NULL,
                    PRIMARY KEY (team_id, channel_id, thread_ts)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS followed_threads_last_seen "
                "ON followed_threads(last_seen)"
            )

    def _prune(self, conn: sqlite3.Connection, *, now: float) -> None:
        conn.execute(
            "DELETE FROM followed_threads WHERE last_seen < ?",
            (now - self.ttl_seconds,),
        )
        row = conn.execute("SELECT COUNT(*) FROM followed_threads").fetchone()
        overflow = int(row[0]) - self.max_threads if row else 0
        if overflow > 0:
            conn.execute(
                """
                DELETE FROM followed_threads
                WHERE rowid IN (
                    SELECT rowid FROM followed_threads
                    ORDER BY last_seen ASC, rowid ASC
                    LIMIT ?
                )
                """,
                (overflow,),
            )

    def follow(
        self,
        team_id: str,
        channel_id: str,
        thread_ts: str,
        *,
        seen_at: float | None = None,
    ) -> None:
        now = self._clock() if seen_at is None else float(seen_at)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO followed_threads(team_id, channel_id, thread_ts, last_seen)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(team_id, channel_id, thread_ts)
                DO UPDATE SET last_seen = excluded.last_seen
                """,
                (str(team_id), str(channel_id), str(thread_ts), now),
            )
            self._prune(conn, now=self._clock())

    def is_followed(self, team_id: str, channel_id: str, thread_ts: str) -> bool:
        now = self._clock()
        with self._connect() as conn:
            self._prune(conn, now=now)
            row = conn.execute(
                """
                SELECT 1 FROM followed_threads
                WHERE team_id = ? AND channel_id = ? AND thread_ts = ?
                """,
                (str(team_id), str(channel_id), str(thread_ts)),
            ).fetchone()
        return row is not None

    def unfollow(self, team_id: str, channel_id: str, thread_ts: str) -> bool:
        """Delete a follow route and report whether one existed."""
        with self._connect() as conn:
            result = conn.execute(
                """
                DELETE FROM followed_threads
                WHERE team_id = ? AND channel_id = ? AND thread_ts = ?
                """,
                (str(team_id), str(channel_id), str(thread_ts)),
            )
        return bool(result.rowcount)

    def touch_if_followed(self, team_id: str, channel_id: str, thread_ts: str) -> bool:
        """Refresh an active follow row atomically and report whether it existed."""
        now = self._clock()
        with self._connect() as conn:
            self._prune(conn, now=now)
            result = conn.execute(
                """
                UPDATE followed_threads SET last_seen = ?
                WHERE team_id = ? AND channel_id = ? AND thread_ts = ?
                """,
                (now, str(team_id), str(channel_id), str(thread_ts)),
            )
        return bool(result.rowcount)

    def count(self) -> int:
        now = self._clock()
        with self._connect() as conn:
            self._prune(conn, now=now)
            row = conn.execute("SELECT COUNT(*) FROM followed_threads").fetchone()
        return int(row[0]) if row else 0


class SlackIngressPolicy:
    """Admission and control policy applied before an event reaches Gateway."""

    def __init__(
        self,
        store: FollowStore,
        *,
        reaction_user_ids: set[str] | None = None,
        reaction_names: set[str] | None = None,
    ) -> None:
        self.store = store
        self.reaction_user_ids = {
            str(user_id).strip()
            for user_id in (reaction_user_ids or set())
            if str(user_id).strip()
        }
        self.reaction_names = {
            str(name).strip().strip(":").lower()
            for name in (reaction_names or set())
            if str(name).strip().strip(":")
        }
        self._control_commands: dict[str, Callable[..., None]] = {}
        self.register_control_command("/mute", self._mute)

    def register_control_command(
        self, command: str, handler: Callable[..., None]
    ) -> None:
        """Register a sidecar-local command that is consumed before forwarding."""
        normalized = command.strip().lower()
        if not normalized.startswith("/") or any(ch.isspace() for ch in normalized):
            raise ValueError("control command must be one slash-prefixed token")
        self._control_commands[normalized] = handler

    def _mute(
        self,
        *,
        team_id: str,
        channel_id: str,
        thread_ts: str,
        event: dict,
    ) -> None:
        del event
        self.store.unfollow(team_id, channel_id, thread_ts)

    def _consume_control_command(
        self,
        text: str,
        mention: str,
        *,
        team_id: str,
        channel_id: str,
        thread_ts: str,
        event: dict,
    ) -> bool:
        if not mention:
            return False
        stripped = text.lstrip()
        if not stripped.startswith(mention):
            return False
        remainder = stripped[len(mention) :].lstrip()
        command = remainder.split(maxsplit=1)[0].lower() if remainder else ""
        handler = self._control_commands.get(command)
        if handler is None:
            return False
        handler(
            team_id=team_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            event=event,
        )
        return True

    def admit_reaction(self, event: dict[str, Any]) -> bool:
        """Admit one configured owner reaction without mutating follow state."""
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "message":
            return False
        user_id = str(event.get("user") or "")
        reaction = str(event.get("reaction") or "").strip().strip(":").lower()
        return bool(
            self.reaction_user_ids
            and self.reaction_names
            and user_id in self.reaction_user_ids
            and reaction in self.reaction_names
            and item.get("channel")
            and item.get("ts")
        )

    def admit(
        self,
        event: dict,
        *,
        team_id: str,
        bot_user_id: str,
        is_one_to_one_dm: bool,
    ) -> bool:
        channel_id = str(event.get("channel") or "")
        ts = str(event.get("ts") or "")
        text = str(event.get("text") or "")
        incoming_thread_ts = str(event.get("thread_ts") or "")
        thread_ts = incoming_thread_ts or ts
        mention = f"<@{bot_user_id}>" if bot_user_id else ""

        # Control commands are sidecar-local state transitions. They run before
        # TTL refresh/admission and are never delivered to Gateway or the LLM.
        if self._consume_control_command(
            text,
            mention,
            team_id=team_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            event=event,
        ):
            return False

        if is_one_to_one_dm:
            return True
        is_root = not incoming_thread_ts or incoming_thread_ts == ts
        if not is_root and self.store.touch_if_followed(team_id, channel_id, thread_ts):
            return True
        if is_root and mention and text.lstrip().startswith(mention):
            self.store.follow(team_id, channel_id, ts)
            return True
        if mention and mention in text:
            return True
        return False


def message_event_to_wire(event: MessageEvent) -> dict[str, Any]:
    """Serialize a normalized event for ``WebSocketRelayTransport``.

    The transport deliberately accepts a plain JSON object.  Keep this mapping
    explicit so new MessageEvent fields are not accidentally exposed, while
    retaining the Slack context needed for feature parity with the direct
    adapter.
    """
    source = event.source.to_dict() if event.source is not None else {}
    return {
        "text": event.text,
        "message_type": event.message_type.value,
        "source": source,
        "message_id": event.message_id,
        "media_urls": list(event.media_urls),
        "media_types": list(event.media_types),
        "reply_to_message_id": event.reply_to_message_id,
        "reply_to_text": event.reply_to_text,
        "reply_to_author_id": event.reply_to_author_id,
        "reply_to_author_name": event.reply_to_author_name,
        "reply_to_is_own_message": event.reply_to_is_own_message,
        "auto_skill": event.auto_skill,
        "channel_prompt": event.channel_prompt,
        "channel_context": event.channel_context,
        "metadata": dict(event.metadata),
    }


class _SingleInstanceLock:
    """Process-wide advisory lock; the OS releases it after abnormal exit."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._handle: Any = None

    def acquire(self) -> None:
        if self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            handle.seek(0)
            if not handle.read(1):
                handle.seek(0)
                handle.write(" ")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError(
                f"Slack ingress sidecar is already running on this machine ({self.path})"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


class SlackIngressServer:
    """Sole Slack Socket Mode owner and local Relay WebSocket server.

    ``slack_adapter`` is the existing production SlackAdapter. Reusing it keeps
    Slack normalization, files, thread context, formatting, and Web API egress
    identical to direct mode; only admission and delivery to Hermes move here.
    """

    def __init__(
        self,
        slack_adapter: Any,
        policy: SlackIngressPolicy,
        *,
        host: str = "127.0.0.1",
        port: int = 8791,
        lock_path: str | Path | None = None,
    ) -> None:
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("Slack ingress currently supports loopback binding only")
        if lock_path is None:
            from hermes_constants import get_default_hermes_root

            lock_path = get_default_hermes_root() / "slack-ingress.lock"
        self.slack_adapter = slack_adapter
        self.policy = policy
        self.host = host
        self.port = int(port)
        self._instance_lock = _SingleInstanceLock(lock_path)
        self._server: Any = None
        self._gateway: Any = None
        self._send_lock = asyncio.Lock()
        self._slack_connection_lock = asyncio.Lock()
        self._slack_connected = False
        self._descriptor = CapabilityDescriptor(
            contract_version=CONTRACT_VERSION,
            platform="slack",
            label="Slack",
            max_message_length=39_000,
            supports_draft_streaming=False,
            supports_edit=True,
            supports_threads=False,
            markdown_dialect="slack",
            len_unit="chars",
            emoji="💬",
            platform_hint="You are on Slack.",
            pii_safe=False,
            supports_context=True,
        )

    @property
    def url(self) -> str:
        if self._server is None or not self._server.sockets:
            raise RuntimeError("Slack ingress server is not running")
        port = int(self._server.sockets[0].getsockname()[1])
        return f"http://{self.host}:{port}"

    async def start(self, *, connect_slack: bool = False) -> None:
        if websockets is None:
            raise RuntimeError("Slack ingress requires the 'websockets' package")
        if connect_slack:
            raise ValueError("Slack may connect only after a Relay handshake")
        if self._server is not None:
            return
        self._instance_lock.acquire()
        try:
            self.slack_adapter.set_external_admission_handler(self._admit_slack_event)
            self.slack_adapter.set_external_reaction_handler(
                self._handle_slack_reaction
            )
            self.slack_adapter.set_message_handler(self.forward_event)
            self._server = await websockets.serve(
                self._handle_connection, self.host, self.port
            )
        except BaseException:
            self._instance_lock.release()
            raise

    async def _ensure_slack_connected(self) -> None:
        async with self._slack_connection_lock:
            if self._slack_connected:
                return
            connected = await self.slack_adapter.connect()
            if connected is False:
                raise RuntimeError("Slack Socket Mode connection failed")
            self._slack_connected = True

    async def _disconnect_slack(self) -> None:
        async with self._slack_connection_lock:
            if not self._slack_connected:
                return
            self._slack_connected = False
            await self.slack_adapter.disconnect()

    async def stop(self) -> None:
        try:
            gateway = self._gateway
            self._gateway = None
            if gateway is not None:
                try:
                    await gateway.close()
                except Exception:
                    pass
            try:
                await self._disconnect_slack()
            finally:
                if self._server is not None:
                    self._server.close()
                    await self._server.wait_closed()
                    self._server = None
        finally:
            self._instance_lock.release()

    async def wait_closed(self) -> None:
        if self._server is None:
            raise RuntimeError("Slack ingress server is not running")
        await self._server.wait_closed()

    def _admit_slack_event(
        self,
        event: dict,
        *,
        body: Any = None,
        team_id: str,
        bot_user_id: str,
        is_one_to_one_dm: bool,
    ) -> bool:
        del body
        if event.get("_slack_ingress_reaction_trigger") is True:
            return True
        return self.policy.admit(
            event,
            team_id=team_id,
            bot_user_id=bot_user_id,
            is_one_to_one_dm=is_one_to_one_dm,
        )

    async def _handle_slack_reaction(
        self, event: dict[str, Any], *, body: Any = None
    ) -> None:
        if self._gateway is None or not self.policy.admit_reaction(event):
            return
        await self.slack_adapter.dispatch_reaction_trigger(event, body=body)

    async def forward_event(self, event: MessageEvent) -> None:
        gateway = self._gateway
        if gateway is None:
            logger.warning(
                "Slack ingress dropped admitted event: Gateway is disconnected"
            )
            return
        await self._send_frame(
            gateway,
            {"type": "inbound", "event": message_event_to_wire(event)},
        )

    async def _handle_connection(self, websocket: Any) -> None:
        request = getattr(websocket, "request", None)
        path = getattr(request, "path", "/relay")
        if str(path).split("?", 1)[0] != "/relay":
            await websocket.close(code=4404, reason="relay endpoint not found")
            return
        if self._gateway is not None:
            await websocket.close(code=4429, reason="Gateway already connected")
            return
        self._gateway = websocket
        try:
            async for payload in websocket:
                if isinstance(payload, bytes):
                    payload = payload.decode("utf-8")
                for line in str(payload).splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        frame = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Slack ingress ignored malformed relay frame")
                        continue
                    if isinstance(frame, dict):
                        await self._handle_frame(websocket, frame)
        finally:
            if self._gateway is websocket:
                self._gateway = None
                await self._disconnect_slack()

    async def _handle_frame(self, websocket: Any, frame: dict[str, Any]) -> None:
        frame_type = frame.get("type")
        if frame_type == "hello":
            try:
                await self._ensure_slack_connected()
            except Exception as exc:
                logger.error("Slack ingress handshake could not connect Slack: %s", exc)
                await websocket.close(code=4503, reason="Slack connection failed")
                return
            await self._send_frame(
                websocket,
                {"type": "descriptor", "descriptor": asdict(self._descriptor)},
            )
            return
        if frame_type == "outbound":
            request_id = str(frame.get("requestId") or "")
            result = await self._execute_outbound(frame.get("action") or {})
            await self._send_frame(
                websocket,
                {
                    "type": "outbound_result",
                    "requestId": request_id,
                    "result": result,
                },
            )
            return
        if frame_type == "going_idle":
            await self._send_frame(websocket, {"type": "going_idle_ack"})

    async def _execute_outbound(self, action: dict[str, Any]) -> dict[str, Any]:
        op = str(action.get("op") or "")
        try:
            if op == "send":
                result = await self.slack_adapter.send(
                    str(action.get("chat_id") or ""),
                    str(action.get("content") or ""),
                    reply_to=action.get("reply_to"),
                    metadata=action.get("metadata") or {},
                )
                return self._send_result(result)
            if op == "edit":
                result = await self.slack_adapter.edit_message(
                    str(action.get("chat_id") or ""),
                    str(action.get("message_id") or ""),
                    str(action.get("content") or ""),
                    finalize=bool(action.get("finalize", False)),
                    metadata=action.get("metadata") or {},
                )
                return self._send_result(result)
            if op == "typing":
                method = (
                    self.slack_adapter.send_typing
                    if action.get("active", True)
                    else self.slack_adapter.stop_typing
                )
                await method(
                    str(action.get("chat_id") or ""),
                    metadata=action.get("metadata") or {},
                )
                return {"success": True}
            if op == "get_chat_info":
                info = await self.slack_adapter.get_chat_info(
                    str(action.get("chat_id") or "")
                )
                return {"success": True, "chat_info": info}
            return {"success": False, "error": f"unsupported Slack relay op: {op}"}
        except Exception as exc:  # noqa: BLE001 - result must correlate every request
            logger.warning("Slack ingress outbound failed: %s", exc, exc_info=True)
            return {"success": False, "error": str(exc)}

    @staticmethod
    def _send_result(result: SendResult) -> dict[str, Any]:
        return {
            "success": bool(result.success),
            "message_id": result.message_id,
            "error": result.error,
            "retryable": bool(result.retryable),
            "retry_after": result.retry_after,
            "continuation_message_ids": list(result.continuation_message_ids),
        }

    async def _send_frame(self, websocket: Any, frame: dict[str, Any]) -> None:
        payload = json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self._send_lock:
            await websocket.send(payload)
