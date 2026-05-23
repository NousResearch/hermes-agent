"""Small local HTTP bridge for Hermes agent-to-agent group chat.

Run with:

    python plugins/agent_bridge/server.py --token-env HERMES_AGENT_BRIDGE_TOKEN

The server intentionally uses only the Python standard library so two Hermes
profiles can share it without installing extra dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import parse


class BridgeState:
    def __init__(self) -> None:
        self.condition = threading.Condition(threading.RLock())
        self.rooms: dict[str, dict[str, Any]] = {}
        self.agents: dict[str, dict[str, Any]] = {}
        self.queues: dict[str, list[dict[str, Any]]] = {}
        self.seen_ids: set[str] = set()
        self.threads: dict[str, dict[str, Any]] = {}

    def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(payload.get("agent_id") or "").strip()
        if not agent_id:
            raise ValueError("agent_id is required")
        rooms = payload.get("rooms") or {}
        if not isinstance(rooms, dict):
            rooms = {}
        with self.condition:
            self.agents[agent_id] = {
                "agent_id": agent_id,
                "display_name": payload.get("display_name") or agent_id,
                "updated_at": time.time(),
            }
            self.queues.setdefault(agent_id, [])
            for room_id, room in rooms.items():
                if isinstance(room, dict):
                    self._merge_room_locked(str(room_id), room)
            self.condition.notify_all()
        return {"ok": True}

    def publish(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_id = str(payload.get("id") or uuid.uuid4().hex)
        room_id = str(payload.get("room_id") or "")
        if not room_id:
            raise ValueError("room_id is required")
        with self.condition:
            room_payload = payload.get("room")
            if isinstance(room_payload, dict):
                self._merge_room_locked(room_id, room_payload)
            room = self.rooms.setdefault(room_id, self._default_room(room_id))
            if event_id in self.seen_ids:
                thread_id = str(payload.get("thread_id") or event_id)
                return {"ok": True, "duplicate": True, "id": event_id, "thread_id": thread_id}
            self.seen_ids.add(event_id)
            self._prune_seen_locked()

            event = dict(payload)
            event["id"] = event_id
            event["room_id"] = room_id
            event["room"] = room
            thread_id, allow_auto_reply = self._update_thread_locked(event, room)
            event["thread_id"] = thread_id
            event["allow_auto_reply"] = allow_auto_reply

            author_id = str(event.get("author_id") or "")
            origin_agent_id = str(event.get("origin_agent_id") or "")
            targets = self._target_agents_locked(room)
            for agent_id in targets:
                if agent_id and agent_id not in {author_id, origin_agent_id}:
                    self.queues.setdefault(agent_id, []).append(event)
            self.condition.notify_all()
        return {"ok": True, "id": event_id, "thread_id": thread_id, "allow_auto_reply": allow_auto_reply}

    def poll(self, agent_id: str, timeout: float = 25.0, limit: int = 20) -> dict[str, Any]:
        deadline = time.time() + max(0.0, timeout)
        with self.condition:
            queue = self.queues.setdefault(agent_id, [])
            while not queue:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=remaining)
            events = queue[:limit]
            del queue[:len(events)]
        return {"ok": True, "events": events}

    def _merge_room_locked(self, room_id: str, room: dict[str, Any]) -> None:
        current = self.rooms.setdefault(room_id, self._default_room(room_id))
        for key in ("external_targets", "participants"):
            value = room.get(key)
            if isinstance(value, list):
                current[key] = value
        for key, default in (("max_bot_messages", 16), ("idle_timeout_seconds", 1800)):
            try:
                current[key] = max(1, int(room.get(key, current.get(key, default))))
            except (TypeError, ValueError):
                current[key] = current.get(key, default)

    @staticmethod
    def _default_room(room_id: str) -> dict[str, Any]:
        return {
            "room_id": room_id,
            "external_targets": [],
            "participants": [],
            "max_bot_messages": 16,
            "idle_timeout_seconds": 1800,
        }

    def _target_agents_locked(self, room: dict[str, Any]) -> list[str]:
        participants = room.get("participants") or []
        targets = [str(p.get("agent_id")) for p in participants if isinstance(p, dict) and p.get("agent_id")]
        return targets or list(self.agents.keys())

    def _update_thread_locked(self, event: dict[str, Any], room: dict[str, Any]) -> tuple[str, bool]:
        now = time.time()
        author_type = str(event.get("author_type") or "human")
        max_bot_messages = max(1, int(room.get("max_bot_messages") or 16))
        idle_timeout = max(30, int(room.get("idle_timeout_seconds") or 1800))
        requested_thread = str(event.get("thread_id") or "")
        if author_type == "human" or not requested_thread:
            thread_id = requested_thread or str(event.get("id") or uuid.uuid4().hex)
            self.threads[thread_id] = {
                "room_id": event.get("room_id"),
                "bot_count": 0,
                "updated_at": now,
                "max_bot_messages": max_bot_messages,
                "idle_timeout_seconds": idle_timeout,
            }
            return thread_id, True

        thread_id = requested_thread
        thread = self.threads.get(thread_id)
        if not thread or now - float(thread.get("updated_at") or 0) > idle_timeout:
            thread = {
                "room_id": event.get("room_id"),
                "bot_count": 0,
                "updated_at": now,
                "max_bot_messages": max_bot_messages,
                "idle_timeout_seconds": idle_timeout,
            }
            self.threads[thread_id] = thread
        thread["bot_count"] = int(thread.get("bot_count") or 0) + 1
        thread["updated_at"] = now
        return thread_id, int(thread["bot_count"]) < max_bot_messages

    def _prune_seen_locked(self) -> None:
        # Keep memory bounded. The seen set is only for short-window de-dupe.
        if len(self.seen_ids) > 10000:
            self.seen_ids = set(list(self.seen_ids)[-5000:])


class BridgeHandler(BaseHTTPRequestHandler):
    server_version = "HermesAgentBridge/0.1"

    @property
    def state(self) -> BridgeState:
        return self.server.state  # type: ignore[attr-defined]

    @property
    def token(self) -> str:
        return self.server.token  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if not self._authorized():
            return self._json({"ok": False, "error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
        parsed = parse.urlparse(self.path)
        if parsed.path == "/health":
            return self._json({"ok": True})
        if parsed.path != "/v1/events":
            return self._json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
        query = parse.parse_qs(parsed.query)
        agent_id = (query.get("agent_id") or [""])[0]
        if not agent_id:
            return self._json({"ok": False, "error": "agent_id is required"}, HTTPStatus.BAD_REQUEST)
        timeout = _float((query.get("timeout") or [25])[0], 25.0)
        limit = max(1, min(100, int(_float((query.get("limit") or [20])[0], 20))))
        return self._json(self.state.poll(agent_id, timeout=timeout, limit=limit))

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if not self._authorized():
            return self._json({"ok": False, "error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
        try:
            payload = self._read_json()
            if self.path == "/v1/register":
                result = self.state.register(payload)
            elif self.path == "/v1/events":
                result = self.state.publish(payload)
            else:
                return self._json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            return self._json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return self._json(result)

    def log_message(self, fmt: str, *args: Any) -> None:
        if getattr(self.server, "verbose", False):  # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    def _authorized(self) -> bool:
        if not self.token:
            return True
        return self.headers.get("Authorization") == f"Bearer {self.token}"

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 2_000_000:
            raise ValueError("payload too large")
        raw = self.rfile.read(length) if length else b"{}"
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON object required")
        return data

    def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_server(host: str, port: int, token: str = "", verbose: bool = False) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), BridgeHandler)
    server.state = BridgeState()  # type: ignore[attr-defined]
    server.token = token  # type: ignore[attr-defined]
    server.verbose = verbose  # type: ignore[attr-defined]
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local Hermes agent bridge server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--token", default="")
    parser.add_argument("--token-env", default="HERMES_AGENT_BRIDGE_TOKEN")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    token = args.token or os.environ.get(args.token_env, "")
    server = build_server(args.host, args.port, token=token, verbose=args.verbose)
    print(f"Hermes agent bridge listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
