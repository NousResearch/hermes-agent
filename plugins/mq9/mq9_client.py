#!/usr/bin/env python3
"""mq9 Python client for Hermes plugin runtime.

This client uses raw NATS text protocol over TCP to avoid compatibility issues
with NATS client feature negotiation across broker versions.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

DEFAULT_NATS_URL = "nats://127.0.0.1:4222"


class Mq9Error(RuntimeError):
    """Raised when mq9 command returns an error."""


@dataclass
class Mailbox:
    mail_address: str
    created: bool | None = None
    already_exists: bool | None = None
    # Legacy field preserved for backwards-compat with earlier PoC scripts.
    is_new: bool | None = None


@dataclass
class FetchedMessage:
    msg_id: int
    payload: str
    priority: str
    create_time: int
    header: dict[str, str] | None = None

    def parse_json(self) -> Any:
        try:
            return json.loads(self.payload)
        except json.JSONDecodeError:
            return self.payload


@dataclass
class CallResult:
    correlation_id: str
    callback_mailbox: str
    response: Any
    raw: FetchedMessage


class Mq9Client:
    """Thin Python client that matches mq9 Agent plugin needs."""

    def __init__(self, nats_url: str = DEFAULT_NATS_URL, request_timeout: float = 5.0) -> None:
        self._nats_url = nats_url
        self._request_timeout = request_timeout
        parsed = urlparse(nats_url)
        if parsed.scheme != "nats":
            raise ValueError(f"unsupported scheme in nats url: {nats_url}")
        if not parsed.hostname or not parsed.port:
            raise ValueError(f"invalid nats url: {nats_url}")
        self._host = parsed.hostname
        self._port = parsed.port

    async def connect(self) -> None:
        # No persistent socket required in this minimal client.
        return None

    async def close(self) -> None:
        return None

    async def __aenter__(self) -> "Mq9Client":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def create_mailbox(
        self,
        *,
        ttl: int = 3600,
        name: str | None = None,
        desc: str | None = None,
        idempotent: bool = True,
        public: bool | None = None,
    ) -> Mailbox:
        req: dict[str, Any] = {"ttl": ttl, "idempotent": bool(idempotent)}
        if name:
            req["name"] = name
        if desc:
            req["desc"] = desc
        if public is not None:
            req["public"] = bool(public)

        reply = await self._request("$mq9.AI.MAILBOX.CREATE", req, retries=3)
        mail_address = reply.get("mail_address") or reply.get("mail_id")
        if not mail_address:
            raise Mq9Error(f"MAILBOX.CREATE invalid response: {reply}")

        created = None
        if "created" in reply:
            created = bool(reply.get("created"))
        elif "already_exists" in reply:
            created = not bool(reply.get("already_exists"))

        already_exists = None
        if "already_exists" in reply:
            already_exists = bool(reply.get("already_exists"))

        return Mailbox(
            mail_address=str(mail_address),
            created=created,
            already_exists=already_exists,
            is_new=reply.get("is_new"),
        )

    async def register_agent(
        self,
        *,
        agent_card: dict[str, Any] | None = None,
        name: str | None = None,
        payload: str | None = None,
    ) -> None:
        if agent_card is not None:
            card_name = name or str(agent_card.get("name", "")).strip()
            if not card_name:
                raise ValueError("agent_card must include a non-empty name")
            card_payload = payload or json.dumps(agent_card, ensure_ascii=False)
            await self._request(
                "$mq9.AI.AGENT.REGISTER",
                {"name": card_name, "payload": card_payload},
                retries=2,
            )
            return
        if not name:
            raise ValueError("name is required when agent_card is not provided")
        await self._request(
            "$mq9.AI.AGENT.REGISTER",
            {"name": name, "payload": payload or ""},
            retries=2,
        )

    async def unregister_agent(self, *, name: str) -> None:
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("name must not be empty")
        await self._request(
            "$mq9.AI.AGENT.UNREGISTER",
            {"name": clean_name},
            retries=2,
        )

    async def discover_agents(
        self,
        *,
        query: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        req = {"query": query, "limit": limit}
        reply = await self._request("$mq9.AI.AGENT.DISCOVER", req, retries=2)
        agents = reply.get("agents", [])
        if not isinstance(agents, list):
            raise Mq9Error(f"AGENT.DISCOVER invalid response: {reply}")
        return [a for a in agents if isinstance(a, dict)]

    async def send_message(
        self,
        mail_address: str,
        payload: dict[str, Any] | str,
        *,
        priority: str = "normal",
        headers: dict[str, str] | None = None,
    ) -> int:
        subject = f"$mq9.AI.MSG.SEND.{mail_address}"
        all_headers = dict(headers or {})
        if priority and priority != "normal":
            all_headers["mq9-priority"] = priority

        reply = await self._request(
            subject,
            payload,
            headers=all_headers or None,
            retries=2,
        )
        msg_id = int(reply.get("msg_id", -1))
        return msg_id

    async def fetch_messages(
        self,
        mail_address: str,
        *,
        group_name: str | None = None,
        deliver: str = "latest",
        max_messages: int = 50,
        max_wait_ms: int = 500,
        force_deliver: bool | None = None,
    ) -> list[FetchedMessage]:
        subject = f"$mq9.AI.MSG.FETCH.{mail_address}"
        req: dict[str, Any] = {
            "group_name": group_name,
            "deliver": deliver,
            "force_deliver": force_deliver,
            "config": {"num_msgs": max_messages, "max_wait_ms": max_wait_ms},
        }
        reply = await self._request(subject, req, retries=2)
        messages = reply.get("messages", [])
        result: list[FetchedMessage] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            result.append(
                FetchedMessage(
                    msg_id=int(item.get("msg_id", 0)),
                    payload=str(item.get("payload", "")),
                    priority=str(item.get("priority", "normal")),
                    create_time=int(item.get("create_time", 0)),
                    header=_parse_headers(item.get("header")),
                )
            )
        return result

    async def ack_message(
        self,
        mail_address: str,
        group_name: str,
        msg_id: int,
        *,
        retries: int = 3,
    ) -> None:
        subject = f"$mq9.AI.MSG.ACK.{mail_address}"
        req = {
            "group_name": group_name,
            "mail_address": mail_address,
            "msg_id": msg_id,
        }
        attempts = max(1, retries)
        for index in range(attempts):
            try:
                await self._request(subject, req, retries=2)
                return
            except Mq9Error as exc:
                is_last = index == attempts - 1
                if is_last or not _is_retryable_fetch_error(str(exc)):
                    raise
                await asyncio.sleep(0.2)

    async def subscribe_loop(
        self,
        mail_address: str,
        handler: Callable[[FetchedMessage], Awaitable[None]],
        *,
        group_name: str | None = None,
        deliver: str = "earliest",
        poll_interval: float = 0.25,
        max_batch: int = 20,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        consumer_group = group_name or f"mq9-sub-{uuid.uuid4().hex[:8]}"
        stopper = stop_event or asyncio.Event()

        while not stopper.is_set():
            try:
                messages = await self.fetch_messages(
                    mail_address,
                    group_name=consumer_group,
                    deliver=deliver,
                    max_messages=max_batch,
                    max_wait_ms=500,
                )
            except Mq9Error as exc:
                # Newly created mailboxes may briefly miss topic cache state;
                # keep polling instead of crashing the passive server loop.
                if _is_retryable_fetch_error(str(exc)):
                    await asyncio.sleep(poll_interval)
                    continue
                raise
            if not messages:
                await asyncio.sleep(poll_interval)
                continue

            for msg in messages:
                await handler(msg)
                await self.ack_message(mail_address, consumer_group, msg.msg_id)

    async def mq9_call(
        self,
        *,
        from_agent: str,
        target_mailbox: str,
        message: dict[str, Any],
        timeout_s: float = 20.0,
    ) -> CallResult:
        callback = await self.create_mailbox(ttl=max(300, int(timeout_s) + 120))
        correlation_id = uuid.uuid4().hex
        envelope = {
            "type": "mq9_call",
            "from": from_agent,
            "reply_to": callback.mail_address,
            "correlation_id": correlation_id,
            "payload": message,
            "ts": int(time.time()),
        }
        await self.send_message(target_mailbox, envelope)

        group = f"mq9-call-{correlation_id[:10]}"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                messages = await self.fetch_messages(
                    callback.mail_address,
                    group_name=group,
                    deliver="earliest",
                    max_messages=20,
                    max_wait_ms=500,
                )
            except Mq9Error as exc:
                if _is_retryable_fetch_error(str(exc)):
                    await asyncio.sleep(0.2)
                    continue
                raise
            for msg in messages:
                body = msg.parse_json()
                is_match = (
                    isinstance(body, dict)
                    and body.get("correlation_id") == correlation_id
                )
                await self.ack_message(callback.mail_address, group, msg.msg_id)
                if is_match:
                    return CallResult(
                        correlation_id=correlation_id,
                        callback_mailbox=callback.mail_address,
                        response=body,
                        raw=msg,
                    )
            await asyncio.sleep(0.2)

        raise TimeoutError(
            f"mq9_call timeout: target={target_mailbox}, correlation_id={correlation_id}"
        )

    async def _request(
        self,
        subject: str,
        payload: dict[str, Any] | str,
        *,
        headers: dict[str, str] | None = None,
        retries: int = 1,
    ) -> dict[str, Any]:
        body = payload if isinstance(payload, str) else json.dumps(payload)
        attempts = max(1, int(retries))
        last_error: BaseException | None = None

        for index in range(attempts):
            try:
                raw = await self._raw_nats_request(
                    subject,
                    body.encode("utf-8"),
                    headers=headers,
                )
            except TimeoutError as exc:
                last_error = Mq9Error(f"{subject} timeout waiting for reply")
                if index < attempts - 1:
                    await asyncio.sleep(0.2 * (index + 1))
                    continue
                raise last_error from exc
            except OSError as exc:
                last_error = Mq9Error(f"{subject} network error: {exc}")
                if index < attempts - 1:
                    await asyncio.sleep(0.2 * (index + 1))
                    continue
                raise last_error from exc

            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError as exc:
                preview = raw[:200].decode("utf-8", errors="replace")
                raise Mq9Error(
                    f"{subject} invalid json reply: {preview!r}"
                ) from exc

            error = str(data.get("error", "") or "").strip()
            if not error:
                return data

            last_error = Mq9Error(f"{subject} failed: {error}")
            if index < attempts - 1 and _is_retryable_broker_error(error):
                await asyncio.sleep(0.2 * (index + 1))
                continue
            raise last_error

        if last_error is not None:
            raise Mq9Error(str(last_error))
        raise Mq9Error(f"{subject} failed with unknown error")

    async def _raw_nats_request(
        self,
        subject: str,
        payload: bytes,
        *,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        reader, writer = await asyncio.open_connection(self._host, self._port)
        inbox = f"_INBOX.{uuid.uuid4().hex}"
        sid = "1"

        writer.write(b'CONNECT {"verbose":false,"pedantic":false,"tls_required":false}\r\n')
        writer.write(f"SUB {inbox} {sid}\r\n".encode("utf-8"))

        if headers:
            header_lines = ["NATS/1.0"] + [f"{k}: {v}" for k, v in headers.items()] + ["", ""]
            header_blob = "\r\n".join(header_lines).encode("utf-8")
            total = len(header_blob) + len(payload)
            writer.write(
                f"HPUB {subject} {inbox} {len(header_blob)} {total}\r\n".encode("utf-8")
            )
            writer.write(header_blob)
            writer.write(payload)
            writer.write(b"\r\n")
        else:
            writer.write(f"PUB {subject} {inbox} {len(payload)}\r\n".encode("utf-8"))
            writer.write(payload)
            writer.write(b"\r\n")

        writer.write(b"PING\r\n")
        await writer.drain()

        try:
            async with asyncio.timeout(self._request_timeout):
                while True:
                    line = await reader.readline()
                    if not line:
                        raise Mq9Error(f"connection closed before reply: subject={subject}")
                    head = line.decode("utf-8", errors="replace").strip()
                    if not head or head.startswith("INFO") or head == "PONG":
                        continue
                    if head == "PING":
                        writer.write(b"PONG\r\n")
                        await writer.drain()
                        continue
                    if head.startswith("-ERR"):
                        raise Mq9Error(f"nats server error: {head}")
                    if head.startswith("MSG "):
                        parts = head.split()
                        if len(parts) < 4:
                            continue
                        # MSG <subject> <sid> [reply-to] <#bytes>
                        if len(parts) == 4:
                            size = int(parts[3])
                        else:
                            size = int(parts[4])
                        data = await reader.readexactly(size)
                        await reader.readexactly(2)  # trailing CRLF
                        return data
                    if head.startswith("HMSG "):
                        parts = head.split()
                        if len(parts) < 5:
                            continue
                        # HMSG <subject> <sid> [reply-to] <#header bytes> <#total bytes>
                        if len(parts) == 5:
                            header_bytes = int(parts[3])
                            total_bytes = int(parts[4])
                        else:
                            header_bytes = int(parts[4])
                            total_bytes = int(parts[5])
                        block = await reader.readexactly(total_bytes)
                        await reader.readexactly(2)  # trailing CRLF
                        return block[header_bytes:]
        finally:
            writer.close()
            await writer.wait_closed()


def _parse_headers(raw: Any) -> dict[str, str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw_bytes = raw.encode("utf-8")
    elif isinstance(raw, list):
        raw_bytes = bytes(raw)
    elif isinstance(raw, bytes):
        raw_bytes = raw
    else:
        return None

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None

    result: dict[str, str] = {}
    for line in text.splitlines()[1:]:
        line = line.strip()
        if not line:
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result or None


def _is_retryable_fetch_error(error_text: str) -> bool:
    lowered = error_text.lower()
    if "timeout waiting for reply" in lowered:
        return True
    return "topic" in lowered and "not found in broker cache" in lowered


def _is_retryable_broker_error(error_text: str) -> bool:
    lowered = error_text.lower()
    if "timeout waiting for reply" in lowered:
        return True
    if "connection reset" in lowered or "connection closed" in lowered:
        return True
    return "topic" in lowered and "not found in broker cache" in lowered
