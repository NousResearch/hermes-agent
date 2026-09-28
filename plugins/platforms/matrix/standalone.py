"""Plaintext Matrix delivery when a native adapter is unavailable."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import json
import os
import re
import time
from typing import Any, TYPE_CHECKING
from urllib.parse import quote

from agent.secret_scope import get_secret
from gateway.config import PlatformConfig
from gateway.platforms._shared import send_error

if TYPE_CHECKING:
    from aiohttp import ClientSession


@dataclass
class _MatrixAPIError(ValueError):
    status: int
    response: dict[str, Any]

    def __str__(self) -> str:
        return f"Matrix API error ({self.status}): {self.response}"


@dataclass
class _HTTPDelivery:
    session: ClientSession
    homeserver: str

    async def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        async with self.session.request(
            method, f"{self.homeserver}/_matrix/client/v3/{path}", **kwargs
        ) as response:
            data = await response.json()
            if not isinstance(data, dict):
                raise ValueError("Matrix API returned an invalid JSON object")
            if response.status not in {200, 201}:
                raise _MatrixAPIError(response.status, data)
            return data

    async def resolve(self, target: str) -> str:
        if target.startswith("@"):
            raise ValueError(
                "MXID delivery is unsupported. Use the DM's room ID or a room alias."
            )
        if target.startswith("!"):
            return target
        if not target.startswith("#"):
            raise ValueError("Invalid Matrix target: use a room ID or alias")
        info = await self.request("GET", f"directory/room/{quote(target, safe='')}")
        room_id = info.get("room_id")
        if not isinstance(room_id, str) or not room_id.startswith("!"):
            raise ValueError(
                "alias did not resolve to a room ID; publish a Local Address on the room"
            )
        joined = await self.request(
            "POST",
            f"join/{quote(room_id, safe='')}",
            params=[("server_name", server) for server in info.get("servers", [])],
            json={},
        )
        if joined.get("room_id") != room_id:
            raise ValueError(
                f"join returned '{joined.get('room_id')}', expected room ID '{room_id}'"
            )
        return room_id

    async def send(
        self, target: str, message: str, thread_id: str | None
    ) -> dict[str, Any]:
        room_id = await self.resolve(target)
        try:
            await self.request(
                "GET", f"rooms/{quote(room_id, safe='')}/state/m.room.encryption"
            )
        except _MatrixAPIError as exc:
            if exc.status != 404 or exc.response.get("errcode") != "M_NOT_FOUND":
                raise
        else:
            raise ValueError(
                f"Room '{room_id}' is encrypted; use a native Matrix adapter with E2EE"
            )
        from gateway.platforms.base import BasePlatformAdapter
        from plugins.platforms.matrix.adapter import (
            MATRIX_MAX_MESSAGE_LENGTH_CEILING,
        )

        for chunk in BasePlatformAdapter.truncate_message(
            message, MATRIX_MAX_MESSAGE_LENGTH_CEILING, lambda text: len(json.dumps(text)) - 2
        ):
            txn_id = f"hermes_{int(time.time() * 1000)}_{os.urandom(4).hex()}"
            data = await self.request(
                "PUT",
                f"rooms/{quote(room_id, safe='')}/send/m.room.message/{txn_id}",
                json=_text_payload(chunk, thread_id),
            )
        return {
            "success": True,
            "platform": "matrix",
            "chat_id": room_id,
            "message_id": data.get("event_id"),
        }


def _text_payload(message: str, thread_id: str | None) -> dict[str, Any]:
    from plugins.platforms.matrix.adapter import (
        _MAX_CONTENT_BYTES,
        _latex_to_tokens,
        _tokens_to_mx_maths,
        _sanitize_matrix_html,
    )

    payload = {"msgtype": "m.text", "body": message}
    if thread_id:
        payload["m.relates_to"] = {
            "rel_type": "m.thread",
            "event_id": thread_id,
            "is_falling_back": True,
            "m.in_reply_to": {"event_id": thread_id},
        }
    if len(json.dumps(payload)) > _MAX_CONTENT_BYTES:
        raise ValueError("Matrix message metadata exceeds the event size limit")
    with suppress(ImportError):
        import markdown

        tokenized, tex_store = _latex_to_tokens(message)
        html = markdown.markdown(tokenized, extensions=["fenced_code", "tables"])
        formatted = {
            "format": "org.matrix.custom.html",
            "formatted_body": _tokens_to_mx_maths(
                _sanitize_matrix_html(re.sub(r"<h[1-6]>(.*?)</h[1-6]>", r"<strong>\1</strong>", html)),
                tex_store,
            ),
        }
        if len(json.dumps({**payload, **formatted})) <= _MAX_CONTENT_BYTES:
            payload.update(formatted)
    return payload


async def standalone_send(
    pconfig: PlatformConfig,
    chat_id: str,
    message: str,
    *,
    thread_id: str | None = None,
    media_files: list | None = None,
    force_document: bool = False,
) -> dict[str, Any]:
    from plugins.platforms.matrix.adapter import _resolve_e2ee_mode

    extra = getattr(pconfig, "extra", {}) or {}
    if _resolve_e2ee_mode(extra) == "required":
        return send_error(
            f"Matrix target '{chat_id}': E2EE is required; use a native Matrix adapter"
        )
    try:
        import aiohttp
    except ImportError:
        return send_error("aiohttp not installed. Run: hermes pm install")
    try:
        homeserver = (
            extra.get("homeserver") or get_secret("MATRIX_HOMESERVER", "") or ""
        ).rstrip("/")
        token = (
            getattr(pconfig, "token", None)
            or get_secret("MATRIX_ACCESS_TOKEN", "")
            or ""
        )
        if not homeserver or not token:
            return send_error(
                "Matrix not configured (MATRIX_HOMESERVER, MATRIX_ACCESS_TOKEN required)"
            )
        headers = {"Authorization": f"Bearer {token}"}
        async with aiohttp.ClientSession(headers=headers) as session:
            delivery = _HTTPDelivery(session, homeserver)
            # The scheduler may submit this coroutine across threads. Bound the task itself.
            return await asyncio.wait_for(
                delivery.send(chat_id, message, thread_id), timeout=45
            )
    except TimeoutError:
        return send_error(f"Matrix target '{chat_id}': API timeout (45s)")
    except Exception as exc:
        return send_error(f"Matrix target '{chat_id}': {exc}")
