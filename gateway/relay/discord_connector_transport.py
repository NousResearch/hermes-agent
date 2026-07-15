"""Credential-free RelayTransport for the local public Discord connector.

The gateway owns neither a Discord token nor a generic Discord API surface.  It
can only poll normalized public-guild events, acknowledge an exact delivery,
prove a public target, read one bounded public-history page, and submit one
bounded public-message send.  Every call uses a fresh Unix connection and
reciprocally authenticates the token-owning connector by UID and the exact
current systemd ``MainPID``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import struct
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from gateway.canonical_writer_client import (
    ExactServerMainPidAuthorizer,
    ServerPeerCredentials,
    linux_server_peer_credentials,
)
from gateway.config import Platform
from gateway.discord_connector_protocol import (
    DISCORD_CONNECTOR_THREAD_TARGET_TYPES,
    MAX_FRAME_BYTES,
    DiscordConnectorEvent,
    DiscordConnectorHistoryAuthority,
    DiscordConnectorHistoryPage,
    DiscordConnectorKind,
    DiscordConnectorProtocolError,
    DiscordConnectorTarget,
    DiscordConnectorTargetType,
    canonical_json_bytes,
    decode_frame,
    request_message,
    sha256_json,
    validate_receipt,
)
from gateway.platforms.base import MessageEvent, MessageType
from gateway.relay.descriptor import CapabilityDescriptor
from gateway.relay.transport import InboundHandler, PassthroughHandler
from gateway.session import SessionSource

logger = logging.getLogger(__name__)

MAX_RESPONSE_BYTES = 128 * 1024
_FRAME_HEADER = struct.Struct("!I")


class DiscordConnectorTransportError(RuntimeError):
    """Stable local-boundary failure without message or credential material."""

    def __init__(self, code: str, *, dispatch_uncertain: bool = False) -> None:
        self.code = code
        self.dispatch_uncertain = dispatch_uncertain
        super().__init__(code)


def _receive_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise OSError("connector connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _receive_response(sock: socket.socket) -> Mapping[str, Any]:
    (size,) = _FRAME_HEADER.unpack(_receive_exact(sock, _FRAME_HEADER.size))
    if size == 0 or size > MAX_RESPONSE_BYTES:
        raise DiscordConnectorProtocolError("invalid_connector_response_size")
    return decode_frame(_receive_exact(sock, size))


def _event_to_gateway(event: DiscordConnectorEvent) -> MessageEvent:
    target = event.target
    is_thread = target.target_type in DISCORD_CONNECTOR_THREAD_TARGET_TYPES
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id=target.channel_id,
        chat_type="thread" if is_thread else "channel",
        chat_name=target.channel_id,
        user_id=event.author_id,
        user_name=event.author_name,
        thread_id=target.channel_id if is_thread else None,
        is_bot=event.author_is_bot,
        scope_id=target.guild_id,
        parent_chat_id=target.parent_channel_id,
        message_id=event.event_id,
        delivered_via_upstream_relay=True,
    )
    return MessageEvent(
        text=event.content,
        message_type=MessageType.TEXT,
        source=source,
        message_id=event.event_id,
        reply_to_message_id=event.reply_to_message_id,
    )


class DiscordConnectorRelayTransport:
    """Existing RelayAdapter backed by the privileged local connector."""

    requires_stable_idempotency_key = True

    def __init__(
        self,
        socket_path: str | os.PathLike[str],
        *,
        server_authorizer: ExactServerMainPidAuthorizer,
        server_peer_getter: Callable[[socket.socket], ServerPeerCredentials] = (
            linux_server_peer_credentials
        ),
        connect_timeout_seconds: float = 2.0,
        request_timeout_seconds: float = 10.0,
        event_wait_ms: int = 1_000,
    ) -> None:
        path = Path(socket_path)
        if not path.is_absolute() or path != Path(os.path.normpath(os.fspath(path))):
            raise ValueError("Discord connector socket path is invalid")
        if not callable(getattr(server_authorizer, "authorize", None)):
            raise TypeError("Discord connector server authorizer is required")
        if not callable(server_peer_getter):
            raise TypeError("Discord connector peer getter is required")
        if not 0 < connect_timeout_seconds <= 30:
            raise ValueError("Discord connector connect timeout is invalid")
        if not 0 < request_timeout_seconds <= 30:
            raise ValueError("Discord connector request timeout is invalid")
        if not 0 <= event_wait_ms <= 5_000:
            raise ValueError("Discord connector event wait is invalid")
        self.socket_path = str(path)
        self.server_authorizer = server_authorizer
        self.server_peer_getter = server_peer_getter
        self.connect_timeout_seconds = float(connect_timeout_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.event_wait_ms = event_wait_ms
        self._inbound: InboundHandler | None = None
        self._passthrough: PassthroughHandler | None = None
        self._descriptor: CapabilityDescriptor | None = None
        self._poller: asyncio.Task[None] | None = None
        self._connected = False
        self._owner_pid = os.getpid()
        # RelayAdapter uses this fixed identity only to tag egress mechanically.
        self._identities = [("discord", "")]

    def _require_owner(self) -> None:
        if os.getpid() != self._owner_pid:
            raise DiscordConnectorTransportError("connector_client_wrong_process")

    def _authorized(self, peer: object) -> bool:
        if not isinstance(peer, ServerPeerCredentials):
            return False
        if peer.pid <= 1 or peer.uid < 0 or peer.gid < 0:
            return False
        try:
            return self.server_authorizer.authorize(peer) is True
        except Exception:
            return False

    def _request_sync(
        self,
        kind: DiscordConnectorKind,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._require_owner()
        message = request_message(kind, payload)
        body = canonical_json_bytes(message)
        if not body or len(body) > MAX_FRAME_BYTES:
            raise DiscordConnectorTransportError("connector_request_frame_invalid")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.set_inheritable(False)
        may_have_dispatched = False
        try:
            sock.settimeout(self.connect_timeout_seconds)
            sock.connect(self.socket_path)
            peer = self.server_peer_getter(sock)
            if not self._authorized(peer):
                raise DiscordConnectorTransportError("connector_server_unauthorized")
            sock.settimeout(self.request_timeout_seconds)
            may_have_dispatched = True
            sock.sendall(_FRAME_HEADER.pack(len(body)) + body)
            response = _receive_response(sock)
            peer_after = self.server_peer_getter(sock)
            if peer_after != peer or not self._authorized(peer_after):
                raise DiscordConnectorTransportError(
                    "connector_server_unauthorized",
                    dispatch_uncertain=kind is DiscordConnectorKind.MESSAGE_SEND,
                )
            return validate_receipt(
                response,
                expected_kind=kind,
                expected_request_id=str(message["request_id"]),
            )
        except DiscordConnectorTransportError:
            raise
        except (OSError, socket.timeout, TypeError, ValueError) as exc:
            raise DiscordConnectorTransportError(
                "connector_transport_failed",
                dispatch_uncertain=(
                    may_have_dispatched
                    and kind is DiscordConnectorKind.MESSAGE_SEND
                ),
            ) from exc
        finally:
            sock.close()

    async def _request(
        self,
        kind: DiscordConnectorKind,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._request_sync, kind, payload)

    def set_inbound_handler(self, handler: InboundHandler) -> None:
        if not callable(handler):
            raise TypeError("Discord connector inbound handler is invalid")
        self._inbound = handler

    def set_passthrough_handler(self, handler: PassthroughHandler) -> None:
        # This connector intentionally has no generic passthrough plane.
        if not callable(handler):
            raise TypeError("Discord connector passthrough handler is invalid")
        self._passthrough = handler

    async def connect(self) -> bool:
        if self._connected:
            return True
        response = await self._request(
            DiscordConnectorKind.HELLO,
            {"consumer": "hermes-relay-adapter"},
        )
        result = response.get("result")
        if response.get("status") != "ok" or not isinstance(result, Mapping):
            return False
        descriptor = result.get("descriptor")
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "contract_version",
            "platform",
            "label",
            "max_message_length",
            "supports_draft_streaming",
            "supports_edit",
            "supports_threads",
            "markdown_dialect",
            "len_unit",
            "emoji",
            "platform_hint",
            "pii_safe",
        }:
            raise DiscordConnectorTransportError("connector_descriptor_invalid")
        parsed = CapabilityDescriptor.from_json(
            json.dumps(dict(descriptor), ensure_ascii=False)
        )
        if (
            parsed.contract_version != 1
            or parsed.platform != "discord"
            or parsed.max_message_length != 2_000
            or parsed.supports_draft_streaming
            or parsed.supports_edit
            or not parsed.supports_threads
            or parsed.len_unit != "chars"
        ):
            raise DiscordConnectorTransportError("connector_descriptor_invalid")
        self._descriptor = parsed
        self._connected = True
        self._poller = asyncio.create_task(
            self._poll_events(), name="discord-public-connector-events"
        )
        return True

    async def disconnect(self) -> None:
        self._connected = False
        poller = self._poller
        self._poller = None
        if poller is not None:
            poller.cancel()
            try:
                await poller
            except asyncio.CancelledError:
                pass

    async def handshake(self) -> CapabilityDescriptor:
        if not self._connected or self._descriptor is None:
            raise DiscordConnectorTransportError("connector_not_connected")
        return self._descriptor

    async def _poll_events(self) -> None:
        while self._connected:
            try:
                response = await self._request(
                    DiscordConnectorKind.EVENT_NEXT,
                    {"wait_ms": self.event_wait_ms},
                )
                status = response.get("status")
                if status == "idle":
                    continue
                if status != "ok":
                    await asyncio.sleep(0.25)
                    continue
                raw = response.get("result")
                if not isinstance(raw, Mapping) or set(raw) != {
                    "delivery_id",
                    "event_id",
                    "event_sha256",
                    "event",
                }:
                    raise DiscordConnectorProtocolError("invalid_event_delivery")
                event = DiscordConnectorEvent.from_mapping(raw["event"])
                if (
                    raw["event_id"] != event.event_id
                    or raw["event_sha256"] != event.sha256
                ):
                    raise DiscordConnectorProtocolError("invalid_event_delivery")
                delivery_id = str(uuid.UUID(str(raw["delivery_id"])))
                if delivery_id != raw["delivery_id"]:
                    raise DiscordConnectorProtocolError("invalid_event_delivery")
                handler = self._inbound
                if handler is None:
                    # No acceptance means no ACK; the lease makes it replayable.
                    await asyncio.sleep(0.25)
                    continue
                await handler(_event_to_gateway(event))
                ack = await self._request(
                    DiscordConnectorKind.EVENT_ACK,
                    {
                        "delivery_id": delivery_id,
                        "event_id": event.event_id,
                        "event_sha256": event.sha256,
                    },
                )
                ack_result = ack.get("result")
                if (
                    ack.get("status") != "ok"
                    or not isinstance(ack_result, Mapping)
                    or set(ack_result) != {"event_id", "event_sha256", "acked"}
                    or ack_result.get("event_id") != event.event_id
                    or ack_result.get("event_sha256") != event.sha256
                    or ack_result.get("acked") is not True
                ):
                    raise DiscordConnectorProtocolError("invalid_event_ack_receipt")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "public Discord connector event poll failed",
                    exc_info=True,
                )
                await asyncio.sleep(0.5)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        response = await self._request(
            DiscordConnectorKind.TARGET_GET, {"channel_id": str(chat_id)}
        )
        raw = response.get("result")
        if response.get("status") != "ok" or not isinstance(raw, Mapping):
            return {"name": str(chat_id), "type": "forbidden"}
        if set(raw) != {"target"}:
            raise DiscordConnectorTransportError("connector_target_receipt_invalid")
        target = DiscordConnectorTarget.from_mapping(raw["target"])
        return {
            "name": target.channel_id,
            "type": (
                "thread"
                if target.target_type in DISCORD_CONNECTOR_THREAD_TARGET_TYPES
                else "channel"
            ),
            "guild_id": target.guild_id,
            "parent_channel_id": target.parent_channel_id,
        }

    def read_guild_history(
        self,
        channel_id: str,
        *,
        limit: int,
        before_message_id: str | None = None,
        after_message_id: str | None = None,
        authority: DiscordConnectorHistoryAuthority,
    ) -> dict[str, Any]:
        """Read one exact authorized page through the credential-owning connector.

        This synchronous method exists for the service-gated model tool.  It
        reuses the same reciprocal MainPID/UID checks and strict receipt parser
        as the relay transport; it does not expose the generic request method.
        """

        if not isinstance(authority, DiscordConnectorHistoryAuthority):
            raise DiscordConnectorTransportError("connector_history_authority_invalid")
        response = self._request_sync(
            DiscordConnectorKind.HISTORY_FETCH,
            {
                "channel_id": channel_id,
                "limit": limit,
                "before_message_id": before_message_id,
                "after_message_id": after_message_id,
                "authority": authority.to_mapping(),
            },
        )
        result = response.get("result")
        if response.get("status") != "ok" or not isinstance(result, Mapping):
            raise DiscordConnectorTransportError("connector_history_blocked")
        if set(result) != {"page", "page_sha256", "authority_sha256"}:
            raise DiscordConnectorTransportError("connector_history_receipt_invalid")
        try:
            page = DiscordConnectorHistoryPage.from_mapping(result["page"])
        except DiscordConnectorProtocolError as exc:
            raise DiscordConnectorTransportError(
                "connector_history_receipt_invalid"
            ) from exc
        if (
            result.get("page_sha256") != page.sha256
            or result.get("authority_sha256") != authority.sha256
            or page.target.channel_id != channel_id
            or page.limit != limit
            or page.before_message_id != before_message_id
            or page.after_message_id != after_message_id
        ):
            raise DiscordConnectorTransportError("connector_history_receipt_invalid")
        return page.to_mapping()

    async def send_outbound(
        self, action: Dict[str, Any], *, platform: Optional[str] = None
    ) -> Dict[str, Any]:
        if set(action) - {"op", "chat_id", "content", "reply_to", "metadata"}:
            return {
                "success": False,
                "error": "unsupported outbound shape",
                "error_kind": "blocked_before_dispatch",
            }
        if action.get("op") != "send" or platform not in {None, "discord"}:
            return {
                "success": False,
                "error": "unsupported connector operation",
                "error_kind": "blocked_before_dispatch",
            }
        metadata = action.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            return {
                "success": False,
                "error": "invalid connector metadata",
                "error_kind": "blocked_before_dispatch",
            }
        allowed_metadata = {
            "scope_id",
            "thread_id",
            "notify",
            "connector_idempotency_key",
            "non_conversational",
            "non_conversational_history",
            # Read-only audit metadata stays in the gateway; it is accepted so
            # normal Canonical Brain receipt recording composes with this edge,
            # but is never serialized into the fixed connector protocol.
            "_canonical_brain_audit",
        }
        if set(metadata) - allowed_metadata:
            return {
                "success": False,
                "error": "unsupported connector metadata",
                "error_kind": "blocked_before_dispatch",
            }
        if "_canonical_brain_audit" in metadata and not isinstance(
            metadata["_canonical_brain_audit"], Mapping
        ):
            return {
                "success": False,
                "error": "invalid connector audit metadata",
                "error_kind": "blocked_before_dispatch",
            }
        try:
            target_receipt = await self._request(
                DiscordConnectorKind.TARGET_GET,
                {"channel_id": str(action.get("chat_id", ""))},
            )
            target_result = target_receipt.get("result")
            if (
                target_receipt.get("status") != "ok"
                or not isinstance(target_result, Mapping)
                or set(target_result) != {"target"}
            ):
                return {
                    "success": False,
                    "error": "Discord target is not an allowed guild surface",
                    "error_kind": "blocked_before_dispatch",
                }
            target = DiscordConnectorTarget.from_mapping(target_result["target"])
            scope_id = metadata.get("scope_id")
            if scope_id is not None and str(scope_id) != target.guild_id:
                return {
                    "success": False,
                    "error": "Discord target scope mismatch",
                    "error_kind": "blocked_before_dispatch",
                }
            thread_id = metadata.get("thread_id")
            if thread_id is not None and str(thread_id) != target.channel_id:
                return {
                    "success": False,
                    "error": "Discord thread binding mismatch",
                    "error_kind": "blocked_before_dispatch",
                }
            key = metadata.get("connector_idempotency_key")
            if key is None:
                key = f"gateway:{uuid.uuid4()}"
            deadline_unix_ms = int(time.time() * 1000) + min(
                int(self.request_timeout_seconds * 1_000), 30_000
            )
            send = await self._request(
                DiscordConnectorKind.MESSAGE_SEND,
                {
                    "idempotency_key": str(key),
                    "target": target.to_mapping(),
                    "content": action.get("content"),
                    "reply_to_message_id": action.get("reply_to"),
                    "deadline_unix_ms": deadline_unix_ms,
                },
            )
        except DiscordConnectorTransportError as exc:
            return {
                "success": False,
                "error": (
                    "Discord dispatch outcome is uncertain"
                    if exc.dispatch_uncertain
                    else exc.code
                ),
                "error_kind": (
                    "dispatch_uncertain"
                    if exc.dispatch_uncertain
                    else (
                        "transient"
                        if exc.code == "connector_transport_failed"
                        else "blocked_before_dispatch"
                    )
                ),
                "retryable": (
                    not exc.dispatch_uncertain
                    and exc.code == "connector_transport_failed"
                ),
            }
        except (DiscordConnectorProtocolError, TypeError, ValueError):
            return {
                "success": False,
                "error": "Discord connector request rejected",
                "error_kind": "blocked_before_dispatch",
            }

        result = send.get("result")
        if not isinstance(result, Mapping) or set(result) != {
            "target",
            "content_sha256",
            "idempotency_key",
            "message_id",
            "readback_verified",
        }:
            return {
                "success": False,
                "error": "Discord receipt is invalid after dispatch",
                "error_kind": "dispatch_uncertain",
            }
        try:
            receipt_target = DiscordConnectorTarget.from_mapping(result["target"])
        except DiscordConnectorProtocolError:
            return {
                "success": False,
                "error": "Discord receipt is invalid after dispatch",
                "error_kind": "dispatch_uncertain",
            }
        if (
            receipt_target != target
            or result.get("idempotency_key") != str(key)
            or result.get("content_sha256")
            != sha256_json({"content": action.get("content")})
        ):
            return {
                "success": False,
                "error": "Discord receipt binding mismatch after dispatch",
                "error_kind": "dispatch_uncertain",
            }
        if (
            send.get("status") != "ok"
            or result.get("readback_verified") is not True
            or not isinstance(result.get("message_id"), str)
            or not result["message_id"].isdigit()
            or result["message_id"].startswith("0")
            or len(result["message_id"]) > 25
        ):
            return {
                "success": False,
                "error": (
                    "Discord dispatch outcome is uncertain"
                    if send.get("status") == "dispatch_uncertain"
                    else "Discord connector blocked the send"
                ),
                "error_kind": (
                    "dispatch_uncertain"
                    if send.get("status") != "blocked"
                    else "blocked_before_dispatch"
                ),
            }
        return {"success": True, "message_id": result["message_id"]}

    async def send_interrupt(
        self, session_key: str, reason: Optional[str] = None
    ) -> None:
        # Interrupts are handled inside the gateway; no connector authority exists.
        return None

    async def go_idle(self, timeout_s: float = 10.0) -> bool:
        return True

    async def send_follow_up(
        self, action: Dict[str, Any], *, platform: Optional[str] = None
    ) -> Dict[str, Any]:
        return {"success": False, "error": "connector follow-up unsupported"}


__all__ = [
    "DiscordConnectorRelayTransport",
    "DiscordConnectorTransportError",
]
