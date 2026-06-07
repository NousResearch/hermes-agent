"""Live Cursor model catalog via GetUsableModels."""

from __future__ import annotations

import logging
import socket
import ssl
import uuid
from urllib.parse import urlsplit

from agent.cursor.constants import (
    CURSOR_API_URL,
    CURSOR_CLIENT_VERSION,
    CURSOR_GET_USABLE_MODELS_PATH,
    normalize_cursor_model_id,
)
from agent.cursor.proto import agent_pb2

logger = logging.getLogger(__name__)


def _parse_usable_models_payload(payload: bytes) -> list[str]:
    response = agent_pb2.GetUsableModelsResponse()
    response.ParseFromString(payload)
    model_ids: list[str] = []
    seen: set[str] = set()
    for details in response.models:
        model_id = normalize_cursor_model_id(details.model_id)
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        model_ids.append(model_id)
    return sorted(model_ids)


def fetch_cursor_usable_models(
    *,
    api_key: str,
    base_url: str = CURSOR_API_URL,
    timeout: float = 12.0,
) -> list[str] | None:
    """Return model IDs from Cursor's GetUsableModels RPC, or None on failure."""

    token = (api_key or "").strip()
    if not token:
        return None

    try:
        import h2.connection
        import h2.events
    except ImportError:
        logger.debug("fetch_cursor_usable_models: h2 extra not installed")
        return None

    parsed = urlsplit(base_url or CURSOR_API_URL)
    host = parsed.hostname or "api2.cursor.sh"
    port = parsed.port or 443

    request_body = agent_pb2.GetUsableModelsRequest().SerializeToString()
    response_buffer = bytearray()

    connection = h2.connection.H2Connection()
    sock: socket.socket | None = None
    ssl_sock: ssl.SSLSocket | None = None

    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.settimeout(timeout)
        context = ssl.create_default_context()
        context.set_alpn_protocols(["h2"])
        ssl_sock = context.wrap_socket(sock, server_hostname=host)
        if ssl_sock.selected_alpn_protocol() != "h2":
            return None

        connection.initiate_connection()
        pending = connection.data_to_send()
        if pending:
            ssl_sock.sendall(pending)

        stream_id = connection.get_next_available_stream_id()
        headers = [
            (":method", "POST"),
            (":path", CURSOR_GET_USABLE_MODELS_PATH),
            (":scheme", "https"),
            (":authority", host),
            ("authorization", f"Bearer {token}"),
            ("content-type", "application/proto"),
            ("connect-protocol-version", "1"),
            ("te", "trailers"),
            ("x-ghost-mode", "true"),
            ("x-cursor-client-version", CURSOR_CLIENT_VERSION),
            ("x-cursor-client-type", "cli"),
            ("x-request-id", str(uuid.uuid4())),
        ]
        connection.send_headers(stream_id, headers, end_stream=False)
        connection.send_data(stream_id, request_body, end_stream=True)
        pending = connection.data_to_send()
        if pending:
            ssl_sock.sendall(pending)

        grpc_status: str | None = None
        while True:
            try:
                chunk = ssl_sock.recv(65535)
            except socket.timeout:
                break
            if not chunk:
                break

            events = connection.receive_data(chunk)
            pending = connection.data_to_send()
            if pending:
                ssl_sock.sendall(pending)

            for event in events:
                if isinstance(event, h2.events.ResponseReceived):
                    status = str(dict(event.headers).get(":status", ""))
                    if status and status != "200":
                        return None
                elif isinstance(event, h2.events.TrailersReceived):
                    grpc_status = str(dict(event.headers).get("grpc-status", "") or "")
                elif isinstance(event, h2.events.DataReceived):
                    connection.acknowledge_received_data(
                        event.flow_controlled_length,
                        event.stream_id,
                    )
                    response_buffer.extend(event.data)
                elif isinstance(event, h2.events.StreamEnded):
                    break

        if grpc_status and grpc_status not in {"", "0"}:
            return None
        if not response_buffer:
            return None

        model_ids = _parse_usable_models_payload(bytes(response_buffer))
        return model_ids or None
    except Exception as exc:
        logger.debug("fetch_cursor_usable_models failed: %s", exc)
        return None
    finally:
        try:
            if ssl_sock is not None:
                ssl_sock.close()
            elif sock is not None:
                sock.close()
        except Exception:
            pass
