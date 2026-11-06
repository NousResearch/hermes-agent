"""Yuanbao send ACK decoding: a WS Response can carry ``head.status == 0`` and still reject the
message in its business payload (protobuf field 1 ``code``, field 2 ``message``), e.g. code 999
for an over-long reply. Treat either signal as a failed delivery instead of a success."""

from __future__ import annotations

import logging
from typing import Optional

from plugins.platforms.yuanbao.yuanbao_proto import _decode_varint

logger = logging.getLogger(__name__)

MALFORMED_SEND_RESPONSE = "malformed send response"


def _decode_varint_strict(data: bytes, pos: int) -> tuple[int, int]:
    """Like ``_decode_varint`` but raises when the varint is truncated (continuation bit left on)."""
    start = pos
    value, pos = _decode_varint(data, pos)
    if pos == start or data[pos - 1] & 0x80:
        raise ValueError("truncated varint")
    return value, pos


def decode_send_ack_payload(data: bytes) -> tuple[int, str]:
    """Send-response protobuf: field 1 varint ``code``, field 2 string ``message``.

    Raises if the payload is truncated/malformed or has no decodable field-1 code.
    """
    pos = 0
    code: Optional[int] = None
    message = ""
    while pos < len(data):
        tag, pos = _decode_varint_strict(data, pos)
        fn, wt = tag >> 3, tag & 7
        if wt == 0:
            val, pos = _decode_varint_strict(data, pos)
            if fn == 1 and code is None:
                code = val
        elif wt == 2:
            length, pos = _decode_varint_strict(data, pos)
            if pos + length > len(data):
                raise ValueError("truncated length-delimited field")
            val = data[pos:pos + length]
            pos += length
            if fn == 2 and not message:
                message = val.decode("utf-8", errors="replace")
        elif wt in (1, 5):
            n = 8 if wt == 1 else 4
            if pos + n > len(data):
                raise ValueError("truncated fixed field")
            pos += n
        else:
            raise ValueError(f"unknown wire type {wt}")
    if code is None:
        raise ValueError("missing business code")
    return code, message


def _format_send_nack(status: Optional[int], code: Optional[int], message: str) -> str:
    msg = (message or "").replace("\n", " ").replace("\r", " ")[:160]
    return f"send rejected: status={0 if status is None else status} code={'-' if code is None else code} message={msg}"


def ack_from_response(adapter_name: str, response) -> dict:
    """Map a WS Response ``{head, data?}`` to ``{success, msg_key|error}``.

    Fail closed on a non-dict response, nonzero ``head.status``, nonzero business ``code`` or
    nonempty malformed ``data``. Fail open (success) when status is missing/0 and ``data`` is
    absent or empty: there is no rejection signal.
    """
    if not isinstance(response, dict):
        logger.warning("[%s] send rejected: non-dict response type=%s", adapter_name, type(response).__name__)
        return {"success": False, "error": MALFORMED_SEND_RESPONSE}
    head = response.get("head") if isinstance(response.get("head"), dict) else {}
    status = head.get("status")
    data = response.get("data")
    if status is not None and status != 0:
        error = _format_send_nack(status, None, "")
        logger.warning("[%s] %s", adapter_name, error)
        return {"success": False, "error": error}
    if data:
        try:
            if not isinstance(data, (bytes, bytearray)):
                raise TypeError(type(data).__name__)
            biz_code, biz_msg = decode_send_ack_payload(bytes(data))
        except (TypeError, ValueError, IndexError):
            logger.warning("[%s] send rejected: status=%s malformed payload", adapter_name, 0 if status is None else status)
            return {"success": False, "error": MALFORMED_SEND_RESPONSE}
        if biz_code:
            error = _format_send_nack(0 if status is None else status, biz_code, biz_msg)
            logger.warning("[%s] %s", adapter_name, error)
            return {"success": False, "error": error}
    return {"success": True, "msg_key": head.get("msg_id", "")}
