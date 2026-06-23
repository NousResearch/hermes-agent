"""
Infoflow Protobuf Frame codec.

Handles encoding/decoding of WebSocket frames for Baidu Infoflow messaging.
Based on the NodeJS SDK frame.proto definition.
"""

from dataclasses import dataclass, field
from typing import List

import betterproto


@dataclass
class Header(betterproto.Message):
    """Header defines frame header information."""
    key: str = betterproto.string_field(1)
    value: str = betterproto.string_field(2)


@dataclass
class Frame(betterproto.Message):
    """Frame defines WebSocket communication frame structure.

    Fields:
        seq_id: Sequence number for request/response matching (uint64)
        log_id: Log trace ID for distributed tracing (string)
        service: Service identifier (int32)
        method: Frame type: 0=CONTROL, 1=DATA, 2=REQUEST, 3=RESPONSE (int32)
        headers: Frame header list (repeated Header)
        payload: Business payload - JSON format, max 64KB (bytes)
    """
    seq_id: int = betterproto.uint64_field(1)
    log_id: str = betterproto.string_field(2)
    service: int = betterproto.int32_field(3)
    method: int = betterproto.int32_field(4)
    headers: List[Header] = betterproto.message_field(5)
    payload: bytes = betterproto.bytes_field(6)


class FrameType:
    """Frame method types."""
    CONTROL = 0
    DATA = 1
    REQUEST = 2
    RESPONSE = 3


def decode_frame(data: bytes) -> Frame:
    """Decode binary data to Frame."""
    frame = Frame()
    frame.parse(data)
    return frame


def encode_frame(frame: Frame) -> bytes:
    """Encode Frame to binary."""
    return bytes(frame)


__all__ = ['Header', 'Frame', 'FrameType', 'decode_frame', 'encode_frame']
