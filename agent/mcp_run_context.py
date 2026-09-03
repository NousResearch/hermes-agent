"""Private, per-run metadata forwarded to opted-in MCP servers.

The value lives in a ContextVar so concurrent API runs cannot overwrite one
another. It is deliberately separate from model messages and tool arguments.
"""

from __future__ import annotations

import base64
import binascii
import copy
import json
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping, Optional


MCP_RUN_METADATA_HEADER = "X-Hermes-MCP-Metadata"
MAX_MCP_RUN_METADATA_BYTES = 8 * 1024
MAX_MCP_RUN_METADATA_DEPTH = 64

_mcp_run_metadata: ContextVar[Optional[dict[str, Any]]] = ContextVar(
    "mcp_run_metadata", default=None
)


def _copy_validated_metadata(value: dict[str, Any]) -> dict[str, Any]:
    """Copy metadata after an iterative nesting-depth check."""
    pending: list[tuple[Any, int]] = [(value, 1)]
    while pending:
        item, depth = pending.pop()
        if isinstance(item, dict):
            if depth > MAX_MCP_RUN_METADATA_DEPTH:
                raise ValueError("MCP metadata header is too deeply nested")
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            if depth > MAX_MCP_RUN_METADATA_DEPTH:
                raise ValueError("MCP metadata header is too deeply nested")
            pending.extend((child, depth + 1) for child in item)
    try:
        return copy.deepcopy(value)
    except RecursionError as exc:
        raise ValueError("MCP metadata header is too deeply nested") from exc


def decode_mcp_run_metadata_header(raw: str) -> dict[str, Any]:
    """Decode an unpadded base64url JSON object from the API header."""
    if not isinstance(raw, str) or not raw:
        raise ValueError("MCP metadata header is empty")
    if len(raw) > ((MAX_MCP_RUN_METADATA_BYTES + 2) // 3) * 4:
        raise ValueError("MCP metadata header is too large")

    padded = raw + "=" * (-len(raw) % 4)
    try:
        payload = base64.b64decode(
            padded.encode("ascii"), altchars=b"-_", validate=True
        )
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise ValueError("MCP metadata header is not valid base64url") from exc

    if len(payload) > MAX_MCP_RUN_METADATA_BYTES:
        raise ValueError("MCP metadata header is too large")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("MCP metadata header is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("MCP metadata header must encode a JSON object")
    return _copy_validated_metadata(value)


@contextmanager
def mcp_run_metadata(
    value: Optional[Mapping[str, Any]],
) -> Iterator[None]:
    """Bind a defensive copy of private MCP metadata for the current run."""
    bound = _copy_validated_metadata(dict(value)) if value is not None else None
    token = _mcp_run_metadata.set(bound)
    try:
        yield
    finally:
        _mcp_run_metadata.reset(token)


def read_mcp_run_metadata() -> Optional[dict[str, Any]]:
    """Return a defensive copy of the current run metadata, if any."""
    value = _mcp_run_metadata.get()
    return copy.deepcopy(value) if value is not None else None


__all__ = [
    "MAX_MCP_RUN_METADATA_BYTES",
    "MAX_MCP_RUN_METADATA_DEPTH",
    "MCP_RUN_METADATA_HEADER",
    "decode_mcp_run_metadata_header",
    "mcp_run_metadata",
    "read_mcp_run_metadata",
]
