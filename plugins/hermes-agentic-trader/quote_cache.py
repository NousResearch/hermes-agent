"""In-memory quote binding for Hermes Agentic Trader.

A write may only reference a quote object that Hermes observed from the
matching quote tool in the same session. The cache intentionally stores no
private keys and is process-local; restart clears it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

QUOTE_TTL_SECONDS = 120.0
MAX_QUOTES = 64

_QUOTE_TOOL_FOR_WRITE = {
    "execute_swap": "get_swap_quote",
    "submit_gasless_swap": "get_gasless_quote",
}


@dataclass(frozen=True)
class QuoteRecord:
    session_id: str
    quote_tool: str
    request_args: dict[str, Any]
    quote_hash: str
    captured_at: float


_lock = threading.RLock()
_records: dict[tuple[str, str, str], QuoteRecord] = {}


def _canonical_hash(value: Mapping[str, Any]) -> Optional[str]:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _decode_json_layers(value: Any, *, max_depth: int = 4) -> Any:
    current = value
    for _ in range(max_depth):
        if isinstance(current, str):
            text = current.strip()
            if not text:
                return current
            try:
                current = json.loads(text)
            except json.JSONDecodeError:
                return current
            continue
        if isinstance(current, dict) and "result" in current:
            candidate = current.get("result")
            if isinstance(candidate, (str, dict)):
                current = candidate
                continue
        break
    return current


def _extract_quote_data(result: Any) -> Optional[dict[str, Any]]:
    payload = _decode_json_layers(result)
    if not isinstance(payload, dict):
        return None

    data = payload.get("data")
    if isinstance(data, dict):
        return data

    if isinstance(payload.get("transaction"), dict):
        return payload
    trade = payload.get("trade")
    if isinstance(trade, dict) and isinstance(trade.get("eip712"), dict):
        return payload
    return None


def _evict(now: float) -> None:
    expired = [
        key
        for key, record in _records.items()
        if now - record.captured_at > QUOTE_TTL_SECONDS
    ]
    for key in expired:
        _records.pop(key, None)

    overflow = len(_records) - MAX_QUOTES
    if overflow > 0:
        oldest = sorted(_records.items(), key=lambda item: item[1].captured_at)
        for key, _record in oldest[:overflow]:
            _records.pop(key, None)


def capture_quote(
    *,
    tool_name: str,
    args: Mapping[str, Any],
    result: Any,
    session_id: Optional[str],
    now: Optional[float] = None,
) -> bool:
    """Capture one quote result and bind it to the request that produced it."""
    if tool_name not in {"get_swap_quote", "get_gasless_quote"}:
        return False
    if not session_id:
        return False

    quote = _extract_quote_data(result)
    if quote is None:
        return False

    quote_hash = _canonical_hash(quote)
    if quote_hash is None:
        return False

    timestamp = time.monotonic() if now is None else float(now)
    record = QuoteRecord(
        session_id=str(session_id),
        quote_tool=tool_name,
        request_args=copy.deepcopy(dict(args)),
        quote_hash=quote_hash,
        captured_at=timestamp,
    )
    key = (record.session_id, record.quote_tool, quote_hash)

    with _lock:
        _evict(timestamp)
        _records[key] = record
        _evict(timestamp)
    return True


def resolve_execution_quote(
    *,
    tool_name: str,
    args: Mapping[str, Any],
    session_id: Optional[str],
    consume: bool = False,
    now: Optional[float] = None,
) -> Optional[QuoteRecord]:
    """Return the matching fresh observed quote, optionally consuming it."""
    quote_tool = _QUOTE_TOOL_FOR_WRITE.get(tool_name)
    if quote_tool is None or not session_id:
        return None

    quote = args.get("quoteData")
    if not isinstance(quote, dict):
        return None

    quote_hash = _canonical_hash(quote)
    if quote_hash is None:
        return None

    timestamp = time.monotonic() if now is None else float(now)
    key = (str(session_id), quote_tool, quote_hash)

    with _lock:
        _evict(timestamp)
        record = _records.get(key)
        if record is None:
            return None
        if consume:
            _records.pop(key, None)
        return record


def clear_quote_cache() -> None:
    """Test/support helper; does not persist anything."""
    with _lock:
        _records.clear()
