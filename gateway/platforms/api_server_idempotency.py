"""Authenticated retry identity and cache for the OpenAI-compatible API routes."""

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Optional, Sequence


class IdempotencyCache:
    """In-memory idempotency cache with TTL and basic LRU semantics."""

    def __init__(self, max_items: int = 1000, ttl_seconds: int = 300):
        self._store = OrderedDict()
        self._inflight: dict[tuple[tuple[str, ...], str, str], asyncio.Task[Any]] = {}
        self._ttl = ttl_seconds
        self._max = max_items

    def _purge(self):
        now = time.time()
        expired = [key for key, value in self._store.items() if now - value["ts"] > self._ttl]
        for key in expired:
            self._store.pop(key, None)
        while len(self._store) > self._max:
            self._store.popitem(last=False)

    async def get_or_set(self, scope: Optional[tuple[str, ...]], key: str, fingerprint: str, compute):
        if scope is None:
            return await compute()

        self._purge()
        store_key = (scope, key)
        item = self._store.get(store_key)
        if item and item["fp"] == fingerprint:
            return item["resp"]
        inflight_key = (scope, key, fingerprint)
        task = self._inflight.get(inflight_key)
        if task is None:
            async def _compute_and_store():
                response = await compute()
                self._store[store_key] = {"resp": response, "fp": fingerprint, "ts": time.time()}
                self._purge()
                return response

            task = asyncio.create_task(_compute_and_store())
            self._inflight[inflight_key] = task

            def _clear_inflight(done_task: asyncio.Task[Any]) -> None:
                if self._inflight.get(inflight_key) is done_task:
                    self._inflight.pop(inflight_key, None)

            task.add_done_callback(_clear_inflight)
        return await asyncio.shield(task)


def make_request_fingerprint(body: dict[str, Any], keys: Sequence[str]) -> str:
    subset = {key: body.get(key) for key in keys}
    return hashlib.sha256(repr(subset).encode("utf-8")).hexdigest()


def make_normalized_request_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def make_fingerprint_digest(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    return hashlib.sha256(b"hermes-api-idempotency-route-value-v1\0" + value.encode("utf-8")).hexdigest()


def _header_scope(request, header_name: str) -> tuple[str, str]:
    if header_name not in request.headers:
        return "absent", ""
    return "present", request.headers.get(header_name, "").strip()


def scope(adapter, request, endpoint: str, profile: Optional[str]) -> Optional[tuple[str, ...]]:
    """Partition retries by authenticated caller, endpoint, and declared session scope."""
    expected_key = adapter._expected_api_key()
    if not expected_key:
        return None

    normalized_profile = (profile or "default").strip() or "default"
    credential_digest = hashlib.sha256(
        b"hermes-api-idempotency-credential-v1\0" + expected_key.encode("utf-8")
    ).hexdigest()
    session_id_state, session_id_value = _header_scope(request, "X-Hermes-Session-Id")
    session_key_state, session_key_value = _header_scope(request, "X-Hermes-Session-Key")
    return (
        endpoint,
        normalized_profile,
        f"credential:{credential_digest}",
        f"session-id:{session_id_state}",
        session_id_value,
        f"session-key:{session_key_state}",
        session_key_value,
    )
