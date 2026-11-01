"""HTTP plumbing for the MemorySync Hermes provider.

Pure standard library on purpose: the provider ships with
``pip_dependencies: []`` in plugin.yaml, so installing it can never
disturb the Hermes runtime's environment — and an upstream SDK release
can never break a user's agent.

Wire contracts match every other MemorySync adapter byte for byte:
verbatim episodic turns via ``POST /v1/memory/add_turn`` with fnv1a64
UTF-16 content-hash speaker seeds (replays converge on one stored row),
hierarchical recall with the ``/v1/memory/query`` fallback, and the
dashboard-key surface (``X-End-User-ID``) for the explicit tools.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_BASE_URL = "https://api.memorysync.io"
SOURCE = "hermes"
MAX_TURN_CHARS = 16000
_TENANT_TTL_SECONDS = 3600.0
USER_AGENT = "memorysync-hermes/1.0.0"


def fnv1a64(value: str) -> str:
    """FNV-1a 64-bit over UTF-16 code units, as fixed-width hex.

    Over UTF-16 code units — not code points, not UTF-8 bytes — so the
    output matches the JavaScript adapters character for character.
    Identical seeds across languages mean a turn persisted by Hermes and
    again by any other MemorySync surface converge on one stored row.
    """
    prime = 0x100000001B3
    mask = 0xFFFFFFFFFFFFFFFF
    h = 0xCBF29CE484222325
    data = value.encode("utf-16-le")
    for i in range(0, len(data), 2):
        unit = data[i] | (data[i + 1] << 8)
        h ^= unit
        h = (h * prime) & mask
    return format(h, "016x")


# Credential shapes the provider refuses to store, checked client-side
# before any network call. Same contract as the OpenClaw plugin.
_SECRET_RE = re.compile(
    r"\b(sk-[A-Za-z0-9]{8,}|ms_[A-Za-z0-9]{8,}|ghp_[A-Za-z0-9]{8,}"
    r"|AKIA[0-9A-Z]{12,}|xox[baprs]-[A-Za-z0-9-]{8,})\b"
    r"|(password|passwd|secret|token|api[_-]?key)\s*[:=]\s*\S{6,}",
    re.IGNORECASE,
)


def looks_like_secret(text: str) -> bool:
    return bool(_SECRET_RE.search(text or ""))


class ClientError(Exception):
    """Any transport or HTTP-level failure. Callers decide silence."""


class CircuitBreaker:
    """After ``threshold`` consecutive failures, stay open ``cooldown_s``.

    A personal agent must not hammer a failing backend on every turn —
    the breaker converts a persistent outage into one cheap boolean
    check per call. Thread-safe: hooks and worker threads share it.
    """

    def __init__(self, threshold: int = 5, cooldown_s: float = 120.0) -> None:
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._failures = 0
        self._open_until = 0.0
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        with self._lock:
            return time.monotonic() < self._open_until

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._open_until = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                self._open_until = time.monotonic() + self._cooldown_s
                self._failures = 0


class MemorySyncClient:
    """Minimal synchronous client for the routes the provider touches."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        user_id: str,
        timeout: float = 5.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.user_id = user_id
        self.timeout = timeout
        self._tenant: Optional[str] = None
        self._tenant_at = 0.0
        self._tenant_lock = threading.Lock()

    # ── transport ────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        *,
        end_user: bool = False,
        timeout: Optional[float] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
        if end_user:
            headers["X-End-User-ID"] = self.user_id
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                payload = response.read()
                status = response.status
        except urllib.error.HTTPError as exc:  # non-2xx still carries a body
            payload = exc.read()
            status = exc.code
        except Exception as exc:  # URLError, timeout, connection refused …
            raise ClientError(str(exc)) from exc
        try:
            parsed = json.loads(payload.decode("utf-8")) if payload else {}
        except Exception:
            parsed = {}
        if not isinstance(parsed, (dict, list)):
            parsed = {}
        return status, parsed if isinstance(parsed, dict) else {"_list": parsed}

    # ── tenant discovery ─────────────────────────────────────────────

    def tenant_id(self) -> str:
        """Tenant for the v1 plane, cached in-process for an hour.

        Keys that cannot list projects (evaluation keys: 401/403)
        resolve to ``"default"`` deterministically, like every other
        MemorySync adapter.
        """
        with self._tenant_lock:
            if self._tenant and time.monotonic() - self._tenant_at < _TENANT_TTL_SECONDS:
                return self._tenant
        status, data = self._request("GET", "/org/projects")
        tenant: Optional[str] = None
        if status in (401, 403):
            tenant = "default"
        elif status == 200:
            rows = data.get("_list") if "_list" in data else data
            if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                raw = rows[0].get("tenant_id")
                if raw:
                    tenant = str(raw)
        if not tenant:
            raise ClientError(f"tenant discovery failed (HTTP {status})")
        with self._tenant_lock:
            self._tenant = tenant
            self._tenant_at = time.monotonic()
        return tenant

    # ── hooks surface: verbatim turns + recall ───────────────────────

    def add_turn(self, *, role: str, text: str, scope: str, session_id: str) -> None:
        trimmed = text if len(text) <= MAX_TURN_CHARS else text[:MAX_TURN_CHARS] + "…"
        status, _ = self._request(
            "POST",
            "/v1/memory/add_turn",
            {
                "tenant_id": self.tenant_id(),
                "user_id": self.user_id,
                "source": SOURCE,
                "text": f"{role}: {trimmed}",
                "speaker": f"{role}@{scope}#h{fnv1a64(f'{role}:{trimmed}')}",
                "metadata": {"session_id": scope, "agent_session": session_id or None},
                "sync_embed": False,
            },
        )
        if status >= 400:
            raise ClientError(f"add_turn HTTP {status}")

    def recall_lines(self, *, prompt: str, k: int, timeout: Optional[float] = None) -> List[str]:
        """Recall as bullet lines: hierarchical first, query fallback.

        Empty list when nothing matches or the server answers over-quota
        silence — the caller cannot tell the difference, which is the
        production contract.
        """
        tenant = self.tenant_id()
        status, data = self._request(
            "POST",
            "/v1/memory/recall",
            {"tenant_id": tenant, "user_id": self.user_id, "prompt": prompt, "k": k},
            timeout=timeout,
        )
        if status == 200:
            context = data.get("context")
            if isinstance(context, str) and context.strip():
                return [line for line in context.strip().splitlines() if line.strip()]
        status, data = self._request(
            "POST",
            "/v1/memory/query",
            {"tenant_id": tenant, "user_id": self.user_id, "prompt": prompt, "k": k},
            timeout=timeout,
        )
        lines: List[str] = []
        if status == 200 and isinstance(data.get("memories"), list):
            for item in data["memories"]:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("raw_text") or item.get("value") or "").strip()
                if text:
                    lines.append(f"- {text}")
        return lines

    # ── tools surface: dashboard-key API ─────────────────────────────

    def search_memories(self, *, query: str, k: int) -> List[Dict[str, Any]]:
        status, data = self._request(
            "POST", "/memory/query", {"query": query, "k": k}, end_user=True
        )
        if status != 200 or not isinstance(data.get("memories"), list):
            if status >= 400:
                raise ClientError(f"query HTTP {status}")
            return []
        out: List[Dict[str, Any]] = []
        for item in data["memories"]:
            if not isinstance(item, dict):
                continue
            text = str(item.get("value") or item.get("raw_text") or item.get("text") or "").strip()
            if text:
                out.append(
                    {
                        # Production answers {"id": ...}; older deployments
                        # {"memory_id": ...}. Tolerate both, prefer explicit.
                        "id": item.get("memory_id", item.get("id")),
                        "text": text,
                        "score": item.get("score"),
                    }
                )
        return out

    def add_memory(
        self, *, text: str, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        body: Dict[str, Any] = {"text": text, "source": SOURCE}
        if tags:
            body["tags"] = list(tags)
        if metadata:
            body["metadata"] = metadata
        status, data = self._request("POST", "/memory/add", body, end_user=True)
        if status >= 400:
            raise ClientError(f"add HTTP {status}")
        # Production answers {"id": ...}; older deployments {"memory_id": ...}.
        return data.get("memory_id", data.get("id"))

    def forget_memory(self, *, memory_id: Any) -> int:
        status, data = self._request(
            "DELETE", "/memory/forget", {"memory_ids": [memory_id]}, end_user=True
        )
        if status >= 400:
            raise ClientError(f"forget HTTP {status}")
        deleted = data.get("deleted")
        return int(deleted) if isinstance(deleted, int) else 0
