"""HMAC-SHA256 handshake + single-use replay guard for the bridge upgrade.

The .NET worker signs ``HMAC(sharedSecret, "{timestampMs}.{callId}")`` and sends
it as two headers on the WebSocket upgrade (see ``config.HEADER_*``). This module
verifies that signature constant-time, enforces a clock-skew/replay window, and
records each accepted ``(callId, ts, signature)`` tuple as **single-use** —
expiring at ``ts + window`` (not ``now + window``) so a future-dated handshake
from worker clock skew cannot outlive its own validity window.

Mirrors ``HmacSigner.cs`` (worker) and the handshake in ``msteams-media-stream.ts``
(the original TypeScript driver).
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass, field


def sign(secret: str, timestamp_ms: int, call_id: str) -> str:
    """Return the lowercase-hex HMAC-SHA256 over ``"{timestamp_ms}.{call_id}"``."""
    payload = f"{timestamp_ms}.{call_id}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256)
    return digest.hexdigest()


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class ReplayGuard:
    """Tracks accepted handshake tuples so each is usable at most once.

    Only *verified* tuples are recorded, so an attacker without the secret cannot
    grow the map. Entries expire at the timestamp's own validity horizon.
    """

    window_ms: int = 60_000
    _seen: dict[str, int] = field(default_factory=dict)  # key -> expiry epoch ms

    def _prune(self, now_ms: int) -> None:
        expired = [k for k, exp in self._seen.items() if exp <= now_ms]
        for k in expired:
            del self._seen[k]

    def check_and_record(self, call_id: str, timestamp_ms: int, signature: str) -> bool:
        """Return True if this tuple is fresh; record it. False if already used."""
        now_ms = _now_ms()
        self._prune(now_ms)
        key = f"{call_id}.{timestamp_ms}.{signature}"
        if key in self._seen:
            return False
        # Expire at the timestamp's OWN horizon (ts + window), not now + window,
        # so a future-dated handshake from worker clock skew cannot outlive its
        # validity window. A stale ts is rejected by the window check in
        # verify_upgrade before it ever reaches here.
        self._seen[key] = timestamp_ms + self.window_ms
        return True


def verify_upgrade(
    *,
    secret: str,
    call_id: str,
    timestamp_header: str | None,
    signature_header: str | None,
    window_ms: int = 60_000,
    replay_guard: ReplayGuard | None = None,
    now_ms: int | None = None,
) -> tuple[bool, str]:
    """Verify a WebSocket upgrade's HMAC headers.

    Returns ``(ok, reason)``. ``reason`` is a short, non-sensitive string for
    logging (never includes the secret or signature). Order of checks: presence,
    timestamp parse + window, constant-time signature compare, then single-use
    replay.
    """
    if not secret:
        return False, "bridge not configured (no shared secret)"
    if not timestamp_header or not signature_header:
        return False, "missing auth headers"

    try:
        ts = int(str(timestamp_header).strip())
    except (TypeError, ValueError):
        return False, "unparseable timestamp"

    current = now_ms if now_ms is not None else _now_ms()
    if abs(current - ts) > window_ms:
        return False, "timestamp outside window"

    expected = sign(secret, ts, call_id)
    if not hmac.compare_digest(expected, str(signature_header).strip().lower()):
        return False, "bad signature"

    if replay_guard is not None and not replay_guard.check_and_record(
        call_id, ts, str(signature_header).strip().lower()
    ):
        return False, "replayed handshake"

    return True, ""
