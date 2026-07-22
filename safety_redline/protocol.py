"""Line-delimited JSON protocol adapter for safety redline.

This is the *wire* half of the safety enforcement. It is intentionally tiny
and dependency-free so it can run inside the image without pulling in the
gateway's asyncio stack for unit tests.

Wire format (line-delimited JSON envelopes):

    {"type": "hello", "ts": ..., "nonce": "...", "body": {peer_id, capabilities, ...}, "hmac": "..."}
    {"type": "ping",  "ts": ..., "nonce": "...", "body": {...}}
    {"type": "safety.report_failure", "ts": ..., "body": {"reason": "..."}}
    {"type": "safety.report_success", "ts": ..., "body": {}}
    {"type": "safety.snapshot", "ts": ..., "body": {}}
    {"type": "safety.reset", "ts": ..., "body": {"operator": "..."}}

The adapter is one-directional in the test suite: we synthesise inbound
messages and assert that ``SafetyRedline`` transitions land where they should.
In production this adapter would be wired into the gateway's existing line
reader (see ``gateway/run.py``) -- that integration is intentionally left out
of this commit because it depends on the gateway's internal message envelope
which has its own HMAC scheme.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

from .redline import SafetyRedline, SafetyState, SafetyEvent


@dataclass
class SafetyRedlineProtocol:
    redline: SafetyRedline

    # -- inbound handlers ---------------------------------------------------

    def handle(self, message: dict) -> Optional[SafetyEvent]:
        mtype = message.get("type")
        if mtype in ("safety.report_failure", "report_failure"):
            reason = (message.get("body") or {}).get("reason", "")
            return self.redline.record_failure(reason=reason)
        if mtype in ("safety.report_success", "report_success"):
            return self.redline.record_success()
        if mtype in ("safety.reset", "reset"):
            operator = (message.get("body") or {}).get("operator", "unknown")
            self.redline.reset()
            return SafetyEvent(
                state=self.redline.state,
                failure_streak=0,
                message=f"reset by {operator}",
                timestamp=0.0,
            )
        if mtype == "safety.snapshot":
            return None  # snapshot is read via redline.snapshot(), not via events
        return None

    # -- outbound helpers ---------------------------------------------------

    def snapshot(self) -> dict:
        return self.redline.snapshot()

    def is_traffic_allowed(self) -> bool:
        return self.redline.is_traffic_allowed()

    # -- utility ------------------------------------------------------------

    def encode(self, message: dict) -> bytes:
        """Encode a message exactly the way the gateway's line reader expects."""
        return (json.dumps(message, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")

    def decode(self, raw: bytes) -> dict:
        """Decode a single message. Raises ``ValueError`` on empty/invalid input."""
        text = raw.decode("utf-8").strip()
        if not text:
            raise ValueError("empty message")
        return json.loads(text)


def make_protocol(
    *,
    pause_threshold: int = 3,
    hard_pause_threshold: int = 4,
    cooldown_seconds: float = 300.0,
    warn_threshold: int = 2,
    notifier: Optional[Callable[[str, str, dict], None]] = None,
) -> SafetyRedlineProtocol:
    """Convenience constructor matching ``SafetyConfig`` semantics."""
    from .redline import SafetyConfig  # local import keeps circular-free
    config = SafetyConfig(
        pause_threshold=pause_threshold,
        hard_pause_threshold=hard_pause_threshold,
        cooldown_seconds=cooldown_seconds,
        warn_threshold=warn_threshold,
        notifier=notifier,
    )
    return SafetyRedlineProtocol(redline=SafetyRedline(config=config))
