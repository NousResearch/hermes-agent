"""Helm L2 outbound secret simulation guard (LOCAL, DEFAULT-OFF).

This is a *simulation* chokepoint, not a real secret detector. It exists so a
security review can exercise the outbound-block path end to end without ever
shipping a real credential pattern or contacting a real platform.

Design constraints (intentional):
  * Disabled by default. The only way to arm it is the env flag
    ``HERMES_L2_SECRET_SIMULATION`` set to a truthy value (1/true/yes/on).
  * The trigger is a single, obviously-fake, local-only sentinel constant
    (:data:`HERMES_L2_FAKE_SECRET_SENTINEL`). It is not a real token shape and
    must never be replaced with one.
  * When armed and a sentinel is present in outbound content, delivery is
    blocked *before* any adapter/platform send, a redacted JSON incident report
    is written locally, and a redacted repair payload is returned. The raw
    sentinel is never written to the report or returned to the caller.

Everything here is pure/local: no network, no real secret scanning.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from utils import env_var_enabled

logger = logging.getLogger(__name__)

# Env flag that arms the simulation. Default-off: absent/falsey == disabled.
SIMULATION_ENV_FLAG = "HERMES_L2_SECRET_SIMULATION"

# Obviously-fake, local-only trigger. This is NOT a credential pattern and must
# never be swapped for one. Its only purpose is to deterministically exercise
# the block path during a security review.
HERMES_L2_FAKE_SECRET_SENTINEL = "HERMES_L2_FAKE_SECRET_SENTINEL__LOCAL_SIM_ONLY"

# What the raw sentinel is replaced with anywhere it would otherwise surface.
_REDACTION_PLACEHOLDER = "[REDACTED_L2_FAKE_SENTINEL]"


def is_simulation_enabled() -> bool:
    """Return True only when the L2 secret simulation is explicitly armed."""
    return env_var_enabled(SIMULATION_ENV_FLAG)


@dataclass
class SimulationVerdict:
    """Outcome of an outbound simulation evaluation.

    ``blocked`` is False for the overwhelmingly common case (disabled, or no
    sentinel present), in which callers proceed with normal delivery untouched.
    """

    blocked: bool
    reason: str = ""
    redacted_content: str = ""
    incident_path: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def repair_payload(self) -> Dict[str, Any]:
        """Redacted result returned to the send-tool caller when blocked."""
        return {
            "success": False,
            "blocked": "l2_secret_simulation",
            "delivered": False,
            "reason": self.reason,
            "redacted_content": self.redacted_content,
            "incident_report": self.incident_path,
        }

    def to_delivery_result(self) -> Dict[str, Any]:
        """Redacted result shaped like a DeliveryRouter platform result."""
        return self.repair_payload


def _redact(text: str) -> str:
    """Strip every occurrence of the fake sentinel from arbitrary text."""
    if not text:
        return text
    return text.replace(HERMES_L2_FAKE_SECRET_SENTINEL, _REDACTION_PLACEHOLDER)


def _incident_dir() -> Path:
    """Profile-aware local incident directory, with a safe ~/.hermes fallback."""
    try:
        from hermes_constants import get_hermes_home

        base = get_hermes_home()
    except Exception:  # pragma: no cover - defensive: never block on home lookup
        base = Path.home() / ".hermes"
    return Path(base) / "security" / "incidents"


def _write_incident(report: Dict[str, Any]) -> Optional[str]:
    """Write a redacted incident report locally; never raise on failure."""
    try:
        target_dir = _incident_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        # Deterministic-ish, collision-resistant filename without Date.now-style
        # APIs (unavailable in some sandboxes). PID + monotonic counter via os.
        seq = report.get("sequence", "0")
        path = target_dir / f"l2_secret_simulation_{os.getpid()}_{seq}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
        return str(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to write L2 simulation incident report: %s", exc)
        return None


# Module-local monotonic counter so repeated incidents in one process don't
# clobber each other's report files (no wall-clock/random APIs needed).
_incident_seq = 0


def evaluate_outbound(
    content: Any,
    *,
    platform: Optional[str] = None,
    target: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SimulationVerdict:
    """Evaluate outbound content against the L2 secret simulation.

    Returns a non-blocking verdict when the gate is disabled (default) or no
    sentinel is present, so existing send/delivery behavior is fully preserved.
    When armed *and* the fake sentinel is present, blocks, writes a redacted
    incident report, and returns a redacted repair payload — before any send.
    """
    if not is_simulation_enabled():
        return SimulationVerdict(blocked=False)

    text = content if isinstance(content, str) else str(content)
    if HERMES_L2_FAKE_SECRET_SENTINEL not in text:
        return SimulationVerdict(blocked=False)

    global _incident_seq
    _incident_seq += 1

    redacted = _redact(text)
    reason = (
        "Outbound message blocked by Helm L2 secret simulation: fake local "
        "sentinel detected in content (this is a simulation, not a real "
        "credential)."
    )
    report = {
        "type": "l2_outbound_secret_simulation",
        "simulation": True,
        "platform": platform,
        "target": _redact(target) if isinstance(target, str) else target,
        "sentinel_present": True,
        "redacted_content": redacted,
        "metadata_keys": sorted(metadata.keys()) if isinstance(metadata, dict) else [],
        "sequence": _incident_seq,
    }
    incident_path = _write_incident(report)

    logger.warning(
        "L2 secret simulation BLOCKED outbound send (platform=%s, target=%s, "
        "incident=%s)",
        platform,
        _redact(target) if isinstance(target, str) else target,
        incident_path,
    )

    return SimulationVerdict(
        blocked=True,
        reason=reason,
        redacted_content=redacted,
        incident_path=incident_path,
        details={"platform": platform, "sequence": _incident_seq},
    )


def guard_outbound(
    content: Any,
    *,
    platform: Optional[str] = None,
    target: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[SimulationVerdict]:
    """Block-or-proceed helper for direct adapter send paths.

    Thin wrapper over :func:`evaluate_outbound` that returns the verdict only
    when delivery should be blocked, and ``None`` for the common
    proceed-as-normal case (gate disabled, or no sentinel present). Direct send
    helpers (``gateway/run.py``, ``gateway/stream_consumer.py``) call this right
    before ``adapter.send``/``send_or_update_status`` so the same chokepoint that
    guards ``DeliveryRouter`` also covers those bypass paths.
    """
    verdict = evaluate_outbound(
        content, platform=platform, target=target, metadata=metadata
    )
    return verdict if verdict.blocked else None


def blocked_send_result(verdict: SimulationVerdict) -> Any:
    """Build a ``SendResult``-shaped blocked result for adapter-send callers.

    Direct stream/status send paths expect a ``SendResult`` (``.success``,
    ``.error``, ``.message_id``) rather than the dict repair payload the
    send-tool/delivery seams return. Shape a failed ``SendResult`` so existing
    callers treat the block as a (non-retryable, non-flood) failed send and do
    not reach the live platform. Imported lazily to avoid a module import cycle.
    """
    from gateway.platforms.base import SendResult

    return SendResult(success=False, error=verdict.reason, retryable=False)
