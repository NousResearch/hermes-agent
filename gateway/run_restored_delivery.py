"""Candidate selection for resumed sessions, including Relay fronted platforms."""
import logging
from typing import Any, Optional

from gateway.config import Platform

logger = logging.getLogger("gateway.run")


def resume_pending_candidates(runner: Any, platform: Optional[Platform] = None) -> Optional[list]:
    """Snapshot pending entries unless enumeration fails or the boot loop breaker trips."""
    relay = runner._primary_adapters().get(Platform.RELAY) if platform == Platform.RELAY else None
    fronts = getattr(relay, "fronts_platform", None)
    try:
        from gateway.delivery_ledger import ledger_enabled, pending_resume_sessions
        # A saved reply still owed for a session means its turn was answered: recovery delivers
        # that reply, and a resume here would generate a second one.
        owed = pending_resume_sessions() if ledger_enabled() else set()
        with runner.session_store._lock:
            runner.session_store._ensure_loaded_locked()
            candidates = [
                entry for entry in runner.session_store._entries.values()
                if entry.resume_pending
                and not entry.suspended
                and entry.origin is not None
                and entry.resume_reason in runner._AUTO_RESUME_REASONS
                and (platform is None or entry.origin.platform == platform
                     or (callable(fronts) and fronts(entry.origin.platform)))
                and entry.session_key not in owed
            ]
    except Exception as exc:
        logger.warning("Failed to enumerate resume-pending sessions: %s", exc, exc_info=True)
        return None
    # Only boots with interrupted sessions count; a tripped boot still serves inbound.
    if candidates:
        try:
            from gateway import restart_loop_guard as guard
            max_restarts, window, max_gap = runner._restart_loop_guard_config()
            if guard.check_and_record(max_restarts, window, max_gap_seconds=max_gap):
                return None
        except Exception as exc:
            logger.debug("Restart-loop guard check skipped: %s", exc, exc_info=True)
    return candidates
