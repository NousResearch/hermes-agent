"""Accounting at completion of an exact heartbeat admission attempt.

Execution means entering the gateway's agent runner after turn preparation,
not reserving an adapter slot, successful model completion, or outbound delivery.
The done callback retains the watch's profile ContextVars and manager claim.
"""
import logging

logger = logging.getLogger("gateway.run")


def settle_heartbeat_attempt(event, manager):
    if not getattr(event, "_heartbeat_execution_started", False):
        try:
            manager.abandon_fire()
        except Exception:
            logger.warning("Failed to refund unexecuted heartbeat", exc_info=True)
