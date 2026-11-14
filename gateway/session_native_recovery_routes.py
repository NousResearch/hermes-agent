"""Rebuild queued native routes from currently verified connector provenance."""
import asyncio
import logging
from typing import Any

from gateway.session_envelope import restore_native_source
from hermes_state_runtime import RuntimeStoreError, latest_session_admission

logger = logging.getLogger(__name__)

# Background retries of a late connection's durable recovery (seconds before each attempt).
_RECOVERY_RETRY_DELAYS = (5, 30, 120, 300)


def _latest_native(authority: Any, owner: str) -> Any:
    """The session's newest native admission payload, or None (one newest-first read, not its history)."""
    row = latest_session_admission(authority.db, session_id=owner,
                                   accept=lambda payload: 'native_text_v1' in payload)
    return row['payload'] if row is not None else None


def recovered_native_source(runner: Any, authority: Any, entry: Any, owner: str) -> Any:
    """Old routing entries may lack a receiving bot; a verified admission names it.

    Never infer a primary transport from an unverified stored profile. The regular
    recovery preflight still checks every queued sender before scheduling any work.
    """
    payload = _latest_native(authority, owner)
    if payload is not None:
        try:
            return restore_native_source(payload, runner)
        except RuntimeStoreError:
            pass  # the authority reports the precise refusal without consuming the queue
    return runner._restored_source(entry)


def verified_restored_relay_source(runner: Any, entry: Any) -> Any:
    """Verify a settled Relay lane too, without trusting an old primary-bot guess."""
    from gateway.session_authorities import serving_authorities

    for authority in serving_authorities(runner):
        payload = _latest_native(authority, authority.logical_owner(entry.session_id))
        if payload is None:
            continue
        envelope = payload['native_text_v1']
        if envelope.get('route') != entry.session_key or 'relay' not in envelope.get('provenance', {}):
            continue
        try:
            return restore_native_source(payload, runner)
        except RuntimeStoreError:
            continue  # no current verified connector: keep the ordinary fail-closed selector
    return None


async def recover_after_late_connection(runner: Any) -> None:
    """A late Relay connection retries every served profile's durable FIFO. A failure here is a
    recovery failure, not a connection failure: it is logged and retried in the background, never
    raised into the reconnect watcher, which would dispose the adapter it has just published."""
    from gateway.session_authorities import serving_authorities

    for authority in serving_authorities(runner):
        if not await _try_recover(runner, authority):
            runner._retain_background_task(asyncio.create_task(_retry_recovery(runner, authority)))


async def _try_recover(runner: Any, authority: Any) -> bool:
    from gateway.run_runtime import recover_profile_native_sessions

    try:
        await recover_profile_native_sessions(runner, authority)
        return True
    except Exception:
        logger.warning("Native recovery after a late Relay connection failed (%s)",
                       getattr(authority, 'profile_id', '?'), exc_info=True)
        return False


async def _retry_recovery(runner: Any, authority: Any) -> None:
    for delay in _RECOVERY_RETRY_DELAYS:
        await asyncio.sleep(delay)
        if not getattr(runner, '_running', True) or await _try_recover(runner, authority):
            return
    logger.error("Native recovery for %s gave up; queued input runs at the next restart or reconnect",
                 getattr(authority, 'profile_id', '?'))
