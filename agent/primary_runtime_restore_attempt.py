"""Restore the live fallback snapshot when a primary runtime attempt fails."""

from contextlib import contextmanager
import logging

from agent.context_compressor_state import defer_compressor_state_writes
from agent.primary_runtime_snapshot import (
    _MISSING, _PRIMARY_RESTORE_SNAPSHOT_FIELDS, _restore_switch_snapshot, _snapshot_switch_state,
)

logger = logging.getLogger(__name__)


@contextmanager
def primary_runtime_restore_attempt(agent):
    snapshot = _snapshot_switch_state(agent, fields=_PRIMARY_RESTORE_SNAPSHOT_FIELDS)
    compressor = agent.context_compressor
    compressor_state = {
        name: value.copy() if isinstance(value, (dict, list, set)) else value
        for name, value in vars(compressor).items()
    }
    transport_cache = getattr(agent, '_transport_cache', _MISSING)
    cached_transports = dict(transport_cache) if isinstance(transport_cache, dict) else None
    try:
        with defer_compressor_state_writes(compressor):
            yield
    except Exception:
        abandoned = [getattr(agent, name, None) for name in ('client', '_anthropic_client')]
        _restore_switch_snapshot(agent, snapshot)
        vars(compressor).clear()
        vars(compressor).update(compressor_state)
        if cached_transports is not None:
            transport_cache.clear()
            transport_cache.update(cached_transports)
        retained = [snapshot.get(name) for name in ('client', '_anthropic_client')]
        retired = set()
        for client in abandoned:
            if client is None or any(client is old for old in retained) or id(client) in retired:
                continue
            retired.add(id(client))
            try:
                agent._retire_shared_openai_client(client, reason='failed_primary_restore')
            except Exception:
                logger.exception('Failed to retire a primary restore client')
        raise
