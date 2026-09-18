"""Recover heartbeat watches from the gateway's canonical persisted routing index."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("gateway.run")


def _watched_homes(runner, default_home) -> list:
    """Every home the sweep's ``_profile_scope_for_source`` can resolve an origin to: the gateway home
    plus, under multiplex, the whole served set INCLUDING ``default`` — a ``-p work`` multiplexer's own
    home is not ``~/.hermes``, so dropping ``default`` here would hide its heartbeats forever."""
    from gateway.run import _multiplex_profile_homes

    homes = [default_home]
    if getattr(getattr(runner, "config", None), "multiplex_profiles", False):
        homes += [home for _name, home in _multiplex_profile_homes(runner.config)]
    return list(dict.fromkeys(Path(home) for home in homes))


async def restore_heartbeat_watches(runner) -> None:
    """Retryable startup/poll scan; failed reads never prune existing watches.

    SessionStore owns one routing index across profiles. Its origin and exact key,
    rather than a second heartbeat routing snapshot, also cover pre-upgrade state.
    Run all storage work off-loop so a cold profile DB cannot block adapters.
    """
    from gateway.run import _profile_runtime_scope
    from gateway.run_idle_gates import profile_has_active_heartbeat
    from hermes_cli.heartbeat import HeartbeatManager
    from hermes_constants import get_hermes_home

    store = runner.session_store

    def scan():
        restored = []
        # The poller may have been spawned by a named profile's /heartbeat command.
        # Anchor even default origins to the gateway home, not inherited context.
        home = getattr(store, "_routing_home", None) or get_hermes_home()
        # Cheap gate: with no heartbeat persisted in any served profile there is nothing to
        # restore — skip the per-origin profile-scope re-parse over every routed session.
        if not any(profile_has_active_heartbeat(h) for h in _watched_homes(runner, home)):
            return restored
        with _profile_runtime_scope(home):
            entries = store.list_sessions()
            for entry in entries:
                if entry.origin is None or not entry.session_id or entry.suspended:
                    continue
                try:
                    with runner._profile_scope_for_source(entry.origin):
                        manager = HeartbeatManager(entry.session_id)
                        if manager.is_active():
                            restored.append((entry.session_key, entry.origin, entry.session_id))
                except Exception:
                    logger.debug("heartbeat restore for %s failed", entry.session_key, exc_info=True)
        return restored

    try:
        candidates = await runner._run_in_executor_with_context(scan)
        for key, source, session_id in candidates:
            # A reset/compression may have published a new owner during the executor hop.
            if store.peek_session_id(key) == session_id:
                runner._register_heartbeat_watch(key, source, session_id)
    except Exception:
        logger.debug("heartbeat restore scan failed; retrying on next poll", exc_info=True)
