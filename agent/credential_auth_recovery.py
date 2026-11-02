"""Credential-pool recovery for an auth failure (401/403) on the dispatched credential.

Nous answers bursts of ``401 {"error": "authentication required"}`` for agent keys that are
still unexpired and valid a few seconds later. Re-minting cannot fix a server-side blip, and
each forced refresh redeems the single-use refresh token, so a Nous 401 episode gets exactly
ONE forced refresh. Any later 401 in the same turn (or a refresh that handed back the bearer
that just failed) waits out ``TRANSIENT_AUTH_BACKOFF_S`` and retries with the store's current
key, adopting a peer's rotation but never POSTing again. A dead grant (refresh rejected)
still fails fast through rotation, and so does an entitlement 403.
"""

from __future__ import annotations

import time
from typing import Any, Optional

# Waits before each retry of an unexpired Nous key that keeps answering 401 (~50 s in total).
TRANSIENT_AUTH_BACKOFF_S = (5.0, 15.0, 30.0)
# Cap same-entry OAuth refreshes on a persistent auth failure, else a single-entry pool re-mints forever.
_MAX_AUTH_REFRESH_ATTEMPTS = 2
_ACTIVITY_TOUCH_S = 30.0
# A 401 this long after the episode's last one starts a new episode (one more refresh allowed):
# hour-long turns must still recover from a key revoked later in the same turn.
_EPISODE_IDLE_S = 120.0


def _ra():
    import run_agent
    return run_agent


def _episode(agent: Any, provider: str, entry_id: Optional[str]) -> dict[str, Any]:
    """Per-turn auth-failure state for one pool entry (the turn loop resets the map)."""
    if getattr(agent, "_auth_pool_refresh_counts", None) is None:
        agent._auth_pool_refresh_counts = {}
    now = time.monotonic()
    episode = agent._auth_pool_refresh_counts.get((provider, entry_id))
    if episode is None or now - episode["last"] > _EPISODE_IDLE_S:
        episode = agent._auth_pool_refresh_counts[(provider, entry_id)] = {"refreshes": 0, "waits": 0}
    episode["last"] = now
    return episode


def nous_pool_refreshed_this_turn(agent: Any) -> bool:
    """True once the pool spent this turn's forced Nous refresh (the singleton path must not re-POST)."""
    counts = getattr(agent, "_auth_pool_refresh_counts", None) or {}
    return any(key[0] == "nous" and state.get("refreshes") for key, state in counts.items())


def _interruptible_wait(agent: Any, seconds: float) -> bool:
    """Sleep in 200 ms slices; False when the user interrupted."""
    deadline = time.monotonic() + seconds
    next_touch = time.monotonic() + _ACTIVITY_TOUCH_S
    while time.monotonic() < deadline:
        if getattr(agent, "_interrupt_requested", False):
            return False
        time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        if time.monotonic() >= next_touch:
            agent._touch_activity("waiting out a Nous 401 on a valid key")
            next_touch = time.monotonic() + _ACTIVITY_TOUCH_S
    return True


def _entry_for(pool: Any, credential_id: Optional[str], api_key_hint: Optional[str]) -> Any:
    entries = pool.entries()
    if credential_id:
        found = next((e for e in entries if e.id == credential_id), None)
        if found is not None:
            return found
    return next((e for e in entries if api_key_hint and e.runtime_api_key == api_key_hint), None)


def _retry_after_backoff(agent: Any, pool: Any, episode: dict[str, Any], *, failed_key: Optional[str],
                         entry_id: Optional[str], credential_id: Optional[str], has_retried_429: bool):
    """Wait out a server-side 401 on an unexpired key, then retry with the store's current key."""
    log = _ra().logger
    step = episode["waits"]
    if step >= len(TRANSIENT_AUTH_BACKOFF_S):
        log.warning(
            "Credential auth failure persists for pool entry %s after one refresh and %s "
            "backoff retries — treating as unrecoverable and allowing fallback to activate.",
            entry_id, step)
        return False, has_retried_429
    wait = TRANSIENT_AUTH_BACKOFF_S[step]
    episode["waits"] = step + 1
    log.info(
        "Credential auth failure on an unexpired key for pool entry %s — transient 401, "
        "retrying in %.0fs (backoff %s/%s, no further refresh)",
        entry_id, wait, step + 1, len(TRANSIENT_AUTH_BACKOFF_S))
    agent._vprint(
        f"{agent.log_prefix}🔐 Nous rejected a still-valid key (transient); retrying in {wait:.0f}s...",
        force=True, diagnostic=True)
    if not _interruptible_wait(agent, wait):
        return True, has_retried_429  # the loop's interrupt check ends the turn before any request
    # A peer process may have rotated meanwhile: adopt what the store holds now (no POST).
    entry = _entry_for(pool, credential_id or entry_id, failed_key)
    if entry is not None:
        if agent._swap_credential(pool._sync_nous_entry_from_auth_store(entry)) is False:
            return False, has_retried_429
    return True, has_retried_429


def recover_auth_failure(agent, pool, *, status_code, has_retried_429, error_context, api_key_hint,
                         credential_id, rotate_and_swap):
    from agent.agent_runtime_helpers import _is_entitlement_403
    log = _ra().logger
    if _is_entitlement_403(agent, status_code, error_context):
        log.info(
            "Credential %s — entitlement-shaped 403 from %s; "
            "skipping pool refresh (account lacks subscription, "
            "not a transient auth failure).", status_code if status_code is not None else "auth",
            agent.provider or "provider",
        )
        return False, has_retried_429
    provider = agent.provider
    nous = provider == "nous"
    entry_id = credential_id
    if nous:
        found = _entry_for(pool, credential_id, api_key_hint)
        entry_id = getattr(found, "id", None) or credential_id
        episode = _episode(agent, provider, entry_id)
        if episode["refreshes"] >= 1:
            # The one forced refresh of this episode already happened: never re-redeem the grant.
            return _retry_after_backoff(
                agent, pool, episode, failed_key=api_key_hint, entry_id=entry_id,
                credential_id=credential_id, has_retried_429=has_retried_429)
    # Refresh the entry that supplied the failing key, not current(): refreshing a healthy entry
    # burns its single-use refresh token for a failure it never had.
    refresh_kwargs = {"api_key_hint": api_key_hint}
    if credential_id:
        refresh_kwargs["credential_id"] = credential_id
    refreshed = pool.try_refresh_matching(**refresh_kwargs)
    if refreshed is None:
        # Refresh failed; rotate (the failed entry is already marked exhausted).
        return (True, False) if rotate_and_swap(401, "auth refresh failed") else (False, has_retried_429)
    refreshed_id = getattr(refreshed, "id", None)
    if nous:
        episode = _episode(agent, provider, refreshed_id or entry_id)
        episode["refreshes"] += 1
        if api_key_hint and getattr(refreshed, "runtime_api_key", None) == api_key_hint:
            # The "refresh" handed back the bearer that just failed: replaying it at once is
            # the 1.5 s give-up; it is a valid key the server is refusing, so back off instead.
            return _retry_after_backoff(
                agent, pool, episode, failed_key=api_key_hint, entry_id=refreshed_id,
                credential_id=credential_id, has_retried_429=has_retried_429)
    elif refreshed_id is not None:
        # try_refresh_matching() reports success even when upstream keeps rejecting; cap same-entry
        # refreshes so a single-entry pool falls through to fallback.
        state = _episode(agent, provider, refreshed_id)
        state["refreshes"] += 1
        if state["refreshes"] > _MAX_AUTH_REFRESH_ATTEMPTS:
            log.warning(
                "Credential auth failure persists after %s refreshes for "
                "pool entry %s — treating as unrecoverable and allowing "
                "fallback to activate.", state["refreshes"] - 1, refreshed_id,
            )
            return False, has_retried_429
    log.info("Credential auth failure — refreshed pool entry %s", refreshed_id or "?")
    if agent._swap_credential(refreshed) is False:
        return False, has_retried_429
    return True, has_retried_429
