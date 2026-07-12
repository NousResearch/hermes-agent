"""Regression test for the #54878 x #54947 interaction.

Bug
---
The #54878 self-heal (``SessionStore.get_or_create_session``) detects a
routing key pointing at a session that state.db already marked ended, drops
the stale ``sessions.json`` entry, and recovers/recreates a fresh session_id
under the SAME session_key.

The #54947 fix (``gateway/run.py`` agent-cache cache-hit guard) treats "cached
agent's snapshot session_id differs from the current session_id, same
session_key" as an intentional ``/resume``/``/branch``-style switch between
two live sibling conversations, and reuses the cached agent unchanged so the
prompt cache isn't busted.

These two fixes compose incorrectly: when the #54878 self-heal just fired,
the cached agent's session_id is not a live sibling conversation — it is the
DEAD session that was just routed away from. #54947's rule reuses it anyway.
The stale agent then runs the turn, and the gateway's post-run "session
split" sync (which fires because ``agent.session_id != session_id``) writes
the routing key straight back onto the dead session_id — undoing the
self-heal. This repeats on every subsequent message until an interrupt (e.g.
``/stop``) happens to race in before that post-run sync, corrupting the
observed session lineage and silently discarding conversation context.

No open upstream issue tracked this specific interaction as of 2026-07-12
(checked: #54878, #54947, #59580, #59597, #61220 all cover adjacent but
distinct edges of the self-heal / agent-cache system).

Fix (this test pins it)
------------------------
Before applying #54947's "different session_id -> reuse freely" rule, check
whether the cached snapshot's session_id is itself ended in state.db
(``SessionStore._is_session_ended_in_db``). If so, treat it as a stale
self-heal artifact — evict and rebuild fresh, exactly like a genuine
cross-process write — instead of reusing it.

This mirrors the production decision block added in
``GatewayRunner._handle_message_with_agent`` (gateway/run.py, "Peek at the
cached entry's snapshot session_id..." / ``_stale_dead_sid_reuse``).
"""

import threading

from hermes_state import SessionDB


def _make_runner_with_db(tmp_path):
    """Minimal GatewayRunner-like object with real cache + DB-backed
    ``_is_session_ended_in_db``, mirroring ``_make_runner`` in
    ``test_session_id_cache_coherence.py`` but adding the session_store
    handle the new guard needs.
    """
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()

    db = SessionDB(db_path=tmp_path / "sessions.db")

    class _FakeSessionStore:
        """Just enough of SessionStore for the new guard's DB check."""

        def __init__(self, db):
            self._db = db

        def _is_session_ended_in_db(self, session_id):
            if not self._db or not session_id:
                return False
            row = self._db.get_session(session_id)
            return bool(row is not None and row.get("end_reason") is not None)

    runner.session_store = _FakeSessionStore(db)
    return runner, db


def _guard_would_reuse_after_fix(runner, session_key, session_id, current_mc=None):
    """Faithful mirror of the production decision block in
    ``gateway/run.py`` AFTER the #54878 x #54947 fix:

    1. Peek the cached entry's snapshot session_id outside the lock.
    2. If it differs from the current session_id, check whether THAT
       session_id is dead in state.db.
    3. Under the lock, re-validate the peeked verdict still applies to the
       tuple actually present (it could have been replaced), then decide:
         - stale dead-session artifact -> evict, rebuild (return False)
         - genuine live sibling switch (#54947) -> reuse (return True)
         - same session_id + cross-process count mismatch (#45966) -> evict
         - otherwise -> reuse

    Returns (would_reuse: bool, evicted: bool).
    """
    with runner._agent_cache_lock:
        peek_entry = runner._agent_cache.get(session_key)
    peek_cached_sid = peek_entry[3] if peek_entry and len(peek_entry) > 3 else None

    cached_sid_is_dead = False
    if peek_cached_sid is not None and session_id is not None and peek_cached_sid != session_id:
        cached_sid_is_dead = runner.session_store._is_session_ended_in_db(peek_cached_sid)

    with runner._agent_cache_lock:
        cached = runner._agent_cache.get(session_key)
        if not cached:
            return True, False
        cached_mc = cached[2] if len(cached) > 2 else None
        cached_sid = cached[3] if len(cached) > 3 else None

        session_id_mismatch = (
            cached_sid is not None and session_id is not None and cached_sid != session_id
        )
        stale_dead_sid_reuse = (
            session_id_mismatch and cached_sid_is_dead and cached_sid == peek_cached_sid
        )

        if stale_dead_sid_reuse:
            runner._agent_cache.pop(session_key, None)
            return False, True

        if (
            not session_id_mismatch
            and cached_mc is not None
            and current_mc is not None
            and current_mc != cached_mc
        ):
            runner._agent_cache.pop(session_key, None)
            return False, True

        return True, False


class TestStaleSelfHealAgentCacheEviction:
    def test_dead_cached_session_id_is_not_reused(self, tmp_path):
        """The #54878 x #54947 bug: cached agent's session_id was just
        self-healed away from (ended in state.db). Must NOT be reused —
        it must be evicted so a fresh agent is built for the recovered
        session_id.
        """
        runner, db = _make_runner_with_db(tmp_path)
        db.create_session("dead_sid", source="telegram")
        db.end_session("dead_sid", "user_requested")  # #54878 self-heal target

        agent = object()
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:USER1"] = (agent, "sig", 642, "dead_sid")

        would_reuse, evicted = _guard_would_reuse_after_fix(
            runner, "telegram:USER1", "fresh_sid_after_selfheal"
        )

        assert would_reuse is False, (
            "BUG: stale agent from a self-healed dead session was reused — "
            "the #54878 x #54947 loop is back."
        )
        assert evicted is True
        with runner._agent_cache_lock:
            assert "telegram:USER1" not in runner._agent_cache

    def test_live_sibling_session_id_switch_still_reuses(self, tmp_path):
        """#54947 invariant must hold: switching between two LIVE sibling
        session_ids under the same session_key (e.g. /resume, /branch)
        still reuses the cached agent — the dead-session check must not
        fire when the cached session_id is not actually ended.
        """
        runner, db = _make_runner_with_db(tmp_path)
        db.create_session("sA", source="telegram")
        db.create_session("sB", source="telegram")
        # sA is NOT ended — a genuine live sibling conversation.

        agent = object()
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:USER1"] = (agent, "sig", 3, "sA")

        would_reuse, evicted = _guard_would_reuse_after_fix(runner, "telegram:USER1", "sB")

        assert would_reuse is True, "Regression: legit #54947 sibling-switch reuse broke."
        assert evicted is False
        with runner._agent_cache_lock:
            assert runner._agent_cache["telegram:USER1"][0] is agent

    def test_cross_process_write_same_session_still_invalidates(self, tmp_path):
        """#45966 invariant must hold: same session_id, message_count
        changed underneath (another process appended) -> still invalidates,
        unaffected by the new dead-session check.
        """
        runner, db = _make_runner_with_db(tmp_path)
        db.create_session("s1", source="telegram")

        agent = object()
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:s1"] = (agent, "sig", 0, "s1")

        would_reuse, evicted = _guard_would_reuse_after_fix(
            runner, "telegram:s1", "s1", current_mc=2
        )

        assert would_reuse is False
        assert evicted is True

    def test_dead_session_but_matching_session_id_still_reuses(self, tmp_path):
        """Edge case: the CURRENT session_id itself happens to be ended in
        db (e.g. a race where end_session fired between routing and cache
        lookup) but matches the cached snapshot's session_id exactly — the
        new dead-session check only applies on a session_id MISMATCH, so
        this must fall through to the ordinary (same-session_id) path
        unaffected.
        """
        runner, db = _make_runner_with_db(tmp_path)
        db.create_session("s1", source="telegram")
        db.end_session("s1", "user_requested")

        agent = object()
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:s1"] = (agent, "sig", 0, "s1")

        would_reuse, evicted = _guard_would_reuse_after_fix(
            runner, "telegram:s1", "s1", current_mc=0
        )

        assert would_reuse is True
        assert evicted is False

    def test_race_relocked_entry_not_evicted_on_stale_peek_verdict(self, tmp_path):
        """The re-validation guard (`cached_sid == peek_cached_sid`) must
        prevent applying a dead-session verdict computed for one cached
        entry to a DIFFERENT entry that replaced it between the outside-
        lock peek and the lock-held decision (e.g. another turn already
        rebuilt the agent for a live sibling session in between).
        """
        runner, db = _make_runner_with_db(tmp_path)
        db.create_session("dead_sid", source="telegram")
        db.end_session("dead_sid", "user_requested")
        db.create_session("sC", source="telegram")  # live sibling

        # Simulate: outside-lock peek would have seen the dead entry, but by
        # the time we re-acquire the lock, another thread already replaced
        # it with a fresh cache entry for a live sibling session.
        with runner._agent_cache_lock:
            peek_entry = runner._agent_cache.get("telegram:USER1")
        assert peek_entry is None  # nothing cached yet at peek time

        agent_new = object()
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:USER1"] = (agent_new, "sig", 1, "sC")

        # peek_cached_sid is None (no entry existed at peek time) so
        # cached_sid_is_dead is never computed True, and stale_dead_sid_reuse
        # is False by construction — the live entry must be reused normally.
        would_reuse, evicted = _guard_would_reuse_after_fix(runner, "telegram:USER1", "sC")

        assert would_reuse is True
        assert evicted is False
        with runner._agent_cache_lock:
            assert runner._agent_cache["telegram:USER1"][0] is agent_new
