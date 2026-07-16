"""Regression: a mirrored delivery must refresh the live session entry.

A cron job with ``attach_to_session=true`` mirrors its brief into the origin
chat's transcript via ``gateway.mirror.mirror_to_session``, which appends to
state.db only.  Before the fix nothing refreshed the live
``SessionEntry.updated_at``, so when the brief landed after an idle/daily
reset boundary the user's immediate reply auto-reset into a fresh session and
lost the mirrored context (observed on a Weixin DM).

The fix: before appending, ``mirror_to_session`` resolves the live routing
target (``gateway.session.resolve_live_mirror_session``).  While the routed
session is still live the store refreshes ``SessionEntry.updated_at`` (and
persists it to the routing index), so the reply stays associated both in the
live gateway and across a restart.  When the background expiry watcher
already finalized the session — or it was ended in state.db — before the
mirror arrived, a touch cannot help (``get_or_create_session`` self-heals /
resets away from ended sessions regardless of ``updated_at``); the store
instead runs the same routing transition the reply will take and the mirror
is appended into the resulting post-reset session.
"""

from datetime import timedelta

import pytest

import gateway.mirror as mirror_mod
from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionSource, SessionStore, _now


@pytest.fixture()
def hermes_tmp(tmp_path, monkeypatch):
    """Isolate state.db and the mirror's legacy sessions.json fallback."""
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(mirror_mod, "_SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(mirror_mod, "_SESSIONS_INDEX", sessions_dir / "sessions.json")
    return tmp_path


def _make_store(tmp_path, policy):
    config = GatewayConfig()
    config.default_reset_policy = policy
    return SessionStore(sessions_dir=tmp_path / "sessions", config=config)


def _weixin_dm_source():
    return SessionSource(
        platform=Platform.WEIXIN,
        chat_id="wx_lulu_dm",
        chat_type="dm",
        user_id="lulu",
        user_name="Lulu",
    )


class TestMirrorRefreshesLiveSession:
    def test_reply_after_mirrored_cron_delivery_stays_in_session(self, hermes_tmp):
        """Idle boundary crossed, then cron mirrors its brief: the user's
        immediate reply must resolve to the mirrored session, not a reset."""
        from cron.scheduler import _maybe_mirror_cron_delivery

        store = _make_store(
            hermes_tmp, SessionResetPolicy(mode="idle", idle_minutes=60)
        )
        source = _weixin_dm_source()
        entry = store.get_or_create_session(source)
        original_session_id = entry.session_id

        # Session goes quiet past the idle boundary.
        entry.updated_at = _now() - timedelta(minutes=180)

        job = {"id": "job_brief", "name": "Morning brief", "attach_to_session": True}
        _maybe_mirror_cron_delivery(
            job,
            "weixin",
            "wx_lulu_dm",
            "Task #2 is due today.",
            user_id="lulu",
            enabled=True,
        )

        # The brief landed in the original session's transcript...
        transcript = store.load_transcript(original_session_id)
        assert any("Task #2" in str(m.get("content")) for m in transcript)

        # ...and the reply right after it joins that same session.
        reply_entry = store.get_or_create_session(source)
        assert reply_entry.session_id == original_session_id
        assert not reply_entry.was_auto_reset

    def test_reply_after_mirror_survives_daily_boundary(self, hermes_tmp):
        """Same association guarantee for a daily reset policy, exercised via
        mirror_to_session directly (the shared choke point send_message and
        all cron mirror paths ride)."""
        from gateway.mirror import mirror_to_session

        store = _make_store(hermes_tmp, SessionResetPolicy(mode="daily", at_hour=4))
        source = _weixin_dm_source()
        entry = store.get_or_create_session(source)
        original_session_id = entry.session_id

        # Last activity two days ago -- unambiguously before today's reset.
        entry.updated_at = _now() - timedelta(days=2)

        assert mirror_to_session(
            "weixin",
            "wx_lulu_dm",
            "[Cron delivery: Morning brief]\nTask #2 is due today.",
            source_label="cron",
            user_id="lulu",
            role="user",
        )

        reply_entry = store.get_or_create_session(source)
        assert reply_entry.session_id == original_session_id

    def test_mirror_touch_persists_across_gateway_restart(self, hermes_tmp):
        """The touch is written to the routing index, so a gateway restarted
        between the mirrored delivery and the user's reply still routes the
        reply into the mirrored session."""
        from gateway.mirror import mirror_to_session

        policy = SessionResetPolicy(mode="idle", idle_minutes=60)
        store = _make_store(hermes_tmp, policy)
        source = _weixin_dm_source()
        entry = store.get_or_create_session(source)
        original_session_id = entry.session_id

        # Persist the stale timestamp, as a real idle gateway would have.
        entry.updated_at = _now() - timedelta(minutes=180)
        store._save_entries()

        assert mirror_to_session(
            "weixin",
            "wx_lulu_dm",
            "[Cron delivery: Morning brief]\nTask #2 is due today.",
            source_label="cron",
            user_id="lulu",
            role="user",
        )

        restarted = _make_store(hermes_tmp, policy)
        reply_entry = restarted.get_or_create_session(source)
        assert reply_entry.session_id == original_session_id

    def test_touch_returns_false_for_unknown_session(self, hermes_tmp):
        store = _make_store(
            hermes_tmp, SessionResetPolicy(mode="idle", idle_minutes=60)
        )
        assert store.touch_session_by_id("no_such_session") is False
        assert store.touch_session_by_id("") is False


class TestMirrorAfterSessionFinalized:
    """The expiry watcher can beat the mirror to the reset boundary.

    ``GatewayRunner._session_expiry_watcher`` runs every 5 minutes: at a
    daily/idle boundary it finalizes the routed session (marks
    ``expiry_finalized``, fires hooks, evicts the agent) before an
    early-morning cron mirror arrives.  Merely appending + touching would
    resurrect or strand the mirror in that session — ``get_or_create_session``
    self-heals/resets away from finalized/ended sessions on the user's reply.
    The mirror must instead land in the post-reset session the reply routes to.
    """

    def test_mirror_after_watcher_finalized_redirects_to_post_reset_session(
        self, hermes_tmp
    ):
        """expiry watcher finalized at the daily boundary → the mirror must
        resolve/create the post-reset session and append there, and the
        user's reply must join that same session."""
        from gateway.mirror import mirror_to_session

        store = _make_store(hermes_tmp, SessionResetPolicy(mode="daily", at_hour=4))
        source = _weixin_dm_source()
        entry = store.get_or_create_session(source)
        original_session_id = entry.session_id

        # Two days quiet, then the background expiry watcher finalizes the
        # session at the daily boundary — before the cron mirror arrives.
        entry.updated_at = _now() - timedelta(days=2)
        store.set_expiry_finalized(entry)

        assert mirror_to_session(
            "weixin",
            "wx_lulu_dm",
            "[Cron delivery: Morning brief]\nTask #2 is due today.",
            source_label="cron",
            user_id="lulu",
            role="user",
        )

        # The reply routes into a fresh post-reset session (the daily reset
        # still applies)...
        reply_entry = store.get_or_create_session(source)
        assert reply_entry.session_id != original_session_id
        assert reply_entry.was_auto_reset

        # ...and the mirrored brief is in THAT session, not the finalized one.
        new_transcript = store.load_transcript(reply_entry.session_id)
        assert any("Task #2" in str(m.get("content")) for m in new_transcript)
        old_transcript = store.load_transcript(original_session_id)
        assert not any("Task #2" in str(m.get("content")) for m in old_transcript)

    def test_mirror_into_db_ended_session_redirects(self, hermes_tmp):
        """The routed session was already ended in state.db (end_reason set)
        while the routing entry survived — the mirror must not append into
        the ended session the reply will self-heal away from (#54878)."""
        import hermes_state
        from gateway.mirror import mirror_to_session

        store = _make_store(
            hermes_tmp, SessionResetPolicy(mode="idle", idle_minutes=60)
        )
        source = _weixin_dm_source()
        entry = store.get_or_create_session(source)
        original_session_id = entry.session_id

        # Session ended in state.db out-of-band (non-recoverable reason), the
        # in-memory/sessions.json routing entry still points at it.
        db = hermes_state.SessionDB()
        try:
            db.end_session(original_session_id, "session_reset")
        finally:
            db.close()

        assert mirror_to_session(
            "weixin",
            "wx_lulu_dm",
            "[Cron delivery: Morning brief]\nTask #2 is due today.",
            source_label="cron",
            user_id="lulu",
            role="user",
        )

        reply_entry = store.get_or_create_session(source)
        assert reply_entry.session_id != original_session_id
        new_transcript = store.load_transcript(reply_entry.session_id)
        assert any("Task #2" in str(m.get("content")) for m in new_transcript)
        old_transcript = store.load_transcript(original_session_id)
        assert not any("Task #2" in str(m.get("content")) for m in old_transcript)
