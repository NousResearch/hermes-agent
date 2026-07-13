"""Tests for expiry_finalized session reset (#63539).

The background expiry watcher sets entry.expiry_finalized=True when a session
crosses its idle/daily boundary.  Before this fix, _should_reset() only checked
time-based conditions (updated_at vs today_reset / idle_deadline) and never
consulted the expiry_finalized flag.  Since update_session() bumps updated_at
on every turn, the daily check always passed because updated_at was always
fresh, so finalized sessions were never reset.

These tests verify that:
- _should_reset() returns "expiry_finalized" when the flag is set
- A finalized session is replaced on next inbound message (get_or_create_session)
- The expiry_finalized flag does NOT override mode="none"
- Active background processes still prevent reset of finalized sessions
- The "expiry_finalized" reason survives the auto_reset_reason roundtrip
"""

from datetime import datetime, timedelta

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionEntry, SessionSource, SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(platform=Platform.TELEGRAM, chat_id="123", user_id="u1"):
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        user_id=user_id,
    )


def _make_store(policy=None, tmp_path=None, has_active_processes_fn=None):
    config = GatewayConfig()
    if policy:
        config.default_reset_policy = policy
    store = SessionStore(
        sessions_dir=tmp_path or "/tmp/test-sessions",
        config=config,
    )
    if has_active_processes_fn is not None:
        store._has_active_processes_fn = has_active_processes_fn
    return store


# ---------------------------------------------------------------------------
# _should_reset checks expiry_finalized
# ---------------------------------------------------------------------------

class TestShouldResetExpiryFinalized:
    def test_finalized_session_returns_expiry_finalized_reason(self, tmp_path):
        """A session with expiry_finalized=True should be reset."""
        store = _make_store(
            SessionResetPolicy(mode="daily", at_hour=4),
            tmp_path,
        )
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),  # just updated — would pass time check
            expiry_finalized=True,
        )
        source = _make_source()
        assert store._should_reset(entry, source) == "expiry_finalized"

    def test_non_finalized_session_not_affected(self, tmp_path):
        """A session with expiry_finalized=False should use normal time checks."""
        store = _make_store(
            SessionResetPolicy(mode="daily", at_hour=4),
            tmp_path,
        )
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            expiry_finalized=False,
        )
        source = _make_source()
        # updated_at is fresh so time-based check should not trigger
        assert store._should_reset(entry, source) is None

    def test_finalized_with_idle_mode(self, tmp_path):
        """expiry_finalized triggers reset even under idle-only mode."""
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=60),
            tmp_path,
        )
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),  # just updated — idle check would pass
            expiry_finalized=True,
        )
        source = _make_source()
        assert store._should_reset(entry, source) == "expiry_finalized"

    def test_finalized_with_both_mode(self, tmp_path):
        """expiry_finalized triggers reset under mode=both."""
        store = _make_store(
            SessionResetPolicy(mode="both", idle_minutes=60, at_hour=4),
            tmp_path,
        )
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            expiry_finalized=True,
        )
        source = _make_source()
        assert store._should_reset(entry, source) == "expiry_finalized"

    def test_finalized_does_not_override_mode_none(self, tmp_path):
        """mode=none should prevent reset even if expiry_finalized is True.

        This preserves the existing contract: mode=none means the session never
        expires, and the expiry watcher should never set expiry_finalized for
        such sessions in the first place.  But if the flag is set (e.g. a race
        or migration artifact), mode=none must still win.
        """
        store = _make_store(
            SessionResetPolicy(mode="none"),
            tmp_path,
        )
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            expiry_finalized=True,
        )
        source = _make_source()
        assert store._should_reset(entry, source) is None

    def test_active_processes_prevent_finalized_reset(self, tmp_path):
        """Active background processes block reset even for finalized sessions."""
        store = _make_store(
            SessionResetPolicy(mode="daily", at_hour=4),
            tmp_path,
            has_active_processes_fn=lambda key: True,  # always active
        )
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            expiry_finalized=True,
        )
        source = _make_source()
        assert store._should_reset(entry, source) is None


# ---------------------------------------------------------------------------
# Integration: get_or_create_session resets a finalized session
# ---------------------------------------------------------------------------

class TestGetOrCreateSessionExpiryFinalized:
    def test_finalized_session_replaced_on_next_message(self, tmp_path):
        """Simulates the #63539 scenario: session finalized by expiry watcher,
        then user sends a new message.  The old session should be replaced."""
        store = _make_store(
            SessionResetPolicy(mode="daily", at_hour=4),
            tmp_path,
        )
        source = _make_source()

        # Create a session
        entry1 = store.get_or_create_session(source)
        old_session_id = entry1.session_id

        # Simulate the expiry watcher marking it as finalized
        with store._lock:
            entry1.expiry_finalized = True
            store._save()

        # Next message should create a fresh session
        entry2 = store.get_or_create_session(source)
        assert entry2.session_id != old_session_id
        assert entry2.was_auto_reset is True
        assert entry2.auto_reset_reason == "expiry_finalized"

    def test_finalized_session_fresh_updated_at_still_resets(self, tmp_path):
        """Even if updated_at was bumped recently, the finalized flag forces
        a reset.  This is the core regression from #63539."""
        store = _make_store(
            SessionResetPolicy(mode="both", idle_minutes=1440, at_hour=4),
            tmp_path,
        )
        source = _make_source()

        entry1 = store.get_or_create_session(source)
        old_id = entry1.session_id

        # Bump updated_at to now (simulating an active session)
        entry1.updated_at = datetime.now()
        # Then mark finalized (expiry watcher fires based on at_hour)
        with store._lock:
            entry1.expiry_finalized = True
            store._save()

        # Despite fresh updated_at, the finalized session must be replaced
        entry2 = store.get_or_create_session(source)
        assert entry2.session_id != old_id
        assert entry2.expiry_finalized is False  # new session starts clean


# ---------------------------------------------------------------------------
# Persistence roundtrip
# ---------------------------------------------------------------------------

class TestExpiryFinalizedReasonRoundtrip:
    def test_expiry_finalized_reason_persists_across_restart(self, tmp_path):
        """auto_reset_reason='expiry_finalized' survives to_dict → from_dict."""
        store = _make_store(
            SessionResetPolicy(mode="daily", at_hour=4),
            tmp_path,
        )
        source = _make_source()

        entry1 = store.get_or_create_session(source)
        with store._lock:
            entry1.expiry_finalized = True
            store._save()

        entry2 = store.get_or_create_session(source)
        assert entry2.auto_reset_reason == "expiry_finalized"

        # Simulate gateway restart
        store._loaded = False
        store._entries.clear()
        store._ensure_loaded()

        reloaded = store._entries.get(entry2.session_key)
        assert reloaded is not None
        assert reloaded.was_auto_reset is True
        assert reloaded.auto_reset_reason == "expiry_finalized"
        assert reloaded.expiry_finalized is False  # new session, not finalized
