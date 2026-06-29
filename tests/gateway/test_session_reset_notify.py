"""Tests for session auto-reset notifications.

Verifies that:
- _should_reset() returns a reason string ("idle" or "daily") instead of bool
- SessionEntry captures auto_reset_reason
- SessionResetPolicy.notify controls whether notifications are sent
- notify_exclude_platforms skips notifications for excluded platforms
"""

from datetime import datetime, timedelta

import ast
import inspect


from gateway.config import (
    GatewayConfig,
    Platform,
    SessionResetPolicy,
)
from gateway import run as gateway_run
from gateway.run import _reset_reason_text, SESSION_RESET_NOTICE_SEND_FAILED
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


def _make_store(policy=None, tmp_path=None):
    config = GatewayConfig()
    if policy:
        config.default_reset_policy = policy
    store = SessionStore(sessions_dir=tmp_path or "/tmp/test-sessions", config=config)
    return store


# ---------------------------------------------------------------------------
# _should_reset returns reason string
# ---------------------------------------------------------------------------

class TestShouldResetReason:
    def test_returns_none_when_not_expired(self, tmp_path):
        store = _make_store(
            SessionResetPolicy(mode="both", idle_minutes=60, at_hour=4),
            tmp_path,
        )
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),  # just updated
        )
        source = _make_source()
        assert store._should_reset(entry, source) is None

    def test_returns_idle_when_idle_expired(self, tmp_path):
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=30),
            tmp_path,
        )
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now() - timedelta(hours=2),
            updated_at=datetime.now() - timedelta(hours=1),  # 60min ago > 30min threshold
        )
        source = _make_source()
        assert store._should_reset(entry, source) == "idle"

    def test_returns_daily_when_daily_boundary_crossed(self, tmp_path):
        now = datetime.now()
        store = _make_store(
            SessionResetPolicy(mode="daily", at_hour=now.hour),
            tmp_path,
        )
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=now - timedelta(days=2),
            updated_at=now - timedelta(days=1),  # last active yesterday
        )
        source = _make_source()
        assert store._should_reset(entry, source) == "daily"

    def test_returns_none_when_mode_is_none(self, tmp_path):
        store = _make_store(
            SessionResetPolicy(mode="none"),
            tmp_path,
        )
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now() - timedelta(days=30),
            updated_at=datetime.now() - timedelta(days=30),
        )
        source = _make_source()
        assert store._should_reset(entry, source) is None


# ---------------------------------------------------------------------------
# SessionEntry captures reason
# ---------------------------------------------------------------------------

class TestSessionEntryReason:
    def test_auto_reset_reason_stored(self, tmp_path):
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=1),
            tmp_path,
        )
        source = _make_source()

        # Create initial session
        entry1 = store.get_or_create_session(source)
        assert not entry1.was_auto_reset

        # Age it past the idle threshold
        entry1.updated_at = datetime.now() - timedelta(minutes=5)
        store._save()

        # Next call should create a new session with reason
        entry2 = store.get_or_create_session(source)
        assert entry2.was_auto_reset is True
        assert entry2.auto_reset_reason == "idle"
        assert entry2.session_id != entry1.session_id

    def test_reset_had_activity_false_when_no_tokens(self, tmp_path):
        """Expired session with no tokens → reset_had_activity=False."""
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=1),
            tmp_path,
        )
        source = _make_source()

        entry1 = store.get_or_create_session(source)
        # No tokens used — session was idle with no conversation
        entry1.updated_at = datetime.now() - timedelta(minutes=5)
        store._save()

        entry2 = store.get_or_create_session(source)
        assert entry2.was_auto_reset is True
        assert entry2.reset_had_activity is False

    def test_reset_had_activity_true_when_tokens_used(self, tmp_path):
        """Expired session with tokens → reset_had_activity=True."""
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=1),
            tmp_path,
        )
        source = _make_source()

        entry1 = store.get_or_create_session(source)
        # Simulate some conversation happened (last_prompt_tokens is the
        # API-reported prompt-token count persisted every turn via
        # update_session; total_tokens is never populated on SessionEntry —
        # see gateway/slash_commands.py).
        entry1.last_prompt_tokens = 5000
        entry1.updated_at = datetime.now() - timedelta(minutes=5)
        store._save()

        entry2 = store.get_or_create_session(source)
        assert entry2.was_auto_reset is True
        assert entry2.reset_had_activity is True


# ---------------------------------------------------------------------------
# SessionResetPolicy notify config
# ---------------------------------------------------------------------------

class TestResetPolicyNotify:
    def test_notify_defaults_true(self):
        policy = SessionResetPolicy()
        assert policy.notify is True

    def test_notify_exclude_defaults(self):
        policy = SessionResetPolicy()
        assert "api_server" in policy.notify_exclude_platforms
        assert "webhook" in policy.notify_exclude_platforms

    def test_from_dict_with_notify_false(self):
        policy = SessionResetPolicy.from_dict({"notify": False})
        assert policy.notify is False

    def test_from_dict_with_custom_excludes(self):
        policy = SessionResetPolicy.from_dict({
            "notify_exclude_platforms": ["api_server", "webhook", "homeassistant"],
        })
        assert "homeassistant" in policy.notify_exclude_platforms

    def test_from_dict_preserves_defaults_on_missing_keys(self):
        policy = SessionResetPolicy.from_dict({})
        assert policy.notify is True
        assert "api_server" in policy.notify_exclude_platforms

    def test_to_dict_roundtrip(self):
        original = SessionResetPolicy(
            mode="idle",
            notify=False,
            notify_exclude_platforms=("api_server",),
        )
        restored = SessionResetPolicy.from_dict(original.to_dict())
        assert restored.notify == original.notify
        assert restored.notify_exclude_platforms == original.notify_exclude_platforms
        assert restored.mode == original.mode


# ---------------------------------------------------------------------------
# SessionEntry to_dict / from_dict roundtrip for auto-reset fields
# ---------------------------------------------------------------------------

class TestSessionEntryAutoResetRoundtrip:
    def test_was_auto_reset_persists_across_roundtrip(self, tmp_path):
        """was_auto_reset=True survives to_dict() → from_dict() (gateway restart)."""
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=1),
            tmp_path,
        )
        source = _make_source()

        entry = store.get_or_create_session(source)
        entry.updated_at = datetime.now() - timedelta(minutes=5)
        store._save()

        entry2 = store.get_or_create_session(source)
        assert entry2.was_auto_reset is True
        assert entry2.auto_reset_reason == "idle"
        assert entry2.session_id != entry.session_id

        # Simulate gateway restart: reload from disk
        store._loaded = False
        store._entries.clear()
        store._ensure_loaded()

        reloaded = store._entries.get(entry2.session_key)
        assert reloaded is not None
        assert reloaded.was_auto_reset is True
        assert reloaded.auto_reset_reason == "idle"

    def test_reset_had_activity_persists_across_roundtrip(self, tmp_path):
        """reset_had_activity survives to_dict() → from_dict() (gateway restart)."""
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=1),
            tmp_path,
        )
        source = _make_source()

        entry = store.get_or_create_session(source)
        entry.last_prompt_tokens = 1000
        entry.updated_at = datetime.now() - timedelta(minutes=5)
        store._save()

        entry2 = store.get_or_create_session(source)
        assert entry2.reset_had_activity is True

        store._loaded = False
        store._entries.clear()
        store._ensure_loaded()

        reloaded = store._entries.get(entry2.session_key)
        assert reloaded is not None
        assert reloaded.reset_had_activity is True

    def test_auto_reset_reason_none_roundtrip(self, tmp_path):
        """auto_reset_reason=None (no reset) survives roundtrip cleanly."""
        store = _make_store(tmp_path=tmp_path)
        source = _make_source()

        entry = store.get_or_create_session(source)
        assert entry.was_auto_reset is False

        store._loaded = False
        store._entries.clear()
        store._ensure_loaded()

        reloaded = store._entries.get(entry.session_key)
        assert reloaded is not None
        assert reloaded.was_auto_reset is False
        assert reloaded.auto_reset_reason is None
        assert reloaded.reset_had_activity is False


# ---------------------------------------------------------------------------
# _reset_reason_text — mode-correct notice wording (idle/daily/both/suspended)
# ---------------------------------------------------------------------------

class TestResetReasonText:
    def test_idle_reason_72h(self):
        policy = SessionResetPolicy(mode="idle", idle_minutes=4320, at_hour=4)
        assert _reset_reason_text("idle", policy) == "inactive for 72h"

    def test_idle_reason_hours_and_minutes(self):
        policy = SessionResetPolicy(mode="idle", idle_minutes=90)
        assert _reset_reason_text("idle", policy) == "inactive for 1h 30m"

    def test_idle_reason_minutes_only(self):
        policy = SessionResetPolicy(mode="idle", idle_minutes=30)
        assert _reset_reason_text("idle", policy) == "inactive for 30m"

    def test_daily_reason_uses_at_hour(self):
        policy = SessionResetPolicy(mode="daily", at_hour=4)
        assert _reset_reason_text("daily", policy) == "daily schedule at 4:00"

    def test_suspended_reason(self):
        policy = SessionResetPolicy()
        assert _reset_reason_text("suspended", policy) == (
            "previous session was stopped or interrupted"
        )

    def test_both_mode_resolves_to_concrete_reason(self):
        """Under mode=both, _should_reset returns the concrete reason that
        tripped (idle or daily), never 'both' — so the helper is correct for
        both-mode without special-casing it."""
        policy = SessionResetPolicy(mode="both", idle_minutes=4320, at_hour=4)
        assert _reset_reason_text("idle", policy) == "inactive for 72h"
        assert _reset_reason_text("daily", policy) == "daily schedule at 4:00"

    def test_unknown_reason_is_neutral_no_raw_token(self):
        """A future/unknown reason yields no clause (caller emits the neutral
        'Session automatically reset.' form) and never echoes the raw token."""
        policy = SessionResetPolicy()
        result = _reset_reason_text("some_future_reason", policy)
        assert result == ""
        assert "some_future_reason" not in result
        assert "inactive" not in result
        assert "daily" not in result

    def test_marker_constant_is_stable(self):
        assert SESSION_RESET_NOTICE_SEND_FAILED == "SESSION_RESET_NOTICE_SEND_FAILED"


# ---------------------------------------------------------------------------
# had_any_turn — durable activity flag survives compaction zeroing
# (PR #104 Greptile P2: last_prompt_tokens is reset to 0 by the transcript-
# compression path; a compressed-then-idle session must still notify.)
# ---------------------------------------------------------------------------

class TestHadAnyTurnDurableFlag:
    def test_update_session_latches_flag_on_real_turn(self, tmp_path):
        store = _make_store(SessionResetPolicy(mode="idle", idle_minutes=1), tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        assert entry.had_any_turn is False
        store.update_session(entry.session_key, last_prompt_tokens=42000)
        assert store._entries[entry.session_key].had_any_turn is True

    def test_compaction_zeroing_does_not_clear_flag(self, tmp_path):
        """update_session(..., last_prompt_tokens=0) is how transcript
        compression marks the value stale — it must NOT set OR clear the
        durable flag."""
        store = _make_store(SessionResetPolicy(mode="idle", idle_minutes=1), tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        store.update_session(entry.session_key, last_prompt_tokens=42000)  # real turn
        store.update_session(entry.session_key, last_prompt_tokens=0)       # compaction
        live = store._entries[entry.session_key]
        assert live.last_prompt_tokens == 0       # compaction zeroed it
        assert live.had_any_turn is True          # but the durable flag survives

    def test_zero_only_session_never_latches_flag(self, tmp_path):
        """A session that only ever saw a 0 token-count (no real turn) must
        not latch the flag — no false 'history cleared'."""
        store = _make_store(SessionResetPolicy(mode="idle", idle_minutes=1), tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        store.update_session(entry.session_key, last_prompt_tokens=0)
        assert store._entries[entry.session_key].had_any_turn is False

    def test_compressed_then_idle_session_still_flags_activity(self, tmp_path):
        """The Greptile P2 case end-to-end: a session has a real turn, gets
        compressed (last_prompt_tokens -> 0), then idles out. The reset MUST
        still report had_activity via the durable flag."""
        store = _make_store(SessionResetPolicy(mode="idle", idle_minutes=1), tmp_path)
        source = _make_source()
        entry1 = store.get_or_create_session(source)
        store.update_session(entry1.session_key, last_prompt_tokens=42000)  # real turn
        store.update_session(entry1.session_key, last_prompt_tokens=0)       # compaction
        # Age it past the idle threshold.
        store._entries[entry1.session_key].updated_at = datetime.now() - timedelta(minutes=5)
        store._save()
        entry2 = store.get_or_create_session(source)
        assert entry2.was_auto_reset is True
        assert entry2.reset_had_activity is True  # would be False under the old gate

    def test_flag_survives_roundtrip(self, tmp_path):
        store = _make_store(SessionResetPolicy(mode="idle", idle_minutes=1), tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        store.update_session(entry.session_key, last_prompt_tokens=42000)
        store._loaded = False
        store._entries.clear()
        store._ensure_loaded()
        assert store._entries[entry.session_key].had_any_turn is True

    def test_legacy_entry_without_flag_falls_back_to_last_prompt_tokens(self, tmp_path):
        """Entries persisted before had_any_turn existed default the flag to
        False but may carry a non-zero last_prompt_tokens — the gate's OR
        fallback must still flag activity for them."""
        store = _make_store(SessionResetPolicy(mode="idle", idle_minutes=1), tmp_path)
        source = _make_source()
        entry1 = store.get_or_create_session(source)
        # Simulate a legacy row: token count set directly, flag never latched.
        entry1.last_prompt_tokens = 5000
        entry1.had_any_turn = False
        entry1.updated_at = datetime.now() - timedelta(minutes=5)
        store._save()
        entry2 = store.get_or_create_session(source)
        assert entry2.reset_had_activity is True


# ---------------------------------------------------------------------------
# AST invariant: the auto-reset NOTICE block wiring (gateway/run.py)
#
# The notice send lives deep in the 17k-line async message handler, so we pin
# its load-bearing structure with an AST invariant (the established pattern —
# see test_35809_auto_reset_clean_context.py) rather than driving the whole
# handler. The behavioral correctness of the gate + wording is covered by the
# real-SessionStore tests above and the _reset_reason_text unit tests.
# ---------------------------------------------------------------------------

def _find_auto_reset_notice_block() -> ast.If:
    """Return the ``if getattr(session_entry, 'was_auto_reset', False):`` block."""
    tree = ast.parse(inspect.getsource(gateway_run))
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        # The 2026-06-29 upstream merge refactored the guard from
        # ``if getattr(session_entry, "was_auto_reset", False):`` (string const in
        # the test) to a capture-then-check: ``_was_auto_reset = getattr(...)`` /
        # ``if _was_auto_reset:`` (a bare Name in the test). Recognize EITHER shape:
        # the test references the string "was_auto_reset" OR the captured name
        # ``_was_auto_reset``/``was_auto_reset``.
        consts = {
            n.value
            for n in ast.walk(node.test)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        names = {
            n.id
            for n in ast.walk(node.test)
            if isinstance(n, ast.Name)
        }
        if "was_auto_reset" in consts or names & {"_was_auto_reset", "was_auto_reset"}:
            calls = {
                sub.func.attr
                for sub in ast.walk(node)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
            }
            if "_reset_reason_text" in calls or "send" in calls:
                candidates.append(node)
    assert candidates, (
        "Could not locate the auto-reset notice block "
        "(if getattr(session_entry,'was_auto_reset',...) ... _reset_reason_text) "
        "in gateway/run.py — the structure changed or the AST walker is stale."
    )
    # The outermost matching block.
    return max(candidates, key=lambda n: len(list(ast.walk(n))))


class TestAutoResetNoticeBlockWiring:
    def test_block_uses_reset_reason_text_helper(self):
        block = _find_auto_reset_notice_block()
        attrs = {
            sub.func.attr
            for sub in ast.walk(block)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
        }
        names = {
            sub.func.id
            for sub in ast.walk(block)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
        }
        assert "_reset_reason_text" in names, (
            "the auto-reset notice block must build its reason text via the "
            "_reset_reason_text helper (mode-correct + testable)."
        )
        assert "send" in attrs, (
            "the auto-reset notice block must adapter.send the notice."
        )

    def test_block_logs_warning_marker_on_send_failure(self):
        """The lost-send path must reference the greppable WARNING marker, and
        the marker must be logged at WARNING (not debug)."""
        block = _find_auto_reset_notice_block()
        marker_refs = [
            n
            for n in ast.walk(block)
            if isinstance(n, ast.Name) and n.id == "SESSION_RESET_NOTICE_SEND_FAILED"
        ]
        assert marker_refs, (
            "the auto-reset notice block must reference "
            "SESSION_RESET_NOTICE_SEND_FAILED on the send-failure path so a "
            "silently-broken adapter is greppable."
        )
        warning_calls = [
            sub
            for sub in ast.walk(block)
            if isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "warning"
        ]
        assert warning_calls, (
            "the lost-send path must logger.warning(...), not logger.debug(...)."
        )

    def test_notice_is_out_of_band_no_history_mutation(self):
        """Invariant I1/I2: the notice is sent via adapter.send only; it must
        NOT be appended into model conversation history (prompt-cache &
        role-alternation safety)."""
        block = _find_auto_reset_notice_block()
        bad_appends = [
            sub
            for sub in ast.walk(block)
            if isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "append"
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id in {"messages", "history"}
        ]
        assert not bad_appends, (
            "the auto-reset notice must be out-of-band (adapter.send), never "
            "appended into messages/history — that would break prompt caching "
            "and role alternation (Invariants I1/I2)."
        )
