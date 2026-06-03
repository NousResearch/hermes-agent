"""Tests for group/channel context loading on session reset.

Acceptance criteria from the task:
- On session reset detection, fetch and inject the recent group message history
  (last N messages, configurable) into the context window before processing
  the triggering message.
- The loaded context must include sender names/IDs, timestamps, and message
  content so Hermes can infer topic continuity.
- Hermes does not treat the next message as a cold start.

Implementation surface:
  gateway/config.py      - SessionResetPolicy.group_context_on_reset
  gateway/session.py     - SessionEntry.prior_session_id
  gateway/run.py         - injection block in _handle_message_with_agent
"""
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionEntry, SessionSource, SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_group_source(chat_id="grp1", user_id="u1", chat_name="Test Group"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="group",
        user_id=user_id,
        user_name="alice",
        chat_name=chat_name,
    )


def _make_dm_source(chat_id="dm1", user_id="u1"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
        user_id=user_id,
        user_name="alice",
    )


def _make_store(tmp_path, group_context_on_reset=0, idle_minutes=1):
    config = GatewayConfig()
    config.default_reset_policy = SessionResetPolicy(
        mode="idle",
        idle_minutes=idle_minutes,
        group_context_on_reset=group_context_on_reset,
    )
    return SessionStore(sessions_dir=tmp_path, config=config)


def _fake_messages():
    """Return a realistic prior-session transcript."""
    now = datetime.now()
    base_ts = (now - timedelta(hours=2)).timestamp()
    return [
        {"role": "user", "content": "Hey, what's the plan for today?",
         "timestamp": base_ts, "user_name": "alice"},
        {"role": "assistant", "content": "We're reviewing the Q3 roadmap. Main topics: capacity planning and hiring.",
         "timestamp": base_ts + 10},
        {"role": "user", "content": "Got it. I'll prep the capacity numbers.",
         "timestamp": base_ts + 60, "user_name": "bob"},
        {"role": "user", "content": "Should we also address the infra backlog?",
         "timestamp": base_ts + 120, "user_name": "alice"},
        {"role": "assistant", "content": "Yes, please add the infra items to the agenda.",
         "timestamp": base_ts + 130},
    ]


# ---------------------------------------------------------------------------
# SessionResetPolicy.group_context_on_reset
# ---------------------------------------------------------------------------

class TestGroupContextOnResetConfig:
    def test_default_is_zero(self):
        policy = SessionResetPolicy()
        assert policy.group_context_on_reset == 0

    def test_to_dict_includes_field(self):
        policy = SessionResetPolicy(group_context_on_reset=10)
        d = policy.to_dict()
        assert d["group_context_on_reset"] == 10

    def test_from_dict_parses_field(self):
        policy = SessionResetPolicy.from_dict({
            "mode": "idle",
            "idle_minutes": 60,
            "group_context_on_reset": 20,
        })
        assert policy.group_context_on_reset == 20

    def test_from_dict_defaults_to_zero_when_absent(self):
        policy = SessionResetPolicy.from_dict({"mode": "idle", "idle_minutes": 30})
        assert policy.group_context_on_reset == 0

    def test_from_dict_ignores_invalid_value(self):
        policy = SessionResetPolicy.from_dict({
            "mode": "idle",
            "group_context_on_reset": "not-a-number",
        })
        assert policy.group_context_on_reset == 0

    def test_from_dict_accepts_string_int(self):
        policy = SessionResetPolicy.from_dict({"group_context_on_reset": "15"})
        assert policy.group_context_on_reset == 15


# ---------------------------------------------------------------------------
# SessionEntry.prior_session_id
# ---------------------------------------------------------------------------

class TestSessionEntryPriorSessionId:
    def test_default_is_none(self):
        entry = SessionEntry(
            session_key="k",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        assert entry.prior_session_id is None

    def test_prior_session_id_not_in_to_dict(self):
        """prior_session_id is runtime-only; not serialised to disk."""
        entry = SessionEntry(
            session_key="k",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            prior_session_id="old-s0",
        )
        d = entry.to_dict()
        # Should not appear in the persisted dict (avoids stale references
        # surviving restarts and being re-consumed on unrelated sessions).
        assert "prior_session_id" not in d


class TestSessionStorePriorSessionIdOnAutoReset:
    def test_prior_session_id_set_on_auto_reset(self, tmp_path):
        """After an idle auto-reset, the new entry carries the old session_id."""
        store = _make_store(tmp_path, group_context_on_reset=5, idle_minutes=1)
        source = _make_group_source()

        # Create initial session
        first_entry = store.get_or_create_session(source)
        first_session_id = first_entry.session_id

        # Artificially age the entry so it appears expired
        first_entry.updated_at = datetime.now() - timedelta(minutes=10)
        store._save()

        # Next access triggers auto-reset
        new_entry = store.get_or_create_session(source)

        assert new_entry.session_id != first_session_id
        assert new_entry.was_auto_reset is True
        assert new_entry.prior_session_id == first_session_id

    def test_prior_session_id_none_on_fresh_session(self, tmp_path):
        """A brand-new session (no prior) never has prior_session_id set."""
        store = _make_store(tmp_path, group_context_on_reset=5, idle_minutes=60)
        source = _make_group_source()
        entry = store.get_or_create_session(source)
        assert entry.prior_session_id is None

    def test_prior_session_id_none_on_normal_access(self, tmp_path):
        """An active (not expired) session access does not set prior_session_id."""
        store = _make_store(tmp_path, group_context_on_reset=5, idle_minutes=60)
        source = _make_group_source()
        store.get_or_create_session(source)
        # Access again within idle window
        entry = store.get_or_create_session(source)
        assert entry.prior_session_id is None
        assert entry.was_auto_reset is False


# ---------------------------------------------------------------------------
# Context injection integration test
#
# We test the injection logic directly without booting the full gateway by
# extracting and replicating the critical path from _handle_message_with_agent.
# ---------------------------------------------------------------------------

class TestGroupContextInjectionLogic:
    """Unit-test the context-injection block in isolation."""

    def _run_injection(self, session_entry, source, load_transcript_fn, config):
        """Mirror of the injection block in _handle_message_with_agent."""
        context_prompt = "## Current Session Context\n(placeholder)"

        _prior_session_id = getattr(session_entry, "prior_session_id", None)
        if _prior_session_id and source.chat_type in ("group", "channel"):
            try:
                _reset_policy = config.get_reset_policy(
                    platform=source.platform,
                    session_type=source.chat_type,
                )
                _gcor = getattr(_reset_policy, "group_context_on_reset", 0)
                if _gcor > 0:
                    _prior_msgs = load_transcript_fn(_prior_session_id)
                    _text_turns = [
                        m for m in (_prior_msgs or [])
                        if m.get("role") in ("user", "assistant")
                        and m.get("content")
                        and not m.get("tool_calls")
                        and not m.get("tool_call_id")
                    ]
                    _tail = _text_turns[-_gcor:]
                    if _tail:
                        import datetime as _dt
                        _lines = ["[Prior conversation context (last {} message{} before session reset):]".format(
                            len(_tail), "s" if len(_tail) != 1 else ""
                        )]
                        for _m in _tail:
                            _role = _m.get("role", "")
                            _ts = _m.get("timestamp")
                            _ts_str = ""
                            if _ts:
                                try:
                                    _ts_str = " @ " + _dt.datetime.fromtimestamp(float(_ts)).strftime("%Y-%m-%d %H:%M")
                                except Exception:
                                    pass
                            _sender = _m.get("user_name") or _m.get("user_id") or (
                                "assistant" if _role == "assistant" else "user"
                            )
                            _content = str(_m.get("content") or "").strip()
                            _lines.append(f"[{_sender}{_ts_str}]: {_content}")
                        _history_block = "\n".join(_lines)
                        context_prompt = _history_block + "\n\n" + context_prompt
            except Exception:
                pass
            finally:
                session_entry.prior_session_id = None

        return context_prompt

    def _make_reset_entry(self, prior_id="prior-s0"):
        return SessionEntry(
            session_key="k",
            session_id="new-s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            chat_type="group",
            was_auto_reset=True,
            prior_session_id=prior_id,
        )

    def test_context_prepended_for_group_on_reset(self):
        """History block appears before the session context prompt."""
        config = GatewayConfig()
        config.default_reset_policy = SessionResetPolicy(group_context_on_reset=5)
        entry = self._make_reset_entry()
        source = _make_group_source()
        msgs = _fake_messages()

        result = self._run_injection(entry, source, lambda _: msgs, config)

        assert result.startswith("[Prior conversation context")
        assert "Current Session Context" in result
        # Content appears
        assert "Q3 roadmap" in result
        assert "capacity planning" in result

    def test_context_includes_sender_names(self):
        config = GatewayConfig()
        config.default_reset_policy = SessionResetPolicy(group_context_on_reset=5)
        entry = self._make_reset_entry()
        source = _make_group_source()

        result = self._run_injection(entry, source, lambda _: _fake_messages(), config)

        assert "[alice" in result
        assert "[bob" in result

    def test_context_includes_timestamps(self):
        config = GatewayConfig()
        config.default_reset_policy = SessionResetPolicy(group_context_on_reset=5)
        entry = self._make_reset_entry()
        source = _make_group_source()

        result = self._run_injection(entry, source, lambda _: _fake_messages(), config)

        # Timestamps formatted as "@ YYYY-MM-DD HH:MM" should appear
        assert " @ 20" in result  # crude year check

    def test_limits_to_n_messages(self):
        """Only the last N messages are injected."""
        config = GatewayConfig()
        config.default_reset_policy = SessionResetPolicy(group_context_on_reset=2)
        entry = self._make_reset_entry()
        source = _make_group_source()
        msgs = _fake_messages()  # 5 messages

        result = self._run_injection(entry, source, lambda _: msgs, config)

        assert "last 2 messages" in result
        # Only the last 2 text messages should appear
        assert "infra backlog" in result            # second-to-last user turn
        assert "add the infra items" in result      # last assistant turn
        # Earlier messages should NOT appear
        assert "Q3 roadmap" not in result

    def test_no_injection_when_disabled(self):
        """group_context_on_reset=0 means no injection."""
        config = GatewayConfig()
        config.default_reset_policy = SessionResetPolicy(group_context_on_reset=0)
        entry = self._make_reset_entry()
        source = _make_group_source()

        result = self._run_injection(entry, source, lambda _: _fake_messages(), config)

        assert "Prior conversation context" not in result
        assert result.startswith("## Current Session Context")

    def test_no_injection_for_dm_chat_type(self):
        """DMs are private 1:1 sessions — no group context injection."""
        config = GatewayConfig()
        config.default_reset_policy = SessionResetPolicy(group_context_on_reset=10)
        entry = self._make_reset_entry()
        source = _make_dm_source()  # chat_type="dm"

        result = self._run_injection(entry, source, lambda _: _fake_messages(), config)

        assert "Prior conversation context" not in result

    def test_no_injection_when_prior_session_id_none(self):
        """If no prior session recorded, no injection block is added."""
        config = GatewayConfig()
        config.default_reset_policy = SessionResetPolicy(group_context_on_reset=5)
        entry = SessionEntry(
            session_key="k",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            prior_session_id=None,
        )
        source = _make_group_source()

        result = self._run_injection(entry, source, lambda _: _fake_messages(), config)

        assert "Prior conversation context" not in result

    def test_prior_session_id_consumed_after_injection(self):
        """The prior_session_id flag is cleared after one use."""
        config = GatewayConfig()
        config.default_reset_policy = SessionResetPolicy(group_context_on_reset=5)
        entry = self._make_reset_entry()
        source = _make_group_source()

        self._run_injection(entry, source, lambda _: _fake_messages(), config)

        assert entry.prior_session_id is None

    def test_injection_skips_tool_calls_and_tool_results(self):
        """Tool messages are filtered out of the injected history."""
        config = GatewayConfig()
        config.default_reset_policy = SessionResetPolicy(group_context_on_reset=10)
        entry = self._make_reset_entry()
        source = _make_group_source()

        msgs = [
            {"role": "user", "content": "Run a search please"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "tc1", "function": {"name": "web_search"}}]},
            {"role": "tool", "content": "Search results here", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "Here are your results: X, Y, Z"},
        ]

        result = self._run_injection(entry, source, lambda _: msgs, config)

        assert "Prior conversation context" in result
        # Tool call and result should not appear
        assert "web_search" not in result
        assert "Search results here" not in result
        # Regular text turns should appear
        assert "Run a search please" in result
        assert "Here are your results" in result

    def test_empty_prior_transcript_produces_no_injection(self):
        """An empty prior session doesn't produce an empty injection block."""
        config = GatewayConfig()
        config.default_reset_policy = SessionResetPolicy(group_context_on_reset=5)
        entry = self._make_reset_entry()
        source = _make_group_source()

        result = self._run_injection(entry, source, lambda _: [], config)

        assert "Prior conversation context" not in result

    def test_load_transcript_exception_is_non_fatal(self):
        """Errors loading the prior transcript don't crash the request."""
        config = GatewayConfig()
        config.default_reset_policy = SessionResetPolicy(group_context_on_reset=5)
        entry = self._make_reset_entry()
        source = _make_group_source()

        def _boom(_):
            raise RuntimeError("DB connection lost")

        # Should not raise
        result = self._run_injection(entry, source, _boom, config)

        # Falls back to unmodified context_prompt
        assert "Current Session Context" in result
        # prior_session_id still consumed in finally block
        assert entry.prior_session_id is None
