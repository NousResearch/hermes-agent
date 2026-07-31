"""Tests for the per-turn date tail delivered via the ``api_content`` sidecar.

Long-lived gateway sessions span days.  The system prompt's
``Conversation started:`` date is a session-start value by design and goes
stale when a session crosses a calendar day, so the current date is delivered
as API-only current-turn user context: ``hermes_time.current_date_line()`` is
folded into the user-message composition in ``build_turn_context`` and rides
the ``api_content`` sidecar channel (the same path plugin context and gateway
notes use).  The stored system prompt, the session DB row, and the outgoing
wire system message are all untouched.

Every date is mocked through ``hermes_time.now`` with a tz-aware datetime.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_loop import _restore_or_build_system_prompt
from agent.prompt_caching import apply_anthropic_cache_control
from agent.turn_context import substitute_api_content
from hermes_time import current_date_line
from tests.agent.test_gateway_turn_sidecar import (
    VC_NOTE,
    _FakeAgent,
    _build,
)


_MAY_17 = datetime(2026, 5, 17, 9, 30, tzinfo=timezone.utc)
_MAY_18 = datetime(2026, 5, 18, 9, 0, tzinfo=timezone.utc)

_DATE_SUNDAY = "Today's date: Sunday, May 17, 2026"
_DATE_MONDAY = "Today's date: Monday, May 18, 2026"


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


class TestCurrentDateLine:
    """Unit coverage for ``hermes_time.current_date_line()``."""

    def test_date_only_format(self):
        with patch("hermes_time.now", return_value=_MAY_17):
            line = current_date_line()
        assert line == _DATE_SUNDAY
        assert re.match(r"^Today's date: \w+, \w+ \d{1,2}, \d{4}$", line)
        assert "09:30" not in line
        assert "9:30" not in line

    def test_line_tracks_mocked_now(self):
        with patch("hermes_time.now", return_value=_MAY_18):
            assert current_date_line() == _DATE_MONDAY


class TestPrologueInjection:
    """The date tail folds into the current turn's user-message composition."""

    def test_string_turn_sidecar_ends_with_date_clean_content_untouched(self):
        agent = _FakeAgent()
        with patch("hermes_time.now", return_value=_MAY_17), patch(
            "hermes_cli.plugins.invoke_hook", return_value=[]
        ):
            ctx = _build(agent)
        msg = ctx.messages[ctx.current_turn_user_idx]
        # Visible content stays clean; the date lives only in the sidecar.
        assert msg["content"] == "hello"
        assert msg["api_content"] == "hello\n\n" + _DATE_SUNDAY

    def test_date_appends_after_plugin_and_gateway_notes(self):
        agent = _FakeAgent()
        agent._gateway_turn_context_notes = VC_NOTE
        with patch("hermes_time.now", return_value=_MAY_17), patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[{"context": "PLUGIN-CTX"}],
        ):
            ctx = _build(agent)
        msg = ctx.messages[ctx.current_turn_user_idx]
        # Order: content -> plugin context -> gateway notes -> date tail.
        assert (
            msg["api_content"]
            == "hello\n\nPLUGIN-CTX\n\n" + VC_NOTE + "\n\n" + _DATE_SUNDAY
        )

    def test_same_day_two_turns_no_duplicate(self):
        with patch("hermes_time.now", return_value=_MAY_17), patch(
            "hermes_cli.plugins.invoke_hook", return_value=[]
        ):
            ctx1 = _build(_FakeAgent(), user_message="first")
            ctx2 = _build(_FakeAgent(), user_message="second")
        m1 = ctx1.messages[ctx1.current_turn_user_idx]
        m2 = ctx2.messages[ctx2.current_turn_user_idx]
        assert m1["api_content"] == "first\n\n" + _DATE_SUNDAY
        assert m2["api_content"] == "second\n\n" + _DATE_SUNDAY
        assert m1["api_content"].count("Today's date:") == 1
        assert m2["api_content"].count("Today's date:") == 1

    def test_cross_day_new_turn_gets_new_date(self):
        with patch("hermes_time.now", return_value=_MAY_17), patch(
            "hermes_cli.plugins.invoke_hook", return_value=[]
        ):
            ctx1 = _build(_FakeAgent(), user_message="first")
        with patch("hermes_time.now", return_value=_MAY_18), patch(
            "hermes_cli.plugins.invoke_hook", return_value=[]
        ):
            ctx2 = _build(_FakeAgent(), user_message="second")
        m1 = ctx1.messages[ctx1.current_turn_user_idx]
        m2 = ctx2.messages[ctx2.current_turn_user_idx]
        assert m1["api_content"].endswith(_DATE_SUNDAY)
        assert m2["api_content"].endswith(_DATE_MONDAY)
        assert "May 17" not in m2["api_content"]

    def test_historical_replay_keeps_sent_bytes(self):
        historical = {
            "role": "user",
            "content": "q1",
            "api_content": "q1\n\nToday's date: Saturday, May 16, 2026",
        }
        api_msg = dict(historical)
        substitute_api_content(api_msg)
        # Replay is byte-identical to what was sent that turn — the date is
        # NOT rewritten to today.
        assert api_msg["content"] == historical["api_content"]
        assert "Today's date: Saturday, May 16, 2026" in api_msg["content"]

    def test_multimodal_turn_no_date(self):
        agent = _FakeAgent()
        content = [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "https://x/img.png"}},
        ]
        with patch("hermes_time.now", return_value=_MAY_17), patch(
            "hermes_cli.plugins.invoke_hook", return_value=[]
        ):
            ctx = _build(agent, user_message=content)
        msg = ctx.messages[ctx.current_turn_user_idx]
        assert "api_content" not in msg
        parts = [p for p in msg["content"] if isinstance(p, dict)]
        assert all("Today's date:" not in (p.get("text") or "") for p in parts)


class TestWireContract:
    """The three outbound payload paths carry the date; the system never does."""

    def test_normal_wire_user_has_date_system_byte_identical(self):
        agent = _FakeAgent()
        with patch("hermes_time.now", return_value=_MAY_17), patch(
            "hermes_cli.plugins.invoke_hook", return_value=[]
        ):
            ctx = _build(agent)
        msg = ctx.messages[ctx.current_turn_user_idx]
        # The main loop sends the stamped sidecar verbatim
        # (conversation_loop 1462-1469); the system message is the cached
        # prompt unchanged (conversation_loop 1552-1556).
        assert msg["api_content"] == "hello\n\n" + _DATE_SUNDAY
        assert agent._cached_system_prompt == "SYSTEM"
        assert "Today's date:" not in agent._cached_system_prompt

    def test_failover_retry_keeps_date_and_identity(self):
        from agent.conversation_loop import _sync_failover_system_message

        static = "You are a helpful assistant.\n\nStable brief.\n"
        prompt = static + "Model: gpt-5.4-mini\nProvider: openai-codex"
        agent = SimpleNamespace(
            _cached_system_prompt=prompt, ephemeral_system_prompt=None
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "hello"},
        ]
        api_messages = apply_anthropic_cache_control(
            messages,
            cache_ttl=None,
            native_anthropic=True,
            static_system_prefix=static,
        )
        # The current-turn user message carries yesterday's date via its
        # sidecar-stamped API copy (stamped at turn start).
        api_messages[1]["content"] = "hello\n\n" + _DATE_SUNDAY

        with patch("hermes_time.now", return_value=_MAY_18):
            _sync_failover_system_message(agent, api_messages, prompt)

        content = api_messages[0]["content"]
        assert isinstance(content, list) and len(content) == 2
        assert all(part.get("cache_control") for part in content)
        assert content[0]["text"] == static
        assert "Today's date:" not in content[0]["text"]
        assert "Today's date:" not in content[1]["text"]
        # Failover sync touches only the system block — the user message
        # keeps the date exactly as composed at turn start.
        assert api_messages[1]["content"] == "hello\n\n" + _DATE_SUNDAY
        assert agent._cached_system_prompt == prompt

    def test_max_iterations_summary_gets_date(self):
        from agent.chat_completion_helpers import handle_max_iterations
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent._cached_system_prompt = "SYS"

        captured = {}

        class _Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return "RAW-RESPONSE"

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=_Completions())
        )
        transport = SimpleNamespace(
            normalize_response=lambda _r: SimpleNamespace(content="SUMMARY")
        )

        messages = [
            {
                "role": "user",
                "content": "q1",
                "api_content": "q1\n\n" + _DATE_SUNDAY,
            },
            {"role": "assistant", "content": "a1"},
        ]
        with patch.object(
            agent, "_ensure_primary_openai_client", return_value=client
        ), patch.object(agent, "_get_transport", return_value=transport), patch(
            "hermes_time.now", return_value=_MAY_17
        ):
            out = handle_max_iterations(agent, messages, 5)

        assert out == "SUMMARY"
        assert captured["messages"][0]["content"] == "SYS"
        assert "Today's date:" not in captured["messages"][0]["content"]
        sent_users = [
            m for m in captured["messages"] if m.get("role") == "user"
        ]
        assert sent_users[0]["content"] == "q1\n\n" + _DATE_SUNDAY
        for m in captured["messages"]:
            assert "api_content" not in m
        # The live history dict is never mutated.
        assert messages[0]["content"] == "q1"
        assert messages[0]["api_content"] == "q1\n\n" + _DATE_SUNDAY


class TestRestoreBoundary:
    """Cross-day restore: cached prompt + DB row stay byte-identical."""

    def test_system_prompt_never_contains_today_line(self):
        agent = _FakeAgent()
        with patch("hermes_time.now", return_value=_MAY_17), patch(
            "hermes_cli.plugins.invoke_hook", return_value=[]
        ):
            _build(agent)
        assert "Today's date:" not in agent._cached_system_prompt

        # A stored prompt with a session-start date restores verbatim.
        stored = (
            "You are Hermes Agent.\n"
            "\n"
            "Conversation started: Saturday, May 16, 2026"
        )
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        restored = MagicMock()
        restored._cached_system_prompt = None
        restored.session_id = "s1"
        restored.model = ""
        restored.provider = ""
        restored.platform = ""
        restored._session_db = db
        restored._use_prompt_caching = False
        restored._build_system_prompt = MagicMock(return_value="BUILT")
        with patch("hermes_time.now", return_value=_MAY_17):
            _restore_or_build_system_prompt(
                restored, None, [{"role": "user", "content": "hi"}]
            )
        assert restored._cached_system_prompt == stored
        assert "Today's date:" not in restored._cached_system_prompt

    def test_cached_and_db_unchanged_across_boundary(self):
        stored = (
            "You are Hermes Agent.\n"
            "\n"
            "Conversation started: Saturday, May 16, 2026\n"
            "Session ID: test-session-id\n"
            "Model: test-model\n"
            "Provider: openrouter\n"
            "Platform: cli"
        )
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = MagicMock()
        agent._cached_system_prompt = None
        agent.session_id = "test-session-id"
        agent.model = "test-model"
        agent.provider = "openrouter"
        agent.platform = "cli"
        agent._session_db = db
        agent._use_prompt_caching = False
        agent._build_system_prompt = MagicMock(return_value="BUILT_PROMPT")

        with patch("hermes_time.now", return_value=_MAY_18):
            _restore_or_build_system_prompt(
                agent, None, [{"role": "user", "content": "hi"}]
            )
            # The current turn's user-message tail uses the new date.
            assert current_date_line() == _DATE_MONDAY

        assert agent._cached_system_prompt == stored
        agent._build_system_prompt.assert_not_called()
        db.update_system_prompt.assert_not_called()
        assert "Today's date:" not in agent._cached_system_prompt
