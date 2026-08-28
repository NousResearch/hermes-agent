"""Empty-transcript sanitizer: stop the per-send heal/WARNING loop (#96870).

``repair_empty_non_final_messages`` is still the single owner of the
wire-copy substitution. The send-time projection now fills empty non-final
turns first so the owner heals 0 on the main loop (same pattern as #88955).
When a caller still hits the owner repeatedly, logging escalates once per
session window instead of flooding errors.log.
"""

from __future__ import annotations

import logging

import pytest

from agent.agent_runtime_helpers import (
    _INTERRUPTED_PLACEHOLDER,
    _empty_heal_log_state,
    fill_empty_non_final_wire_payload,
    repair_empty_non_final_messages,
)
from hermes_logging import clear_session_context, set_session_context


@pytest.fixture(autouse=True)
def _reset_heal_log():
    _empty_heal_log_state.clear()
    clear_session_context()
    yield
    _empty_heal_log_state.clear()
    clear_session_context()


def _poisoned_rows():
    return [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "next"},
    ]


class TestFillEmptyNonFinalWirePayload:
    def test_fills_empty_non_final_assistant(self):
        msg = {"role": "assistant", "content": ""}
        assert fill_empty_non_final_wire_payload(msg, is_final=False) is True
        assert msg["content"] == _INTERRUPTED_PLACEHOLDER

    def test_fills_empty_non_final_user(self):
        msg = {"role": "user", "content": None}
        assert fill_empty_non_final_wire_payload(msg, is_final=False) is True
        assert msg["content"] == _INTERRUPTED_PLACEHOLDER

    def test_skips_final_turn(self):
        msg = {"role": "assistant", "content": ""}
        assert fill_empty_non_final_wire_payload(msg, is_final=True) is False
        assert msg["content"] == ""

    def test_skips_tool_call_turn(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "x", "arguments": "{}"}}],
        }
        assert fill_empty_non_final_wire_payload(msg, is_final=False) is False
        assert msg["content"] == ""

    def test_skips_codex_commentary_carrier(self):
        msg = {
            "role": "assistant",
            "content": "",
            "codex_message_items": [{"type": "text", "text": "hi"}],
        }
        assert fill_empty_non_final_wire_payload(msg, is_final=False) is False
        assert msg["content"] == ""

    def test_skips_already_populated_content(self):
        msg = {"role": "assistant", "content": "hello"}
        assert fill_empty_non_final_wire_payload(msg, is_final=False) is False
        assert msg["content"] == "hello"


class TestHealLogEscalation:
    def test_warning_then_one_error_then_silence(self, monkeypatch, caplog):
        import agent.agent_runtime_helpers as arh

        monkeypatch.setattr(arh, "_EMPTY_HEAL_ESCALATE_AFTER", 3)
        set_session_context("sess-heal")
        durable = _poisoned_rows()

        with caplog.at_level(logging.WARNING, logger="run_agent"):
            for _ in range(5):
                out = repair_empty_non_final_messages(
                    [dict(m) for m in durable]
                )
                assert out[1]["content"] == _INTERRUPTED_PLACEHOLDER
                assert durable[1]["content"] == ""

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(warnings) == 2
        assert len(errors) == 1
        assert "healed" in warnings[0].getMessage()
        assert "session window" in errors[0].getMessage()
        assert "/new" in errors[0].getMessage()

    def test_sessions_do_not_share_heal_counters(self, monkeypatch, caplog):
        import agent.agent_runtime_helpers as arh

        monkeypatch.setattr(arh, "_EMPTY_HEAL_ESCALATE_AFTER", 3)
        with caplog.at_level(logging.WARNING, logger="run_agent"):
            set_session_context("sess-a")
            repair_empty_non_final_messages([dict(m) for m in _poisoned_rows()])
            set_session_context("sess-b")
            repair_empty_non_final_messages([dict(m) for m in _poisoned_rows()])

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(warnings) == 2
        assert errors == []

    def test_owner_still_heals_wire_copy_only(self):
        durable = _poisoned_rows()
        out = repair_empty_non_final_messages(durable)
        assert out[1]["content"] == _INTERRUPTED_PLACEHOLDER
        assert durable[1]["content"] == ""
        assert out is not durable


class TestProjectionStopsReheal:
    def _loop_agent(self):
        from unittest.mock import MagicMock, patch

        from run_agent import AIAgent

        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        return agent

    def test_unmarked_empty_assistant_is_filled_before_sanitizer(self):
        """The per-turn WARNING spam IS the bug: projection must fill the
        unmarked empty row so the sanitizer heals 0 (#96870 / #88955)."""
        from unittest.mock import patch

        import agent.agent_runtime_helpers as _arh
        from tests.run_agent.test_run_agent import _mock_response

        agent = self._loop_agent()
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="ok", finish_reason="stop"),
        ]
        sanitizer_healed = []
        _real_repair = _arh.repair_empty_non_final_messages

        def _spy_repair(messages, *a, **k):
            empty = [
                (m.get("role"), m.get("content"))
                for m in messages
                if isinstance(m, dict)
                and m.get("role") in ("user", "assistant")
                and not (m.get("content") or "").strip()
                and not m.get("tool_calls")
            ]
            sanitizer_healed.append(empty)
            return _real_repair(messages, *a, **k)

        history = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "continue", "finish_reason": "stop"},
            {"role": "assistant", "content": "earlier reply", "finish_reason": "stop"},
        ]

        with (
            patch.object(agent, "_flush_messages_to_session_db"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(
                _arh, "repair_empty_non_final_messages", side_effect=_spy_repair
            ),
        ):
            agent.run_conversation("next question", conversation_history=history)

        assert sanitizer_healed, "sanitizer was never invoked — test is vacuous"
        for empty in sanitizer_healed:
            assert empty == [], (
                "sanitizer still received an empty non-final row — "
                "the re-heal loop is back (#96870)"
            )

        wire = agent.client.chat.completions.create.call_args.kwargs["messages"]
        wire_assistants = [m for m in wire if m.get("role") == "assistant"]
        assert wire_assistants[0]["content"] == _INTERRUPTED_PLACEHOLDER
        assert history[1]["content"] == ""
        assert "api_content" not in history[1]
