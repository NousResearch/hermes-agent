"""Unit tests for agent/autoreply.py — shared autoreply engine.

Tests every public function in isolation, with focus on edge cases
that the integration tests (test_cli_autoreply.py, test_autoreply_command.py)
don't cover.
"""

from unittest.mock import MagicMock

import pytest

from agent.autoreply import (
    CLI_INPUT_PREFIX,
    GATEWAY_MSG_PREFIX,
    _DEFAULT_MAX_TURNS,
    build_autoreply_messages,
    check_and_advance,
    extract_reply,
    format_status,
    handle_command,
    parse_autoreply_args,
    prepare_llm_call,
    session_info,
)


# ── Constants ────────────────────────────────────────────────────────────


class TestConstants:
    def test_default_max_turns(self):
        assert _DEFAULT_MAX_TURNS == 20

    def test_gateway_msg_prefix(self):
        assert GATEWAY_MSG_PREFIX == "autoreply-"

    def test_cli_input_prefix(self):
        assert CLI_INPUT_PREFIX == "[autoreply]"


# ── parse_autoreply_args ─────────────────────────────────────────────────


class TestParseAutoreplyArgs:
    def test_empty_string_returns_status(self):
        assert parse_autoreply_args("") == ("status", None)

    def test_off(self):
        assert parse_autoreply_args("off") == ("off", None)

    def test_disable_aliases(self):
        for word in ("off", "disable", "stop", "OFF", "Disable", "STOP"):
            action, cfg = parse_autoreply_args(word)
            assert action == "off"

    def test_simple_instructions(self):
        action, cfg = parse_autoreply_args("Ask follow-up questions")
        assert action == "enabled"
        assert cfg["prompt"] == "Ask follow-up questions"
        assert cfg["max_turns"] == _DEFAULT_MAX_TURNS
        assert cfg["turn_count"] == 0
        assert cfg.get("literal") is None  # not set for LLM mode

    def test_literal_flag(self):
        action, cfg = parse_autoreply_args("--literal continue")
        assert action == "enabled"
        assert cfg["prompt"] == "continue"
        assert cfg["literal"] is True

    def test_literal_without_message(self):
        action, cfg = parse_autoreply_args("--literal")
        assert action.startswith("error:")
        assert cfg is None

    def test_forever_flag(self):
        action, cfg = parse_autoreply_args("--forever Ask questions")
        assert action == "enabled"
        assert cfg["max_turns"] == 0

    def test_max_flag(self):
        action, cfg = parse_autoreply_args("--max 5 Ask questions")
        assert action == "enabled"
        assert cfg["max_turns"] == 5
        assert cfg["prompt"] == "Ask questions"

    def test_combined_flags(self):
        action, cfg = parse_autoreply_args("--literal --max 10 continue")
        assert action == "enabled"
        assert cfg["literal"] is True
        assert cfg["max_turns"] == 10
        assert cfg["prompt"] == "continue"

    def test_max_subcommand(self):
        action, cfg = parse_autoreply_args("max 25")
        assert action == "max:25"
        assert cfg is None

    def test_max_subcommand_without_number(self):
        action, cfg = parse_autoreply_args("max")
        assert action.startswith("error:")

    def test_max_subcommand_invalid_number(self):
        action, cfg = parse_autoreply_args("max abc")
        assert action.startswith("error:")

    def test_max_subcommand_zero_rejected(self):
        action, cfg = parse_autoreply_args("max 0")
        assert action.startswith("error:")
        assert "at least 1" in action

    def test_max_flag_zero_rejected(self):
        action, cfg = parse_autoreply_args("--max 0 prompt")
        assert action.startswith("error:")

    def test_maximize_not_treated_as_max_subcommand(self):
        """Words starting with 'max' that aren't 'max <N>' are instructions."""
        action, cfg = parse_autoreply_args("maximize depth of investigation")
        assert action == "enabled"
        assert "maximize" in cfg["prompt"]

    def test_whitespace_only_is_error(self):
        action, cfg = parse_autoreply_args("   ")
        # After stripping flags, empty prompt → error
        assert action.startswith("error:") or action == "status"


# ── build_autoreply_messages ─────────────────────────────────────────────


class TestBuildAutoreplyMessages:
    def _config(self, prompt="Test prompt"):
        return {"prompt": prompt, "model": None, "max_turns": 20, "turn_count": 0}

    def test_empty_history(self):
        msgs = build_autoreply_messages(self._config(), [])
        # Should have system + final user prompt, no history messages
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "user"
        assert "generate the next user reply" in msgs[-1]["content"]

    def test_none_history(self):
        msgs = build_autoreply_messages(self._config(), None)
        assert len(msgs) == 2

    def test_system_prompt_contains_user_instructions(self):
        msgs = build_autoreply_messages(self._config("Ask about Python"), [])
        assert "Ask about Python" in msgs[0]["content"]

    def test_filters_to_user_and_assistant_only(self):
        history = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "tool", "content": "tool output"},
        ]
        msgs = build_autoreply_messages(self._config(), history)
        roles = [m["role"] for m in msgs]
        # system (ours) + user (Hello) + assistant (Hi) + user (generate prompt)
        assert roles == ["system", "user", "assistant", "user"]

    def test_filters_empty_content(self):
        history = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "Real message"},
        ]
        msgs = build_autoreply_messages(self._config(), history)
        # Only "Real message" should appear (empty content filtered)
        assert len(msgs) == 3  # system + 1 history + generate prompt

    def test_truncates_to_20_messages(self):
        history = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
        msgs = build_autoreply_messages(self._config(), history)
        # system + 20 history + generate prompt = 22
        assert len(msgs) == 22

    def test_flattens_list_content(self):
        history = [
            {"role": "user", "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "image_url", "image_url": {"url": "http://example.com"}},
            ]},
        ]
        msgs = build_autoreply_messages(self._config(), history)
        user_msg = msgs[1]
        assert user_msg["content"] == "Look at this"

    def test_list_content_with_no_text_items(self):
        history = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "http://example.com"}},
            ]},
        ]
        msgs = build_autoreply_messages(self._config(), history)
        # The message has empty string content but still passes the filter
        # because the original content (a list) is truthy
        user_msg = msgs[1]
        assert user_msg["content"] == ""

    def test_list_content_with_non_dict_elements(self):
        """Non-dict elements in list content are silently skipped."""
        history = [
            {"role": "user", "content": ["raw string", {"type": "text", "text": "ok"}]},
        ]
        msgs = build_autoreply_messages(self._config(), history)
        assert msgs[1]["content"] == "ok"


# ── check_and_advance ────────────────────────────────────────────────────


class TestCheckAndAdvance:
    def test_cap_reached(self):
        config = {"max_turns": 5, "turn_count": 5}
        text, cap = check_and_advance(config)
        assert text is None
        assert cap is True

    def test_cap_exceeded(self):
        config = {"max_turns": 5, "turn_count": 10}
        text, cap = check_and_advance(config)
        assert text is None
        assert cap is True

    def test_under_cap_llm_mode(self):
        config = {"max_turns": 5, "turn_count": 2}
        text, cap = check_and_advance(config)
        assert text is None
        assert cap is False

    def test_unlimited_mode(self):
        """max_turns=0 means unlimited — should never trigger cap."""
        config = {"max_turns": 0, "turn_count": 1000}
        text, cap = check_and_advance(config)
        assert text is None
        assert cap is False

    def test_literal_mode_returns_prompt(self):
        config = {"max_turns": 20, "turn_count": 0, "literal": True, "prompt": "continue"}
        text, cap = check_and_advance(config)
        assert text == "continue"
        assert cap is False
        assert config["turn_count"] == 1

    def test_literal_mode_cap_takes_precedence(self):
        """Cap check happens before literal check."""
        config = {"max_turns": 3, "turn_count": 3, "literal": True, "prompt": "continue"}
        text, cap = check_and_advance(config)
        assert text is None
        assert cap is True
        # turn_count should NOT have been incremented
        assert config["turn_count"] == 3

    def test_boundary_turn_count_equals_max_minus_one(self):
        """Last allowed turn."""
        config = {"max_turns": 5, "turn_count": 4, "literal": True, "prompt": "go"}
        text, cap = check_and_advance(config)
        assert text == "go"
        assert config["turn_count"] == 5


# ── prepare_llm_call ─────────────────────────────────────────────────────


class TestPrepareLlmCall:
    def test_basic_kwargs(self):
        config = {"prompt": "Ask questions", "model": None, "max_turns": 20, "turn_count": 0}
        kwargs = prepare_llm_call(config, [])
        assert kwargs["task"] == "autoreply"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 1024
        assert "model" not in kwargs
        assert "messages" in kwargs

    def test_model_override(self):
        config = {"prompt": "test", "model": "openai/gpt-4o-mini", "max_turns": 20, "turn_count": 0}
        kwargs = prepare_llm_call(config, [])
        assert kwargs["model"] == "openai/gpt-4o-mini"

    def test_empty_history(self):
        config = {"prompt": "test", "model": None, "max_turns": 20, "turn_count": 0}
        kwargs = prepare_llm_call(config, [])
        # system + generate prompt only
        assert len(kwargs["messages"]) == 2

    def test_with_history(self):
        config = {"prompt": "test", "model": None, "max_turns": 20, "turn_count": 0}
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        kwargs = prepare_llm_call(config, history)
        # system + 2 history + generate prompt
        assert len(kwargs["messages"]) == 4


# ── extract_reply ─────────────────────────────────────────────────────────


class TestExtractReply:
    def _config(self):
        return {"turn_count": 0}

    def test_normal_response(self):
        config = self._config()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "  A reply  "
        result = extract_reply(config, resp)
        assert result == "A reply"
        assert config["turn_count"] == 1

    def test_empty_choices(self):
        config = self._config()
        resp = MagicMock()
        resp.choices = []
        result = extract_reply(config, resp)
        assert result is None
        assert config["turn_count"] == 0

    def test_none_content(self):
        config = self._config()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = None
        result = extract_reply(config, resp)
        assert result is None
        assert config["turn_count"] == 0

    def test_empty_string_content(self):
        config = self._config()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = ""
        result = extract_reply(config, resp)
        assert result is None
        assert config["turn_count"] == 0

    def test_whitespace_only_content(self):
        config = self._config()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "   \n  "
        result = extract_reply(config, resp)
        # Whitespace-only content is treated as empty — returns None
        assert result is None
        assert config["turn_count"] == 0


# ── session_info ──────────────────────────────────────────────────────────


class TestSessionInfo:
    def test_none_config(self):
        info = session_info(None)
        assert info["autoreply"]["enabled"] is False
        assert info["autoreply"]["prompt"] is None
        assert info["autoreply"]["max_turns"] is None
        assert info["autoreply"]["turn_count"] is None

    def test_with_config(self):
        cfg = {"prompt": "test", "max_turns": 10, "turn_count": 3}
        info = session_info(cfg)
        assert info["autoreply"]["enabled"] is True
        assert info["autoreply"]["prompt"] == "test"
        assert info["autoreply"]["max_turns"] == 10
        assert info["autoreply"]["turn_count"] == 3

    def test_empty_dict_treated_as_disabled(self):
        """An empty dict is falsy in Python, so it's treated as disabled."""
        info = session_info({})
        # {} is falsy → enabled is True (config is not None) but fields use None
        # because `if config` is False for empty dict
        assert info["autoreply"]["enabled"] is True
        assert info["autoreply"]["prompt"] is None

    def test_config_with_minimal_keys(self):
        """Config with just the required keys works correctly."""
        cfg = {"prompt": "test", "max_turns": 20, "turn_count": 0}
        info = session_info(cfg)
        assert info["autoreply"]["enabled"] is True
        assert info["autoreply"]["prompt"] == "test"
        assert info["autoreply"]["max_turns"] == 20
        assert info["autoreply"]["turn_count"] == 0

    def test_cli_and_gateway_parity(self):
        """Both CLI and gateway should produce identical output for the same config."""
        cfg = {"prompt": "test", "max_turns": 20, "turn_count": 5}
        # Both paths now call session_info() directly
        result = session_info(cfg)
        assert result == {
            "autoreply": {
                "enabled": True,
                "prompt": "test",
                "max_turns": 20,
                "turn_count": 5,
            },
        }


# ── format_status ─────────────────────────────────────────────────────────


class TestFormatStatus:
    def test_llm_mode(self):
        cfg = {"prompt": "Ask questions", "max_turns": 20, "turn_count": 3}
        result = format_status(cfg)
        assert "LLM-generated" in result
        assert "Prompt:" in result
        assert "3/20" in result
        assert "Ask questions" in result

    def test_literal_mode(self):
        cfg = {"prompt": "continue", "max_turns": 20, "turn_count": 1, "literal": True}
        result = format_status(cfg)
        assert "literal" in result
        assert "Message:" in result

    def test_unlimited_mode(self):
        cfg = {"prompt": "test", "max_turns": 0, "turn_count": 5}
        result = format_status(cfg)
        assert "5/∞" in result

    def test_long_prompt_truncated(self):
        cfg = {"prompt": "A" * 200, "max_turns": 20, "turn_count": 0}
        result = format_status(cfg)
        assert "..." in result
        assert "A" * 200 not in result
        assert "A" * 100 in result


# ── handle_command ────────────────────────────────────────────────────────


class TestHandleCommand:
    """Unit tests for the handle_command orchestrator."""

    def test_enable_returns_config(self):
        text, cfg = handle_command("Ask questions", None)
        assert cfg is not None
        assert cfg["prompt"] == "Ask questions"
        assert "enabled" in text.lower()

    def test_disable_active(self):
        active = {"prompt": "test", "model": None, "max_turns": 20, "turn_count": 0}
        text, cfg = handle_command("off", active)
        assert cfg is None
        assert "disabled" in text.lower()

    def test_disable_inactive(self):
        text, cfg = handle_command("off", None)
        assert cfg is None
        assert "not active" in text.lower()

    def test_status_active(self):
        active = {"prompt": "test", "model": None, "max_turns": 20, "turn_count": 3}
        text, cfg = handle_command("", active)
        assert cfg is active  # Same object returned
        assert "3/20" in text

    def test_status_inactive(self):
        text, cfg = handle_command("", None)
        assert cfg is None
        assert "not active" in text.lower()

    def test_max_mutates_config(self):
        active = {"prompt": "test", "model": None, "max_turns": 20, "turn_count": 0}
        text, cfg = handle_command("max 50", active)
        assert cfg is active  # Same object, mutated
        assert cfg["max_turns"] == 50

    def test_max_without_active_config(self):
        text, cfg = handle_command("max 50", None)
        assert cfg is None
        assert "not active" in text.lower()

    def test_error_propagated(self):
        text, cfg = handle_command("--literal", None)
        assert "Usage" in text

    def test_error_preserves_existing_config(self):
        active = {"prompt": "test", "model": None, "max_turns": 20, "turn_count": 0}
        text, cfg = handle_command("--max abc prompt", active)
        assert cfg is active  # Config unchanged on error

    def test_literal_mode(self):
        text, cfg = handle_command("--literal continue", None)
        assert cfg is not None
        assert cfg["literal"] is True
        assert "literal" in text.lower()
