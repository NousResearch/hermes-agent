"""Tests for ``gateway.agent_mention`` — parse ``@<agent>`` inline routing."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.agent_mention import parse_agent_mention
from gateway.agent_registry import AgentRegistry, reset_default_registry


@pytest.fixture
def registry(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    (root / "profiles").mkdir()
    (root / "profiles" / "coder").mkdir()
    (root / "profiles" / "data-sci").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    reset_default_registry()
    yield AgentRegistry()
    reset_default_registry()


class TestPlainMessage:
    def test_no_mention(self, registry):
        result = parse_agent_mention("hello world", registry)
        assert result.target_agent is None
        assert result.stripped_text == "hello world"

    def test_empty(self, registry):
        result = parse_agent_mention("", registry)
        assert result.target_agent is None
        assert result.stripped_text == ""

    def test_none_input(self, registry):
        result = parse_agent_mention(None, registry)  # type: ignore[arg-type]
        assert result.target_agent is None
        assert result.stripped_text == ""

    def test_email_inside_message_not_a_mention(self, registry):
        """Mid-sentence ``@`` (e.g. email addresses, handles) must not trigger routing."""
        text = "ping alice@host.com about coder please"
        result = parse_agent_mention(text, registry)
        assert result.target_agent is None
        assert result.stripped_text == text


class TestKnownMention:
    def test_routes_to_known_agent(self, registry):
        result = parse_agent_mention("@coder fix this bug", registry)
        assert result.target_agent is not None
        assert result.target_agent.name == "coder"
        assert result.stripped_text == "fix this bug"
        assert result.raw_mention == "@coder"

    def test_case_insensitive_target(self, registry):
        result = parse_agent_mention("@Coder fix this", registry)
        assert result.target_agent is not None
        assert result.target_agent.name == "coder"

    def test_strips_only_leading_whitespace(self, registry):
        result = parse_agent_mention("  @coder fix this", registry)
        assert result.target_agent is not None
        assert result.stripped_text == "fix this"

    def test_handles_hyphenated_name(self, registry):
        result = parse_agent_mention("@data-sci analyze foo.csv", registry)
        assert result.target_agent is not None
        assert result.target_agent.name == "data-sci"
        assert result.stripped_text == "analyze foo.csv"

    def test_default_agent_via_mention(self, registry):
        result = parse_agent_mention("@default hello", registry)
        assert result.target_agent is not None
        assert result.target_agent.name == "default"


class TestUnknownMention:
    def test_unknown_agent_passes_through(self, registry):
        """Mention syntax matched but target not registered — keep original text.

        Important: users still write ``@alice`` to address a person, and
        we mustn't eat that message.
        """
        text = "@alice hello"
        result = parse_agent_mention(text, registry)
        assert result.target_agent is None
        assert result.stripped_text == text

    def test_partial_match_not_a_mention(self, registry):
        # ``@`` alone, no name, no space → not a mention
        result = parse_agent_mention("@", registry)
        assert result.target_agent is None

    def test_at_with_no_separator(self, registry):
        # ``@coder`` with no following whitespace+body → not a routing
        # mention (we'd otherwise eat the entire token as an agent name)
        result = parse_agent_mention("@coder", registry)
        assert result.target_agent is None


class TestEdgeCases:
    def test_multiline_message_after_mention(self, registry):
        text = "@coder fix\nplease\nthank you"
        result = parse_agent_mention(text, registry)
        assert result.target_agent is not None
        assert result.target_agent.name == "coder"
        assert "fix" in result.stripped_text
        assert "please" in result.stripped_text

    def test_tab_separator_between_name_and_body(self, registry):
        result = parse_agent_mention("@coder\tdo it", registry)
        assert result.target_agent is not None
        assert result.stripped_text == "do it"

    def test_only_first_token_consumed(self, registry):
        # Only the first @<name> is parsed; subsequent @s in the body
        # are normal text.
        result = parse_agent_mention("@coder ping @data-sci with results", registry)
        assert result.target_agent.name == "coder"
        assert result.stripped_text == "ping @data-sci with results"
