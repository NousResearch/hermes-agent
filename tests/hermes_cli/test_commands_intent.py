"""Tests for natural language prompt intent matching and keyword slash command discovery."""

from __future__ import annotations

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from hermes_cli.commands_completion import SlashCommandCompleter
from hermes_cli.commands_intent import match_prompt_intent, match_slash_keywords


def _completions(completer: SlashCommandCompleter, text: str):
    return list(
        completer.get_completions(
            Document(text=text),
            CompleteEvent(completion_requested=True),
        )
    )


class TestPromptIntentMatching:
    """Verify natural language prompts map accurately to canonical commands."""

    def test_session_reset_intents(self):
        prompts = [
            "reset chat",
            "clear conversation",
            "new session",
            "nayi chat",
            "chat reset",
            "nueva sesion",
        ]
        for p in prompts:
            matches = match_prompt_intent(p)
            assert len(matches) >= 1, f"Expected intent match for '{p}'"
            assert matches[0].command == "new"

    def test_model_switching_intents(self):
        prompts = [
            "switch model",
            "change model",
            "model badlo",
            "cambiar modelo",
        ]
        for p in prompts:
            matches = match_prompt_intent(p)
            assert len(matches) >= 1, f"Expected intent match for '{p}'"
            assert matches[0].command == "model"

    def test_cost_and_spending_intents(self):
        prompts = [
            "session cost",
            "total spend",
            "kitna kharcha hua",
            "cost kitni",
        ]
        for p in prompts:
            matches = match_prompt_intent(p)
            assert len(matches) >= 1, f"Expected intent match for '{p}'"
            assert matches[0].command == "cost"

    def test_token_usage_intents(self):
        prompts = [
            "token usage",
            "how many tokens",
            "tokens kitne",
        ]
        for p in prompts:
            matches = match_prompt_intent(p)
            assert len(matches) >= 1, f"Expected intent match for '{p}'"
            assert matches[0].command == "usage"

    def test_short_input_ignored(self):
        """Input shorter than 3 characters must not trigger intent completions."""
        assert match_prompt_intent("re") == []
        assert match_prompt_intent("a") == []
        assert match_prompt_intent("") == []

    def test_long_prompt_ignored(self):
        """Standard conversational prompts must not be treated as meta-commands."""
        long_text = "Please write a Python function that uses pandas to aggregate sales data by month and compute percentage growth."
        assert match_prompt_intent(long_text) == []

    def test_unrelated_text_ignored(self):
        assert match_prompt_intent("hello there how are you today") == []


class TestSlashKeywordMatching:
    """Verify slash search words match keywords and synonyms."""

    def test_token_matches_usage(self):
        matches = match_slash_keywords("token")
        cmds = [m[0] for m in matches]
        assert "usage" in cmds

    def test_pricing_matches_cost(self):
        matches = match_slash_keywords("pricing")
        cmds = [m[0] for m in matches]
        assert "cost" in cmds

    def test_clean_matches_new_and_clear(self):
        matches = match_slash_keywords("clean")
        cmds = [m[0] for m in matches]
        assert "new" in cmds or "clear" in cmds

    def test_keyword_ranking_exact_over_prefix_and_substring(self, monkeypatch):
        test_tags = {
            "cmd_sub": ["something_long"],
            "cmd_prefix": ["long_prefix"],
            "cmd_exact": ["long"],
        }
        monkeypatch.setattr("hermes_cli.commands_intent.KEYWORD_TAGS", test_tags)
        matches = match_slash_keywords("long")
        cmds = [m[0] for m in matches]
        assert cmds == ["cmd_exact", "cmd_prefix", "cmd_sub"]


class TestCompleterIntegration:
    """Verify SlashCommandCompleter integrates intent and keyword suggestions."""

    def test_natural_language_replaces_text(self):
        completer = SlashCommandCompleter()
        results = _completions(completer, "reset chat")
        assert len(results) >= 1
        top = results[0]
        assert top.text == "/new "
        assert top.display_text == "/new"
        assert top.start_position == -len("reset chat")

    def test_short_input_no_intent_completions(self):
        completer = SlashCommandCompleter()
        assert _completions(completer, "re") == []
        assert _completions(completer, "a") == []
        assert _completions(completer, "") == []

    def test_multilingual_hindi_intent(self):
        completer = SlashCommandCompleter()
        results = _completions(completer, "model badlo")
        assert len(results) >= 1
        assert results[0].text == "/model "

    def test_slash_keyword_discovery(self):
        completer = SlashCommandCompleter()
        results = _completions(completer, "/pricing")
        assert any(c.text == "cost" for c in results)

    def test_prefix_priority_preserved(self):
        """Prefix matches must still take highest precedence."""
        completer = SlashCommandCompleter()
        results = _completions(completer, "/he")
        texts = [c.text for c in results]
        assert "help" in texts
        assert "heartbeat" in texts
