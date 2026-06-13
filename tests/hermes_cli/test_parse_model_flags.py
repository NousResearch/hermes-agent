"""Unit tests for parse_model_flags flag stripping.

The TUI model picker appends ``--tui-session`` to session-scoped switch
commands (the default when the user does not toggle ``--global``).
``parse_model_flags`` must strip this marker before model-name parsing
so it doesn't leak into ``model_input`` and trigger the "Model names
cannot contain spaces" validation error.
"""

import pytest

from hermes_cli.model_switch import parse_model_flags


class TestParseModelFlagsTuiSession:
    """--tui-session flag handling."""

    def test_tui_session_stripped_from_model_name(self):
        """The flag must not appear in the returned model_input."""
        model, provider, is_global, _ = parse_model_flags(
            "glm/glm-5.2 --provider 9router --tui-session"
        )
        assert "--tui-session" not in model
        assert model == "glm/glm-5.2"

    def test_tui_session_preserves_provider(self):
        """Provider flag must still be extracted alongside --tui-session."""
        _, provider, _, _ = parse_model_flags(
            "glm/glm-5.2 --provider custom --tui-session"
        )
        assert provider == "custom"

    def test_tui_session_is_not_global(self):
        """--tui-session explicitly means session-only (is_global=False)."""
        _, _, is_global, _ = parse_model_flags(
            "glm/glm-5.2 --provider 9router --tui-session"
        )
        assert is_global is False

    def test_tui_session_with_unicode_dash(self):
        """Unicode dash variants of --tui-session must be normalized."""
        model, _, _, _ = parse_model_flags(
            "glm/glm-5.2 \u2014tui-session"
        )
        assert "--tui-session" not in model
        assert model == "glm/glm-5.2"

    def test_bare_model_without_flags(self):
        """Baseline: bare model name with no flags."""
        model, provider, is_global, refresh = parse_model_flags("sonnet")
        assert model == "sonnet"
        assert provider == ""
        assert is_global is False
        assert refresh is False

    def test_global_flag_still_works(self):
        """--global must still set is_global=True."""
        _, _, is_global, _ = parse_model_flags("sonnet --global")
        assert is_global is True

    def test_both_tui_session_and_global(self):
        """If both flags appear, --global wins for is_global."""
        _, _, is_global, _ = parse_model_flags(
            "sonnet --global --tui-session"
        )
        assert is_global is True

    def test_refresh_flag_still_works(self):
        """--refresh must still set force_refresh=True."""
        _, _, _, refresh = parse_model_flags("--refresh")
        assert refresh is True

    def test_multi_word_model_after_stripping(self):
        """After stripping --tui-session, a clean model name must remain."""
        model, provider, is_global, _ = parse_model_flags(
            "meta-llama/Llama-3.3-70B-Instruct --provider openrouter --tui-session"
        )
        assert model == "meta-llama/Llama-3.3-70B-Instruct"
        assert provider == "openrouter"
        assert is_global is False
