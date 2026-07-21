"""
Tests for CLI vim mode prompt indicator (_get_cli_vim_mode_fragment).

Uses the HermesCLI.__new__ pattern (see test_cli_approval_ui.py) to
create a minimal CLI stub without invoking the full constructor.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from cli import HermesCLI, CLI_CONFIG


@pytest.fixture(autouse=True)
def enable_vi_mode(monkeypatch):
    """Make tests independent of the user's real display.vi_mode setting."""
    monkeypatch.setitem(CLI_CONFIG.setdefault("display", {}), "vi_mode", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cli_stub(**overrides) -> HermesCLI:
    """Return a bare HermesCLI with only the prompt-rendering attrs set."""
    obj = HermesCLI.__new__(HermesCLI)
    obj._get_tui_terminal_width = lambda: 120
    obj._use_minimal_tui_chrome = lambda width: False
    obj._voice_recording = False
    obj._voice_processing = False
    obj._sudo_state = None
    obj._secret_state = None
    obj._approval_state = None
    obj._slash_confirm_state = None
    obj._clarify_freetext = False
    obj._clarify_state = None
    obj._command_running = False
    obj._agent_running = False
    obj._voice_mode = False

    for k, v in overrides.items():
        setattr(obj, k, v)

    return obj


def _stub_app(input_mode):
    """Return a SimpleNamespace mimicking prompt_toolkit's Application."""
    return SimpleNamespace(
        vi_state=SimpleNamespace(input_mode=input_mode)
    )


# ---------------------------------------------------------------------------
# Tests: _get_cli_vim_mode_fragment
# ---------------------------------------------------------------------------

class TestCliVimModeFragment:
    """Unit tests for _get_cli_vim_mode_fragment()."""

    def test_insert_mode(self):
        """INSERT fragment when vi_state.input_mode is INSERT."""
        import prompt_toolkit.key_binding.vi_state as vi_module
        obj = _make_cli_stub(_app=_stub_app(vi_module.InputMode.INSERT))
        frags = obj._get_cli_vim_mode_fragment()
        assert frags == [("class:vim-insert", "INSERT ")]

    def test_normal_mode(self):
        """NORMAL fragment when vi_state.input_mode is NAVIGATION."""
        import prompt_toolkit.key_binding.vi_state as vi_module
        obj = _make_cli_stub(_app=_stub_app(vi_module.InputMode.NAVIGATION))
        frags = obj._get_cli_vim_mode_fragment()
        assert frags == [("class:vim-normal", "NORMAL ")]

    def test_replace_mode(self):
        """REPLACE fragment when vi_state.input_mode is REPLACE."""
        import prompt_toolkit.key_binding.vi_state as vi_module
        obj = _make_cli_stub(_app=_stub_app(vi_module.InputMode.REPLACE))
        frags = obj._get_cli_vim_mode_fragment()
        assert frags == [("class:vim-replace", "REPLACE ")]

    def test_replace_single_mode(self):
        """REPLACE1 fragment when vi_state.input_mode is REPLACE_SINGLE."""
        import prompt_toolkit.key_binding.vi_state as vi_module
        obj = _make_cli_stub(_app=_stub_app(vi_module.InputMode.REPLACE_SINGLE))
        frags = obj._get_cli_vim_mode_fragment()
        assert frags == [("class:vim-replace", "REPLACE1 ")]

    def test_unknown_mode_defaults_to_insert(self):
        """Unknown or None input_mode falls back to INSERT."""
        import prompt_toolkit.key_binding.vi_state as vi_module
        obj = _make_cli_stub(_app=_stub_app(None))
        frags = obj._get_cli_vim_mode_fragment()
        assert frags == [("class:vim-insert", "INSERT ")]

    def test_no_app_returns_insert(self):
        """No _app attribute safely returns INSERT."""
        obj = _make_cli_stub()
        # Remove _app if one was set
        if hasattr(obj, '_app'):
            del obj._app
        frags = obj._get_cli_vim_mode_fragment()
        assert frags == [("class:vim-insert", "INSERT ")]

    def test_no_vi_state_returns_insert(self):
        """_app exists but lacks vi_state safely returns INSERT."""
        obj = _make_cli_stub(_app=SimpleNamespace(vi_state=None))
        frags = obj._get_cli_vim_mode_fragment()
        assert frags == [("class:vim-insert", "INSERT ")]

    def test_vi_mode_disabled_returns_empty(self, monkeypatch):
        """Empty list when vi_mode config is False."""
        monkeypatch.setitem(
            CLI_CONFIG.setdefault("display", {}),
            "vi_mode",
            False,
        )
        obj = _make_cli_stub(_app=_stub_app(None))
        frags = obj._get_cli_vim_mode_fragment()
        assert frags == []

    def test_vi_mode_enabled_returns_indicator(self, monkeypatch):
        """Non-empty fragment when vi_mode config is True."""
        monkeypatch.setitem(
            CLI_CONFIG.setdefault("display", {}),
            "vi_mode",
            True,
        )
        import prompt_toolkit.key_binding.vi_state as vi_module
        obj = _make_cli_stub(_app=_stub_app(vi_module.InputMode.INSERT))
        frags = obj._get_cli_vim_mode_fragment()
        assert len(frags) == 1
        assert frags[0][1] == "INSERT "


# ---------------------------------------------------------------------------
# Tests: _get_tui_prompt_fragments integration
# ---------------------------------------------------------------------------

class TestCliPromptFragmentIntegration:
    """Vim fragment is correctly prepended to prompt fragments."""

    def _make_cli_stub(self, **kw):
        return _make_cli_stub(**kw)

    def test_normal_prompt_prepends_vim_indicator(self):
        """Default prompt gets vim fragment as first entry."""
        import prompt_toolkit.key_binding.vi_state as vi_module

        obj = self._make_cli_stub(
            _app=_stub_app(vi_module.InputMode.INSERT),
            _get_tui_prompt_symbols=lambda: ("❯ ", "❯ "),
        )
        obj._get_tui_prompt_symbols = lambda: ("❯ ", "❯ ")
        frags = obj._get_tui_prompt_fragments()
        assert len(frags) >= 2
        assert frags[0] == ("class:vim-insert", "INSERT ")
        assert frags[1][1] == "❯ "

    def test_voice_state_prepends_vim_indicator(self):
        """Voice-recording state still shows vim indicator."""
        import prompt_toolkit.key_binding.vi_state as vi_module

        obj = self._make_cli_stub(
            _app=_stub_app(vi_module.InputMode.NAVIGATION),
            _voice_recording=True,
        )
        frags = obj._get_tui_prompt_fragments()
        assert frags[0] == ("class:vim-normal", "NORMAL ")

    def test_prompt_text_includes_vim_label(self):
        """_get_tui_prompt_text() includes the vim mode label."""
        import prompt_toolkit.key_binding.vi_state as vi_module

        obj = self._make_cli_stub(
            _app=_stub_app(vi_module.InputMode.NAVIGATION),
        )
        text = obj._get_tui_prompt_text()
        assert "NORMAL" in text

    def test_insert_text_shows_insert(self):
        """_get_tui_prompt_text() shows INSERT in insert mode."""
        import prompt_toolkit.key_binding.vi_state as vi_module

        obj = self._make_cli_stub(
            _app=_stub_app(vi_module.InputMode.INSERT),
        )
        text = obj._get_tui_prompt_text()
        assert "INSERT" in text
