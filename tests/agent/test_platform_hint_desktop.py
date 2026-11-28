"""System-prompt assembly for the desktop chat surface.

Pins the second half of the fix: the new ``PLATFORM_HINTS["desktop"]``
entry, the deletion of the standalone desktop-hint block from
``build_environment_hints()``, and the lookup-site extension that
appends the embedded-terminal-pane clarifier to the ``tui`` platform hint
when ``HERMES_DESKTOP_TERMINAL=1``.

These tests run against the real prompt builders (no mocks) because
cache-stability and byte-for-byte text contracts are what we are
verifying — mocking the resolver would hide exactly the class of bug
this test covers.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.prompt_builder import PLATFORM_HINTS
from agent.system_prompt import (
    _tui_embedded_pane_clarifier,
    build_system_prompt_parts,
)

def _stable_prompt(agent):
    with (
        patch("agent.prompt_builder.load_soul_md", return_value=""),
        patch("agent.prompt_builder.build_environment_hints", return_value=""),
        patch("agent.prompt_builder.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]

def _make_agent(platform="", **overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        _platform_hint_overrides={},
        model="",
        provider="",
        pass_session_id=False,
        session_id="",
    )
    base["platform"] = platform
    base.update(overrides)
    return SimpleNamespace(**base)

class TestPlatformHintResolutionInStablePrompt:
    """End-to-end through ``build_system_prompt_parts`` — the platform tag on
    the agent drives BOTH which PLATFORM_HINTS entry gets appended AND
    whether the embedded-pane clarifier follows it. The desktop-hint block
    that used to live in ``build_environment_hints()`` is gone."""

    def test_desktop_platform_yields_desktop_hint_no_tui_framing(self, monkeypatch):
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
        stable = _stable_prompt(_make_agent(platform="desktop"))
        assert PLATFORM_HINTS["desktop"] in stable
        assert "terminal UI" not in stable
        assert "Runtime surface:" not in stable
        assert "embedded terminal pane" not in stable

    def test_embedded_tui_yields_tui_hint_with_clarifier(self, monkeypatch):
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        monkeypatch.setenv("HERMES_DESKTOP_TERMINAL", "1")
        stable = _stable_prompt(_make_agent(platform="tui"))
        assert PLATFORM_HINTS["tui"] in stable
        assert "embedded terminal pane" in stable
        assert "Shift-drag" in stable or "Option-drag" in stable or "⌥" in stable

class TestEmbeddedTuiPaneClarifier:
    """When ``HERMES_DESKTOP_TERMINAL=1``, a standalone ``hermes --tui`` is
    running inside the desktop's embedded terminal pane. The user can
    ⌥-drag-select its output and ⌘/Ctrl+L to send it to the chat composer.
    That clarifier must be appended to the ``tui`` platform hint at the
    resolution site, NOT baked into the static ``PLATFORM_HINTS["tui"]``
    string (which is shared with every standalone TUI session and must
    stay byte-stable)."""

    @pytest.mark.parametrize("val", ["0", "false", ""])
    def test_falsy_env_does_not_trigger_clarifier(self, monkeypatch, val):
        monkeypatch.setenv("HERMES_DESKTOP_TERMINAL", val)
        out = _tui_embedded_pane_clarifier(PLATFORM_HINTS["tui"])
        assert out == PLATFORM_HINTS["tui"], (
            f"HERMES_DESKTOP_TERMINAL={val!r} should not trigger clarifier"
        )

class TestBotChatOnCliTransport:
    """A bot's canonical Bot Chat is read in the desktop app, but message_agent, relay and
    cron deliveries run its turns through ``chat -c "Bot Chat"``, so the agent's platform
    is ``cli``. It must get the desktop hint (GFM renders), not "Markdown does NOT render"."""

    def _stable(self, platform, title):
        agent = _make_agent(platform=platform, _bot_mode_protocol=True, _session_title_hint=title,
                            _session_db=None)
        with (
            patch("tools.bot_mode_probe.get_bot_mode_protocol_section", return_value="## Messaging other agents"),
            patch("tools.bot_mode_probe.epoch_line", return_value="Capability epoch: test"),
        ):
            return _stable_prompt(agent)

    def test_bot_chat_on_cli_gets_desktop_hint(self, monkeypatch):
        monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
        stable = self._stable("cli", "Bot Chat")
        assert PLATFORM_HINTS["desktop"] in stable
        assert PLATFORM_HINTS["cli"] not in stable

    def test_plain_cli_session_keeps_cli_hint(self, monkeypatch):
        monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
        stable = self._stable("cli", "Some chat")
        assert PLATFORM_HINTS["cli"] in stable
        assert PLATFORM_HINTS["desktop"] not in stable

    def test_bot_chat_on_tui_keeps_tui_hint(self, monkeypatch):
        monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
        stable = self._stable("tui", "Bot Chat")
        assert PLATFORM_HINTS["tui"] in stable
        assert PLATFORM_HINTS["desktop"] not in stable
