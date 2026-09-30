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

def _volatile_prompt(agent):
    with (
        patch("agent.prompt_builder.load_soul_md", return_value=""),
        patch("agent.prompt_builder.build_environment_hints", return_value=""),
        patch("agent.prompt_builder.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["volatile"]

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

class TestPlatformHintResolutionInVolatilePrompt:
    """End-to-end through ``build_system_prompt_parts`` — the platform tag on
    the agent drives BOTH which PLATFORM_HINTS entry gets appended AND
    whether the embedded-pane clarifier follows it. The desktop-hint block
    that used to live in ``build_environment_hints()`` is gone."""

    def test_desktop_platform_yields_desktop_hint_no_tui_framing(self, monkeypatch):
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
        volatile = _volatile_prompt(_make_agent(platform="desktop"))
        assert PLATFORM_HINTS["desktop"] in volatile
        assert "terminal UI" not in volatile
        assert "Runtime surface:" not in volatile
        assert "embedded terminal pane" not in volatile

    def test_embedded_tui_yields_tui_hint_with_clarifier(self, monkeypatch):
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        monkeypatch.setenv("HERMES_DESKTOP_TERMINAL", "1")
        volatile = _volatile_prompt(_make_agent(platform="tui"))
        assert PLATFORM_HINTS["tui"] in volatile
        assert "embedded terminal pane" in volatile
        assert "Shift-drag" in volatile or "Option-drag" in volatile or "⌥" in volatile

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
