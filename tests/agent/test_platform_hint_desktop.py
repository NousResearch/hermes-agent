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

from agent.prompt_builder import PLATFORM_HINTS, build_environment_hints
from agent.system_prompt import (
    _tui_embedded_pane_clarifier,
    build_system_prompt_parts,
)


def _stable_prompt(agent):
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
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


class TestDesktopHintEntry:
    def test_desktop_key_exists(self):
        """The map must carry a "desktop" entry — without it the platform
        hint lookup falls through to an empty string and the agent gets no
        surface framing at all on the desktop chat surface."""
        assert "desktop" in PLATFORM_HINTS

    def test_desktop_hint_disambiguates_from_terminal(self):
        """The agent must be told it is in a graphical chat surface, NOT a
        terminal. This is the line that kills the contradiction with the
        old tui mis-tag."""
        hint = PLATFORM_HINTS["desktop"]
        lowered = hint.lower()
        assert "desktop" in lowered
        assert "not a terminal" in lowered
        assert "graphical chat surface" in lowered

    def test_desktop_hint_advertises_markdown(self):
        """The desktop renderer supports full GFM (verified via the
        Streamdown pipeline in apps/desktop). The hint must steer the
        agent toward markdown, not away from it like the cli/tui hints do."""
        hint = PLATFORM_HINTS["desktop"]
        assert "markdown" in hint.lower()

    def test_desktop_hint_advertises_media_delivery(self):
        """The desktop chat intercepts MEDIA:/abs/path like telegram — images
        inline, audio/video inline players, other files as download links.
        Without this line the agent falls back to the cli/tui "state the
        path in text" model, which is the wrong UX for the desktop surface."""
        hint = PLATFORM_HINTS["desktop"]
        assert "MEDIA:" in hint

    def test_desktop_hint_advertises_inline_image_urls(self):
        hint = PLATFORM_HINTS["desktop"]
        assert "![alt](url)" in hint

    def test_desktop_hint_does_not_inherit_tui_cron_local_only_block(self):
        """The desktop chat surface's cron delivery semantics differ from
        the standalone TUI — desktop runs its own cron ticker in-process
        (hermes_cli/web_server.py under HERMES_DESKTOP=1). We deliberately
        do NOT parrot the tui "LOCAL-ONLY … no live-delivery channel" block
        into the desktop hint, since partially-correct cron guidance is
        exactly the bug class we are fixing. Cron guidance for desktop is
        deferred to a follow-up issue."""
        hint = PLATFORM_HINTS["desktop"]
        assert "LOCAL-ONLY" not in hint


class TestDesktopHintBlockRemoved:
    """The standalone desktop-hint block that used to live in
    ``build_environment_hints()`` (lines ~1130-1144) was a band-aid for the
    missing ``PLATFORM_HINTS["desktop"]`` entry. Once that entry exists, the
    block is dead code that competes with the platform hint's claim of
    what surface the agent is on. It must be gone."""

    def test_build_environment_hints_has_no_runtime_surface_line(self, monkeypatch):
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
        from agent.prompt_builder import _clear_backend_probe_cache
        _clear_backend_probe_cache()
        hints = build_environment_hints()
        assert "Runtime surface:" not in hints
        assert "desktop GUI app" not in hints

    def test_build_environment_hints_has_no_embedded_pane_clarifier(self, monkeypatch):
        """The ⌥-drag / ⌘+L embedded-pane clarifier moves to the platform-hint
        resolution site (system_prompt.py), not build_environment_hints()."""
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        monkeypatch.setenv("HERMES_DESKTOP_TERMINAL", "1")
        from agent.prompt_builder import _clear_backend_probe_cache
        _clear_backend_probe_cache()
        hints = build_environment_hints()
        assert "embedded terminal pane" not in hints
        assert "Shift-drag" not in hints


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

    def test_standalone_tui_yields_plain_tui_hint_no_clarifier(self, monkeypatch):
        monkeypatch.delenv("HERMES_DESKTOP", raising=False)
        monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
        stable = _stable_prompt(_make_agent(platform="tui"))
        assert PLATFORM_HINTS["tui"] in stable
        assert "embedded terminal pane" not in stable

    def test_embedded_tui_yields_tui_hint_with_clarifier(self, monkeypatch):
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        monkeypatch.setenv("HERMES_DESKTOP_TERMINAL", "1")
        stable = _stable_prompt(_make_agent(platform="tui"))
        assert PLATFORM_HINTS["tui"] in stable
        assert "embedded terminal pane" in stable
        assert "Shift-drag" in stable or "Option-drag" in stable or "⌥" in stable

    def test_embedded_clarifier_does_not_attach_to_desktop_platform(self, monkeypatch):
        """Critical regression: even when HERMES_DESKTOP_TERMINAL=1, a
        desktop-tagged session must NOT get the embedded-pane clarifier —
        the clarifier describes the *embedded terminal pane*, which a
        desktop chat session is not."""
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        monkeypatch.setenv("HERMES_DESKTOP_TERMINAL", "1")
        stable = _stable_prompt(_make_agent(platform="desktop"))
        assert "embedded terminal pane" not in stable


class TestEmbeddedTuiPaneClarifier:
    """When ``HERMES_DESKTOP_TERMINAL=1``, a standalone ``hermes --tui`` is
    running inside the desktop's embedded terminal pane. The user can
    ⌥-drag-select its output and ⌘/Ctrl+L to send it to the chat composer.
    That clarifier must be appended to the ``tui`` platform hint at the
    resolution site, NOT baked into the static ``PLATFORM_HINTS["tui"]``
    string (which is shared with every standalone TUI session and must
    stay byte-stable)."""

    def test_tui_standalone_hint_byte_stable_without_env(self, monkeypatch):
        """Without HERMES_DESKTOP_TERMINAL, the clarifier is a no-op and the
        resolved tui hint is exactly the static PLATFORM_HINTS["tui"]
        string. Cache-stable for every standalone TUI session."""
        monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
        out = _tui_embedded_pane_clarifier(PLATFORM_HINTS["tui"])
        assert out == PLATFORM_HINTS["tui"]

    def test_embedded_pane_clarifier_appended_when_env_set(self, monkeypatch):
        monkeypatch.setenv("HERMES_DESKTOP_TERMINAL", "1")
        out = _tui_embedded_pane_clarifier(PLATFORM_HINTS["tui"])
        assert out.startswith(PLATFORM_HINTS["tui"])
        assert "embedded terminal pane" in out
        assert "Shift-drag" in out or "Option-drag" in out or "⌥" in out

    def test_embedded_pane_clarifier_idempotent(self, monkeypatch):
        """Calling the clarifier twice must NOT double-append the sentence.
        Cache-stability: the resolver is called once per session build, so
        re-applying on an already-augmented hint is a no-op."""
        monkeypatch.setenv("HERMES_DESKTOP_TERMINAL", "1")
        once = _tui_embedded_pane_clarifier(PLATFORM_HINTS["tui"])
        twice = _tui_embedded_pane_clarifier(once)
        assert once == twice

    def test_embedded_pane_clarifier_does_not_touch_empty_hint(self, monkeypatch):
        """Defensive: if the tui hint is somehow empty (e.g. overridden to
        empty by config), do not synthesize a clarifier-only hint — that
        would put a desktop-pane reference in the prompt without the tui
        surface framing it sits under."""
        monkeypatch.setenv("HERMES_DESKTOP_TERMINAL", "1")
        out = _tui_embedded_pane_clarifier("")
        assert out == ""

    @pytest.mark.parametrize("val", ["0", "false", "no", "", "0", "False"])
    def test_falsy_env_does_not_trigger_clarifier(self, monkeypatch, val):
        monkeypatch.setenv("HERMES_DESKTOP_TERMINAL", val)
        out = _tui_embedded_pane_clarifier(PLATFORM_HINTS["tui"])
        assert out == PLATFORM_HINTS["tui"], (
            f"HERMES_DESKTOP_TERMINAL={val!r} should not trigger clarifier"
        )


class TestContradictionGone:
    """The original contradiction: a single assembled system prompt
    contained both ``You are running in the Hermes terminal UI (TUI).`` and
    ``Runtime surface: you're running inside the Hermes desktop GUI app.``.
    After the fix, no single session's prompt can carry both."""

    def test_desktop_chat_session_has_no_tui_framing(self, monkeypatch):
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
        assert "tui" in PLATFORM_HINTS
        assert "desktop" in PLATFORM_HINTS
        desktop_hint = PLATFORM_HINTS["desktop"]
        tui_hint = PLATFORM_HINTS["tui"]
        assert "terminal UI" not in desktop_hint
        assert "terminal UI" in tui_hint

    def test_tui_hint_does_not_carry_desktop_marker(self):
        tui_hint = PLATFORM_HINTS["tui"]
        assert "desktop GUI app" not in tui_hint
        assert "Runtime surface:" not in tui_hint
