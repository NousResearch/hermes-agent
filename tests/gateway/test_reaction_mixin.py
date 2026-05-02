"""Tests for the platform-agnostic DynamicReactionMixin.

Tests the configurable behavior: persona emoji, dynamic reactions toggle,
cooldown, and the full lifecycle state machine — independent of any platform.
"""

import time
from types import SimpleNamespace
from typing import Any, Dict, Hashable, Optional
from unittest.mock import AsyncMock, patch as mock_patch

import pytest

from gateway.platforms.base import ProcessingOutcome
from gateway.platforms.reaction_mixin import DynamicReactionMixin


# ── Fake adapter that implements the mixin primitives ────────────────────

class FakeAdapter(DynamicReactionMixin):
    """Minimal adapter that wires the mixin to in-memory tracking."""

    def __init__(self, extra: Optional[Dict] = None, replace_mode: bool = False):
        self.config = SimpleNamespace(extra=extra or {})
        self._reaction_replace_mode = replace_mode
        self._reactions_on = True

        # Track all calls for assertions
        self.added: list = []
        self.removed: list = []
        self.replaced: list = []

        self._init_reaction_mixin()

    def _reactions_enabled(self) -> bool:
        return self._reactions_on

    async def _reaction_add(self, msg_ref: Any, emoji: str) -> bool:
        self.added.append((msg_ref, emoji))
        return True

    async def _reaction_remove(self, msg_ref: Any, emoji: str) -> bool:
        self.removed.append((msg_ref, emoji))
        return True

    async def _reaction_set(self, msg_ref: Any, emoji: str) -> bool:
        self.replaced.append((msg_ref, emoji))
        return True

    def _reaction_resolve_message(self, event: Any) -> Any:
        return getattr(event, "msg_ref", None)

    def _reaction_msg_key(self, event: Any) -> Optional[Hashable]:
        return getattr(event, "msg_key", None)


def _make_event(msg_ref="msg1", msg_key="key1"):
    return SimpleNamespace(msg_ref=msg_ref, msg_key=msg_key)


# ── Config resolution ────────────────────────────────────────────────────


class TestPersonaEmojiResolution:
    """Persona emoji should be configurable at platform and global level."""

    def test_default_persona_emoji(self):
        """Without any config, persona emoji defaults to 👀."""
        with mock_patch("hermes_cli.config.load_config", return_value={}):
            adapter = FakeAdapter()
        assert adapter._rxn_persona_emoji == "👀"

    def test_platform_config_overrides_default(self):
        """Platform-level persona_emoji takes priority."""
        adapter = FakeAdapter(extra={"persona_emoji": "🤖"})
        assert adapter._rxn_persona_emoji == "🤖"

    def test_global_config_used_when_no_platform_override(self):
        """Global config.yaml persona_emoji is used when platform doesn't set one."""
        with mock_patch("hermes_cli.config.load_config", return_value={"persona_emoji": "🦊"}):
            adapter = FakeAdapter()
        assert adapter._rxn_persona_emoji == "🦊"

    def test_platform_overrides_global(self):
        """Platform-level wins over global config."""
        with mock_patch("hermes_cli.config.load_config", return_value={"persona_emoji": "🦊"}):
            adapter = FakeAdapter(extra={"persona_emoji": "🤖"})
        assert adapter._rxn_persona_emoji == "🤖"


class TestDynamicReactionsResolution:
    """dynamic_reactions flag should be configurable."""

    def test_enabled_by_default(self):
        """Dynamic reactions are on by default."""
        with mock_patch("hermes_cli.config.load_config", return_value={}):
            adapter = FakeAdapter()
        assert adapter._rxn_dynamic is True

    def test_disabled_via_platform_config(self):
        """Platform-level dynamic_reactions=false disables tool swapping."""
        adapter = FakeAdapter(extra={"dynamic_reactions": False})
        assert adapter._rxn_dynamic is False

    def test_disabled_via_global_config(self):
        """Global dynamic_reactions=false disables tool swapping."""
        with mock_patch("hermes_cli.config.load_config", return_value={"dynamic_reactions": False}):
            adapter = FakeAdapter()
        assert adapter._rxn_dynamic is False

    def test_disabled_when_reactions_off(self):
        """Dynamic reactions are off when base reactions are disabled."""
        adapter = FakeAdapter()
        adapter._reactions_on = False
        adapter._init_reaction_mixin()
        assert adapter._rxn_dynamic is False


class TestCooldownResolution:
    """reaction_cooldown should be configurable."""

    def test_default_cooldown(self):
        adapter = FakeAdapter()
        assert adapter._rxn_cooldown == 1.0

    def test_custom_cooldown(self):
        adapter = FakeAdapter(extra={"reaction_cooldown": 2.5})
        assert adapter._rxn_cooldown == 2.5


# ── Lifecycle: add-remove mode (Discord, Slack, Matrix) ──────────────────


class TestAddRemoveLifecycle:
    """Full lifecycle with add/remove reactions (Discord-style)."""

    @pytest.mark.asyncio
    async def test_processing_start_adds_persona_emoji(self):
        adapter = FakeAdapter(extra={"persona_emoji": "🤖"})
        event = _make_event()

        await adapter._rxn_on_processing_start(event)

        assert adapter.added == [("msg1", "🤖")]
        assert adapter._rxn_active["key1"] == "🤖"

    @pytest.mark.asyncio
    async def test_processing_start_skipped_when_disabled(self):
        adapter = FakeAdapter()
        adapter._reactions_on = False
        adapter._init_reaction_mixin()
        event = _make_event()

        await adapter._rxn_on_processing_start(event)

        assert adapter.added == []

    @pytest.mark.asyncio
    async def test_tool_call_swaps_emoji(self):
        adapter = FakeAdapter(extra={"persona_emoji": "🤖"})
        adapter._rxn_cooldown = 0  # disable cooldown for test
        event = _make_event()

        await adapter._rxn_on_processing_start(event)
        adapter.added.clear()

        await adapter._rxn_on_tool_call_start(event, "terminal")

        # Should add tool emoji and remove persona
        assert len(adapter.added) == 1
        assert adapter.added[0][0] == "msg1"
        assert len(adapter.removed) == 1
        assert adapter.removed[0] == ("msg1", "🤖")

    @pytest.mark.asyncio
    async def test_tool_call_skipped_when_dynamic_disabled(self):
        adapter = FakeAdapter(extra={"dynamic_reactions": False})
        event = _make_event()

        await adapter._rxn_on_processing_start(event)
        adapter.added.clear()

        await adapter._rxn_on_tool_call_start(event, "terminal")

        assert adapter.added == []
        assert adapter.removed == []

    @pytest.mark.asyncio
    async def test_cooldown_prevents_rapid_swaps(self):
        adapter = FakeAdapter(extra={"persona_emoji": "🤖", "reaction_cooldown": 10.0})
        event = _make_event()

        await adapter._rxn_on_processing_start(event)
        adapter.added.clear()

        # First tool call should work
        await adapter._rxn_on_tool_call_start(event, "terminal")
        assert len(adapter.added) == 1

        # Second tool call within cooldown should be skipped
        adapter.added.clear()
        await adapter._rxn_on_tool_call_start(event, "read_file")
        assert adapter.added == []

    @pytest.mark.asyncio
    async def test_success_restores_persona_emoji(self):
        adapter = FakeAdapter(extra={"persona_emoji": "🤖"})
        adapter._rxn_cooldown = 0
        event = _make_event()

        await adapter._rxn_on_processing_start(event)
        await adapter._rxn_on_tool_call_start(event, "terminal")
        adapter.added.clear()
        adapter.removed.clear()

        await adapter._rxn_on_processing_complete(event, ProcessingOutcome.SUCCESS)

        # Should swap tool emoji → persona emoji
        assert any(emoji == "🤖" for _, emoji in adapter.added)
        assert len(adapter.removed) >= 1

    @pytest.mark.asyncio
    async def test_success_no_swap_when_already_persona(self):
        """When no tool calls happened, success shouldn't remove+re-add."""
        adapter = FakeAdapter(extra={"persona_emoji": "🤖"})
        event = _make_event()

        await adapter._rxn_on_processing_start(event)
        adapter.added.clear()

        await adapter._rxn_on_processing_complete(event, ProcessingOutcome.SUCCESS)

        # Persona → persona: no swap needed
        assert adapter.added == []
        assert adapter.removed == []

    @pytest.mark.asyncio
    async def test_failure_shows_error_emoji(self):
        adapter = FakeAdapter(extra={"persona_emoji": "🤖"})
        event = _make_event()

        await adapter._rxn_on_processing_start(event)
        adapter.added.clear()
        adapter.removed.clear()

        await adapter._rxn_on_processing_complete(event, ProcessingOutcome.FAILURE)

        assert any(emoji == "❌" for _, emoji in adapter.added)
        assert any(emoji == "🤖" for _, emoji in adapter.removed)

    @pytest.mark.asyncio
    async def test_cancelled_removes_reaction(self):
        adapter = FakeAdapter(extra={"persona_emoji": "🤖"})
        event = _make_event()

        await adapter._rxn_on_processing_start(event)
        adapter.added.clear()
        adapter.removed.clear()

        await adapter._rxn_on_processing_complete(event, ProcessingOutcome.CANCELLED)

        assert ("msg1", "🤖") in adapter.removed
        assert adapter.added == []

    @pytest.mark.asyncio
    async def test_cleanup_after_complete(self):
        """Tracking dicts should be cleaned up after processing completes."""
        adapter = FakeAdapter(extra={"persona_emoji": "🤖"})
        event = _make_event()

        await adapter._rxn_on_processing_start(event)
        assert "key1" in adapter._rxn_active

        await adapter._rxn_on_processing_complete(event, ProcessingOutcome.SUCCESS)
        assert "key1" not in adapter._rxn_active
        assert "key1" not in adapter._rxn_msg_refs
        assert "key1" not in adapter._rxn_last_swap


# ── Lifecycle: replace mode (Telegram) ───────────────────────────────────


class TestReplaceLifecycle:
    """Full lifecycle with replace-all reactions (Telegram-style)."""

    @pytest.mark.asyncio
    async def test_processing_start_sets_persona(self):
        adapter = FakeAdapter(extra={"persona_emoji": "🦊"}, replace_mode=True)
        event = _make_event()

        await adapter._rxn_on_processing_start(event)

        assert adapter.replaced == [("msg1", "🦊")]
        assert adapter.added == []

    @pytest.mark.asyncio
    async def test_tool_call_replaces_emoji(self):
        adapter = FakeAdapter(extra={"persona_emoji": "🦊"}, replace_mode=True)
        adapter._rxn_cooldown = 0
        event = _make_event()

        await adapter._rxn_on_processing_start(event)
        adapter.replaced.clear()

        await adapter._rxn_on_tool_call_start(event, "terminal")

        # Replace mode: single set call, no add+remove
        assert len(adapter.replaced) == 1
        assert adapter.added == []
        assert adapter.removed == []

    @pytest.mark.asyncio
    async def test_success_replaces_with_persona(self):
        adapter = FakeAdapter(extra={"persona_emoji": "🦊"}, replace_mode=True)
        adapter._rxn_cooldown = 0
        event = _make_event()

        await adapter._rxn_on_processing_start(event)
        await adapter._rxn_on_tool_call_start(event, "terminal")
        adapter.replaced.clear()

        await adapter._rxn_on_processing_complete(event, ProcessingOutcome.SUCCESS)

        assert any(emoji == "🦊" for _, emoji in adapter.replaced)


# ── Emoji translation ────────────────────────────────────────────────────


class TestEmojiTranslation:
    """Platforms can override _reaction_translate_emoji for format conversion."""

    @pytest.mark.asyncio
    async def test_custom_translation(self):
        """Subclass emoji translation is applied to all reactions."""

        class SlackLikeAdapter(FakeAdapter):
            def _reaction_translate_emoji(self, emoji: str) -> Optional[str]:
                return {"👀": "eyes", "❌": "x"}.get(emoji, "gear")

        adapter = SlackLikeAdapter(extra={"persona_emoji": "👀"})
        event = _make_event()

        await adapter._rxn_on_processing_start(event)

        assert adapter.added == [("msg1", "eyes")]

    @pytest.mark.asyncio
    async def test_unsupported_emoji_falls_back(self):
        """When translation returns None, fallback emoji is used."""

        class LimitedAdapter(FakeAdapter):
            def _reaction_translate_emoji(self, emoji: str) -> Optional[str]:
                if emoji == "👀":
                    return "👀"
                return None  # unsupported

        adapter = LimitedAdapter(extra={"persona_emoji": "👀"})
        adapter._rxn_cooldown = 0
        event = _make_event()

        await adapter._rxn_on_processing_start(event)
        adapter.added.clear()

        await adapter._rxn_on_tool_call_start(event, "terminal")

        # Tool emoji unsupported → falls back to ⚙️ → also None → "⚙️"
        # The mixin tries translate(raw) then translate("⚙️") then "⚙️"
        assert len(adapter.added) == 1


# ── Edge cases ───────────────────────────────────────────────────────────


class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_no_msg_ref_is_noop(self):
        """Events that don't resolve to a message are silently skipped."""
        adapter = FakeAdapter()
        event = SimpleNamespace(msg_ref=None, msg_key=None)

        await adapter._rxn_on_processing_start(event)
        await adapter._rxn_on_tool_call_start(event, "terminal")
        await adapter._rxn_on_processing_complete(event, ProcessingOutcome.SUCCESS)

        assert adapter.added == []
        assert adapter.removed == []

    @pytest.mark.asyncio
    async def test_complete_without_start_is_safe(self):
        """on_processing_complete without prior start doesn't crash."""
        adapter = FakeAdapter(extra={"persona_emoji": "🤖"})
        event = _make_event()

        await adapter._rxn_on_processing_complete(event, ProcessingOutcome.SUCCESS)

        # Should add persona emoji even without prior tracking
        assert any(emoji == "🤖" for _, emoji in adapter.added)

    @pytest.mark.asyncio
    async def test_mixin_not_initialized_is_noop(self):
        """If _init_reaction_mixin was never called, all hooks are no-ops."""
        adapter = FakeAdapter.__new__(FakeAdapter)
        adapter.added = []
        adapter.removed = []
        event = _make_event()

        await adapter._rxn_on_processing_start(event)
        await adapter._rxn_on_tool_call_start(event, "terminal")
        await adapter._rxn_on_processing_complete(event, ProcessingOutcome.SUCCESS)

        assert adapter.added == []
        assert adapter.removed == []
