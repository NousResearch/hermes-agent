"""Seam tests for the SlackReactionHooksMixin extraction (god-file slice R3-S1).

Region R3 consensus (epic #78647, target #78638): the reaction-primitives
cluster (``_add_reaction`` / ``_remove_reaction`` / ``_reactions_enabled`` /
``on_processing_start`` / ``on_processing_complete``) moved from
``plugins/platforms/slack/adapter.py`` lines 3711-3778 into
``plugins/platforms/slack/reaction_hooks_mixin.py``.

This file pins the mixin-first MRO seam (decisive: the base-class
``on_processing_start`` / ``on_processing_complete`` stubs would otherwise
silently win the override) and exercises the moved behavior through the final
``SlackAdapter`` class:

1. Identity: the hook methods resolve to the mixin module, the mixin sits at
   ``__mro__[1]`` ahead of ``BasePlatformAdapter``, and the moved methods are
   gone from ``SlackAdapter.__dict__`` (no duplicate definitions left behind).
2. Behavior: reaction add/remove dispatch through ``_get_client`` (primary and
   team-scoped), the ``SLACK_REACTIONS`` env gate, and the full
   processing-hook lifecycle (start → eyes, complete → white_check_mark/x),
   reached exactly the way the framework reaches it: by string name via
   ``_run_processing_hook``.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[3])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    ProcessingOutcome,
)
from gateway.session import SessionSource
from plugins.platforms.slack.adapter import SlackAdapter
from plugins.platforms.slack.reaction_hooks_mixin import SlackReactionHooksMixin

MOVED_METHODS = [
    "_add_reaction",
    "_remove_reaction",
    "_reactions_enabled",
    "on_processing_start",
    "on_processing_complete",
]


@pytest.fixture()
def adapter():
    """SlackAdapter instance with the Slack client mocked (mirrors
    tests/gateway/test_slack.py fixture)."""
    config = PlatformConfig(enabled=True, token="***")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.reactions_add = AsyncMock()
    a._app.client.reactions_remove = AsyncMock()
    a._bot_user_id = "U_BOT"
    return a


def _event(chat_id="C123", scope_id="T1", message_id="ts1") -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_id=message_id,
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id=chat_id,
            scope_id=scope_id,
        ),
    )


# ---------------------------------------------------------------------------
# Identity: mixin-first MRO seam
# ---------------------------------------------------------------------------


class TestMixinIdentity:
    def test_mixin_is_first_mro_base(self):
        # Mixin-first is decisive: the base stubs must never win the override.
        assert SlackAdapter.__mro__[1] is SlackReactionHooksMixin
        assert SlackAdapter.__mro__[2] is BasePlatformAdapter

    def test_hook_methods_resolve_to_mixin_module(self):
        for name in MOVED_METHODS:
            method = getattr(SlackAdapter, name)
            assert method.__module__ == (
                "plugins.platforms.slack.reaction_hooks_mixin"
            ), f"{name} did not resolve to the mixin module"

    def test_hook_methods_are_not_redefined_in_adapter_dict(self):
        # The verbatim move must not leave duplicate definitions behind.
        for name in MOVED_METHODS:
            assert name not in SlackAdapter.__dict__, (
                f"{name} still defined in SlackAdapter.__dict__"
            )

    def test_base_stub_not_shadowed_on_base_class(self):
        # The base adapter keeps its own (no-op) stubs; only the Slack
        # subclass gets the mixin override. Compare unbound functions (the
        # base class is abstract, so no instance is constructed).
        base_stub = BasePlatformAdapter.on_processing_start
        assert base_stub.__module__ == "gateway.platforms.base"
        assert SlackAdapter.on_processing_start is not base_stub
        assert SlackAdapter.on_processing_start.__module__ == (
            "plugins.platforms.slack.reaction_hooks_mixin"
        )

    def test_identity_probe_through_instance(self):
        # The exact probe from the consensus seam plan: bound-method module.
        a = SlackAdapter(PlatformConfig(enabled=True, token="***"))
        assert a.on_processing_start.__func__.__module__ == (
            "plugins.platforms.slack.reaction_hooks_mixin"
        )


# ---------------------------------------------------------------------------
# Behavior: reaction primitives through the final class
# ---------------------------------------------------------------------------


class TestReactionPrimitives:
    @pytest.mark.asyncio
    async def test_add_reaction_dispatches_to_primary_client(self, adapter):
        result = await adapter._add_reaction("C123", "ts1", "eyes")
        assert result is True
        adapter._app.client.reactions_add.assert_called_once_with(
            channel="C123", timestamp="ts1", name="eyes"
        )

    @pytest.mark.asyncio
    async def test_remove_reaction_dispatches_to_primary_client(self, adapter):
        result = await adapter._remove_reaction("C123", "ts1", "eyes")
        assert result is True
        adapter._app.client.reactions_remove.assert_called_once_with(
            channel="C123", timestamp="ts1", name="eyes"
        )

    @pytest.mark.asyncio
    async def test_add_reaction_dispatches_to_team_scoped_client(self, adapter):
        team_client = AsyncMock()
        team_client.reactions_add = AsyncMock()
        adapter._team_clients = {"T1": team_client}
        adapter._channel_team = {"C1": "T1"}

        result = await adapter._add_reaction("C1", "ts1", "eyes", team_id="T1")
        assert result is True
        team_client.reactions_add.assert_called_once_with(
            channel="C1", timestamp="ts1", name="eyes"
        )

    @pytest.mark.asyncio
    async def test_add_reaction_survives_api_error(self, adapter):
        adapter._app.client.reactions_add = AsyncMock(
            side_effect=Exception("rate limited")
        )
        result = await adapter._add_reaction("C123", "ts1", "eyes")
        assert result is False

    @pytest.mark.asyncio
    async def test_add_reaction_noops_without_app(self, adapter):
        adapter._app = None
        result = await adapter._add_reaction("C123", "ts1", "eyes")
        assert result is False

    def test_reactions_enabled_gate(self, adapter, monkeypatch):
        assert adapter._reactions_enabled() is True
        monkeypatch.setenv("SLACK_REACTIONS", "false")
        assert adapter._reactions_enabled() is False
        monkeypatch.setenv("SLACK_REACTIONS", "0")
        assert adapter._reactions_enabled() is False


# ---------------------------------------------------------------------------
# Behavior: processing-hook lifecycle via string dispatch (framework path)
# ---------------------------------------------------------------------------


class TestProcessingHooks:
    def _arm_hook(self, adapter, event):
        """Register the message for reaction lifecycle the way inbound
        routing does, and return the string-dispatched hook callable."""
        marker = adapter._workspace_message_marker(
            str(event.source.scope_id or ""), event.message_id
        )
        adapter._reacting_message_ids.add(marker)
        return marker

    @pytest.mark.asyncio
    async def test_on_processing_start_adds_eyes(self, adapter):
        event = _event()
        marker = self._arm_hook(adapter, event)

        # Framework reaches the hook by string name — same as
        # base._run_processing_hook("on_processing_start", event).
        hook = getattr(adapter, "on_processing_start")
        await hook(event)

        adapter._app.client.reactions_add.assert_awaited_once_with(
            channel="C123", timestamp="ts1", name="eyes"
        )
        assert marker in adapter._reacting_message_ids  # not yet finalized

    @pytest.mark.asyncio
    async def test_on_processing_complete_swaps_eyes_for_success(self, adapter):
        event = _event()
        self._arm_hook(adapter, event)

        hook = getattr(adapter, "on_processing_complete")
        await hook(event, ProcessingOutcome.SUCCESS)

        adapter._app.client.reactions_remove.assert_awaited_once_with(
            channel="C123", timestamp="ts1", name="eyes"
        )
        adapter._app.client.reactions_add.assert_awaited_once_with(
            channel="C123", timestamp="ts1", name="white_check_mark"
        )
        assert adapter._reacting_message_ids == set()  # marker released

    @pytest.mark.asyncio
    async def test_on_processing_complete_failure_marks_x(self, adapter):
        event = _event()
        self._arm_hook(adapter, event)

        hook = getattr(adapter, "on_processing_complete")
        await hook(event, ProcessingOutcome.FAILURE)

        adapter._app.client.reactions_remove.assert_awaited_once_with(
            channel="C123", timestamp="ts1", name="eyes"
        )
        adapter._app.client.reactions_add.assert_awaited_once_with(
            channel="C123", timestamp="ts1", name="x"
        )
        assert adapter._reacting_message_ids == set()

    @pytest.mark.asyncio
    async def test_hooks_noop_when_reactions_disabled(self, adapter, monkeypatch):
        monkeypatch.setenv("SLACK_REACTIONS", "false")
        event = _event()
        self._arm_hook(adapter, event)

        await getattr(adapter, "on_processing_start")(event)
        await getattr(adapter, "on_processing_complete")(
            event, ProcessingOutcome.SUCCESS
        )

        adapter._app.client.reactions_add.assert_not_called()
        adapter._app.client.reactions_remove.assert_not_called()

    @pytest.mark.asyncio
    async def test_hooks_noop_for_unregistered_message(self, adapter):
        # A message that was never routed for reactions must not react.
        event = _event()
        await getattr(adapter, "on_processing_start")(event)
        adapter._app.client.reactions_add.assert_not_called()


# Keep the module importable without a running event loop if pytest-asyncio
# is configured in strict mode; nothing further needed.
