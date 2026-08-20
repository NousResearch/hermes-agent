"""Tests for Discord clarify button rendering and resolution.

Mirrors test_telegram_clarify_buttons.py for the Discord ``send_clarify``
override and the ``ClarifyChoiceView`` callbacks. Discord uses ``discord.ui.View``
button callbacks (closures) rather than a string-prefixed callback_query
dispatcher like Telegram — the auth + resolution path is the same:

  · numeric choice → resolve_gateway_clarify(clarify_id, choice_text)
  · "Other" button → mark_awaiting_text(clarify_id) so the text-intercept
    captures the next user message in this session
  · already-resolved or unauthorized → ephemeral "this prompt..." reply
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Repo root importable
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

# Triggers the shared discord mock from tests/gateway/conftest.py before
# importing the production module.
from plugins.platforms.discord.adapter import (  # noqa: E402
    ClarifyChoiceView,
    DiscordAdapter,
)
from gateway.config import PlatformConfig  # noqa: E402
from gateway.platforms.base import utf16_len  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(*, allowed_users=None, allowed_roles=None):
    config = PlatformConfig(enabled=True, token="test-token", extra={})
    adapter = DiscordAdapter(config)
    adapter._client = MagicMock()
    adapter._allowed_user_ids = set(allowed_users or [])
    adapter._allowed_role_ids = set(allowed_roles or [])
    return adapter


def _clear_clarify_state():
    from tools import clarify_gateway as cm
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


def _make_interaction(*, user_id="42", display_name="Tester", roles=None,
                      include_message=True):
    """Build a mock discord.Interaction with response.edit_message /
    send_message / defer all coroutine-callable."""
    user = SimpleNamespace(
        id=user_id,
        display_name=display_name,
        roles=[SimpleNamespace(id=r) for r in (roles or [])],
    )
    response = SimpleNamespace(
        edit_message=AsyncMock(),
        send_message=AsyncMock(),
        defer=AsyncMock(),
    )
    if include_message:
        embed = MagicMock()
        embed.color = None
        embed.set_footer = MagicMock()
        message = SimpleNamespace(embeds=[embed])
    else:
        message = None
    return SimpleNamespace(user=user, response=response, message=message)


# ===========================================================================
# ClarifyChoiceView construction
# ===========================================================================

class TestClarifyChoiceViewConstruction:
    """The view should build numeric buttons plus an Other button."""


    def test_truncates_long_choice_label(self):
        long_choice = "x" * 200
        view = ClarifyChoiceView(
            choices=[long_choice],
            clarify_id="cidZ",
            allowed_user_ids=set(),
        )
        # 78 chars + single-char ellipsis in the body, plus "1. " prefix.
        # Uses U+2026 (…) instead of "..." to fit the 80-char Discord cap.
        first_label = view.children[0].label
        assert first_label.startswith("1. ")
        assert first_label.endswith("\u2026")
        # Final label total <= 80 (Discord cap on button labels)
        assert len(first_label) <= 80


    def test_truncates_long_no_space_choice_on_soft_boundary(self):
        # A long choice with soft boundaries (commas, hyphens) but no spaces
        # should still cut on a soft boundary, not mid-word. We use an input
        # where position 76 is NOT a soft boundary — the test only passes
        # if the renderer actively searches backward for a soft char
        # rather than blindly cutting at the budget limit.
        long_choice = "a" * 30 + "-" + "b" * 30 + "-" + "c" * 30 + "-" + "d" * 30
        # 30a-30b-30c-30d = 30 + 1 + 30 + 1 + 30 + 1 + 30 = 123 chars
        # Position 76 is 'b' (a mid-word alpha). The renderer must look back
        # for a '-' to cut on.
        view = ClarifyChoiceView(
            choices=[long_choice],
            clarify_id="cidSB",
            allowed_user_ids=set(),
        )
        first_label = view.children[0].label
        assert first_label.endswith("\u2026")
        assert len(first_label) <= 80
        body = first_label[len("1. "):].rstrip("\u2026")
        last_char = body[-1]
        assert last_char in {"-", ",", ".", ")", " "}, (
            f"Label cuts mid-word at {last_char!r}: {first_label!r}"
        )


# ===========================================================================
# Choice callback → resolve_gateway_clarify
# ===========================================================================

class TestClarifyChoiceResolve:
    """Clicking a numeric button should resolve the clarify entry."""

    def setup_method(self):
        _clear_clarify_state()


    @pytest.mark.asyncio
    async def test_unauthorized_user_rejected(self):
        from tools import clarify_gateway as cm
        cm.register("cidC", "sk-C", "Pick", ["x"])

        # Allowlist set, user not in it
        view = ClarifyChoiceView(
            choices=["x"],
            clarify_id="cidC",
            allowed_user_ids={"99999"},  # not 42
        )

        interaction = _make_interaction(user_id="42")
        await view._resolve_choice(interaction, index=0, choice="x")

        # Ephemeral rejection, no resolution, no edit
        interaction.response.send_message.assert_called_once()
        kwargs = interaction.response.send_message.call_args.kwargs
        assert kwargs.get("ephemeral") is True
        interaction.response.edit_message.assert_not_called()
        with cm._lock:
            entry = cm._entries.get("cidC")
        assert entry is not None
        assert not entry.event.is_set()


# ===========================================================================
# "Other" button → mark_awaiting_text
# ===========================================================================

class TestClarifyOtherButton:
    """Clicking Other should flip the entry into text-capture mode."""

    def setup_method(self):
        _clear_clarify_state()


    @pytest.mark.asyncio
    async def test_other_unauthorized_user_rejected(self):
        from tools import clarify_gateway as cm
        cm.register("cidE", "sk-E", "Pick", ["x"])

        view = ClarifyChoiceView(
            choices=["x"],
            clarify_id="cidE",
            allowed_user_ids={"99999"},
        )

        interaction = _make_interaction(user_id="42")
        await view._on_other(interaction)

        # Rejected; entry NOT awaiting text
        interaction.response.send_message.assert_called_once()
        pending = cm.get_pending_for_session("sk-E")
        assert pending is None or pending.awaiting_text is False


# ===========================================================================
# DiscordAdapter.send_clarify integration
# ===========================================================================

class TestDiscordSendClarify:
    """Verify send_clarify renders an embed and (optionally) attaches the view."""

    def setup_method(self):
        _clear_clarify_state()

    @pytest.mark.asyncio
    async def test_multi_choice_attaches_view(self):
        adapter = _make_adapter(allowed_users={"42"})
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 123456
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        result = await adapter.send_clarify(
            chat_id="9001",
            question="Pick a color",
            choices=["red", "green", "blue"],
            clarify_id="cidM",
            session_key="sk-M",
        )

        assert result.success is True
        assert result.message_id == "123456"
        # Verify channel.send was called with embed + view kwargs
        channel.send.assert_called_once()
        kwargs = channel.send.call_args.kwargs
        assert "embed" in kwargs
        assert "view" in kwargs
        assert isinstance(kwargs["view"], ClarifyChoiceView)
        # 3 choice buttons + 1 Other
        assert len(kwargs["view"].children) == 4

    @pytest.mark.asyncio
    async def test_open_ended_omits_view(self):
        adapter = _make_adapter()
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 222
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        result = await adapter.send_clarify(
            chat_id="9001",
            question="What is your name?",
            choices=None,
            clarify_id="cidOE",
            session_key="sk-OE",
        )

        assert result.success is True
        channel.send.assert_called_once()
        kwargs = channel.send.call_args.kwargs
        # Open-ended path renders embed but no view (text-capture handles reply)
        assert "embed" in kwargs
        assert "view" not in kwargs


    @pytest.mark.asyncio
    async def test_unwrap_does_not_pick_value_or_name_alone(self):
        # 'name' and 'value' are Discord-component-shaped fields that could
        # accidentally appear in dicts not intended as choices (e.g., a
        # developer-error in the gateway wiring). The renderer should not
        # surface them as button labels — only the well-known LLM tool-call
        # keys (label, description, text, title) should win.
        adapter = _make_adapter()
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 888
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        await adapter.send_clarify(
            chat_id="9001",
            question="?",
            choices=[
                {"name": "only_name_here"},   # should be filtered out
                {"value": "only_value_here"},  # should be filtered out
                {"description": "real choice"},
            ],
            clarify_id="cidNV",
            session_key="sk-NV",
        )
        kwargs = channel.send.call_args.kwargs
        view = kwargs["view"]
        choice_labels = [b.label for b in view.children[:-1]]  # exclude Other
        # Only the well-formed dict survives.
        assert len(choice_labels) == 1, (
            f"Expected 1 choice, got {len(choice_labels)}: {choice_labels!r}"
        )
        assert "real choice" in choice_labels[0]
        for label in choice_labels:
            assert "only_name_here" not in label, f"name leaked: {label!r}"
            assert "only_value_here" not in label, f"value leaked: {label!r}"


# ===========================================================================
# ClarifyChoiceView.on_timeout — expiry must release the agent
# ===========================================================================

def _make_view_message():
    """Mock the prompt message the view repaints on expiry."""
    embed = MagicMock()
    embed.color = None
    embed.set_footer = MagicMock()
    return SimpleNamespace(embeds=[embed], edit=AsyncMock())


def _expiry_tasks():
    """Live expiry tasks. getattr keeps the behavioural assertions below the
    real failure point when the release logic is missing entirely."""
    from plugins.platforms.discord import adapter as adapter_mod
    return list(getattr(adapter_mod, "_CLARIFY_EXPIRY_TASKS", ()))


async def _drain_expiry_tasks():
    """Await whatever ``on_timeout`` scheduled, so assertions see the result."""
    tasks = _expiry_tasks()
    for task in tasks:
        await task
    return tasks


def _cancel_expiry_tasks():
    for task in _expiry_tasks():
        task.cancel()


class TestClarifyChoiceViewTimeout:
    """An expired view must never leave the agent parked on a dead prompt.

    The view timeout only kills the buttons; the agent thread stays inside
    ``clarify_gateway.wait_for_response`` until ``agent.clarify_timeout``
    fires (an hour by default, never when set to 0). Until then the prompt
    is unanswerable: the buttons are disabled and typed prose is rejected
    by the multi-choice coercion, so every follow-up message queues behind
    the very turn it was meant to unblock.
    """

    def setup_method(self):
        _clear_clarify_state()
        _cancel_expiry_tasks()

    def teardown_method(self):
        _cancel_expiry_tasks()

    @pytest.mark.asyncio
    async def test_timeout_flips_live_entry_to_awaiting_text(self):
        from tools import clarify_gateway as cm
        cm.register("cidT1", "sk-T1", "Pick", ["x", "y"])

        view = ClarifyChoiceView(
            choices=["x", "y"], clarify_id="cidT1", allowed_user_ids={"42"},
        )
        view._message = _make_view_message()

        await view.on_timeout()

        # Entry survives the button expiry and now accepts free text.
        with cm._lock:
            entry = cm._entries.get("cidT1")
        assert entry is not None
        assert entry.awaiting_text is True
        assert not entry.event.is_set()
        assert all(b.disabled for b in view.children)
        footer = view._message.embeds[0].set_footer.call_args.kwargs["text"]
        assert "reply with a message" in footer.lower()

    @pytest.mark.asyncio
    async def test_typed_prose_answers_the_prompt_after_timeout(self):
        """Regression: prose was rejected while the buttons were dead."""
        from tools import clarify_gateway as cm
        cm.register("cidT2", "sk-T2", "What first?", ["Fix the bug", "Do the docs"])

        # Live multi-choice prompt: prose is (correctly) not an answer.
        assert cm.resolve_text_response_for_session("sk-T2", "cancel that") is False

        view = ClarifyChoiceView(
            choices=["Fix the bug", "Do the docs"],
            clarify_id="cidT2",
            allowed_user_ids={"42"},
        )
        view._message = _make_view_message()
        await view.on_timeout()

        # Buttons are gone, so the same prose must now resolve the clarify.
        assert cm.resolve_text_response_for_session("sk-T2", "cancel that") is True
        with cm._lock:
            entry = cm._entries.get("cidT2")
        assert entry.response == "cancel that"
        assert entry.event.is_set()

    @pytest.mark.asyncio
    async def test_timeout_releases_agent_when_grace_disabled(self, monkeypatch):
        from plugins.platforms.discord import adapter as adapter_mod
        from tools import clarify_gateway as cm
        monkeypatch.setattr(adapter_mod, "_read_clarify_text_grace", lambda: 0)
        cm.register("cidT3", "sk-T3", "Pick", ["x", "y"])

        view = ClarifyChoiceView(
            choices=["x", "y"], clarify_id="cidT3", allowed_user_ids={"42"},
        )
        view._message = _make_view_message()

        await view.on_timeout()
        await _drain_expiry_tasks()

        # Empty response = "user did not answer"; the waiter unblocks.
        with cm._lock:
            entry = cm._entries.get("cidT3")
        assert entry is not None
        assert entry.response == ""
        assert entry.event.is_set()

    @pytest.mark.asyncio
    async def test_grace_task_leaves_answered_prompt_alone(self, monkeypatch):
        """A reply during the grace window wins; the task must not overwrite it."""
        from plugins.platforms.discord import adapter as adapter_mod
        from tools import clarify_gateway as cm
        monkeypatch.setattr(adapter_mod, "_read_clarify_text_grace", lambda: 0)
        cm.register("cidT4", "sk-T4", "Pick", ["x", "y"])

        view = ClarifyChoiceView(
            choices=["x", "y"], clarify_id="cidT4", allowed_user_ids={"42"},
        )
        view._message = _make_view_message()
        await view.on_timeout()

        # Simulate wait_for_response returning: the waiter pops its entry.
        with cm._lock:
            entry = cm._entries.pop("cidT4")
            cm._session_index.pop("sk-T4", None)
        entry.response = "y"

        await _drain_expiry_tasks()
        assert entry.response == "y"

    @pytest.mark.asyncio
    async def test_timeout_without_entry_is_a_plain_expiry(self):
        view = ClarifyChoiceView(
            choices=["x"], clarify_id="cidGoneT", allowed_user_ids={"42"},
        )
        view._message = _make_view_message()

        await view.on_timeout()

        footer = view._message.embeds[0].set_footer.call_args.kwargs["text"]
        assert "no action taken" in footer.lower()
        assert not _expiry_tasks()

    @pytest.mark.asyncio
    async def test_timeout_survives_missing_message_reference(self):
        from tools import clarify_gateway as cm
        cm.register("cidT5", "sk-T5", "Pick", ["x"])

        view = ClarifyChoiceView(
            choices=["x"], clarify_id="cidT5", allowed_user_ids={"42"},
        )
        # No view._message (send_clarify never stored one).
        await view.on_timeout()

        with cm._lock:
            assert cm._entries.get("cidT5") is not None
