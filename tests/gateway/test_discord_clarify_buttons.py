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

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

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

    def test_renders_n_choice_buttons_plus_other(self):
        view = ClarifyChoiceView(
            choices=["apple", "banana", "cherry"],
            clarify_id="cidX",
            allowed_user_ids={"42"},
        )
        # 3 numeric + 1 "Other"
        assert len(view.children) == 4
        labels = [b.label for b in view.children]
        assert labels[0].startswith("1. apple")
        assert labels[1].startswith("2. banana")
        assert labels[2].startswith("3. cherry")
        assert "Other" in labels[3]
        # custom_ids encode clarify_id + index/other
        ids = [b.custom_id for b in view.children]
        assert ids[0] == "clarify:cidX:0"
        assert ids[1] == "clarify:cidX:1"
        assert ids[2] == "clarify:cidX:2"
        assert ids[3] == "clarify:cidX:other"

    def test_caps_at_24_choices_plus_other(self):
        choices = [f"choice-{i}" for i in range(50)]
        view = ClarifyChoiceView(
            choices=choices,
            clarify_id="cidY",
            allowed_user_ids=set(),
        )
        # Discord limit is 25 components; we cap choices at 24 + 1 Other = 25
        assert len(view.children) == 25
        assert "Other" in view.children[-1].label

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

    def test_truncates_emoji_choice_label_by_utf16_limit(self):
        long_choice = "\U0001f600" * 80
        view = ClarifyChoiceView(
            choices=[long_choice],
            clarify_id="cidEmoji",
            allowed_user_ids=set(),
        )

        first_label = view.children[0].label
        assert first_label.startswith("1. ")
        assert first_label.endswith("\u2026")
        assert utf16_len(first_label) <= 80

    def test_truncates_long_choice_label_breaks_on_word_boundary(self):
        # Long choice with spaces — should cut at the last whole word so the
        # trailing text stays readable on Discord mobile.
        long_choice = (
            "Tight, well-illustrated, covers all 3 audiences "
            "(patients, families, curious general readers)"
        )
        view = ClarifyChoiceView(
            choices=[long_choice],
            clarify_id="cidW",
            allowed_user_ids=set(),
        )
        first_label = view.children[0].label
        assert first_label.startswith("1. ")
        assert first_label.endswith("\u2026")
        # No mid-word fragment before the ellipsis.
        assert not first_label.rstrip("\u2026").endswith("(")

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
    async def test_choice_resolves_with_canonical_choice_text(self):
        from tools import clarify_gateway as cm
        cm.register("cidA", "sk-A", "Pick", ["red", "green", "blue"])

        view = ClarifyChoiceView(
            choices=["red", "green", "blue"],
            clarify_id="cidA",
            allowed_user_ids={"42"},
        )

        interaction = _make_interaction(user_id="42")
        await view._resolve_choice(interaction, index=1, choice="green")

        # Resolved through clarify primitive
        with cm._lock:
            entry = cm._entries.get("cidA")
        assert entry is not None
        assert entry.response == "green"
        assert entry.event.is_set()
        # Buttons disabled
        assert all(b.disabled for b in view.children)
        # Embed updated + edit_message called
        interaction.response.edit_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_choice_falls_back_to_label_text_when_entry_missing(self):
        """If the gateway entry vanished (race / stale view), the button's
        own choice text is used as the response."""
        # Note: no cm.register() — entry intentionally absent

        view = ClarifyChoiceView(
            choices=["alpha"],
            clarify_id="cidGone",
            allowed_user_ids={"42"},  # matches _make_interaction's user; empty = fail-closed
        )
        interaction = _make_interaction()
        # Doesn't raise; resolve_gateway_clarify returns False quietly
        await view._resolve_choice(interaction, index=0, choice="alpha")
        # Still marks the view resolved + disables buttons
        assert view.resolved is True
        assert all(b.disabled for b in view.children)

    @pytest.mark.asyncio
    async def test_already_resolved_sends_ephemeral_reply(self):
        view = ClarifyChoiceView(
            choices=["a", "b"],
            clarify_id="cidB",
            allowed_user_ids=set(),
        )
        view.resolved = True

        interaction = _make_interaction()
        await view._resolve_choice(interaction, index=0, choice="a")

        interaction.response.send_message.assert_called_once()
        kwargs = interaction.response.send_message.call_args.kwargs
        assert kwargs.get("ephemeral") is True
        # No resolve was called
        interaction.response.edit_message.assert_not_called()

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
    async def test_other_flips_entry_to_awaiting_text(self):
        from tools import clarify_gateway as cm
        cm.register("cidD", "sk-D", "Pick", ["x", "y"])

        view = ClarifyChoiceView(
            choices=["x", "y"],
            clarify_id="cidD",
            allowed_user_ids={"42"},  # matches _make_interaction's user; empty = fail-closed
        )

        interaction = _make_interaction()
        await view._on_other(interaction)

        # Entry awaiting_text now
        pending = cm.get_pending_for_session("sk-D")
        assert pending is not None
        assert pending.clarify_id == "cidD"
        assert pending.awaiting_text is True
        # Entry still pending (not resolved)
        with cm._lock:
            entry = cm._entries.get("cidD")
        assert entry is not None
        assert not entry.event.is_set()
        # View locked + buttons disabled
        assert view.resolved is True
        assert all(b.disabled for b in view.children)
        interaction.response.edit_message.assert_called_once()

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
    async def test_routes_to_thread_when_metadata_thread_id_set(self):
        adapter = _make_adapter()
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 333
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        await adapter.send_clarify(
            chat_id="9001",
            question="?",
            choices=["a"],
            clarify_id="cidT",
            session_key="sk-T",
            metadata={"thread_id": "7777"},
        )

        # Channel lookup should resolve to thread id, not chat_id
        adapter._client.get_channel.assert_called_once_with(7777)

    @pytest.mark.asyncio
    async def test_not_connected_returns_failure(self):
        adapter = _make_adapter()
        adapter._client = None
        result = await adapter.send_clarify(
            chat_id="9001",
            question="?",
            choices=["a"],
            clarify_id="cidNC",
            session_key="sk-NC",
        )
        assert result.success is False
        assert "Not connected" in (result.error or "")

    @pytest.mark.asyncio
    async def test_filters_empty_and_whitespace_choices(self):
        adapter = _make_adapter()
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 444
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        await adapter.send_clarify(
            chat_id="9001",
            question="?",
            choices=["", "  ", "real-choice", None],
            clarify_id="cidF",
            session_key="sk-F",
        )
        kwargs = channel.send.call_args.kwargs
        view = kwargs["view"]
        # Only 1 real choice + 1 Other = 2 children
        assert len(view.children) == 2
        assert "real-choice" in view.children[0].label

    @pytest.mark.asyncio
    async def test_unwraps_dict_choices_to_description(self):
        # LLMs sometimes emit [{"description": "..."}] instead of bare strings
        # — the renderer must unwrap common dict shapes, not str() the whole
        # dict into a Python repr on the button label.
        adapter = _make_adapter()
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 555
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        malformed = [
            {"description": "Tight, well-illustrated"},
            {"label": "Use label key"},
            {"text": "Use text key"},
            "normal-string",  # strings still pass through
        ]
        await adapter.send_clarify(
            chat_id="9001",
            question="?",
            choices=malformed,
            clarify_id="cidU",
            session_key="sk-U",
        )
        kwargs = channel.send.call_args.kwargs
        view = kwargs["view"]
        labels = [b.label for b in view.children[:-1]]  # exclude Other
        # No raw Python repr should leak onto any label.
        for label in labels:
            assert "{'" not in label
            assert "':" not in label
        # Each dict unwrapped to its inner string.
        assert any("Tight, well-illustrated" in lbl for lbl in labels)
        assert any("Use label key" in lbl for lbl in labels)
        assert any("Use text key" in lbl for lbl in labels)
        assert any("normal-string" in lbl for lbl in labels)

    @pytest.mark.asyncio
    async def test_unwrap_prefers_description_over_name_in_multi_key_dict(self):
        # When the LLM emits both 'name' (often a short identifier in
        # OpenAI-style tool calls) and 'description' (the user-facing text),
        # the renderer must surface 'description'. The user should never see
        # a 4-char model identifier on a button label.
        adapter = _make_adapter()
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 666
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        await adapter.send_clarify(
            chat_id="9001",
            question="?",
            choices=[{"name": "tight", "description": "Tight, well-illustrated"}],
            clarify_id="cidN",
            session_key="sk-N",
        )
        kwargs = channel.send.call_args.kwargs
        view = kwargs["view"]
        choice_label = view.children[0].label
        assert "Tight, well-illustrated" in choice_label
        # The 'name' value (a short identifier) must NOT have leaked.
        body = choice_label.split("1. ", 1)[1].rstrip("\u2026")
        assert "tight" not in body, f"'name' leaked onto button: {choice_label!r}"

    @pytest.mark.asyncio
    async def test_unwrap_prefers_label_over_description(self):
        # When both 'label' and 'description' are present, 'label' wins.
        # 'label' is the canonical short user-facing text in most LLM tool
        # conventions; 'description' is the longer explanation.
        adapter = _make_adapter()
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 777
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        await adapter.send_clarify(
            chat_id="9001",
            question="?",
            choices=[{"label": "Short", "description": "Long verbose explanation"}],
            clarify_id="cidL",
            session_key="sk-L",
        )
        kwargs = channel.send.call_args.kwargs
        view = kwargs["view"]
        choice_label = view.children[0].label
        assert "Short" in choice_label
        # The longer description must NOT have leaked.
        assert "Long verbose" not in choice_label, (
            f"'description' leaked over 'label': {choice_label!r}"
        )

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
