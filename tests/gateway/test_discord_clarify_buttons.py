"""Tests for Discord clarify button rendering and resolution.

Mirrors test_telegram_clarify_buttons.py for the Discord ``send_clarify``
override and the ``ClarifyChoiceView`` callbacks. Discord uses ``discord.ui.View``
button callbacks (closures) rather than a string-prefixed callback_query
dispatcher like Telegram — the auth + resolution path is the same:

  · choice button → resolve_gateway_clarify(clarify_id, choice_text)
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


def _choices_field_value(embed) -> str:
    """Pull the 'Choices' embed field's rendered text (full option list).

    The test conftest's ``_FakeEmbed.add_field`` stores fields as plain
    dicts (``{"name": ..., "value": ..., "inline": ...}``); support both
    that shape and objects with ``.name``/``.value`` attributes in case a
    real ``discord.Embed`` is ever passed in.
    """
    for field in embed.fields:
        if isinstance(field, dict):
            if field.get("name") == "Choices":
                return field.get("value") or ""
        elif getattr(field, "name", None) == "Choices":
            return field.value
    return ""


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
    """The view should build short action buttons plus an Other button."""

    def test_renders_n_choice_buttons_plus_other(self):
        view = ClarifyChoiceView(
            choices=[
                "Approve — apply the exact visible change",
                "Revise — change nothing and request edits",
                "Show diff — display the exact patch before deciding",
            ],
            clarify_id="cidX",
            allowed_user_ids={"42"},
        )
        # 3 short action labels + 1 "Other"
        assert len(view.children) == 4
        labels = [b.label for b in view.children]
        # The number links each short button to the complete numbered row in
        # the message body. The explanatory tail never lives only on a button.
        assert labels[0] == "1 · Approve"
        assert labels[1] == "2 · Revise"
        assert labels[2] == "3 · Show diff"
        assert "Other" in labels[3]
        # Equal-significance choices follow Discord's secondary-button guidance.
        other_style = getattr(view.children[-1], "style", None)
        assert all(
            getattr(button, "style", None) == other_style
            for button in view.children[:-1]
        )
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

    def test_button_label_uses_short_preview_when_choice_has_no_separator(self):
        long_choice = "x" * 200
        view = ClarifyChoiceView(
            choices=[long_choice],
            clarify_id="cidZ",
            allowed_user_ids=set(),
        )
        first_label = view.children[0].label
        assert first_label.startswith("1 · ")
        assert first_label.endswith("…")
        assert len(first_label.encode("utf-16-le")) // 2 <= 38

    def test_button_label_short_for_emoji_choice(self):
        long_choice = "\U0001f600" * 80
        view = ClarifyChoiceView(
            choices=[long_choice],
            clarify_id="cidEmoji",
            allowed_user_ids=set(),
        )
        first_label = view.children[0].label
        assert first_label.startswith("1 · ")
        assert first_label.endswith("…")
        assert len(first_label.encode("utf-16-le")) // 2 <= 38

    def test_button_label_uses_text_before_explanation_separator(self):
        long_choice = (
            "Tight layout — well-illustrated and covers all 3 audiences "
            "(patients, families, curious general readers)"
        )
        view = ClarifyChoiceView(
            choices=[long_choice],
            clarify_id="cidW",
            allowed_user_ids=set(),
        )
        first_label = view.children[0].label
        assert first_label == "1 · Tight layout"


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
        # Short action label is linked to the complete row by its number.
        assert view.children[0].label == "1 · real-choice"
        embed = kwargs["embed"]
        assert "real-choice" in _choices_field_value(embed)

    @pytest.mark.asyncio
    async def test_unwraps_dict_choices_to_description(self):
        # LLMs sometimes emit [{"description": "..."}] instead of bare strings
        # — the renderer must unwrap common dict shapes, not str() the whole
        # dict into a Python repr anywhere user-facing (embed field text).
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
        embed = kwargs["embed"]
        option_text = _choices_field_value(embed)
        # No raw Python repr should leak into the rendered choice text.
        assert "{'" not in option_text
        assert "':" not in option_text
        # Each dict unwrapped to its inner string.
        assert "Tight, well-illustrated" in option_text
        assert "Use label key" in option_text
        assert "Use text key" in option_text
        assert "normal-string" in option_text
        # Button labels keep a compact preview of the same numbered choices.
        labels = [b.label for b in kwargs["view"].children[:-1]]
        assert labels == [
            "1 · Tight, well-illustrated",
            "2 · Use label key",
            "3 · Use text key",
            "4 · normal-string",
        ]

    @pytest.mark.asyncio
    async def test_unwrap_prefers_description_over_name_in_multi_key_dict(self):
        # When the LLM emits both 'name' (often a short identifier in
        # OpenAI-style tool calls) and 'description' (the user-facing text),
        # the renderer must surface 'description'. The user should never see
        # a 4-char model identifier in the rendered choice text.
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
        embed = kwargs["embed"]
        option_text = _choices_field_value(embed)
        assert "Tight, well-illustrated" in option_text
        # The 'name' value (a short identifier) must NOT have leaked as its
        # own standalone token (it's fine that "tight" is a substring of
        # "Tight" case-insensitively — check the literal lowercase form
        # isn't present as a separate word).
        assert "\ntight\n" not in option_text.lower()

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
        embed = kwargs["embed"]
        option_text = _choices_field_value(embed)
        assert "Short" in option_text
        # The longer description must NOT have leaked.
        assert "Long verbose" not in option_text, (
            f"'description' leaked over 'label': {option_text!r}"
        )

    @pytest.mark.asyncio
    async def test_unwrap_does_not_pick_value_or_name_alone(self):
        # 'name' and 'value' are Discord-component-shaped fields that could
        # accidentally appear in dicts not intended as choices (e.g., a
        # developer-error in the gateway wiring). The renderer should not
        # surface them in the rendered choice text — only the well-known
        # LLM tool-call keys (label, description, text, title) should win.
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
        # Only the well-formed dict survives as an actual button.
        choice_buttons = view.children[:-1]  # exclude Other
        assert len(choice_buttons) == 1, (
            f"Expected 1 choice, got {len(choice_buttons)}"
        )
        embed = kwargs["embed"]
        option_text = _choices_field_value(embed)
        assert "real choice" in option_text
        assert "only_name_here" not in option_text
        assert "only_value_here" not in option_text

    @pytest.mark.asyncio
    async def test_choices_field_value_never_exceeds_discord_embed_field_cap(self):
        # Regression test: the "Choices" embed field value is the numbered
        # option list PLUS a fixed instruction suffix ("Pick a button
        # below, or click..."). Truncating only the option list to 1024
        # chars and then appending the suffix can push the *combined*
        # value over Discord's real 1024-char embed-field cap, which
        # causes channel.send() to raise and send_clarify() to report a
        # failed send on exactly the long-choice-list case this feature
        # exists to handle.
        adapter = _make_adapter()
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 999
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        # 24 choices (the max allowed) of substantial length guarantees the
        # raw numbered option list alone is well over 1024 chars, forcing
        # the truncation path to run.
        long_choices = [f"Option number {i} with some extra descriptive text" for i in range(24)]

        result = await adapter.send_clarify(
            chat_id="9001",
            question="Pick one",
            choices=long_choices,
            clarify_id="cidCap",
            session_key="sk-Cap",
        )

        assert result.success is True
        kwargs = channel.send.call_args.kwargs
        embed = kwargs["embed"]
        option_text = _choices_field_value(embed)
        assert len(option_text) <= 1024, (
            f"Choices embed field value is {len(option_text)} chars, "
            "exceeding Discord's 1024-char embed field cap"
        )
        # The instruction suffix must still be present and intact even when
        # the option list had to be truncated to make room for it.
        assert option_text.endswith(
            "Pick a button below, or click ✏️ Other to type a custom answer."
        )
