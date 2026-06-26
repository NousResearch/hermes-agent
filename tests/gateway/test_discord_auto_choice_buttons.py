"""Tests for Discord auto-detected choice buttons.

Covers the path where an ordinary agent reply already poses a question with
a short option list and the adapter appends clickable buttons — without any
clarify tool / gateway involvement:

  · ``_detect_inline_choices`` — numbered/circled/lettered/bulleted parsing,
    gated on an explicit ``? ``/``?? `` prefix so incidental lists don't
    sprout buttons; returns ``(choices, multi_select)``
  · ``AutoChoiceView`` — one button per option plus ``✏️ Other``; single-
    select click injects the option as a fresh user turn, multi-select toggles
    and ``✅ Confirm`` injects the joined answer, ``Other`` dismisses for typing
  · ``DiscordAdapter._inject_user_choice`` — builds a MessageEvent matching
    the channel/thread/user and dispatches via ``handle_message``
  · ``DiscordAdapter._maybe_send_choice_buttons`` — attaches the view only
    when 2+ choices are detected and the feature flag is on
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
    AutoChoiceView,
    DiscordAdapter,
    _detect_inline_choices,
    _fit_button_label,
)
from gateway.config import PlatformConfig  # noqa: E402
import discord  # noqa: E402  (mocked via conftest, imported after adapter)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(*, allowed_users=None, allowed_roles=None, extra=None):
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = DiscordAdapter(config)
    adapter._client = MagicMock()
    adapter._allowed_user_ids = set(allowed_users or [])
    adapter._allowed_role_ids = set(allowed_roles or [])
    return adapter


def _make_interaction(*, user_id="42", display_name="Tester", roles=None,
                      channel=None):
    user = SimpleNamespace(
        id=user_id,
        name=display_name,
        display_name=display_name,
        bot=False,
        roles=[SimpleNamespace(id=r) for r in (roles or [])],
    )
    response = SimpleNamespace(
        edit_message=AsyncMock(),
        send_message=AsyncMock(),
        defer=AsyncMock(),
    )
    followup = SimpleNamespace(send=AsyncMock())
    return SimpleNamespace(
        user=user, response=response, followup=followup, channel=channel,
    )


# ===========================================================================
# _detect_inline_choices
# ===========================================================================

class TestDetectInlineChoices:

    def test_numbered_list_with_single_prefix(self):
        content = "? 你想要哪個？\n1. 查天氣\n2. 查股價\n3. 翻譯文件"
        assert _detect_inline_choices(content) == (
            ["查天氣", "查股價", "翻譯文件"], False,
        )

    def test_multi_select_prefix(self):
        content = "?? 你喜歡哪些程式語言？\n1. Python\n2. TypeScript\n3. Rust"
        assert _detect_inline_choices(content) == (
            ["Python", "TypeScript", "Rust"], True,
        )

    def test_circled_numbers(self):
        content = "? 選一個\n① 查天氣\n② 查股價"
        assert _detect_inline_choices(content) == (["查天氣", "查股價"], False)

    def test_lettered_choices(self):
        content = "? Pick one\nA) Weather\nB) Stocks\nC) Translate"
        assert _detect_inline_choices(content) == (
            ["Weather", "Stocks", "Translate"], False,
        )

    def test_bulleted_choices_with_prefix(self):
        content = "? What next\n- Weather\n- Stocks"
        assert _detect_inline_choices(content) == (["Weather", "Stocks"], False)

    def test_no_prefix_skips_numbered(self):
        # Incidental numbered list (deploy steps) must NOT become buttons.
        content = "Deploy steps\n1. build\n2. push\n3. ship"
        assert _detect_inline_choices(content) == ([], False)

    def test_colon_intro_without_prefix_skips(self):
        # A colon-introduced guess list is purely informational — no prefix,
        # so no buttons. This is the over-eager case the rewrite kills.
        content = "推測可能是：\n1. Discord 內建 UI\n2. 其他 bot"
        assert _detect_inline_choices(content) == ([], False)

    def test_inline_or_question_without_prefix_skips(self):
        # Inline "A or B?" prose no longer triggers without a prefix.
        content = "Do you want weather or stocks or translation?"
        assert _detect_inline_choices(content) == ([], False)

    def test_numbered_preferred_over_bullets(self):
        # Both numbered and bullet lines present — numbered wins.
        content = "? Which\n1. alpha\n2. beta\n- ignored"
        assert _detect_inline_choices(content) == (["alpha", "beta"], False)

    def test_strips_markdown_emphasis(self):
        content = "? Choose\n1. **bold opt**\n2. `code opt`"
        assert _detect_inline_choices(content) == (["bold opt", "code opt"], False)

    def test_dedupes_repeated_options(self):
        content = "? Which\n1. same\n2. same\n3. other"
        assert _detect_inline_choices(content) == (["same", "other"], False)

    def test_single_choice_below_threshold(self):
        content = "? Only one\n1. just this"
        assert _detect_inline_choices(content) == ([], False)

    def test_version_numbers_not_matched(self):
        content = "? Released\nv1.2.3 is out\nv1.3.0 next"
        assert _detect_inline_choices(content) == ([], False)

    def test_empty_content(self):
        assert _detect_inline_choices("") == ([], False)
        assert _detect_inline_choices("   ") == ([], False)

    def test_caps_at_max_choices(self):
        lines = "\n".join(f"{i}. opt{i}" for i in range(1, 40))
        content = "? Pick\n" + lines
        choices, multi = _detect_inline_choices(content, max_choices=24)
        assert len(choices) == 24
        assert multi is False


# ===========================================================================
# _fit_button_label
# ===========================================================================

class TestFitButtonLabel:

    def test_short_label_unchanged(self):
        assert _fit_button_label("1. ", "apple") == "1. apple"

    def test_long_label_truncated_with_ellipsis(self):
        label = _fit_button_label("1. ", "x" * 200)
        assert len(label) <= 80
        assert label.endswith("…")

    def test_cuts_at_word_boundary(self):
        choice = "the quick brown fox jumps over the lazy dog " * 3
        label = _fit_button_label("1. ", choice)
        assert len(label) <= 80
        assert label.endswith("…")


# ===========================================================================
# AutoChoiceView construction
# ===========================================================================

class TestAutoChoiceViewConstruction:

    def test_builds_buttons_plus_other(self):
        view = AutoChoiceView(
            adapter=MagicMock(),
            choices=["查天氣", "查股價", "翻譯文件"],
            allowed_user_ids={"42"},
        )
        assert len(view.children) == 4
        labels = [b.label for b in view.children]
        assert labels[0].startswith("1. 查天氣")
        assert labels[1].startswith("2. 查股價")
        assert labels[2].startswith("3. 翻譯文件")
        assert "Other" in labels[3]
        ids = [b.custom_id for b in view.children]
        assert ids[:3] == ["autochoice:0", "autochoice:1", "autochoice:2"]
        assert ids[3] == "autochoice:other"

    def test_caps_at_24_choices_plus_other(self):
        view = AutoChoiceView(
            adapter=MagicMock(),
            choices=[f"choice-{i}" for i in range(50)],
            allowed_user_ids=set(),
        )
        assert len(view.children) == 25
        assert "Other" in view.children[-1].label

    def test_single_select_has_no_confirm_button(self):
        view = AutoChoiceView(
            adapter=MagicMock(),
            choices=["a", "b"],
            allowed_user_ids=set(),
        )
        ids = [b.custom_id for b in view.children]
        assert "autochoice:confirm" not in ids
        assert view.multi_select is False

    def test_multi_select_adds_confirm_button(self):
        view = AutoChoiceView(
            adapter=MagicMock(),
            choices=["Python", "TypeScript", "Rust"],
            allowed_user_ids={"42"},
            multi_select=True,
        )
        # 3 choices + Confirm + Other
        assert len(view.children) == 5
        ids = [b.custom_id for b in view.children]
        assert ids[3] == "autochoice:confirm"
        assert ids[4] == "autochoice:other"
        confirm = view.children[3]
        assert confirm.label == "✅ Confirm (0 selected)"


# ===========================================================================
# AutoChoiceView resolution (async)
# ===========================================================================

class TestAutoChoiceResolve:

    @pytest.mark.asyncio
    async def test_choice_click_injects_user_message(self):
        adapter = _make_adapter(allowed_users={"42"})
        adapter._inject_user_choice = AsyncMock()
        view = AutoChoiceView(
            adapter=adapter,
            choices=["查天氣", "查股價"],
            allowed_user_ids={"42"},
        )
        interaction = _make_interaction(user_id="42")
        await view._resolve_choice(interaction, 0, "查天氣")

        assert view.resolved is True
        assert all(c.disabled for c in view.children)
        adapter._inject_user_choice.assert_awaited_once_with(interaction, "查天氣")

    @pytest.mark.asyncio
    async def test_already_resolved_sends_ephemeral(self):
        adapter = _make_adapter(allowed_users={"42"})
        adapter._inject_user_choice = AsyncMock()
        view = AutoChoiceView(adapter=adapter, choices=["a", "b"],
                              allowed_user_ids={"42"})
        view.resolved = True
        interaction = _make_interaction(user_id="42")
        await view._resolve_choice(interaction, 0, "a")

        interaction.response.send_message.assert_awaited_once()
        assert interaction.response.send_message.call_args.kwargs.get("ephemeral") is True
        adapter._inject_user_choice.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unauthorized_user_rejected(self):
        adapter = _make_adapter(allowed_users={"42"})
        adapter._inject_user_choice = AsyncMock()
        view = AutoChoiceView(adapter=adapter, choices=["a", "b"],
                              allowed_user_ids={"42"})
        interaction = _make_interaction(user_id="999")  # not allowlisted
        await view._resolve_choice(interaction, 0, "a")

        interaction.response.send_message.assert_awaited_once()
        adapter._inject_user_choice.assert_not_awaited()
        assert view.resolved is False

    @pytest.mark.asyncio
    async def test_other_dismisses_without_injection(self):
        adapter = _make_adapter(allowed_users={"42"})
        adapter._inject_user_choice = AsyncMock()
        view = AutoChoiceView(adapter=adapter, choices=["a", "b"],
                              allowed_user_ids={"42"})
        interaction = _make_interaction(user_id="42")
        await view._on_other(interaction)

        assert view.resolved is True
        assert all(c.disabled for c in view.children)
        adapter._inject_user_choice.assert_not_awaited()


# ===========================================================================
# AutoChoiceView multi-select (async)
# ===========================================================================

class TestAutoChoiceMultiSelect:

    def _multi_view(self, adapter):
        return AutoChoiceView(
            adapter=adapter,
            choices=["Python", "TypeScript", "Rust"],
            allowed_user_ids={"42"},
            multi_select=True,
        )

    @pytest.mark.asyncio
    async def test_toggle_marks_selected_without_injecting(self):
        adapter = _make_adapter(allowed_users={"42"})
        adapter._inject_user_choice = AsyncMock()
        view = self._multi_view(adapter)
        interaction = _make_interaction(user_id="42")

        await view._resolve_choice(interaction, 0, "Python")

        assert view._selected == {0}
        assert view.resolved is False
        assert view._choice_buttons[0].style == discord.ButtonStyle.success
        assert view._choice_buttons[1].style == discord.ButtonStyle.primary
        assert view._confirm_btn.label == "✅ Confirm (1 selected)"
        interaction.response.edit_message.assert_awaited_once()
        adapter._inject_user_choice.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_toggle_twice_deselects(self):
        adapter = _make_adapter(allowed_users={"42"})
        adapter._inject_user_choice = AsyncMock()
        view = self._multi_view(adapter)
        interaction = _make_interaction(user_id="42")

        await view._resolve_choice(interaction, 1, "TypeScript")
        await view._resolve_choice(interaction, 1, "TypeScript")

        assert view._selected == set()
        assert view._choice_buttons[1].style == discord.ButtonStyle.primary
        assert view._confirm_btn.label == "✅ Confirm (0 selected)"

    @pytest.mark.asyncio
    async def test_confirm_injects_joined_answer(self):
        adapter = _make_adapter(allowed_users={"42"})
        adapter._inject_user_choice = AsyncMock()
        view = self._multi_view(adapter)
        interaction = _make_interaction(user_id="42")

        await view._resolve_choice(interaction, 2, "Rust")
        await view._resolve_choice(interaction, 0, "Python")
        await view._on_confirm(interaction)

        assert view.resolved is True
        assert all(c.disabled for c in view.children)
        # Joined in display order with the ideographic comma.
        adapter._inject_user_choice.assert_awaited_once_with(
            interaction, "Python、Rust",
        )

    @pytest.mark.asyncio
    async def test_confirm_with_nothing_selected_warns(self):
        adapter = _make_adapter(allowed_users={"42"})
        adapter._inject_user_choice = AsyncMock()
        view = self._multi_view(adapter)
        interaction = _make_interaction(user_id="42")

        await view._on_confirm(interaction)

        interaction.response.send_message.assert_awaited_once()
        assert interaction.response.send_message.call_args.kwargs.get("ephemeral") is True
        assert view.resolved is False
        adapter._inject_user_choice.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_confirm_unauthorized_rejected(self):
        adapter = _make_adapter(allowed_users={"42"})
        adapter._inject_user_choice = AsyncMock()
        view = self._multi_view(adapter)
        view._selected = {0}
        interaction = _make_interaction(user_id="999")

        await view._on_confirm(interaction)

        interaction.response.send_message.assert_awaited_once()
        adapter._inject_user_choice.assert_not_awaited()
        assert view.resolved is False


# ===========================================================================
# _inject_user_choice (async)
# ===========================================================================

class TestInjectUserChoice:

    @pytest.mark.asyncio
    async def test_builds_event_and_dispatches(self):
        adapter = _make_adapter(allowed_users={"42"})
        adapter.handle_message = AsyncMock()

        # Plain text channel (not a Thread / DMChannel mock subtype).
        guild = SimpleNamespace(id=999, name="Guild")
        channel = SimpleNamespace(id=123, name="general", guild=guild)
        adapter._get_effective_topic = MagicMock(return_value=None)

        interaction = _make_interaction(user_id="42", display_name="Sam",
                                        channel=channel)
        await adapter._inject_user_choice(interaction, "查天氣")

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "查天氣"
        assert event.raw_message is None
        assert event.source.chat_id == "123"
        assert event.source.user_id == "42"
        assert event.source.user_name == "Sam"

    @pytest.mark.asyncio
    async def test_missing_channel_is_noop(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        interaction = _make_interaction(channel=None)
        await adapter._inject_user_choice(interaction, "x")
        adapter.handle_message.assert_not_awaited()


# ===========================================================================
# _maybe_send_choice_buttons (async)
# ===========================================================================

class TestMaybeSendChoiceButtons:

    @pytest.mark.asyncio
    async def test_attaches_view_when_choices_detected(self):
        adapter = _make_adapter(allowed_users={"42"})
        channel = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(id=1)))
        adapter._client.get_channel = MagicMock(return_value=channel)

        await adapter._maybe_send_choice_buttons(
            chat_id="123",
            content="? 你想要哪個？\n1. 查天氣\n2. 查股價",
            reply_to=None,
            metadata=None,
        )

        channel.send.assert_awaited_once()
        view = channel.send.await_args.kwargs.get("view")
        assert isinstance(view, AutoChoiceView)
        # 2 choices + Other
        assert len(view.children) == 3
        # Embed echoes the question text (prefix stripped).
        embed = channel.send.await_args.kwargs.get("embed")
        assert "你想要哪個？" in embed.description

    @pytest.mark.asyncio
    async def test_attaches_multi_select_view(self):
        adapter = _make_adapter(allowed_users={"42"})
        channel = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(id=1)))
        adapter._client.get_channel = MagicMock(return_value=channel)

        await adapter._maybe_send_choice_buttons(
            chat_id="123",
            content="?? 你喜歡哪些？\n1. Python\n2. Rust",
            reply_to=None,
            metadata=None,
        )

        channel.send.assert_awaited_once()
        view = channel.send.await_args.kwargs.get("view")
        assert isinstance(view, AutoChoiceView)
        assert view.multi_select is True
        # 2 choices + Confirm + Other
        assert len(view.children) == 4
        embed = channel.send.await_args.kwargs.get("embed")
        assert "multi-select" in embed.title

    @pytest.mark.asyncio
    async def test_no_buttons_when_no_choices(self):
        adapter = _make_adapter()
        channel = SimpleNamespace(send=AsyncMock())
        adapter._client.get_channel = MagicMock(return_value=channel)

        await adapter._maybe_send_choice_buttons(
            chat_id="123",
            content="Just a normal reply with no options.",
            reply_to=None,
            metadata=None,
        )
        channel.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_disabled_via_config_flag(self):
        adapter = _make_adapter(extra={"auto_choice_buttons": False})
        channel = SimpleNamespace(send=AsyncMock())
        adapter._client.get_channel = MagicMock(return_value=channel)

        await adapter._maybe_send_choice_buttons(
            chat_id="123",
            content="? Which\n1. a\n2. b",
            reply_to=None,
            metadata=None,
        )
        channel.send.assert_not_awaited()
