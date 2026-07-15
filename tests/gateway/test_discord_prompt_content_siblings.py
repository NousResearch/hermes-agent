"""Sibling coverage for the embed-invisibility fix (send_exec_approval got it
in the same PR): slash confirm, clarify, and update prompts must also mirror
their payload into plain message content, since embeds don't render on some
Discord clients (web/mobile)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import i18n
from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import ClarifyChoiceView, DiscordAdapter


def _capture_channel(adapter):
    sent = {}

    async def fake_send(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(id=1234)

    channel = SimpleNamespace(send=AsyncMock(side_effect=fake_send))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )
    return sent


@pytest.mark.asyncio
async def test_slash_confirm_mirrors_message_into_content():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    result = await adapter.send_slash_confirm(
        chat_id="555",
        title="Reset session?",
        message="This will clear the current conversation history.",
        session_key="discord:555",
        confirm_id="c1",
    )

    assert result.success is True
    assert sent["view"] is not None
    assert sent["embed"] is not None
    assert "Reset session?" in sent["content"]
    assert "clear the current conversation history" in sent["content"]


@pytest.mark.asyncio
async def test_slash_confirm_truncates_long_message_in_content():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    result = await adapter.send_slash_confirm(
        chat_id="555",
        title="Confirm",
        message="y" * 5000,
        session_key="discord:555",
        confirm_id="c2",
    )

    assert result.success is True
    assert len(sent["content"]) <= adapter.MAX_MESSAGE_LENGTH
    assert "... [truncated]" in sent["content"]


@pytest.mark.asyncio
async def test_clarify_with_choices_mirrors_question_into_content():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    result = await adapter.send_clarify(
        chat_id="555",
        question="Which environment should I deploy to?",
        choices=["staging", "production"],
        clarify_id="cl1",
        session_key="discord:555",
    )

    assert result.success is True
    assert sent["view"] is not None
    assert "Hermes needs your input" in sent["content"]
    assert "Which environment should I deploy to?" in sent["content"]
    assert "Pick one below" in sent["content"]


@pytest.mark.asyncio
async def test_clarify_without_choices_mirrors_question_and_reply_hint():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    result = await adapter.send_clarify(
        chat_id="555",
        question="What should the cron schedule be?",
        choices=[],
        clarify_id="cl2",
        session_key="discord:555",
    )

    assert result.success is True
    assert sent.get("view") is None
    assert "What should the cron schedule be?" in sent["content"]
    assert "Reply in this channel" in sent["content"]


@pytest.mark.asyncio
async def test_update_prompt_mirrors_prompt_into_content():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    result = await adapter.send_update_prompt(
        chat_id="555",
        prompt="Restore stashed changes?",
        default="yes",
        session_key="discord:555",
    )

    assert result.success is True
    assert sent["view"] is not None
    assert "Update Needs Your Input" in sent["content"]
    assert "Restore stashed changes?" in sent["content"]
    assert "(default: yes)" in sent["content"]


@pytest.mark.asyncio
async def test_update_prompt_native_fields_use_active_language(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "ja")
    i18n.reset_language_cache()
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    result = await adapter.send_update_prompt(
        chat_id="555",
        prompt="Restore local changes?",
        default="custom value",
        session_key="discord:555",
    )

    assert result.success is True
    assert sent["embed"].title == "⚕ 更新には入力が必要です"
    assert "Restore local changes?" in sent["embed"].description
    assert "（デフォルト: custom value）" in sent["embed"].description

    view = sent["view"]
    view._respond = AsyncMock()
    await view.yes_btn(object(), object())
    await view.no_btn(object(), object())
    assert [call.args[3] for call in view._respond.await_args_list] == [
        "はい", "いいえ",
    ]
    if view.children:
        assert [child.label for child in view.children] == ["はい", "いいえ"]


@pytest.mark.asyncio
async def test_update_prompt_callback_uses_active_language_after_click(tmp_path, monkeypatch):
    """Discord's post-click footer and duplicate response stay localized."""
    monkeypatch.setenv("HERMES_LANGUAGE", "ja")
    i18n.reset_language_cache()
    try:
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        sent = _capture_channel(adapter)
        result = await adapter.send_update_prompt(
            chat_id="555",
            prompt="Restore local changes?",
            default="custom value",
            session_key="discord:555",
        )
        assert result.success is True

        view = sent["view"]
        view.allowed_user_ids = {"u1"}
        interaction = SimpleNamespace(
            user=SimpleNamespace(id="u1", display_name="花子", roles=[]),
            message=SimpleNamespace(embeds=[sent["embed"]]),
            response=AsyncMock(),
        )

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            await view.yes_btn(interaction, object())

        assert sent["embed"].footer["text"] == "はい（回答者: 花子）"
        assert (tmp_path / ".update_response").read_text() == "y"

        duplicate = SimpleNamespace(response=AsyncMock())
        await view.no_btn(duplicate, object())
        assert duplicate.response.send_message.call_args.args[0] == (
            "このプロンプトには既に回答済みです。"
        )
    finally:
        i18n.reset_language_cache()


@pytest.mark.asyncio
async def test_clarify_callback_footer_uses_active_language(monkeypatch):
    """The Discord callback footer does not leak the English answered-by label."""
    monkeypatch.setenv("HERMES_LANGUAGE", "ja")
    i18n.reset_language_cache()
    try:
        view = ClarifyChoiceView(
            choices=["staging"],
            clarify_id="clarify-ja",
            allowed_user_ids={"u1"},
        )
        embed = MagicMock()
        interaction = SimpleNamespace(
            user=SimpleNamespace(id="u1", display_name="花子", roles=[]),
            message=SimpleNamespace(embeds=[embed]),
            response=AsyncMock(),
        )

        with patch("tools.clarify_gateway.resolve_gateway_clarify", return_value=True):
            await view._resolve_choice(interaction, 0, "staging")

        assert embed.set_footer.call_args.kwargs["text"] == "回答者: 花子（staging）"
    finally:
        i18n.reset_language_cache()
