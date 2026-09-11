"""Telegram rendering and authorization tests for the `/wisdom` controller."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.wisdom_command import (
    WisdomAction,
    WisdomCommandContext,
    WisdomItem,
    WisdomView,
)
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


@pytest.mark.asyncio
async def test_mediation_rich_digest_preserves_each_complete_assessment():
    from hermes_wisdom.mediation_view import advice_view
    from plugins.platforms.slack.wisdom_blocks import (
        render_wisdom_blocks,
        wisdom_fallback_text,
    )

    items = [
        {
            "advice": {
                "title": f"Skill {index}",
                "relevance": "digest",
                "explanation": "Existing workflow overlap. " * 15
                + f"Final assessment {index}: no installation needed.",
            }
        }
        for index in range(3)
    ]
    view = advice_view(items)
    adapter = _adapter()
    adapter._bot.do_api_request = AsyncMock(
        return_value={"message_id": 123, "chat": {"id": 42}}
    )
    await adapter.send_wisdom_mediation(
        view, source=SimpleNamespace(chat_id="42", thread_id=None)
    )
    html = adapter._bot.do_api_request.call_args.kwargs["api_kwargs"]["rich_message"][
        "html"
    ]
    slack = str(render_wisdom_blocks(view))
    for index in range(3):
        ending = f"Final assessment {index}: no installation needed."
        assert ending in html
        assert ending in slack
        assert ending in wisdom_fallback_text(view)
    assert len(html) <= 4096
    assert "Team skill activity" in html
    assert html.count("Hermes assessment:") == 3
    assert "Also received" not in html


def test_mediation_full_details_still_escape_untrusted_content():
    view = WisdomView(
        title="Collective Wisdom",
        items=[WisdomItem(title="<skill>", detail="<script>untrusted & text</script>")],
    )
    html = TelegramAdapter._wisdom_command_html(view, full_details=True)
    assert "<script>" not in html
    assert "&lt;script&gt;untrusted &amp; text&lt;/script&gt;" in html


@pytest.mark.asyncio
async def test_mediation_oversized_html_uses_text_without_dropping_items():
    from hermes_wisdom.mediation_view import advice_view

    view = advice_view([
        {
            "advice": {
                "title": f"Skill {index}",
                "relevance": "digest",
                "explanation": "&" * 500 + f" final assessment {index}",
            }
        }
        for index in range(3)
    ])
    adapter = _adapter()
    adapter._send_message_with_thread_fallback = AsyncMock(
        return_value={"message_id": 124, "chat": {"id": 42}}
    )
    await adapter.send_wisdom_mediation(
        view, source=SimpleNamespace(chat_id="42", thread_id=None)
    )
    adapter._bot.do_api_request.assert_not_called()
    text = adapter._send_message_with_thread_fallback.call_args.kwargs["text"]
    for index in range(3):
        assert f"final assessment {index}" in text


class _Runner:
    def __init__(self, slash_error: str | None = None) -> None:
        self.slash_error = slash_error
        self.checked: list[str] = []

    async def _handle_message(self, _event) -> None:
        return None

    def _is_user_authorized(self, _source) -> bool:
        return True

    def _check_slash_access(self, _source, command: str) -> str | None:
        self.checked.append(command)
        return self.slash_error


def test_wisdom_callback_obeys_command_allowlist():
    adapter = _adapter()
    runner = _Runner(slash_error="disabled")
    adapter._message_handler = runner._handle_message

    assert not adapter._is_callback_user_authorized(
        "42",
        chat_id="42",
        chat_type="dm",
        command="wisdom",
    )
    assert runner.checked == ["wisdom"]


def test_wisdom_callback_fails_closed_when_slash_policy_check_errors():
    adapter = _adapter()
    runner = _Runner()
    runner._check_slash_access = MagicMock(
        side_effect=RuntimeError("policy unavailable")
    )
    adapter._message_handler = runner._handle_message

    assert not adapter._is_callback_user_authorized(
        "42",
        chat_id="42",
        chat_type="dm",
        command="wisdom",
    )


def test_wisdom_rich_card_escapes_untrusted_text_and_embeds_controls():
    view = WisdomView(
        "<script>Collective Wisdom</script>",
        "A" * 1200,
        items=[
            WisdomItem(
                "<b>untrusted</b>",
                "details <tag> " + "B" * 600,
                actions=[
                    WisdomAction(
                        "Install",
                        callback_data="wi:cmd:abc",
                        primary=True,
                    ),
                    WisdomAction(
                        "View",
                        url="https://portal.example/skill?a=1&b=2",
                    ),
                ],
            )
        ],
    )

    rendered = TelegramAdapter._wisdom_command_html(view)

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;b&gt;untrusted&lt;/b&gt;" in rendered
    assert 'type="callback_data"' in rendered
    assert 'data="wi:cmd:abc"' in rendered
    assert 'type="url"' in rendered
    assert "&amp;b=2" in rendered
    assert '<tg-button-row align="left">' in rendered
    assert "</p><tg-button-row" in rendered
    assert "<p><tg-button" not in rendered
    assert "<br/><tg-button" not in rendered
    assert len(rendered) < 4096


def test_wisdom_rich_card_stays_below_telegram_limit_for_large_page():
    view = WisdomView(
        "Collective Wisdom",
        "S" * 5000,
        items=[
            WisdomItem(
                f"skill-{index}-" + "T" * 400,
                "D" * 3000,
                actions=[
                    WisdomAction("Install", callback_data=f"wi:cmd:{index}"),
                    WisdomAction("View", url=f"https://portal.example/{index}"),
                ],
            )
            for index in range(5)
        ],
    )

    assert len(TelegramAdapter._wisdom_command_html(view)) < 4096


def test_wisdom_back_control_is_separate_and_first():
    view = WisdomView(
        "Skill details",
        "A shared skill",
        actions=[WisdomAction("Install", callback_data="wi:cmd:install")],
        navigation_actions=[WisdomAction("← Back", callback_data="wi:cmd:back")],
    )

    rendered = TelegramAdapter._wisdom_command_html(view)
    with (
        patch(
            "telegram.InlineKeyboardButton",
            side_effect=lambda text, **kwargs: SimpleNamespace(text=text, **kwargs),
        ),
        patch(
            "telegram.InlineKeyboardMarkup",
            side_effect=lambda rows: SimpleNamespace(inline_keyboard=rows),
        ),
    ):
        keyboard = TelegramAdapter._wisdom_command_keyboard(view)

    assert rendered.index("← Back") < rendered.index("A shared skill")
    assert rendered.index("← Back") < rendered.index("Install")
    assert rendered.count('<tg-button-row align="left">') == 2
    assert rendered.index("</tg-button-row>") < rendered.index("A shared skill")
    assert keyboard is not None
    assert [button.text for button in keyboard.inline_keyboard[0]] == ["← Back"]
    assert [button.text for button in keyboard.inline_keyboard[1]] == ["Install"]


def test_wisdom_telegram_url_controls_drop_redundant_arrow():
    view = WisdomView(
        "Skill details",
        actions=[
            WisdomAction(
                "View in Portal ↗",
                url="https://portal.example/skill",
            )
        ],
    )

    rendered = TelegramAdapter._wisdom_command_html(view)
    with (
        patch(
            "telegram.InlineKeyboardButton",
            side_effect=lambda text, **kwargs: SimpleNamespace(text=text, **kwargs),
        ),
        patch(
            "telegram.InlineKeyboardMarkup",
            side_effect=lambda rows: SimpleNamespace(inline_keyboard=rows),
        ),
    ):
        keyboard = TelegramAdapter._wisdom_command_keyboard(view)

    assert ">View in Portal</tg-button>" in rendered
    assert "↗" not in rendered
    assert keyboard is not None
    assert keyboard.inline_keyboard[0][0].text == "View in Portal"


@pytest.mark.asyncio
async def test_group_continuation_is_always_rendered_as_dm_deep_link():
    adapter = _adapter()
    adapter._bot.username = "HermesTestBot"
    view = WisdomView(
        "Shared skills",
        actions=[
            WisdomAction(
                "Continue in DM",
                "continue_dm",
                {"raw_args": "versions skill-1"},
            ),
            WisdomAction("Next", "browse_page", {"page": 1}),
        ],
    )
    context = WisdomCommandContext(
        user_id="42",
        chat_id="group-1",
        profile="default",
        organization_id="org-1",
        is_group=True,
    )

    await adapter._prepare_wisdom_command_view(view, context)

    continuation = view.actions[0]
    assert continuation.operation is None
    assert continuation.callback_data is None
    assert continuation.url.startswith("https://t.me/HermesTestBot?start=wisdom_")
    assert view.actions[1].callback_data.startswith("wi:cmd:")


@pytest.mark.asyncio
async def test_send_wisdom_command_uses_rich_card_and_short_callbacks():
    adapter = _adapter()
    adapter._owner_profile = "default"
    adapter._run_wisdom_profile_operation = AsyncMock(side_effect=lambda fn: fn())
    adapter._bot.do_api_request = AsyncMock(return_value={"message_id": 123})
    source = SimpleNamespace(
        user_id="42",
        chat_id="42",
        chat_type="dm",
    )
    view = WisdomView(
        "Collective Wisdom",
        actions=[WisdomAction("Browse", "browse", primary=True)],
    )
    service = SimpleNamespace(store=SimpleNamespace(active_org_id=lambda: "org-1"))

    with (
        patch("hermes_wisdom.service.WisdomService", return_value=service),
        patch(
            "gateway.wisdom_command.WisdomCommandController.execute",
            return_value=view,
        ),
    ):
        await adapter.send_wisdom_command("", source=source)

    payload = adapter._bot.do_api_request.call_args.kwargs["api_kwargs"]
    html = payload["rich_message"]["html"]
    assert payload["chat_id"] == 42
    assert "Collective Wisdom" in html
    assert "wi:cmd:" in html
    assert '<tg-button-row align="left">' in html
    callback = view.actions[0].callback_data
    assert callback is not None
    assert len(callback.encode("utf-8")) <= 64


@pytest.mark.asyncio
async def test_wisdom_command_failure_does_not_echo_upstream_error():
    adapter = _adapter()
    adapter._run_wisdom_profile_operation = AsyncMock(
        side_effect=RuntimeError("Authorization: Bearer secret-token")
    )
    adapter._send_message_with_thread_fallback = AsyncMock()
    source = SimpleNamespace(user_id="42", chat_id="42", chat_type="dm")

    await adapter.send_wisdom_command("browse", source=source)

    text = adapter._send_message_with_thread_fallback.call_args.kwargs["text"]
    assert "secret-token" not in text
    assert "temporarily unavailable" in text
