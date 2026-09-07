from html.parser import HTMLParser
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.wisdom_command import (
    WisdomCommandContext,
    WisdomCommandController,
    bind_view_callbacks,
)
from tests.wisdom.test_mute_controls import controls  # noqa: F401
from tests.wisdom.test_preferences import preferences  # noqa: F401


@pytest.fixture
def mute_view(controls, monkeypatch):
    p, service, _ = controls
    monkeypatch.setattr(
        "hermes_wisdom.preferences.WisdomPreferences", lambda service: p
    )
    view = WisdomCommandController().execute(
        "mute",
        service,
        WisdomCommandContext(
            user_id="user",
            chat_id="dm",
            profile="demo",
            organization_id="org",
            is_group=False,
        ),
    )
    bind_view_callbacks(
        view,
        WisdomCommandContext(
            user_id="user",
            chat_id="dm",
            profile="demo",
            organization_id="org",
            is_group=False,
        ),
    )
    return view


def test_telegram_rich_and_fallback_have_same_scoped_choices(mute_view, monkeypatch):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    # The shared gateway fixture stubs the optional Telegram SDK.
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.InlineKeyboardButton",
        lambda text, **kwargs: SimpleNamespace(text=text, **kwargs),
    )
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.InlineKeyboardMarkup",
        lambda rows: SimpleNamespace(inline_keyboard=rows),
    )

    class Rows(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows = []

        def handle_starttag(self, tag, attrs):
            if tag == "tg-button-row":
                self.rows.append([])
            if tag == "tg-button":
                self.rows[-1].append(dict(attrs))

    rendered = TelegramAdapter._wisdom_command_html(mute_view)
    parser = Rows()
    parser.feed(rendered)
    assert [len(row) for row in parser.rows] == [3, 2]
    assert parser.rows[-1][-1]["style"] == "primary"
    rich_targets = {button["data"] for row in parser.rows for button in row}
    assert all(target.startswith("wi:cmd:") for target in rich_targets)
    keyboard = TelegramAdapter._wisdom_command_keyboard(mute_view)
    fallback_targets = {
        button.callback_data for row in keyboard.inline_keyboard for button in row
    }
    assert rich_targets == fallback_targets
    assert len(rich_targets) == 5
    assert keyboard.inline_keyboard[-1][-1].text == "Turn on"
    assert all(len(row) <= 2 for row in keyboard.inline_keyboard)


def test_slack_preserves_duration_groups_and_primary_rightmost(mute_view):
    from plugins.platforms.slack.wisdom_blocks import (
        render_wisdom_blocks,
        wisdom_fallback_text,
    )

    blocks = render_wisdom_blocks(mute_view)
    rows = [block["elements"] for block in blocks if block["type"] == "actions"]
    assert [len(row) for row in rows] == [3, 2]
    assert [button["text"]["text"] for row in rows for button in row] == [
        "1 day",
        "1 week",
        "30 days",
        "Indefinitely",
        "Turn on",
    ]
    assert all(button["value"].startswith("wi:cmd:") for row in rows for button in row)
    assert rows[-1][-1]["style"] == "primary"
    assert "across clients in this organization" in wisdom_fallback_text(mute_view)


def test_pending_sync_is_visible_in_both_renderers(controls, monkeypatch):
    from plugins.platforms.slack.wisdom_blocks import wisdom_fallback_text
    from plugins.platforms.telegram.adapter import TelegramAdapter

    p, service, _ = controls
    service.client.set_recommendation_mute.side_effect = TimeoutError
    monkeypatch.setattr(
        "hermes_wisdom.preferences.WisdomPreferences", lambda service: p
    )
    view = WisdomCommandController().execute(
        "mute 1w",
        service,
        WisdomCommandContext(
            user_id="user",
            chat_id="dm",
            profile="demo",
            organization_id="org",
            is_group=False,
        ),
    )
    assert "waiting to sync" in TelegramAdapter._wisdom_command_html(view)
    assert "waiting to sync" in wisdom_fallback_text(view)
    assert "notifications are enabled" in view.summary


@pytest.mark.asyncio
async def test_legacy_telegram_callbacks_check_authorization_before_dispatch(
    monkeypatch,
):
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._is_callback_user_authorized = Mock(return_value=False)
    adapter._run_wisdom_profile_operation = AsyncMock()
    query = SimpleNamespace(
        from_user=SimpleNamespace(id="user", first_name="Test"),
        message=SimpleNamespace(chat_id="dm", chat=SimpleNamespace(type="private")),
        answer=AsyncMock(),
    )
    await adapter._handle_wisdom_agent_callback(query, "wa:mute:old:30d")
    adapter._run_wisdom_profile_operation.assert_not_called()
    assert adapter._is_callback_user_authorized.call_args.kwargs["command"] == "wisdom"
    assert "not authorized" in query.answer.call_args.kwargs["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chat_type,private", [("private", True), ("group", False), ("", False)]
)
async def test_legacy_mute_duration_opens_fresh_menu_without_applying(
    controls, monkeypatch, chat_type, private
):
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    p, service, _ = controls
    monkeypatch.setattr(
        "hermes_wisdom.preferences.WisdomPreferences", lambda service: p
    )
    monkeypatch.setattr("hermes_wisdom.service.WisdomService", lambda: service)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._is_callback_user_authorized = Mock(return_value=True)
    adapter._run_wisdom_profile_operation = AsyncMock(side_effect=lambda fn: fn())
    adapter._prepare_wisdom_command_view = AsyncMock()
    adapter._edit_wisdom_command_view = AsyncMock()
    query = SimpleNamespace(
        from_user=SimpleNamespace(id="user", first_name="Test"),
        message=SimpleNamespace(chat_id="dm", chat=SimpleNamespace(type=chat_type)),
        answer=AsyncMock(),
    )
    await adapter._handle_wisdom_agent_callback(query, "wa:mute:old:30d")
    view = adapter._edit_wisdom_command_view.call_args.args[1]
    assert p.mute_status("org") is None
    service.client.set_recommendation_mute.assert_not_called()
    if private:
        assert view.actions[-1].label == "Turn on"
        assert view.items[0].actions[0].operation == "mute_choice"
    else:
        assert view.actions[0].label == "Continue in DM"
