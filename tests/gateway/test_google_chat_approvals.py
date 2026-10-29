"""Google Chat exec-approval cards: Card v2 buttons, click parsing (native and add-on
envelopes), the default-deny clicker gate, and card retirement on click or timeout."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.base import ExecApprovalPrompt, SendResult
# Imported first: installs the google-* module shims the adapter import needs.
from tests.gateway.test_google_chat import _base_config, _make_chat_envelope, _make_pubsub_message

from plugins.platforms.google_chat.adapter import GoogleChatAdapter, _ThreadCountStore
from plugins.platforms.google_chat.approvals import APPROVAL_FUNCTION, parse_card_click

TOPIC = "projects/test-project/topics/chat-events"

CARD = "spaces/S/messages/CARD.1"
BOSS = "boss@example.com"


@pytest.fixture()
def adapter(tmp_path):
    a = GoogleChatAdapter(_base_config())
    a._chat_api = MagicMock()
    a._new_authed_http = MagicMock(return_value=MagicMock())
    a._thread_count_store = _ThreadCountStore(tmp_path / "google_chat_thread_counts.json")
    a._subscription_topic = TOPIC
    return a


@pytest.fixture(autouse=True)
def _allowlist(monkeypatch):
    for var in ("GOOGLE_CHAT_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GOOGLE_CHAT_ALLOWED_USERS", BOSS)


def _prompt(command="<terminal> (plugin approval rule)", description="Owner's OK needed"):
    return ExecApprovalPrompt(
        chat_id="spaces/S", session_key="agent:main:google_chat:dm:spaces/S", text="", command=command,
        description=description, smart_denied=False,
        actions=[("Allow Once", "once", "primary"), ("Deny", "deny", "danger")])


def _params(choice, action="approval"):
    return {"hermes_action": action, "choice": choice} if action else {"choice": choice}


def _native_click(choice="once", email=BOSS, message=CARD, space="spaces/S", action="approval"):
    params = _params(choice, action)
    return {
        "type": "CARD_CLICKED",
        "action": {"actionMethodName": APPROVAL_FUNCTION, "parameters": [{"key": k, "value": v} for k, v in params.items()]},
        "common": {"invokedFunction": APPROVAL_FUNCTION, "parameters": params},
        "message": {"name": message},
        "space": {"name": space, "spaceType": "DIRECT_MESSAGE"},
        "user": {"name": "users/1", "email": email, "displayName": "Boss", "type": "HUMAN"},
    }


def _addon_click(choice="once", email=BOSS):
    """Shape of a real click on an add-on app over Pub/Sub (captured 2026-10): no
    ``invokedFunction``, no ``ce-type`` attribute, the space repeated inside the message."""
    space = {"name": "spaces/S", "type": "ROOM", "spaceType": "SPACE", "displayName": "Team"}
    return {
        "commonEventObject": {"userLocale": "fr", "hostApp": "CHAT", "platform": "WEB",
                              "timeZone": {"id": "Europe/Paris", "offset": 3600000}, "parameters": _params(choice)},
        "chat": {
            "user": {"name": "users/1", "displayName": "Boss", "email": email, "type": "HUMAN"},
            "eventTime": "2026-10-26T09:46:28.555721Z",
            "buttonClickedPayload": {
                "space": space,
                "message": {"name": CARD, "sender": {"name": "users/9", "displayName": "Hermes", "type": "BOT"},
                            "thread": {"name": "spaces/S/threads/T"}, "space": space},
            },
        },
    }


async def _post_card(adapter):
    adapter._create_message = AsyncMock(return_value=SendResult(success=True, message_id=CARD))
    adapter._patch_message = AsyncMock(return_value=SendResult(success=True, message_id=CARD))
    result = await adapter._send_exec_approval_prompt(_prompt())
    assert result.success
    return adapter._create_message.call_args.args[1]["cardsV2"][0]


def _patched_widgets(adapter):
    body = adapter._patch_message.call_args.args[1]
    return body["cardsV2"][0]["card"]["sections"][0]["widgets"]


class TestApprovalCard:
    def test_adapter_renders_native_buttons(self, adapter):
        assert type(adapter).supports_exec_approval_buttons()

    @pytest.mark.asyncio
    async def test_card_carries_choices_but_not_the_session_key(self, adapter):
        card = await _post_card(adapter)
        widgets = card["card"]["sections"][0]["widgets"]
        buttons = widgets[-1]["buttonList"]["buttons"]
        assert [b["text"] for b in buttons] == ["Allow Once", "Deny"]
        actions = [b["onClick"]["action"] for b in buttons]
        # Chat delivers the click only when the button names the app's own Pub/Sub topic.
        assert {a["function"] for a in actions} == {TOPIC}
        assert [a["parameters"] for a in actions] == [
            [{"key": "hermes_action", "value": "approval"}, {"key": "choice", "value": "once"}],
            [{"key": "hermes_action", "value": "approval"}, {"key": "choice", "value": "deny"}]]
        assert "agent:main" not in json.dumps(card)
        assert adapter._approval_cards[CARD]["session_key"] == "agent:main:google_chat:dm:spaces/S"

    @pytest.mark.asyncio
    async def test_http_inbound_buttons_name_the_endpoint(self, adapter):
        adapter._subscription_topic = ""
        adapter._http_events_url = "https://bot.example.com/google-chat/events"
        card = await _post_card(adapter)
        functions = {b["onClick"]["action"]["function"]
                     for b in card["card"]["sections"][0]["widgets"][-1]["buttonList"]["buttons"]}
        assert functions == {"https://bot.example.com/google-chat/events"}

    @pytest.mark.asyncio
    async def test_command_markup_is_escaped(self, adapter):
        adapter._create_message = AsyncMock(return_value=SendResult(success=True, message_id=CARD))
        await adapter._send_exec_approval_prompt(_prompt(command="cat <secret> | wc -l"))
        assert "cat &lt;secret&gt; | wc -l" in json.dumps(adapter._create_message.call_args.args[1])

    @pytest.mark.asyncio
    async def test_plugin_rule_label_is_left_out(self, adapter):
        card = await _post_card(adapter)
        text = json.dumps(card)
        assert "plugin approval rule" not in text
        assert "Owner&#x27;s OK needed" in text or "Owner's OK needed" in text

    @pytest.mark.asyncio
    async def test_failed_send_keeps_no_state(self, adapter):
        adapter._create_message = AsyncMock(side_effect=RuntimeError("boom"))
        result = await adapter._send_exec_approval_prompt(_prompt())
        assert not result.success
        assert adapter._approval_cards == {}


class TestParseCardClick:
    def test_native_event(self):
        click = parse_card_click(_native_click("session"))
        assert (click.function, click.parameters["choice"], click.message_name, click.space_name, click.user_email) == (
            APPROVAL_FUNCTION, "session", CARD, "spaces/S", BOSS)
        assert click.space_type == "DIRECT_MESSAGE"

    def test_addon_event(self):
        click = parse_card_click(_addon_click("deny"))
        assert (click.parameters, click.message_name, click.space_name, click.user_email) == (
            {"hermes_action": "approval", "choice": "deny"}, CARD, "spaces/S", BOSS)
        assert click.function == ""

    def test_messages_are_not_clicks(self):
        assert parse_card_click(_make_chat_envelope(text="hi")) is None
        assert parse_card_click({"type": "MESSAGE", "message": {"name": "m"}}) is None
        assert parse_card_click("not a dict") is None


class TestCardClick:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("envelope", [_native_click(), _addon_click()])
    async def test_allowed_click_resolves_and_retires_the_card(self, adapter, envelope):
        await _post_card(adapter)
        with patch("tools.approval.resolve_gateway_approval", return_value=1) as resolve:
            await adapter._handle_card_click(parse_card_click(envelope))
        resolve.assert_called_once_with("agent:main:google_chat:dm:spaces/S", "once")
        widgets = _patched_widgets(adapter)
        assert not any("buttonList" in w for w in widgets)
        assert "Allow Once · Boss" in widgets[-1]["textParagraph"]["text"]
        assert CARD not in adapter._approval_cards

    @pytest.mark.asyncio
    async def test_unauthorized_click_changes_nothing(self, adapter):
        await _post_card(adapter)
        with patch("tools.approval.resolve_gateway_approval") as resolve:
            await adapter._handle_card_click(parse_card_click(_native_click(email="intruder@example.com")))
        resolve.assert_not_called()
        adapter._patch_message.assert_not_called()
        assert CARD in adapter._approval_cards

    @pytest.mark.asyncio
    async def test_no_allowlist_denies_by_default(self, adapter, monkeypatch):
        monkeypatch.delenv("GOOGLE_CHAT_ALLOWED_USERS")
        await _post_card(adapter)
        with patch("tools.approval.resolve_gateway_approval") as resolve:
            await adapter._handle_card_click(parse_card_click(_native_click()))
        resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_gateway_check_takes_precedence(self, adapter):
        await _post_card(adapter)
        adapter._authorization_check = lambda user_id, chat_type, chat_id, **_: False
        with patch("tools.approval.resolve_gateway_approval") as resolve:
            await adapter._handle_card_click(parse_card_click(_native_click()))
        resolve.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("envelope", [
        _native_click(space="spaces/OTHER"),
        _native_click(message="spaces/S/messages/UNKNOWN"),
        _native_click(choice="sudo"),
        _native_click(action=None),
        _native_click(action="clarify"),
    ])
    async def test_foreign_unknown_or_malformed_clicks_are_ignored(self, adapter, envelope):
        await _post_card(adapter)
        with patch("tools.approval.resolve_gateway_approval") as resolve:
            await adapter._handle_card_click(parse_card_click(envelope))
        resolve.assert_not_called()
        assert CARD in adapter._approval_cards

    @pytest.mark.asyncio
    async def test_second_click_does_not_resolve_again(self, adapter):
        await _post_card(adapter)
        with patch("tools.approval.resolve_gateway_approval", return_value=1) as resolve:
            await adapter._handle_card_click(parse_card_click(_native_click()))
            await adapter._handle_card_click(parse_card_click(_native_click(choice="always")))
        resolve.assert_called_once()

    @pytest.mark.asyncio
    async def test_click_after_timeout_shows_expired(self, adapter):
        await _post_card(adapter)
        with patch("tools.approval.resolve_gateway_approval", return_value=0):
            await adapter._handle_card_click(parse_card_click(_native_click()))
        assert "⌛" in _patched_widgets(adapter)[-1]["textParagraph"]["text"]


class TestRouting:
    def test_pubsub_click_is_routed_and_acked(self, adapter):
        msg = _make_pubsub_message(_native_click())
        with patch.object(adapter, "_submit_on_loop") as submit, \
                patch.object(adapter, "_handle_card_click", new=AsyncMock()) as handler:
            adapter._on_pubsub_message(msg)
        submit.assert_called_once()
        submit.call_args.args[0].close()
        handler.assert_called_once()
        assert handler.call_args.args[0].message_name == CARD
        msg.ack.assert_called_once()

    @pytest.mark.asyncio
    async def test_http_click_is_handled_without_dispatch(self, adapter):
        adapter._dispatch_message = AsyncMock()
        with patch.object(adapter, "_handle_card_click", new=AsyncMock()) as handler:
            assert await adapter.dispatch_http_event(_addon_click()) == {}
        handler.assert_awaited_once()
        adapter._dispatch_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_timeout_edit_retires_the_card_instead_of_patching_text(self, adapter):
        await _post_card(adapter)
        result = await adapter.edit_message("spaces/S", CARD, "⌛ Approval timed out\n```\ncmd\n```")
        assert result.success
        body = adapter._patch_message.call_args.args[1]
        assert "text" not in body
        assert "⌛ Approval timed out" in _patched_widgets(adapter)[-1]["textParagraph"]["text"]
        assert CARD not in adapter._approval_cards

    @pytest.mark.asyncio
    async def test_failed_card_update_sends_the_outcome(self, adapter):
        await _post_card(adapter)
        adapter._patch_message = AsyncMock(side_effect=RuntimeError("503"))
        adapter.send = AsyncMock(return_value=SendResult(success=True))
        with patch("tools.approval.resolve_gateway_approval", return_value=1):
            await adapter._handle_card_click(parse_card_click(_native_click()))
        assert "Allow Once · Boss" in adapter.send.call_args.args[1]

    @pytest.mark.asyncio
    async def test_resolver_error_keeps_the_card_live(self, adapter):
        await _post_card(adapter)
        with patch("tools.approval.resolve_gateway_approval", side_effect=RuntimeError("boom")):
            await adapter._handle_card_click(parse_card_click(_native_click()))
        adapter._patch_message.assert_not_called()
        assert CARD in adapter._approval_cards

    @pytest.mark.asyncio
    async def test_card_memory_is_bounded(self, adapter, monkeypatch):
        import plugins.platforms.google_chat.adapter as gc_adapter
        monkeypatch.setattr(gc_adapter, "_APPROVAL_CARDS_MAX", 2)
        adapter._create_message = AsyncMock(side_effect=[
            SendResult(success=True, message_id=f"spaces/S/messages/{i}") for i in range(3)])
        for _ in range(3):
            await adapter._send_exec_approval_prompt(_prompt())
        assert list(adapter._approval_cards) == ["spaces/S/messages/1", "spaces/S/messages/2"]
