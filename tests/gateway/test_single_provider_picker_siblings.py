"""Regression for #131555: skip only redundant Slack/Discord provider menus.

Real adapter sends and picker callbacks use fake transports; pricing is stubbed
so selecting a fixture model cannot trigger a catalog API request.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.i18n import t
from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter, ModelPickerView
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture(params=["single", "multiple", "empty-models", "empty"])
def providers(request):
    provider = {
        "slug": "custom:fixture-endpoint",
        "name": "Fixture Endpoint",
        "models": [f"fixture/model-{i}" for i in range(80)],
        "total_models": 90,
    }
    if request.param == "empty":
        return []
    if request.param == "empty-models":
        provider["models"] = []
        return [provider]
    if request.param == "multiple":
        return [provider, {"slug": "other", "name": "Other", "models": ["other/model"]}]
    return [provider]


class _SlackAuthRunner:
    async def handle(self, event):
        pass

    def _is_user_authorized(self, source):
        return source.user_id == "U1"


@pytest.fixture
def slack():
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="fixture-token"))
    adapter._app = MagicMock()
    adapter._team_clients = {"T1": AsyncMock()}
    adapter._team_bot_user_ids = {"T1": "U_BOT"}
    adapter._channel_team = {"C1": "T1"}
    adapter._team_clients["T1"].chat_postMessage.return_value = {"ts": "1.2"}
    adapter.set_message_handler(_SlackAuthRunner().handle)
    return adapter, adapter._team_clients["T1"]


def _slack_controls(blocks):
    return {element["action_id"]: element for block in blocks
            for element in block.get("elements", [])}


@pytest.mark.asyncio
async def test_slack_picker_entry_and_selection(slack, providers):
    adapter, client = slack
    selected = AsyncMock(return_value="Switched fixture model")
    result = await adapter.send_model_picker(
        "C1", providers, "old/model", "other-current", "session", selected,
        metadata={"team_id": "T1", "thread_id": "thread.1"},
    )
    if not providers:
        assert not result.success
        client.chat_postMessage.assert_not_awaited()
        assert not adapter._model_picker_state
        return

    assert result.success, result.error
    sent = client.chat_postMessage.call_args.kwargs
    assert sent["thread_ts"] == "thread.1"
    state = adapter._model_picker_state[("T1", "1.2")]
    controls = _slack_controls(sent["blocks"])
    direct = len(providers) == 1 and bool(providers[0]["models"])
    assert state["stage"] == ("model" if direct else "provider")
    assert state["selected_provider_slug"] == (providers[0]["slug"] if direct else "")
    assert "hermes_model_cancel" in controls
    body = {"message": {"ts": "1.2"}, "channel": {"id": "C1"},
            "user": {"id": "U1", "name": "fixture-user"}, "team_id": "T1"}
    ack = AsyncMock()
    if not direct:
        assert "hermes_model_provider" in controls
        assert "hermes_model_model" not in controls
        if not providers[0]["models"]:
            selected.assert_not_awaited()
            return
        await adapter._handle_model_picker_action(
            ack, body, {"action_id": "hermes_model_provider", "selected_option": {"value": "0"}},
        )
        controls = _slack_controls(client.chat_update.call_args.kwargs["blocks"])
    else:
        assert "hermes_model_provider" not in controls
        assert sent["blocks"] == adapter._build_model_picker_model_blocks(providers, providers[0]["slug"])
        assert sent["text"] == t("platform.slack.picker.fallback_model", provider=providers[0]["name"])
    assert ("hermes_model_back" in controls) is (not direct)
    if not direct:
        await adapter._handle_model_picker_action(ack, body, {"action_id": "hermes_model_back"})
        assert state["stage"] == "provider"
        assert state["selected_provider_slug"] == ""
        assert client.chat_update.call_args.kwargs["blocks"] == sent["blocks"]
        await adapter._handle_model_picker_action(
            ack, body, {"action_id": "hermes_model_provider", "selected_option": {"value": "0"}},
        )
    options = controls["hermes_model_model"]["options"]
    assert [o["value"] for o in options] == [str(i) for i in range(len(providers[0]["models"]))]
    await adapter._handle_model_picker_action(
        ack, body, {"action_id": "hermes_model_model", "selected_option": options[-1]},
    )
    selected.assert_awaited_once_with("C1", providers[0]["models"][-1], providers[0]["slug"])
    assert not adapter._model_picker_state


@pytest.fixture
def discord(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token"))
    adapter._client = MagicMock()
    adapter._allowed_user_ids = {"123"}
    adapter._allowed_role_ids = set()
    channel = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(id=789)))
    adapter._resolve_channel = AsyncMock(return_value=channel)
    monkeypatch.setattr(ModelPickerView, "_expensive_warning_for", AsyncMock(return_value=None))
    return adapter, channel


def _interaction(value):
    return SimpleNamespace(
        user=SimpleNamespace(id=123), channel_id=456, data={"values": [value]},
        response=SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock()),
        edit_original_response=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_discord_picker_entry_and_selection(discord, providers):
    adapter, channel = discord
    selected = AsyncMock(return_value="Switched fixture model")
    result = await adapter.send_model_picker(
        "456", providers, "old/model", "other-current", "session", selected,
    )
    assert result.success, result.error
    sent = channel.send.call_args.kwargs
    view = sent["view"]
    assert view._message.id == 789
    controls = {child.custom_id: child for child in view.children}
    direct = len(providers) == 1 and bool(providers[0]["models"])
    assert view._selected_provider == (providers[0]["slug"] if direct else "")
    if not providers:
        assert not controls
        return
    if not direct:
        assert "model_provider_select" in controls
        assert not any(key.startswith("model_model_select_") for key in controls)
        if not providers[0]["models"]:
            selected.assert_not_awaited()
            return
        interaction = _interaction(providers[0]["slug"])
        await controls["model_provider_select"].callback(interaction)
        model_embed = interaction.response.edit_message.call_args.kwargs["embed"]
        controls = {child.custom_id: child for child in view.children}
    else:
        assert "model_provider_select" not in controls
        model_embed = sent["embed"]
    assert ("model_back" in controls) is (not direct)
    if not direct:
        back = _interaction("")
        await controls["model_back"].callback(back)
        assert back.response.edit_message.call_args.kwargs["embed"].description == sent["embed"].description
        provider_select = next(child for child in view.children if child.custom_id == "model_provider_select")
        await provider_select.callback(_interaction(providers[0]["slug"]))
        controls = {child.custom_id: child for child in view.children}
    assert "model_cancel2" in controls
    model_selects = [item for key, item in controls.items() if key.startswith("model_model_select_")]
    rendered = [option.value for item in model_selects for option in item.options]
    assert rendered == providers[0]["models"][:75]
    extra = "\n*" + t("platform.discord.picker.more_available", count=str(providers[0]["total_models"] - len(rendered))) + "*"
    assert model_embed.description == t("platform.discord.picker.select_model", provider=providers[0]["name"], extra=extra)
    interaction = _interaction(rendered[-1])
    await model_selects[-1].callback(interaction)
    selected.assert_awaited_once_with("456", rendered[-1], providers[0]["slug"])
    assert view.resolved and not view.children
    assert interaction.response.edit_message.call_args.kwargs["view"] is None
