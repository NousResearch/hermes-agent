"""Teams plugin cards: a click on a plugin's ``Action.Execute`` button reaches the plugin that registered
its namespace (``ctx.register_card_action_handler``), a plugin slash command's ``CardReply`` goes out as a
card, and exec-approval clicks resolve exactly as before. Contract: ``gateway/plugin_card_actions.py``.

The card models are the real SDK's when ``microsoft-teams-apps`` is installed; otherwise stand-ins with the
SDK card models' pydantic config (unknown keys kept, ``extra="allow"``)."""

import asyncio
import json
import logging
import threading
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, ConfigDict, SerializeAsAny

from tests.gateway.test_teams import TeamsAdapter, _make_config, _teams_mod
from agent.i18n import t
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.plugin_card_actions import MAX_CARD_BYTES, CardActionResult, CardReply, prepare_card
from gateway.run import GatewayRunner
from gateway.session_context import get_session_env
from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
import plugins.platforms.teams.card_actions as teams_card_actions

OWNER = "aad-owner"
CHAT = "19:review@thread.v2"
FAILED = "That action failed; try again or use the typed command."


class _Element(BaseModel):
    model_config = ConfigDict(extra="allow")


class _StandInCard(_Element):
    type: str = "AdaptiveCard"
    body: Optional[list[SerializeAsAny[_Element]]] = None
    actions: Optional[list[SerializeAsAny[_Element]]] = None


class _StandInCardResponse(BaseModel):
    status_code: int = 200
    type: str = "application/vnd.microsoft.card.adaptive"
    value: _StandInCard


class _StandInMessageResponse(BaseModel):
    status_code: int = 200
    type: str = "application/vnd.microsoft.activity.message"
    value: str


@pytest.fixture
def card_models(monkeypatch):
    """Bind pydantic card models into the adapter unless the real SDK already is."""
    if not hasattr(_teams_mod.AdaptiveCard, "model_validate"):
        monkeypatch.setattr(_teams_mod, "AdaptiveCard", _StandInCard)
        monkeypatch.setattr(_teams_mod, "AdaptiveCardActionCardResponse", _StandInCardResponse)
        monkeypatch.setattr(_teams_mod, "AdaptiveCardActionMessageResponse", _StandInMessageResponse)


def _wire(response) -> dict:
    """The invoke response body as the SDK's HTTP server serializes it."""
    dump = getattr(response, "model_dump", None)
    body = dump(exclude_none=True)["body"] if dump else response.body.model_dump(by_alias=True, exclude_none=True)
    return json.loads(json.dumps(body))


@pytest.fixture
def manager(monkeypatch):
    """The process plugin manager the gateway resolves namespaces in (discovery already done)."""
    plugins_mod._reset_plugin_managers_for_tests()
    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)
    yield mgr
    plugins_mod._reset_plugin_managers_for_tests()


def _register(mgr: PluginManager, namespace: str, callback, plugin: str = "brain-review") -> None:
    ctx = PluginContext(manifest=PluginManifest(name=plugin, version="0.1.0", description="test"), manager=mgr)
    ctx.register_card_action_handler(namespace, callback)


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv("TEAMS_ALLOWED_USERS", OWNER)
    adapter = TeamsAdapter(_make_config(client_id="bot-id", client_secret="secret", tenant_id="tenant"))
    adapter._app = MagicMock()
    adapter._app.send = AsyncMock(return_value=SimpleNamespace(id="card-msg-1"))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={adapter.platform: PlatformConfig(enabled=True)})
    runner._draining = False
    runner.adapters = {adapter.platform: adapter}
    adapter.gateway_runner = runner
    return adapter


def _click(data: dict, *, aad: str = OWNER, reply_to: str = "card-msg-1"):
    activity = SimpleNamespace(
        value=SimpleNamespace(action=SimpleNamespace(data=data)),
        from_=SimpleNamespace(id="29:channel-account", aad_object_id=aad, name="Owner"),
        conversation=SimpleNamespace(id=CHAT, conversation_type="personal", name="Review", tenant_id="tenant"),
        reply_to_id=reply_to)
    return SimpleNamespace(activity=activity)


def _review_card(*, note: str = "") -> dict:
    """The consumer's shape: per-item buttons, an Edit ShowCard with a text input, a Move choice set."""
    return {
        "type": "AdaptiveCard", "version": "1.5",
        "body": [{"type": "TextBlock", "text": "Proposal 42" + note, "wrap": True}],
        "actions": [
            {"type": "Action.Execute", "title": "Approve", "verb": "brain",
             "data": {"hermes_plugin_action": "brain", "op": "approve", "item": "42", "v": "h1"}},
            {"type": "Action.ShowCard", "title": "Edit", "card": {
                "type": "AdaptiveCard",
                "body": [{"type": "Input.Text", "id": "text", "isMultiline": True, "value": "draft"},
                         {"type": "Input.ChoiceSet", "id": "target", "value": "inbox",
                          "choices": [{"title": "Inbox", "value": "inbox"}, {"title": "Archive", "value": "archive"}]}],
                "actions": [{"type": "Action.Execute", "title": "Save", "verb": "brain",
                             "data": {"hermes_plugin_action": "brain", "op": "edit", "item": "42", "v": "h1"}}]}},
        ]}


@pytest.mark.anyio
async def test_exec_approval_clicks_resolve_as_before(adapter, manager, monkeypatch):
    _register(manager, "brain", lambda event: pytest.fail("a plugin saw an approval click"))
    resolved = []
    monkeypatch.setattr("tools.approval.has_blocking_approval", lambda key: key == "agent:main:teams:dm:x")
    monkeypatch.setattr("tools.approval.resolve_gateway_approval", lambda key, choice: resolved.append((key, choice)))
    approve = {"hermes_action": "approve_session", "session_key": "agent:main:teams:dm:x", "cmd": "make", "desc": "d"}

    answered = await adapter._on_card_action(_click(approve))
    assert resolved == [("agent:main:teams:dm:x", "session")]
    assert answered.status == 200 and isinstance(answered.body.value, _teams_mod.AdaptiveCard)

    stranger = await adapter._on_card_action(_click({**approve, "hermes_action": "deny"}, aad="aad-stranger"))
    unknown = await adapter._on_card_action(_click({"hermes_action": "approve_once"}))
    expired = await adapter._on_card_action(_click({**approve, "session_key": "agent:main:teams:dm:gone"}))
    assert stranger.body.value == t("platform.shared.not_authorized")
    assert unknown.body.value == t("platform.teams.approval.unknown_action")
    assert isinstance(expired.body.value, _teams_mod.AdaptiveCard)
    assert resolved == [("agent:main:teams:dm:x", "session")]


@pytest.mark.anyio
async def test_a_click_reaches_the_plugin_with_the_verified_clicker_inputs_and_session(adapter, manager, card_models):
    loop_thread = threading.current_thread()
    seen = {}

    def on_click(event):
        seen["event"], seen["thread"] = event, threading.current_thread()
        seen["env"] = {k: get_session_env(k) for k in (
            "HERMES_SESSION_PLATFORM", "HERMES_SESSION_USER_ID", "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_KEY")}
        return "Approved."

    _register(manager, "brain", on_click)
    data = {"hermes_plugin_action": "brain", "hermes_plugin_data": {"op": "edit", "item": "42"},
            "text": "new wording", "target": "archive"}
    answered = await adapter._on_card_action(_click(data))

    event = seen["event"]
    chat_source = adapter._source_for_activity(_click({}).activity, None)
    expected_key = adapter.gateway_runner._session_key_for_source(chat_source)
    assert (event.platform, event.namespace, event.user_id, event.user_name) == ("teams", "brain", OWNER, "Owner")
    assert (event.chat_id, event.message_id, event.session_key) == (CHAT, "card-msg-1", expected_key)
    assert event.action_data == {"op": "edit", "item": "42"}
    assert event.inputs == {"text": "new wording", "target": "archive"}
    assert seen["env"] == {"HERMES_SESSION_PLATFORM": "teams", "HERMES_SESSION_USER_ID": OWNER,
                           "HERMES_SESSION_CHAT_ID": CHAT, "HERMES_SESSION_KEY": expected_key}
    assert seen["thread"] is not loop_thread  # sync handlers run on the gateway pool
    assert get_session_env("HERMES_SESSION_USER_ID") == ""  # bound for the handler only
    wire = _wire(answered)
    assert (wire["type"], wire["value"]) == ("application/vnd.microsoft.activity.message", "Approved.")


@pytest.mark.anyio
async def test_a_card_reply_round_trips_into_a_click_with_its_button_data_and_inputs(adapter, manager, card_models,
                                                                                     monkeypatch):
    """The plugin writes flat button data; Teams merges input values into it; the plugin gets them apart."""
    seen = []
    _register(manager, "brain", lambda event: seen.append(event) or "Saved.")
    reply = CardReply(card=_review_card(), text="Proposal 42: approve, edit or move")
    monkeypatch.setattr(plugins_mod, "get_plugin_command_handler",
                        lambda name: (lambda raw_args: reply) if name == "brain" else None)
    source = adapter._source_for_activity(_click({}).activity, "m1")

    handled, reply, _ = await adapter.gateway_runner._hm_dispatch_quick_and_plugin_commands(
        MessageEvent(text="/brain", source=source, message_id="m1"), source, "brain")

    assert (handled, reply) == (True, None)  # delivered as a card: no text reply
    chat_id, sent = adapter._app.send.await_args.args
    sent = sent.model_dump(by_alias=True, exclude_none=True)
    assert chat_id == CHAT and sent["fallbackText"] == "Proposal 42: approve, edit or move"
    save = sent["actions"][1]["card"]["actions"][0]["data"]
    assert save == {"hermes_plugin_action": "brain", "hermes_plugin_data": {"op": "edit", "item": "42", "v": "h1"}}

    await adapter._on_card_action(_click({**save, "text": "new wording", "target": "archive"}))
    assert seen[0].action_data == {"op": "edit", "item": "42", "v": "h1"}
    assert seen[0].inputs == {"text": "new wording", "target": "archive"}


@pytest.mark.anyio
async def test_an_async_handler_is_awaited(adapter, manager, card_models):
    async def on_click(event):
        return CardActionResult(text=f"done {event.action_data['item']}")

    _register(manager, "brain", on_click)
    answered = await adapter._on_card_action(_click({"hermes_plugin_action": "brain", "item": "7"}))
    assert _wire(answered)["value"] == "done 7"


@pytest.mark.anyio
@pytest.mark.parametrize("allowed", ["aad-someone-else", None])
async def test_a_clicker_outside_the_allowlist_never_reaches_the_plugin(adapter, manager, monkeypatch, allowed):
    if allowed is None:  # unset allowlist: default deny
        monkeypatch.delenv("TEAMS_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("TEAMS_ALLOW_ALL_USERS", raising=False)
    else:
        monkeypatch.setenv("TEAMS_ALLOWED_USERS", allowed)
    _register(manager, "brain", lambda event: pytest.fail("the plugin saw an unauthorized click"))

    answered = await adapter._on_card_action(_click({"hermes_plugin_action": "brain", "item": "42"}))

    expected = t("platform.shared.not_authorized") if allowed else t("platform.teams.approval.requires_allowlist")
    assert answered.body.value == expected


@pytest.mark.anyio
@pytest.mark.parametrize("namespace", ["nobody", "", ["brain"], None])
async def test_an_unknown_namespace_is_an_unknown_action(adapter, manager, namespace):
    _register(manager, "brain", lambda event: pytest.fail("routed to the wrong plugin"))
    answered = await adapter._on_card_action(_click({"hermes_plugin_action": namespace}))
    assert answered.body.value == t("platform.teams.approval.unknown_action")


@pytest.mark.anyio
async def test_a_raising_handler_answers_the_failure_text_and_logs_no_content(adapter, manager, card_models, caplog):
    def on_click(event):
        raise RuntimeError("boom")

    _register(manager, "brain", on_click)
    with caplog.at_level(logging.DEBUG):
        answered = await adapter._on_card_action(_click(
            {"hermes_plugin_action": "brain", "hermes_plugin_data": {"item": "secret-item"}, "text": "secret-input"}))

    assert answered.status == 200 and _wire(answered)["value"] == FAILED
    assert "brain-review" in caplog.text and "RuntimeError" in caplog.text
    assert "secret-item" not in caplog.text and "secret-input" not in caplog.text


@pytest.mark.anyio
async def test_a_handler_that_outlives_the_invoke_deadline_answers_the_failure_text(adapter, manager, card_models,
                                                                                    monkeypatch):
    monkeypatch.setattr(teams_card_actions, "INVOKE_TIMEOUT_S", 0.2)
    release = threading.Event()
    _register(manager, "slow-sync", lambda event: release.wait(30))

    async def never(event):
        await asyncio.Event().wait()

    _register(manager, "slow-async", never)
    try:
        sync_answer = await adapter._on_card_action(_click({"hermes_plugin_action": "slow-sync"}))
        async_answer = await adapter._on_card_action(_click({"hermes_plugin_action": "slow-async"}))
    finally:
        release.set()
    assert _wire(sync_answer)["value"] == FAILED and _wire(async_answer)["value"] == FAILED


@pytest.mark.anyio
async def test_a_card_answer_replaces_the_clicked_card_with_every_element_kept(adapter, manager, card_models):
    card = _review_card(note=" (approved)")
    _register(manager, "brain", lambda event: card)

    answered = await adapter._on_card_action(_click({"hermes_plugin_action": "brain", "op": "approve"}))

    wire = _wire(answered)
    assert wire["type"] == "application/vnd.microsoft.card.adaptive"
    prepared = prepare_card(card)
    assert wire["value"]["body"] == prepared["body"]
    assert wire["value"]["actions"] == prepared["actions"]
    edit = wire["value"]["actions"][1]
    assert edit["type"] == "Action.ShowCard"
    assert [e["type"] for e in edit["card"]["body"]] == ["Input.Text", "Input.ChoiceSet"]
    assert edit["card"]["body"][1]["choices"] == card["actions"][1]["card"]["body"][1]["choices"]


@pytest.mark.anyio
async def test_an_oversized_card_answer_becomes_text(adapter, manager, card_models, caplog):
    huge = {"type": "AdaptiveCard", "body": [{"type": "TextBlock", "text": "secret-body " * (MAX_CARD_BYTES // 8)}]}
    _register(manager, "brain", lambda event: huge)

    with caplog.at_level(logging.WARNING):
        answered = await adapter._on_card_action(_click({"hermes_plugin_action": "brain"}))

    wire = _wire(answered)
    assert wire["type"] == "application/vnd.microsoft.activity.message"
    assert str(MAX_CARD_BYTES) in wire["value"]
    assert "brain-review" in caplog.text and "secret-body" not in caplog.text


@pytest.mark.anyio
async def test_a_card_the_sdk_refuses_answers_the_failure_text(adapter, manager, card_models):
    _register(manager, "brain", lambda event: {"type": "AdaptiveCard", "body": "not a list"})
    answered = await adapter._on_card_action(_click({"hermes_plugin_action": "brain"}))
    assert _wire(answered)["value"] == FAILED


@pytest.mark.anyio
async def test_an_oversized_card_reply_is_sent_as_its_text(adapter, monkeypatch, card_models):
    huge = {"type": "AdaptiveCard", "body": [{"type": "TextBlock", "text": "x" * MAX_CARD_BYTES}]}
    monkeypatch.setattr(plugins_mod, "get_plugin_command_handler",
                        lambda name: lambda raw_args: CardReply(card=huge, text="Too many proposals; see /brain list"))
    source = adapter._source_for_activity(_click({}).activity, "m1")

    handled, reply, _ = await adapter.gateway_runner._hm_dispatch_quick_and_plugin_commands(
        MessageEvent(text="/brain", source=source, message_id="m1"), source, "brain")

    assert (handled, reply) == (True, "Too many proposals; see /brain list")
    adapter._app.send.assert_not_awaited()
