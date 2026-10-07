"""Optional approval context against the approval-card safety rules main added since the PR's base.

The model-provided context (purpose/effect/risk) is unverified display text. It must not change what
a prompt offers or who may answer it:

* a request is born Deny-only and a card lifts that only once it shows the whole command
  (677e352ad2, 295c325c35, f0ac114d2d): context never lifts it, and never costs a card its Approve
  button by pushing the preview into a full-command send that could fail;
* a text prompt on an adapter whose send cuts long text keeps the instructions-first layout with
  every choice, or leaves the context out (a836c50d1e); a prompt that already offers only Deny
  gets no context and its request stays Deny-only;
* id-less and stale card taps settle nothing (3ef096d916, 5ab58da1f6) and the requester binding
  (466591fe65) holds, with context on the request.

Each case drives the real chain (``check_all_command_guards`` in a gateway session, the registered
``TurnRunner._approval_notify_sync``, the real adapter) into a fake transport.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import tools.approval as approval_module
from gateway.config import PlatformConfig
from gateway.run_turn_runner import TurnRunner
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.whatsapp_cloud.adapter import WhatsAppCloudAdapter
from tests.gateway._plugin_adapter_loader import load_plugin_adapter
from tools import approval_context
from tools.approval import _build_enhanced_description_with_context, check_all_command_guards
from tools.approval_context import reset_current_session_key, set_current_session_key
from tools.approval_detection import detect_dangerous_command
from tools.approval_gateway_wait import _ApprovalEntry

_HEAD = "—— Model-provided context (unverified) ——"
_TAIL = "—— End unverified context ——"
_SESSION = "agent:main:whatsapp:dm:seams"
_LONG_CONTEXT = {"purpose": "p" * 900, "effect": "e" * 900, "risk": "r" * 900}


@pytest.fixture(autouse=True)
def _gateway_session(monkeypatch):
    for key in ("HERMES_INTERACTIVE", "HERMES_EXEC_ASK", "HERMES_YOLO_MODE", "DISCORD_APPROVAL_MENTIONS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    token = set_current_session_key(_SESSION)
    yield
    reset_current_session_key(token)
    approval_module._gateway_queues.clear()
    approval_module._gateway_notify_cbs.clear()
    approval_module._session_approved.clear()


def _entry():
    return approval_module._gateway_queues[_SESSION][0]


def _at_card(record):
    """What the request looks like when its card is posted (on a Deny-only one, an approve tap is tried
    and must settle nothing), then a Deny tap on that card ends it."""
    entry = _entry()
    rid = entry.data["request_id"]
    state = {"deny_only": entry.card_deny_only}
    if entry.card_deny_only:
        state["approve_tap"] = approval_module.resolve_gateway_approval_card(_SESSION, "once", rid)
    record.append(state)
    assert approval_module.resolve_gateway_approval_card(_SESSION, "deny", rid) == 1


def _whatsapp(record, full_command_ok):
    adapter = WhatsAppCloudAdapter(PlatformConfig(
        enabled=True, extra={"phone_number_id": "1234567890", "access_token": "***"}))
    cards = []

    async def post(_url, **kwargs):
        payload = kwargs["json"]
        if payload["type"] != "interactive":  # the full-command message ahead of a shortened preview
            if not full_command_ok:
                raise ConnectionError("full command lost")
            return SimpleNamespace(status_code=200, json=lambda: {"messages": [{"id": "wamid.full"}]})
        interactive = payload["interactive"]
        cards.append((interactive["body"]["text"],
                      [{"approve": "once"}.get(c, c) for c in
                       (b["reply"]["id"].rsplit(":", 1)[1] for b in interactive["action"]["buttons"])]))
        _at_card(record)
        return SimpleNamespace(status_code=200, json=lambda: {"messages": [{"id": "wamid.card"}]})

    adapter._http_client = SimpleNamespace(post=post)
    return adapter, cards


def _discord(record, full_command_ok):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._allowed_user_ids = {"111111111111111111"}
    cards = []

    async def channel_send(**kwargs):
        view = kwargs.get("view")
        if view is None:  # the full-command message ahead of a shortened preview
            if not full_command_ok:
                raise ConnectionError("full command lost")
            return SimpleNamespace(id=1)
        removed = view.__dict__.get("_removed_items", [])  # recorded by the remove_item patch below
        cards.append((kwargs["content"], [c for c in ("once", "deny") if c == "deny" or "allow_once" not in removed]))
        _at_card(record)
        return SimpleNamespace(id=1234)

    channel = SimpleNamespace(send=AsyncMock(side_effect=channel_send))
    adapter._client = SimpleNamespace(get_channel=lambda _chat_id: channel, fetch_channel=AsyncMock())
    return adapter, cards


def _runner(adapter, chat_id="555"):
    def schedule(coro, _label):
        loop = asyncio.new_event_loop()
        try:
            value = loop.run_until_complete(coro)
        finally:
            loop.close()
        return SimpleNamespace(result=lambda timeout=None: value)

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(_status_adapter=adapter, _status_chat_id=chat_id,
                                  _status_thread_metadata=None, session_key=_SESSION)
    runner._close_native_stream_boundary = lambda reason: None
    runner._schedule = schedule
    return runner


def _guard(adapter, command, context=None):
    approval_module.register_gateway_notify(_SESSION, _runner(adapter)._approval_notify_sync)
    is_dangerous, pattern_key, description = detect_dangerous_command(command)
    try:
        with patch.object(approval_module, "detect_dangerous_command",
                          return_value=(is_dangerous, pattern_key, description)):
            return check_all_command_guards(command, "local", approval_context=context)
    finally:
        approval_module.unregister_gateway_notify(_SESSION)


@pytest.fixture(autouse=True)
def _record_discord_removed_buttons(monkeypatch):
    """The test double for ``discord.ui.View`` has no ``remove_item``: record what a Deny-only card drops."""
    from plugins.platforms.discord.adapter import ExecApprovalView

    monkeypatch.setattr(ExecApprovalView, "remove_item",
                        lambda self, item: self.__dict__.setdefault("_removed_items", []).append(item.__name__),
                        raising=False)


def _send_card(make, command, context=None, *, full_command_ok):
    record = []
    adapter, cards = make(record, full_command_ok)
    result = _guard(adapter, command, context)
    assert result["approved"] is False  # denied from the card
    assert len(cards) == 1 and len(record) == 1
    return cards[0], record[0]


# WhatsApp: 1,024-char body; Discord: 2,000-char message. The first command fits the plain card whole
# with little room left; the second cannot be shown whole on either card.
_FITS = {_whatsapp: "rm -rf /tmp/" + "a" * 600, _discord: "rm -rf /tmp/" + "a" * 1650}
_CUT = {_whatsapp: "rm -rf /tmp/" + "a" * 1000, _discord: "rm -rf /tmp/" + "a" * 1900}


@pytest.mark.parametrize("make", [_whatsapp, _discord], ids=["whatsapp", "discord"])
def test_context_never_costs_a_card_its_approve_button(make):
    # The full-command send fails, so a card whose preview context had shortened would go Deny-only.
    plain, plain_state = _send_card(make, _FITS[make], full_command_ok=False)
    card, state = _send_card(make, _FITS[make], _LONG_CONTEXT, full_command_ok=False)
    assert plain_state == state == {"deny_only": False}  # Approve is live on both cards
    assert card[1] == plain[1] == ["once", "deny"]
    assert _FITS[make] in card[0]  # the whole command, as on the plain card
    assert _HEAD in card[0]  # precondition: the context had to be fitted beside it
    if _HEAD in card[0]:
        assert _TAIL in card[0][card[0].index(_HEAD):]


@pytest.mark.parametrize("make", [_whatsapp, _discord], ids=["whatsapp", "discord"])
def test_context_never_lifts_a_deny_only_card(make):
    plain, plain_state = _send_card(make, _CUT[make], full_command_ok=False)
    card, state = _send_card(make, _CUT[make], _LONG_CONTEXT, full_command_ok=False)
    assert plain_state == state == {"deny_only": True, "approve_tap": 0}  # an approve tap settles nothing
    assert card == plain  # no context squeezed beside a shortened preview
    assert card[1] == ["deny"]


# ── Text prompts on adapters whose send cuts long text (ntfy 4,096 chars, LINE 4,096, Photon 8,000).

def _ntfy():
    a = load_plugin_adapter("ntfy").NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t"}))
    wire = []

    async def post(url, content=None, headers=None, timeout=None):
        wire.append(content.decode())
        return SimpleNamespace(status_code=200, headers={}, json=lambda: {"id": "x"}, text="")
    a._http_client = SimpleNamespace(post=post)
    return a, wire


def _line():
    a = load_plugin_adapter("line").LineAdapter(PlatformConfig(enabled=True, extra={}))
    wire = []

    async def send_messages(chat_id, messages, **_k):
        wire.append("\n".join(m["text"] for m in messages))
        return SimpleNamespace(success=True, message_id="m")
    a._client, a._send_messages = object(), send_messages
    return a, wire


def _photon():
    from plugins.platforms.photon.adapter import PhotonAdapter
    a = PhotonAdapter(PlatformConfig(enabled=True, extra={}))
    wire = []

    async def post_send(path, body, **_k):
        wire.append(body["text"])
        return SimpleNamespace(success=True, message_id="m")
    a._post_send = post_send
    return a, wire


_SCANNER = "recursive delete"


@pytest.fixture()
def pending():
    entry = _ApprovalEntry({"request_id": "r1", "command": "x", "description": _SCANNER})
    entry.card_deny_only = True  # born Deny-only, as _await_gateway_decision queues it
    with approval_module._lock:
        approval_module._gateway_queues[_SESSION] = [entry]
    yield entry
    with approval_module._lock:
        approval_module._gateway_queues.clear()


def _text_prompt(make, command, context=None):
    adapter, wire = make()
    adapter.pause_typing_for_chat = lambda _c: None
    data = {"command": command, "description": _SCANNER, "request_id": "r1"}
    if context:
        data.update(explanation=context, description=_build_enhanced_description_with_context(_SCANNER, context))
    runner = _runner(adapter, chat_id="chat")
    runner._ctx._run_still_current = lambda: True
    runner._render_approval_sync(data)
    return adapter, wire[-1]


_CUTTING = pytest.mark.parametrize("make", [_ntfy, _line, _photon], ids=["ntfy", "line", "photon"])


@_CUTTING
@pytest.mark.parametrize("context", [{"purpose": "clean a temp path", "risk": "files are gone"}, _LONG_CONTEXT],
                         ids=["short", "long"])
def test_cutting_text_prompt_keeps_every_choice_with_context(make, context, pending):
    _adapter, plain = _text_prompt(make, "rm -rf /tmp/x")
    adapter, content = _text_prompt(make, "rm -rf /tmp/x", context)
    assert len(content) <= adapter.max_message_length_for_chat("chat")
    assert "/approve" in content and "/deny" in content
    assert _HEAD in content and _TAIL in content[content.index(_HEAD):]
    block = content[content.index(_HEAD):content.index(_TAIL) + len(_TAIL)]
    assert content.replace("\n\n" + block, "", 1) == plain  # the plain prompt plus one closed annotation
    assert pending.card_deny_only is True  # the text path never lifts the mark; only a card that shows it


@_CUTTING
def test_context_never_turns_a_cutting_text_prompt_deny_only(make, pending):
    adapter, plain = _text_prompt(make, "x")
    command = "rm -rf /tmp/" + "a" * (adapter.max_message_length_for_chat("chat") - len(plain) - 40)
    _adapter, plain = _text_prompt(make, command)
    assert "/approve" in plain  # precondition: every choice fits, with ~40 characters to spare
    _adapter, content = _text_prompt(make, command, _LONG_CONTEXT)
    assert "/approve" in content and command in content  # every choice, and the whole command
    assert len(content) <= adapter.max_message_length_for_chat("chat")
    if _HEAD in content:
        assert _TAIL in content[content.index(_HEAD):]
    else:
        assert content == plain


@_CUTTING
@pytest.mark.parametrize("shape", ["fits-beside-deny", "shortened"])
def test_deny_only_text_prompt_gets_no_context_and_stays_deny_only(make, shape, pending):
    adapter, plain = _text_prompt(make, "x")
    room = adapter.max_message_length_for_chat("chat") - len(plain)
    command = "rm -rf /tmp/" + "a" * (room + 1 if shape == "fits-beside-deny" else 30000)
    _adapter, plain = _text_prompt(make, command)
    assert "/approve" not in plain and "/deny" in plain  # precondition: upstream offers only Deny
    pending.card_deny_only = True
    _adapter, content = _text_prompt(make, command, _LONG_CONTEXT)
    assert content == plain and _HEAD not in content
    assert pending.card_deny_only is True
    assert approval_module.resolve_gateway_approval_card(_SESSION, "once", "r1") == 0


# ── Taps that settle nothing, and the requester binding, with context on the request.

def _notify_then(action):
    def notify(data):
        assert data["explanation"] and _HEAD in data["description"]  # the prompt carries the context
        action(data["request_id"])
    return notify


def test_idless_and_stale_card_taps_settle_nothing_with_context():
    taps = []

    def tap(rid):
        taps.append(approval_module.resolve_gateway_approval_card(_SESSION, "once", None))
        taps.append(approval_module.resolve_gateway_approval_card(_SESSION, "once", "stale-request"))
        approval_module.resolve_gateway_approval_card(_SESSION, "deny", rid)

    approval_module.register_gateway_notify(_SESSION, _notify_then(tap))
    try:
        result = check_all_command_guards("rm -rf /tmp/example", "local", approval_context=_LONG_CONTEXT)
    finally:
        approval_module.unregister_gateway_notify(_SESSION)
    assert taps == [0, 0]
    assert result["approved"] is False


def test_requester_binding_holds_with_context(monkeypatch):
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"mode": "manual", "approvers": []})
    taps = []

    def tap(rid):
        assert approval_module.list_gateway_approval_authority(_SESSION)[0]["requester_id"] == "owner"
        taps.append(approval_module.resolve_gateway_approval_card(_SESSION, "once", rid, actor_id="intruder"))
        taps.append(approval_module.resolve_gateway_approval_card(_SESSION, "deny", rid, actor_id="owner"))

    tokens = set_session_vars(platform="whatsapp", user_id="owner", user_name="Owner", session_key=_SESSION)
    approval_module.register_gateway_notify(_SESSION, _notify_then(tap))
    try:
        result = check_all_command_guards("rm -rf /tmp/example", "local", approval_context=_LONG_CONTEXT)
    finally:
        approval_module.unregister_gateway_notify(_SESSION)
        clear_session_vars(tokens)
    assert taps == [0, 1]  # the intruder's approve settles nothing; the requester's deny settles it
    assert result["approved"] is False


# ── A turn owned by a session authority (hosted rooms, local sessions) publishes the request to its
# viewers after the chat prompt (``_approval_notify_sync``): a failed chat send is not the end of it.

class _Authority:
    def __init__(self):
        self.registered = []

    def check_approval_generation(self, session_id, generation):
        return None

    def register_approval(self, session_id, generation, route, data, *, timeout_s=None):
        self.registered.append((session_id, generation, route, data["request_id"]))


class _ApiLikeAdapter:
    """No native card; ``send`` fails like the API server's (HTTP request/response, not send())."""
    typed_command_prefix = "/"

    def __init__(self, result):
        self.result, self.sent = result, []

    def pause_typing_for_chat(self, chat_id):
        return None

    async def send(self, chat_id, content, metadata=None, **_kw):
        self.sent.append(content)
        return self.result


@pytest.mark.parametrize("owned", [True, False], ids=["owned", "plain"])
@pytest.mark.parametrize("outcome", ["failed", "declined"])
def test_failed_chat_prompt_still_reaches_the_owning_authority(owned, outcome):
    from gateway.platforms.base import SendResult
    from gateway.relay.egress import EGRESS_DECLINE_CODE
    from gateway.run_turn_runner_approval_delivery import ApprovalDeliveryError

    result = (SendResult(success=False, error="API server uses HTTP request/response, not send()")
              if outcome == "failed" else
              SendResult(success=False, error="declined", raw_response={"success": False, "code": EGRESS_DECLINE_CODE}))
    adapter, authority = _ApiLikeAdapter(result), _Authority()
    runner = _runner(adapter)
    if owned:
        runner._approval_owner = (authority, "sess-1", 1)
    data = {"command": "rm -rf /tmp/x", "request_id": "r-owned",
            "explanation": {"purpose": "clean"}, "description": f"recursive delete\n\n{_HEAD}\nPurpose: clean\n{_TAIL}"}
    if owned and outcome == "failed":
        runner._approval_notify_sync(data)  # main's behaviour: logged, then published to the viewers
        assert authority.registered == [("sess-1", 1, _SESSION, "r-owned")]
    else:  # no other surface, or a definitive refusal: fail closed before any observer can authorize
        with pytest.raises(ApprovalDeliveryError):
            runner._approval_notify_sync(data)
        assert authority.registered == []
    assert len(adapter.sent) == 1
