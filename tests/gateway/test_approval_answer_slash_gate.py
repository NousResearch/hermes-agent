"""Answering a prompt in a chat counts as running its command (#120506).

An exec-approval answer is ``/approve`` / ``/deny`` and a slash-confirm answer runs the confirmed
command, so with slash admin gating on (``allow_admin_from``) and the default ``approvers: ["*"]``
a non-admin's answer is refused on every path: a bare "always" typed while the turn is busy, a typed
confirm reply, and each adapter's native button. The admin's answer goes through, and a chat without
``allow_admin_from`` lets every admitted member answer, as before.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_agent.gateway.config import Platform, PlatformConfig
from hermes_agent.gateway.platforms.event import MessageEvent, MessageType
from hermes_agent.gateway.session import SessionSource

ADMIN, MEMBER = "111", "222"


@pytest.fixture(autouse=True)
def _clean_state():
    from hermes_agent.tools import approval, slash_confirm

    approval._gateway_queues.clear()
    slash_confirm._pending.clear()
    yield
    approval._gateway_queues.clear()
    slash_confirm._pending.clear()


def _gate(admins):
    return {"allow_admin_from": admins} if admins else {}


def _raise_approval(session_key, platform, requester=ADMIN):
    """Enqueue an approval the way the agent thread does, inside the requester's DM turn."""
    from hermes_agent.gateway.session_context import clear_session_vars, set_session_vars
    from hermes_agent.tools import approval
    from hermes_agent.tools.approval_gateway_wait import _ApprovalEntry

    tokens = set_session_vars(platform=platform, chat_type="dm", user_id=requester, user_name="Req",
                              session_key=session_key)
    try:
        entry = _ApprovalEntry({"command": "rm -rf /srv/data"})
    finally:
        clear_session_vars(tokens)
    entry.card_deny_only = False  # its card showed the whole command
    approval._gateway_queues.setdefault(session_key, []).append(entry)
    return entry


def _event(uid, text, chat_id="c1"):
    return MessageEvent(text=text, message_type=MessageType.TEXT, message_id="m1",
                        source=SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm",
                                             user_id=uid, user_name=f"u{uid}"))


def _runner(extra):
    from tests.gateway.test_plaintext_approval_routing import _make_runner

    runner, _adapter = _make_runner()
    runner.config.platforms[Platform.TELEGRAM].extra.update(extra)
    return runner


# ── one driver per answer path: (setup(extra) -> answer(uid) -> answered?) ─────────────────


def _bare_always(extra):
    runner = _runner(extra)
    key = runner._session_key_for_source(_event(ADMIN, "").source)
    entry = _raise_approval(key, "telegram")

    def answer(uid):
        asyncio.run(runner._handle_active_session_busy_message(_event(uid, "always"), key))
        return entry.result == "always"
    return answer


def _typed_confirm(extra):
    from hermes_agent.tools import slash_confirm

    runner = _runner(extra)
    key = runner._session_key_for_source(_event(ADMIN, "").source)
    ran = []

    async def handler(choice):
        ran.append(choice)
        return "reset done"

    def answer(uid):
        slash_confirm.register(key, "1", "reset", handler, authority={
            "requester_id": ADMIN, "requester_name": "Req", "chat_type": "dm", "approvers": ["*"]})
        asyncio.run(runner._hm_slash_confirm_reply(_event(uid, "/approve"), key))
        return ran == ["once"]
    return answer


def _telegram_button(extra):
    from hermes_agent.plugins.platforms.telegram.adapter import TelegramAdapter

    key = "agent:main:telegram:dm:12345"
    entry = _raise_approval(key, "telegram")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra=extra))
    adapter._bot = AsyncMock()
    adapter._approval_state[5] = key
    adapter._approval_request_ids[5] = entry.data["request_id"]

    def answer(uid):
        query = AsyncMock()
        query.data = "ea:always:5"
        query.message = MagicMock(chat_id=12345, message_thread_id=None)
        query.from_user = SimpleNamespace(id=int(uid), first_name=f"u{uid}")
        asyncio.run(adapter._handle_callback_query(SimpleNamespace(callback_query=query), MagicMock()))
        return entry.result == "always"
    return answer


def _slack_button(extra):
    from tests.gateway.test_slack_approval_buttons import _attach_auth_runner, _make_adapter

    key = "agent:main:slack:dm:C1"
    entry = _raise_approval(key, "slack")
    adapter = _make_adapter()
    adapter.config.extra.update(extra)
    _attach_auth_runner(adapter)
    adapter._approval_resolved["1.2"] = False
    adapter._approval_request_ids["1.2"] = entry.data["request_id"]

    def answer(uid):
        body = {"message": {"ts": "1.2", "blocks": []}, "channel": {"id": "C1"}, "user": {"name": uid, "id": uid}}
        asyncio.run(adapter._handle_approval_action(AsyncMock(), body, {"action_id": "hermes_approve_once", "value": key}))
        return entry.result == "once"
    return answer


def _feishu_card(extra, monkeypatch):
    import tests.gateway.test_feishu_approval_buttons as fx

    key = "agent:main:feishu:dm:oc_1"
    entry = _raise_approval(key, "feishu")
    adapter = fx._make_adapter()
    adapter.config.extra.update(extra)
    adapter._loop = MagicMock(is_closed=MagicMock(return_value=False))
    adapter.set_authorization_check(lambda *_a, **_kw: True)
    adapter._approval_state[1] = {"session_key": key, "message_id": "m", "chat_id": "oc_1",
                                  "request_id": entry.data["request_id"]}
    monkeypatch.setattr(fx.feishu_module, "P2CardActionTriggerResponse", fx._FakeP2Response)
    monkeypatch.setattr(fx.feishu_module, "CallBackCard", fx._FakeCallBackCard)
    monkeypatch.setattr(fx.feishu_module, "CallBackToast", SimpleNamespace, raising=False)

    def submit(coro, _loop):
        asyncio.run(coro)
        return SimpleNamespace(add_done_callback=lambda *_a, **_k: None)

    def answer(uid):
        data = fx._make_card_action_data({"hermes_action": "approve_once", "approval_id": 1}, chat_id="oc_1", open_id=uid)
        with patch("asyncio.run_coroutine_threadsafe", side_effect=submit):
            response = adapter._on_card_action_trigger(data)
        if entry.result is None:  # refused: the card keeps its buttons (Greptile 4504999357)
            assert response.card is None and 1 in adapter._approval_state
        return entry.result == "once"
    return answer


def _whatsapp_cloud_tap(extra, monkeypatch):
    from tests.gateway.test_whatsapp_cloud import _make_adapter, _mock_httpx_response

    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")
    key = "agent:main:whatsapp_cloud:dm:111"
    entry = _raise_approval(key, "whatsapp_cloud")
    adapter = _make_adapter()
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    adapter._http_client = MagicMock()
    adapter._http_client.post = AsyncMock(return_value=_mock_httpx_response(200, {"messages": [{"id": "x"}]}))
    adapter._exec_approval_state["a1"] = key
    adapter._approval_request_ids()["a1"] = entry.data["request_id"]

    def answer(uid):
        raw = {"from": uid, "type": "interactive",
               "interactive": {"type": "button_reply", "button_reply": {"id": "appr:a1:approve", "title": "Approve"}}}
        assert asyncio.run(adapter._dispatch_interactive_reply(raw, {})) is True
        return entry.result is not None
    return answer


_PATHS = {"bare_always": _bare_always, "typed_confirm": _typed_confirm, "telegram_button": _telegram_button,
          "slack_button": _slack_button, "feishu_card": _feishu_card, "whatsapp_cloud_tap": _whatsapp_cloud_tap}


@pytest.mark.parametrize("path", sorted(_PATHS))
@pytest.mark.parametrize(("admins", "member_answers"), [([ADMIN], False), ([], True)], ids=["gated", "ungated"])
def test_a_non_admin_answers_only_where_they_could_type_the_command(path, admins, member_answers, monkeypatch):
    """Gated: the member's answer leaves the prompt pending and the admin's then settles it.
    Ungated (no ``allow_admin_from``): the member answers, as on main."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    driver = _PATHS[path]
    extra = _gate(admins)
    answer = driver(extra, monkeypatch) if driver.__code__.co_argcount == 2 else driver(extra)
    assert answer(MEMBER) is member_answers
    if not member_answers:
        assert answer(ADMIN) is True


@pytest.mark.parametrize(("approvers", "user_cmds", "who", "choice", "allowed"), [
    (["*"], ["approve"], MEMBER, "once", True),     # user_allowed_commands admits the member, as for /approve
    (["*"], ["approve"], MEMBER, "deny", True),     # ...and may refuse what it may approve
    (["*"], ["deny"], MEMBER, "once", False),       # deny alone never approves
    ([], [], MEMBER, "once", True),                 # narrowed approvers wins: the non-admin requester answers
    ([], ["approve"], "333", "once", False),        # ...and nobody else does, whatever user_allowed_commands says
    ([], [], ADMIN, "deny", True),                  # admins always answer
])
def test_narrowed_approvers_take_precedence_over_the_slash_gate(approvers, user_cmds, who, choice, allowed):
    from hermes_agent.gateway.approval_actor import approval_refusal
    from tests.gateway.test_approval_requester_binding import write_approvers

    write_approvers(approvers)
    _raise_approval("k", "telegram", requester=MEMBER)
    extra = {"allow_admin_from": [ADMIN], "user_allowed_commands": user_cmds}
    assert (approval_refusal("k", who, extra=extra, choice=choice) is None) is allowed
    if approvers == ["*"]:  # the typed command passes the same slash gate the button does
        typed = "/deny" if choice == "deny" else "/approve"
        assert (_runner(extra)._check_slash_access(_event(who, typed).source, typed[1:]) is None) is allowed
        from hermes_agent.gateway.slash_access import policy_for_runner_source  # /whoami and /help list what runs
        listed = policy_for_runner_source(_runner(extra), _event(who, typed).source).user_runnable_commands
        assert (typed[1:] in listed or who == ADMIN) is allowed
