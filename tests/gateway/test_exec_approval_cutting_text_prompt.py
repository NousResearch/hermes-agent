"""The plain-text approval prompt on adapters whose ``send`` cuts long text (ntfy, DingTalk webhook,
Photon, LINE): the answer instructions and the deadline survive the cut, and a command that cannot
be shown whole leaves only Deny, the same rule the button cards follow (#62359)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.run_turn_runner import TurnRunner
from tests.gateway._plugin_adapter_loader import load_plugin_adapter
from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry

SESSION = "agent:main:x:dm:chat"
TAIL = "&& rm -rf ~ # TAIL_MARKER"


def _ntfy():
    a = load_plugin_adapter("ntfy").NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t"}))
    wire = []

    async def post(url, content=None, headers=None, timeout=None):
        wire.append(content.decode())
        return SimpleNamespace(status_code=200, headers={}, json=lambda: {"id": "x"}, text="")
    a._http_client = SimpleNamespace(post=post)
    return a, wire


def _dingtalk():
    a = load_plugin_adapter("dingtalk").DingTalkAdapter(PlatformConfig(enabled=True, extra={}))
    wire = []

    async def post(url, json=None, timeout=None, **_k):
        wire.append(json["markdown"]["text"])
        return SimpleNamespace(status_code=200, text="{}", json=lambda: {"errcode": 0})
    a._http_client = SimpleNamespace(post=post)
    a._get_valid_webhook = lambda _chat: ("https://oapi.dingtalk.com/hook",)
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


def _line():
    a = load_plugin_adapter("line").LineAdapter(PlatformConfig(enabled=True, extra={}))
    wire = []

    async def send_messages(chat_id, messages, **_k):
        wire.append("\n".join(m["text"] for m in messages))
        return SimpleNamespace(success=True, message_id="m")
    a._client, a._send_messages = object(), send_messages
    return a, wire


def _render(adapter, command: str) -> None:
    """Drive the production text-prompt path (``TurnRunner._render_approval_sync``) on ``adapter``."""
    loop = asyncio.new_event_loop()
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(_status_adapter=adapter, _status_chat_id="chat", session_key=SESSION,
                                _status_thread_metadata=None, _run_still_current=lambda: True)
    turn._schedule = lambda coro, _m: SimpleNamespace(result=lambda timeout=None: loop.run_until_complete(coro))
    turn._close_native_stream_boundary = lambda *_a, **_k: True
    try:
        turn._render_approval_sync({"command": command, "description": "recursive delete", "request_id": "r1"})
    finally:
        loop.close()


@pytest.fixture()
def pending():
    entry = _ApprovalEntry({"request_id": "r1", "command": "x"})
    with approval._lock:
        approval._gateway_queues[SESSION] = [entry]
    yield entry
    with approval._lock:
        approval._gateway_queues.clear()


@pytest.mark.parametrize("make", [_ntfy, _dingtalk, _photon, _line], ids=["ntfy", "dingtalk", "photon", "line"])
def test_long_command_keeps_instructions_and_offers_only_deny(make, pending):
    adapter, wire = make()
    adapter.pause_typing_for_chat = lambda _c: None
    command = "echo " + "A" * 30000 + " " + TAIL
    _render(adapter, command)
    delivered = wire[-1]
    assert "/deny" in delivered and "/approve" not in delivered  # instructions survive; Deny alone
    assert "Preview shortened" in delivered and TAIL not in delivered
    assert pending.card_deny_only is True
    assert approval.resolve_gateway_approval_card(SESSION, "once", "r1") == 0  # a crafted approve resolves nothing


@pytest.mark.parametrize("make", [_ntfy, _dingtalk, _photon, _line], ids=["ntfy", "dingtalk", "photon", "line"])
def test_short_command_is_shown_whole_with_every_choice(make, pending):
    adapter, wire = make()
    adapter.pause_typing_for_chat = lambda _c: None
    _render(adapter, "rm -rf /tmp/x " + TAIL)
    delivered = wire[-1]
    assert TAIL in delivered and "/approve" in delivered and "/deny" in delivered
    assert not pending.card_deny_only


@pytest.mark.parametrize("command", ["echo " + "`" * 2100 + " ; rm -rf ~", "`" * 3000, "`" * 5000],
                         ids=["echo-2100-ticks", "3000-ticks", "5000-ticks"])
def test_a_long_backtick_run_never_leaves_the_fence_open_past_the_cut(command, pending):
    """The fence is sized from the shortened command, so the cut can never drop the closing fence."""
    import re

    adapter, wire = _ntfy()
    adapter.pause_typing_for_chat = lambda _c: None
    _render(adapter, command)
    delivered = wire[-1]
    assert len(delivered) <= adapter.max_message_length_for_chat("chat") and "Preview shortened" in delivered
    fence = re.search(r"\n(`{3,})\n", delivered).group(1)
    assert delivered.endswith("\n" + fence)  # the closing fence survived the cut
    shown = delivered[delivered.index("\n" + fence + "\n") + len(fence) + 2:-len(fence) - 1]
    assert max(map(len, re.findall("`+", shown)), default=0) < len(fence)  # and nothing shown closes it early
    assert pending.card_deny_only is True


@pytest.mark.parametrize("make", [_photon, _ntfy, _line], ids=["photon", "ntfy", "line"])
def test_command_that_fits_only_beside_deny_is_shown_whole_and_not_called_shortened(make, pending):
    """One character too long for every choice, short enough for Deny alone: the whole command is
    shown, only Deny is offered, and the notice says why instead of claiming a shortened preview."""
    adapter, wire = make()
    adapter.pause_typing_for_chat = lambda _c: None
    _render(adapter, TAIL)
    room = adapter.max_message_length_for_chat("chat") - len(wire[-1])
    command = "A" * (room + 1) + TAIL
    _render(adapter, command)
    delivered = wire[-1]
    assert "\n" + command in delivered and "/approve" not in delivered and "/deny" in delivered
    assert "Preview shortened" not in delivered and "only Deny is offered" in delivered
    assert pending.card_deny_only is True
