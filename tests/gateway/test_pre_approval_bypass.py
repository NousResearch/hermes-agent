"""``pre_approval_bypass`` on the gateway loop: a messaging ``/yolo`` and the session service's yolo control
are refused before anything is written when a plugin says no, a routing entry's stored bypass is not
re-armed for a sender the plugin refuses, and switching approvals back on is never asked."""

from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, SessionStore
from hermes_state_runtime import RuntimeStoreError
from tools.approval import disable_session_yolo, is_session_yolo_enabled

SENTENCE = "Only an admin can turn off approvals here."


@pytest.fixture
def hook(monkeypatch):
    """Records each ask; refuses while ``answer["block"]`` is set (``answer["message"]`` its sentence)."""
    calls, answer = [], {"block": True, "message": SENTENCE}

    async def ainvoke_hook(name, **kwargs):
        if name != "pre_approval_bypass":
            return []
        calls.append(kwargs)
        return [{"action": "block", "message": answer["message"]}] if answer["block"] else []
    monkeypatch.setattr("hermes_cli.plugins.ainvoke_hook", ainvoke_hook)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    yield calls, answer


@pytest.fixture
def runner(tmp_path):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = None
    runner.session_store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    runner.session_store._db = None
    event = MessageEvent(text="/yolo", source=SessionSource(
        platform=Platform.TELEGRAM, user_id="user-7", chat_id="chat-a", user_name="tester", chat_type="dm"))
    key = runner._session_key_for_source(event.source)
    yield runner, event, key
    disable_session_yolo(key)


@pytest.mark.asyncio
async def test_messaging_yolo_is_refused_with_the_plugins_sentence_and_off_is_never_asked(hook, runner):
    calls, answer = hook
    gateway, event, key = runner

    reply = await gateway._handle_yolo_command(event)
    assert getattr(reply, "text", reply) == SENTENCE
    assert not is_session_yolo_enabled(key) and gateway.session_store.get_or_create_session(event.source).yolo is not True
    assert calls == [{"surface": "command", "scope": "session", "session_key": key, "identity": None,
                      "platform": "telegram", "user_id": "user-7", "session_owner": ""}]

    answer["message"] = None  # a block without a sentence: the gateway's own hook-refusal line
    reply = await gateway._handle_yolo_command(event)
    assert "/yolo" in getattr(reply, "text", reply) and not is_session_yolo_enabled(key)

    answer["block"] = False
    await gateway._handle_yolo_command(event)
    assert is_session_yolo_enabled(key)
    answer["block"], asked = True, len(calls)
    await gateway._handle_yolo_command(event)  # back on: never asked
    assert not is_session_yolo_enabled(key) and len(calls) == asked


@pytest.mark.asyncio
async def test_session_service_yolo_is_refused_permission_denied_before_anything_changes(hook, runner):
    from gateway.session_busy_controls import busy_config
    calls, _answer = hook
    gateway, event, key = runner
    authority = SimpleNamespace(authorize=lambda *_a: None, profile_id="p", runner=gateway,
                                sessions={"sid": SimpleNamespace(route=key, source=event.source)})
    connection = SimpleNamespace(actor="a", authority=authority,
                                 _identity={"user_id": "member-7", "provider": "oidc", "profile_scope": "host"})

    with pytest.raises(RuntimeStoreError) as refused:
        await busy_config(connection, SimpleNamespace(session_id="sid"), {"session_id": "sid", "key": "yolo"}, write=True)
    assert (refused.value.reason, refused.value.detail) == ("permission_denied", SENTENCE)
    assert not is_session_yolo_enabled(key) and gateway.session_store.get_or_create_session(event.source).yolo is not True
    assert calls[-1]["surface"] == "session_api" and calls[-1]["identity"] == {"user_id": "member-7", "provider": "oidc"}


@pytest.mark.asyncio
async def test_a_stored_bypass_is_not_re_armed_for_a_sender_the_plugin_refuses(hook, runner, monkeypatch):
    """A turn restores the routing entry's flag before it runs; the restore asks for the turn's sender."""
    import gateway.run_turn_prepare as prepare_module
    calls, answer = hook
    gateway, event, key = runner
    entry = gateway.session_store.get_or_create_session(event.source)
    entry.yolo = True

    async def opened(*_args):
        return False, False

    def stop(*_args, **_kwargs):
        raise LookupError("stop after the yolo restore")
    gateway._hmwa_open_session = opened
    monkeypatch.setattr(prepare_module, "build_session_context", stop)

    with pytest.raises(LookupError):
        await gateway._hmwa_prepare_turn(event, event.source, entry, key, key, 0)
    assert not is_session_yolo_enabled(key)
    assert calls[-1]["surface"] == "restore" and calls[-1]["user_id"] == "user-7"

    disable_session_yolo(key)
    from tools import approval
    approval._launch_yolo_applied.discard(key)  # a fresh boundary: the next turn restores again
    answer["block"] = False
    with pytest.raises(LookupError):
        await gateway._hmwa_prepare_turn(event, event.source, entry, key, key, 0)
    assert is_session_yolo_enabled(key)
