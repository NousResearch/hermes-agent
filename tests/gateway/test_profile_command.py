"""/profile command: list profiles + persisted per-chat binding (salvaged from #24914)."""

from gateway.chat_bindings import ChatBindings, chat_binding_key
from gateway.session import SessionSource
from gateway.platforms.base import Platform


def _src(chat_id="100", thread_id=None):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, thread_id=thread_id, chat_type="group", user_id="u1")


def test_binding_key_is_stable_and_distinct():
    a = chat_binding_key(_src("100"))
    assert chat_binding_key(_src("100")) == a
    assert chat_binding_key(_src("200")) != a
    assert chat_binding_key(_src("100", thread_id="t1")) != a


def test_set_get_persists_across_instances(tmp_path):
    path = tmp_path / "chat_bindings.json"
    key = chat_binding_key(_src())
    ChatBindings(path).set(key, "coder")
    assert ChatBindings(path).get(key) == "coder"  # fresh instance reads from disk


def test_clear_removes_binding(tmp_path):
    path = tmp_path / "chat_bindings.json"
    key = chat_binding_key(_src())
    b = ChatBindings(path)
    b.set(key, "coder")
    b.clear(key)
    assert b.get(key) is None


def test_missing_file_is_empty(tmp_path):
    assert ChatBindings(tmp_path / "nope.json").get("any") is None


import pytest
import gateway.run as gateway_run
from gateway.config import GatewayConfig
from gateway.platforms.base import MessageEvent, MessageType


def _runner(tmp_path):
    r = object.__new__(gateway_run.GatewayRunner)
    r.config = GatewayConfig(sessions_dir=str(tmp_path))
    r._chat_bindings_store = None
    return r


def _cmd_event(text):
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=_src())


@pytest.mark.asyncio
async def test_profile_bind_persists_and_routes(tmp_path, monkeypatch):
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda n: True)
    monkeypatch.setattr(gateway_run, "t", lambda *a, **k: "", raising=False)
    r = _runner(tmp_path)
    reply = await r._handle_profile_command(_cmd_event("/profile coder"))
    assert "coder" in reply
    # Persisted: a fresh binding store sees it.
    from gateway.chat_bindings import ChatBindings, chat_binding_key

    assert ChatBindings(tmp_path / "chat_bindings.json").get(chat_binding_key(_src())) == "coder"


@pytest.mark.asyncio
async def test_profile_unknown_name_is_rejected(tmp_path, monkeypatch):
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda n: False)
    r = _runner(tmp_path)
    reply = await r._handle_profile_command(_cmd_event("/profile ghost"))
    assert "No such profile" in reply
