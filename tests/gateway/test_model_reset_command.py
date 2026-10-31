"""Gateway ``/model reset``: a persisted session override is cleared without losing the chat."""

import json
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore, build_session_key


def _event(text: str, source: SessionSource) -> MessageEvent:
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


@pytest.mark.asyncio
async def test_reset_clears_a_persisted_override_after_restart(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    home = tmp_path / "home"
    home.mkdir()
    # JSON is valid YAML; CI ships no pyyaml for test fixtures.
    (home / "config.yaml").write_text(json.dumps({"model": {"default": "cfg-model", "provider": "cfg-prov"}}))
    monkeypatch.setattr(gateway_run, "_hermes_home", home)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1", chat_type="dm")
    key = build_session_key(source)
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    store.get_or_create_session(source)
    store.set_model_override(key, {"model": "pinned-model", "provider": "pinned-prov"})

    # A fresh runner: the override exists only in the store, as after a gateway restart.
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner.session_store = store
    runner._session_db = None
    evicted = []
    runner._evict_cached_agent = lambda k, **_kw: evicted.append(k)
    runner._normalize_source_for_session_key = lambda s: s
    runner._session_key_for_source = lambda s: build_session_key(s)
    runner._pending_one_turn_model_restores[key] = {"had_override": False, "override": None}

    reply = await runner._handle_model_command(_event("/model reset", source))

    assert "cfg-model" in reply
    assert store.get_model_override(key) is None
    assert runner._session_model_overrides.get(key) is None
    assert runner._pending_one_turn_model_restores.get(key) is None  # an armed --once can't re-plant it
    assert evicted == [key]
    assert "pinned-model" in runner._pending_model_notes[key]

    again = await runner._handle_model_command(_event("/model reset", source))
    assert again != reply and "cfg-model" in again  # nothing left to clear: the "none active" copy
    assert evicted == [key]
