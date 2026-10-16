"""A stored messaging session remains attachable after its live owner is gone."""

import json
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_attach_restores_cold_persisted_messaging_session(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal, SessionRef

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(run, "_load_gateway_config", lambda: {"platform_toolsets": {"cli": []}})
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _primary_profile_name="default",
                             _resolve_profile_home_for_source=lambda source: tmp_path / source.profile,
                             config=SimpleNamespace(multiplex_profiles=True))
    source = SessionSource(Platform.FEISHU, "group-1", chat_type="group", user_id="member-1", profile="runtime")
    route = store._generate_session_key(source)
    session_id = "cold-feishu"
    store._db.create_session(session_id, source="feishu", session_key=route, chat_id=source.chat_id,
                             chat_type=source.chat_type, user_id=source.user_id,
                             origin_json=json.dumps(source.to_dict()), transport_profile="receiver")
    store._db.append_message(session_id, "assistant", "persisted reply")

    authority = await initialize_session_authority(runner, profile_id="fixture", instance_id="owner")
    actor = Principal("desktop-user", "fixture", frozenset({"session:read"}), "desktop-ws")

    snapshot = await authority.attach(actor, SessionRef("fixture", session_id))

    assert session_id in authority.sessions
    assert authority.sessions[session_id].source.to_dict() == source.to_dict()
    assert store._entries[route].transport_profile == "receiver"
    from gateway.session_identity import identity_of
    assert identity_of(authority.sessions[session_id].source).transport_profile == "receiver"
    assert [message["content"] for message in snapshot.history] == ["persisted reply"]
