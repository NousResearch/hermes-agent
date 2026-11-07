"""``/model reset``: a messaging chat's persisted override is cleared, and a local (TUI/Desktop/CLI) session's
owner pin returns to the profile's configured default, without losing the conversation."""

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


@pytest.mark.asyncio
async def test_owner_reset_returns_a_local_session_to_the_configured_default_read_fresh(monkeypatch):
    """The TUI, Desktop and the attached CLI send ``session.mutate model {model: 'reset'}`` to the owner."""
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin
    from gateway.session_policy import policy_for_source
    from hermes_constants import get_hermes_home
    from tests.gateway.test_follow_profile_followups import _profile
    from tests.gateway.test_session_follow_profile_config import _owner
    import gateway.run as gateway_run

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    _profile(home, "cfg-a", 1, "k")
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    runner._resolve_session_agent_runtime = (
        lambda **kw: GatewayTurnPrepareMixin._resolve_session_agent_runtime(runner, **kw))
    try:
        created = await owner.dispatch({"id": "c", "method": "session.create", "params": dict(
            request_id="c", source="tui", cwd=str(home), toolsets=[])})
        sid = created["result"]["session_id"]
        live = authority.sessions[sid]

        async def pick(model):
            snap = authority.db.get_session(sid)
            response = await owner.dispatch({"id": model, "method": "session.mutate", "params": dict(
                session_id=sid, request_id=f"{model}-{snap['runtime_revision']}", operation="model",
                payload={"model": model}, expected_revision=snap["runtime_revision"],
                expected_generation=snap["runtime_generation"])})
            return response["result"]

        await pick("other-model")
        assert policy_for_source(runner, live.source).model == "other-model"
        _profile(home, "cfg-b", 1, "k")  # edited after launch: reset reads config.yaml now
        await pick("reset")
        assert policy_for_source(runner, live.source).model == "cfg-b"
        revision = authority.db.get_session(sid)["runtime_revision"]
        again = await pick("reset")
        assert again["status"] == "noop" and "cfg-b" in again["note"]  # nothing to reset, nothing written
        assert authority.db.get_session(sid)["runtime_revision"] == revision
    finally:
        await owner.close()
        store.close_all_db_handles()
