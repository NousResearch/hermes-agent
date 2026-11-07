"""An explicit compression-route reset repairs an old chat without replacing its history or main route."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from tests.gateway.test_config_secret_scope import (
    _older_gateway_reference,
    _store_reference,
    _stored_policy,
)
from tests.gateway.test_session_resume_config_refresh import (
    _call,
    _connect,
    _owner_authority,
)


def _config(model="retired", key="old-compression-key"):
    return {
        "model": {
            "provider": "custom",
            "default": "main-model",
            "base_url": "http://127.0.0.1:9/v1",
            "api_key": "main-key",
        },
        "platform_toolsets": {"cli": []},
        "auxiliary": {
            "compression": {"provider": "openrouter", "model": model, "api_key": key},
            "title_generation": {
                "provider": "openrouter",
                "model": "title-model",
                "api_key": "title-key",
            },
        },
    }


def _write(home, config):
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")


async def _session(tmp_path, monkeypatch, legacy):
    from gateway import run
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, "_hermes_home", home)
    _write(home, _config())
    store, authority = await _owner_authority(home)
    owner = _connect(authority)
    sid = (
        await _call(
            owner,
            "session.create",
            request_id="create",
            source="cli",
            cwd=str(tmp_path),
        )
    )["session_id"]
    store._db.append_message(sid, "user", "Keep my original question")
    store._db.append_message(sid, "assistant", "Keep my original answer")
    store._db.update_system_prompt(sid, "Keep the cached system prompt")
    store._db.update_session_tool_names(
        sid, [{"type": "function", "function": {"name": "kept-tool"}}]
    )
    if legacy:
        _store_reference(
            store, sid, _older_gateway_reference(_stored_policy(store, sid))
        )
    return home, store, authority, owner, sid


def _mutation(store, sid, request="reset"):
    row = store._db.get_session(sid)
    return dict(
        session_id=sid,
        request_id=request,
        operation="compression_model",
        payload={},
        expected_revision=row["runtime_revision"],
        expected_generation=row["runtime_generation"],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("automatic", [False, True])
async def test_reset_survives_resume_and_restart_without_changing_history_or_other_credentials(
    tmp_path,
    monkeypatch,
    legacy,
    automatic,
):
    from gateway.session_policy import policy_scope
    from agent.auxiliary_task_config import _get_auxiliary_task_config

    home, store, authority, owner, sid = await _session(tmp_path, monkeypatch, legacy)
    try:
        original = _stored_policy(store, sid)
        messages = store._db.get_messages(sid)
        session = store._db.get_session(sid)
        from agent.context_compressor import ContextCompressor

        compressor = ContextCompressor(
            "main-model", quiet_mode=True, config_context_length=256000
        )
        compressor._summary_model_fallen_back = True
        compressor._previous_summary = "keep the previous summary"
        resident = SimpleNamespace(context_compressor=compressor, model="main-model")
        monkeypatch.setattr(
            authority.runner, "_cached_agent_for", lambda _route: resident
        )
        evicted = []
        monkeypatch.setattr(authority.runner, "_evict_cached_agent", evicted.append)
        current = _config("replacement", "replacement-key")
        if automatic:
            current["auxiliary"].pop("compression")
        _write(home, current)
        # A config edit alone does not move an existing explicit pin.
        assert original.config()["auxiliary"]["compression"]["model"] == "retired"
        params = _mutation(store, sid)
        result = await _call(owner, "session.mutate", **params)
        assert result["operation"] == "compression_model"
        assert not compressor._summary_model_fallen_back
        assert compressor._previous_summary == "keep the previous summary"
        assert not evicted
        # A retry arriving after another turn has used the new route must not clear its fallback.
        compressor._summary_model_fallen_back = True
        assert await _call(owner, "session.mutate", **params) == result
        assert compressor._summary_model_fallen_back
        changed = _stored_policy(store, sid)
        assert {
            k: v
            for k, v in asdict(changed).items()
            if k not in {"config_json", "config_secret_ref"}
        } == {
            k: v
            for k, v in asdict(original).items()
            if k not in {"config_json", "config_secret_ref"}
        }
        assert changed.model == original.model
        assert changed.config(authority)["model"] == original.config(authority)["model"]
        assert (
            changed.config(authority)["auxiliary"]["title_generation"]["api_key"]
            == "title-key"
        )
        assert "replacement-key" not in changed.config_json
        assert store._db.get_messages(sid) == messages
        assert store._db.get_session(sid)["system_prompt"] == session["system_prompt"]
        assert store._db.get_session(sid)["tool_names"] == session["tool_names"]
        await owner.close()
        store.close_all_db_handles()
        store, cold = await _owner_authority(home, "restarted")
        owner = _connect(cold)
        await _call(owner, "session.resume", session_id=sid, omit_messages=True)
        restored = _stored_policy(store, sid)
        with policy_scope(restored, authority=cold):
            route = _get_auxiliary_task_config("compression")
        if automatic:
            assert not route.get("model")
        else:
            assert route["model"] == "replacement"
            assert route["api_key"] == "replacement-key"
        assert restored.config(cold)["model"]["api_key"] == "main-key"
        assert (
            restored.config(cold)["auxiliary"]["title_generation"]["api_key"]
            == "title-key"
        )
        assert store._db.get_messages(sid) == messages
        assert store._db.get_session(sid)["system_prompt"] == session["system_prompt"]
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_reset_preserves_a_rotated_conversations_parent_and_live_segment(
    tmp_path, monkeypatch
):
    home, store, authority, owner, sid = await _session(tmp_path, monkeypatch, False)
    child = sid + "-compressed"
    try:
        assert store._db.try_acquire_compression_lock(sid, "test-compression")
        store._db.publish_compression_child(
            parent_session_id=sid,
            child_session_id=child,
            source=store._db.get_session(sid)["source"],
            messages=[{"role": "user", "content": "The retained compression handoff"}],
            system_prompt="Keep the cached system prompt",
            compression_lock_holder="test-compression",
        )
        store._db.release_compression_lock(sid, "test-compression")
        resumed = await _call(
            owner, "session.resume", session_id=child, omit_messages=True
        )
        assert resumed["session_id"] == sid
        before = {key: store._db.get_messages(key) for key in (sid, child)}
        _write(home, _config("replacement", "replacement-key"))
        await _call(owner, "session.mutate", **_mutation(store, sid))
        await owner.close()
        store.close_all_db_handles()
        store, authority = await _owner_authority(home, "cold")
        owner = _connect(authority)
        resumed = await _call(
            owner, "session.resume", session_id=child, omit_messages=True
        )
        assert resumed["session_id"] == sid
        assert {key: store._db.get_messages(key) for key in (sid, child)} == before
        assert (
            _stored_policy(store, sid).config(authority)["auxiliary"]["compression"][
                "model"
            ]
            == "replacement"
        )
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_reset_refuses_a_stale_or_unfenced_request(tmp_path, monkeypatch):
    _home, store, _authority, owner, sid = await _session(tmp_path, monkeypatch, False)
    try:
        original = _stored_policy(store, sid)
        params = _mutation(store, sid)
        params["expected_revision"] += 1
        reply = await owner.dispatch(
            {"id": 1, "method": "session.mutate", "params": params}
        )
        assert reply["error"]["data"]["reason"] == "revision_conflict"
        params = _mutation(store, sid)
        params.pop("expected_generation")
        reply = await owner.dispatch(
            {"id": 2, "method": "session.mutate", "params": params}
        )
        assert reply["error"]["data"]["reason"] == "invalid_params"
        assert _stored_policy(store, sid) == original
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_reset_refuses_running_work_without_changing_the_policy(
    tmp_path, monkeypatch
):
    import hermes_state_runtime as rt

    _home, store, authority, owner, sid = await _session(tmp_path, monkeypatch, False)
    try:
        original = _stored_policy(store, sid)
        admission = rt.admit_session_input(
            store._db,
            epoch=authority.epoch,
            principal_id="human",
            session_id=sid,
            request_id="busy-turn",
            payload={"text": "running work"},
        )
        rt.claim_session_input(store._db, epoch=authority.epoch, session_id=sid)
        reply = await owner.dispatch(
            {"id": 1, "method": "session.mutate", "params": _mutation(store, sid)}
        )
        assert reply["error"]["message"] == "session_busy", reply
        assert _stored_policy(store, sid) == original
        assert (
            rt.list_session_admissions(store._db, session_id=sid)[0]["admission_id"]
            == admission["admission_id"]
        )
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_reset_reads_the_owning_profile_and_refuses_other_sessions(
    tmp_path, monkeypatch
):
    from gateway.session_contract import SessionRef
    from gateway.session_mutations import mutate_session
    from hermes_state_runtime import RuntimeStoreError

    home, store, authority, owner, sid = await _session(tmp_path, monkeypatch, False)
    try:
        _write(home, _config("own-profile", "own-profile-key"))
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        _write(foreign, _config("wrong-profile", "wrong-profile-key"))
        monkeypatch.setenv("HERMES_HOME", str(foreign))
        await _call(owner, "session.mutate", **_mutation(store, sid))
        assert (
            _stored_policy(store, sid).config(authority)["auxiliary"]["compression"][
                "api_key"
            ]
            == "own-profile-key"
        )
        with pytest.raises(RuntimeStoreError, match="profile_mismatch"):
            await mutate_session(
                authority,
                owner.actor,
                SessionRef(str(foreign), sid),
                _mutation(store, sid, "foreign"),
            )
    finally:
        await owner.close()
        store.close_all_db_handles()
