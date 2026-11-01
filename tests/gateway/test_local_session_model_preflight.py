"""Model-less local launches must refuse before persisting an unrestorable receipt (#144027)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    home = tmp_path / "profile-home"
    home.mkdir()
    (home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(run, "_hermes_home", home)
    store = SessionStore(home / "sessions", GatewayConfig())
    runner = SimpleNamespace(
        session_store=store, _session_db=store._db, adapters={}, _draining=False,
        _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None,
        _evict_cached_agent=lambda route: None)
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id="model-preflight")
    connection = AuthorityConnection(authority, object(), {"user_id": "owner"})
    try:
        yield home, authority, connection.actor
    finally:
        await connection.close()
        authority.db.close()


def _write_config(home, config):
    path = home / "config.yaml"
    path.write_text(json.dumps({"platform_toolsets": {"cli": []}, **config}), encoding="utf-8")
    return path.read_text(encoding="utf-8")


def _params(home):
    return {"request_id": "oneshot", "source": "oneshot", "cwd": str(home), "toolsets": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [
    {"model": {"streaming": True}}, {"model": {"default": ""}}, {"model": ""}, {},
    {"model": {"default": "  "}}, {"model": "  "},
], ids=["streaming-only", "empty-default", "empty-scalar", "missing-model", "blank-default", "blank-scalar"])
async def test_missing_model_refuses_before_any_local_session_is_committed(owner, config):
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import local_identity
    from hermes_cli.anon_auth import free_inference_active
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_runtime import RuntimeStoreError

    home, authority, actor = owner
    original_config = _write_config(home, config)
    assert not free_inference_active()
    params = _params(home)
    sid = local_identity(authority.profile_id, actor.subject, params["request_id"])
    with pytest.raises(RuntimeStoreError) as caught:
        create_local_session(authority, actor, params)
    assert caught.value.reason == "invalid_params"
    assert caught.value.field == "model"
    assert "model.default" in caught.value.detail and "-m" in caught.value.detail
    assert authority.db.get_session(sid) is None
    with authority.db._read_ctx() as conn:
        assert conn.execute("SELECT 1 FROM state_meta WHERE key=?", (POLICY_PREFIX + sid,)).fetchone() is None
        assert conn.execute("SELECT 1 FROM gateway_routing").fetchone() is None
    assert not authority.sessions
    assert (home / "config.yaml").read_text(encoding="utf-8") == original_config


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["configured", "explicit", "welcome-trial", "exact-retry"])
async def test_resolvable_model_and_exact_retry_keep_restorable_frozen_policy(owner, route):
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import restore_local_session
    from gateway.session_policy import restore_policy
    from hermes_cli.anon_auth import GUEST_MODEL, WELCOME_CLAIM_FLAG, free_inference_active
    from hermes_state_local import local_receipt

    home, authority, actor = owner
    configured = route in {"configured", "exact-retry"}
    _write_config(home, {"model": {"default": "configured-model"} if configured else {"streaming": True}})
    params = _params(home)
    if route == "explicit":
        params["model"] = "cli-model"
    elif route == "welcome-trial":
        (home / "auth.json").write_text(json.dumps({"providers": {"nous": {
            "auth_method": "anonymous", WELCOME_CLAIM_FLAG: True,
        }}}), encoding="utf-8")
        assert free_inference_active()
    else:
        assert not free_inference_active()
    ref = create_local_session(authority, actor, dict(params))
    receipt = local_receipt(authority.db, ref.session_id)
    expected = {"configured": "configured-model", "explicit": "cli-model",
                "welcome-trial": GUEST_MODEL, "exact-retry": "configured-model"}[route]
    assert restore_policy(receipt["policy"]).model == expected
    assert authority.db.get_session(ref.session_id)["model"] == expected
    assert authority.db.get_session(ref.session_id)["source"] == "oneshot"
    assert restore_local_session(authority, ref.session_id) == ref
    if route == "exact-retry":
        _write_config(home, {"model": {"streaming": True}})
        assert create_local_session(authority, actor, dict(params)) == ref
        assert local_receipt(authority.db, ref.session_id) == receipt
        assert restore_policy(local_receipt(authority.db, ref.session_id)["policy"]).model == expected
