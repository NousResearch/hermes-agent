"""Named custom provider identity survives gateway sync and canonical resume reads."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from gateway.run_turn_prepare import GatewayTurnPrepareMixin
from hermes_state import SessionDB
from tui_gateway.server import _session_info, _stored_session_runtime_overrides, _runtime_model_config
from gateway.session_local_inventory import agent_inventory


SHARED_URL = "https://relay.example/v1"
CONFIG = {
    "model": {"default": "alpha-model", "provider": "custom:alpha"},
    "custom_providers": [
        {"name": "alpha", "base_url": SHARED_URL, "api_key": "sk-alpha", "model": "alpha-model"},
        {"name": "beta", "base_url": SHARED_URL, "api_key": "sk-beta", "model": "beta-model"},
    ],
}


def _agent(*, model="beta-model", provider="custom", requested_provider="custom:beta"):
    return SimpleNamespace(
        model=model,
        provider=provider,
        requested_provider=requested_provider,
        base_url=SHARED_URL,
        api_mode="chat_completions",
        _fallback_activated=False,
    )


def _runner(db):
    return SimpleNamespace(_session_db=SimpleNamespace(_db=db))


@pytest.mark.parametrize("alpha_model", ["alpha-model", "beta-model"])
def test_named_custom_resolver_sync_row_reader_and_canonical_projection_keep_owner(tmp_path, alpha_model):
    """The selected slug chooses its key even when entries share endpoint and model."""
    from hermes_cli import runtime_provider
    from hermes_cli.config import atomic_config_write, get_config_path

    config = deepcopy(CONFIG)
    config["custom_providers"][0]["model"] = alpha_model
    atomic_config_write(get_config_path(), config)
    runtime = runtime_provider.resolve_runtime_provider(requested="custom:beta", target_model="beta-model")
    assert runtime["provider"] == "custom"
    assert runtime["requested_provider"] == "custom:beta"
    assert runtime["api_key"] == "sk-beta"

    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s1", source="telegram", model="old-model")
        GatewayTurnPrepareMixin._sync_session_model_from_agent(_runner(db), "s1", _agent())

        row = db.get_session("s1")
        route = SessionDB.session_gateway_runtime(row)
        assert route["provider"] == "custom:beta"
        assert route["base_url"] == SHARED_URL
        assert "sk-beta" not in str(row["model_config"])
        assert "api_key" not in str(row["model_config"])

        resume = _stored_session_runtime_overrides(row)
        assert resume["provider_override"] == "custom:beta"
        assert resume["model_override"]["provider"] == "custom:beta"
        assert resume["model_override"]["base_url"] == SHARED_URL
        restored = runtime_provider.resolve_runtime_provider(
            requested=resume["provider_override"], target_model=resume["model_override"]["model"])
        assert restored["api_key"] == "sk-beta"

        # Canonical session.info's provider field is the durable chosen identity, not
        # the resolved adapter class ``custom``.
        assert agent_inventory(_agent())["provider"] == "custom:beta"
    finally:
        db.close()


@pytest.mark.parametrize("requested", ["", "auto", "custom"])
def test_bare_custom_sync_uses_canonical_endpoint_fallback(tmp_path, monkeypatch, requested):
    """Legacy/bare runtime shapes keep the existing URL-based fallback semantics."""
    from hermes_cli import runtime_provider

    monkeypatch.setattr(runtime_provider, "load_config", lambda: CONFIG)
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s1", source="telegram", model="old-model")
        GatewayTurnPrepareMixin._sync_session_model_from_agent(
            _runner(db), "s1", _agent(requested_provider=requested)
        )
        row = db.get_session("s1")
        assert SessionDB.session_gateway_runtime(row)["provider"] == "custom:alpha"
    finally:
        db.close()


def test_tui_runtime_config_and_session_info_keep_requested_custom_slug(monkeypatch):
    from hermes_cli.config import atomic_config_write, get_config_path
    from hermes_constants import get_hermes_home

    atomic_config_write(get_config_path(), CONFIG)
    agent = _agent()
    monkeypatch.setattr("tui_gateway.server._load_approval_mode", lambda: "manual")
    config = _runtime_model_config(agent)
    assert config["provider"] == "custom:beta"
    info = _session_info(agent, {"agent": agent, "session_key": "s1", "profile_home": str(get_hermes_home())})
    assert info["provider"] == "custom:beta"
    assert "sk-beta" not in str(config)

def test_tui_non_custom_adapter_keeps_adapter_identity(monkeypatch):
    from hermes_cli import runtime_provider

    def fail_lookup(**_kwargs):
        raise AssertionError("custom identity lookup must not run for a native adapter")

    monkeypatch.setattr(runtime_provider, "canonical_custom_identity", fail_lookup)
    agent = _agent(provider="anthropic", requested_provider="custom:beta")
    assert _runtime_model_config(agent)["provider"] == "anthropic"


@pytest.mark.asyncio
async def test_canonical_session_info_snapshot_uses_named_custom_identity(tmp_path, monkeypatch):
    from gateway.session_contract import SessionRef
    from gateway.session_local_inventory import snapshot_info
    from tests.gateway.test_surface_admission import _authority

    from hermes_cli.config import atomic_config_write, get_config_path

    agent = _agent()
    authority = await _authority(tmp_path, monkeypatch, lambda _event: "ok")
    atomic_config_write(get_config_path(), CONFIG)
    authority.runner._cached_agent_for = lambda _route: agent
    info = await snapshot_info(authority, SessionRef("p", "s"))
    assert info["provider"] == "custom:beta"


def test_non_custom_adapter_does_not_take_requested_custom_identity(tmp_path, monkeypatch):
    """A native/fallback adapter owns its route even if a stale request label is present."""
    from hermes_cli import runtime_provider

    def fail_lookup(*, base_url=None, config_provider=None, model=None):
        raise AssertionError("custom identity lookup must not run for a native adapter")

    monkeypatch.setattr(runtime_provider, "canonical_custom_identity", fail_lookup)
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s1", source="telegram", model="old-model")
        GatewayTurnPrepareMixin._sync_session_model_from_agent(
            _runner(db), "s1", _agent(provider="anthropic", requested_provider="custom:beta")
        )
        row = db.get_session("s1")
        assert SessionDB.session_gateway_runtime(row)["provider"] == "anthropic"
    finally:
        db.close()


OLD_URL = "https://old.example/v1"
NEW_URL = "https://new.example/v1"


def _edited_entry_agent(*, old_owner=True, old_url=OLD_URL, new_url=NEW_URL):
    """A live session on ``custom:alpha`` whose entry was then moved to another endpoint and key; with
    ``old_owner`` another entry still serves the old endpoint, without it nothing does."""
    from hermes_cli import runtime_provider
    from hermes_cli.config import atomic_config_write, get_config_path

    config = {
        "model": {"default": "m", "provider": "custom:alpha"},
        "custom_providers": [
            {"name": "alpha", "base_url": old_url, "api_key": "sk-old", "model": "m"},
            {"name": "old-owner", "base_url": old_url, "api_key": "sk-old", "model": "m"},
        ][:2 if old_owner else 1],
    }
    atomic_config_write(get_config_path(), config)
    runtime = runtime_provider.resolve_runtime_provider(requested="custom:alpha", target_model="m")
    agent = SimpleNamespace(model="m", provider=runtime["provider"], requested_provider=runtime["requested_provider"],
                            base_url=runtime["base_url"], api_mode=runtime["api_mode"], _fallback_activated=False)
    config["custom_providers"][0].update(base_url=new_url, api_key="sk-new")
    atomic_config_write(get_config_path(), config)
    return agent


def _resumed_route(row):
    from tui_gateway.server import _resolve_agent_model_runtime

    overrides = _stored_session_runtime_overrides(row)
    _, restored = _resolve_agent_model_runtime(overrides["model_override"], overrides.get("provider_override"))
    return restored["base_url"], restored["api_key"]


def test_gateway_sync_does_not_pair_an_edited_entry_key_with_the_old_endpoint(tmp_path):
    agent = _edited_entry_agent()
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s1", source="telegram", model="m")
        GatewayTurnPrepareMixin._sync_session_model_from_agent(_runner(db), "s1", agent)
        assert _resumed_route(db.get_session("s1")) == (OLD_URL, "sk-old")
    finally:
        db.close()


def test_tui_runtime_config_does_not_pair_an_edited_entry_key_with_the_old_endpoint():
    import json

    agent = _edited_entry_agent()
    row = {"model": "m", "model_config": json.dumps(_runtime_model_config(agent))}
    assert _resumed_route(row) == (OLD_URL, "sk-old")


@pytest.mark.parametrize("old_url,new_url", [(OLD_URL, NEW_URL), ("https://relay.example/TenantA/v1", "https://relay.example/tenanta/v1")],
                         ids=["new-host", "path-case-only"])
def test_moved_entry_with_no_old_owner_is_not_saved_beside_the_old_endpoint(tmp_path, old_url, new_url):
    """Neither the requested-entry check nor the endpoint-owner and model fallbacks hand the moved
    entry's name back for an endpoint it left; a path differing only in case is another endpoint."""
    import json

    agent = _edited_entry_agent(old_owner=False, old_url=old_url, new_url=new_url)
    assert _runtime_model_config(agent)["provider"] == "custom"
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s1", source="telegram", model="m")
        GatewayTurnPrepareMixin._sync_session_model_from_agent(_runner(db), "s1", agent)
        assert SessionDB.session_gateway_runtime(db.get_session("s1"))["provider"] == "custom"
        assert _resumed_route(db.get_session("s1")) != (old_url, "sk-new")
    finally:
        db.close()
    row = {"model": "m", "model_config": json.dumps(_runtime_model_config(agent))}
    assert _resumed_route(row) != (old_url, "sk-new")


def test_live_unpersisted_session_reports_the_requested_entry(monkeypatch):
    """A built session reattached before its first turn reports the entry it runs, not the adapter class."""
    from hermes_cli.config import atomic_config_write, get_config_path
    from hermes_constants import get_hermes_home
    from tui_gateway.server import _live_session_identity

    atomic_config_write(get_config_path(), CONFIG)
    session = {"agent": _agent(), "session_key": "not-persisted", "profile_home": str(get_hermes_home())}
    assert _live_session_identity(session) == ("beta-model", "custom:beta")

