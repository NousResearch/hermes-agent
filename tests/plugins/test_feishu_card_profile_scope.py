"""Identity aliases cannot borrow a grant from a different routed profile."""

from pathlib import Path

import pytest

from agent import secret_scope
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore
from gateway.profile_routing import ProfileRoute
from gateway.run import GatewayRunner
from hermes_cli import profiles
from plugins.platforms.feishu.adapter import FeishuAdapter


@pytest.fixture
def mux(tmp_path, monkeypatch):
    home = tmp_path / "hh"
    secondary_home = home / "profiles" / "secondary"
    secondary_home.mkdir(parents=True)
    (home / ".env").write_text("")
    (secondary_home / ".env").write_text("")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("FEISHU_ALLOWED_USERS", "FEISHU_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    served = [("default", home), ("secondary", secondary_home)]
    monkeypatch.setattr(profiles, "profiles_to_serve", lambda **_kwargs: served)
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: home if name == "default" else home / "profiles" / name)
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in {"default", "secondary"})

    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        adapter = FeishuAdapter(PlatformConfig(enabled=True))
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner.config.platforms = {Platform.FEISHU: adapter.config}
        runner._primary_profile_name = "default"
        runner._profile_adapters = {"secondary": {}}
        runner.adapters = {Platform.FEISHU: adapter}
        runner.pairing_store = PairingStore(profile="default")
        secondary = PairingStore(profile="secondary")
        runner.pairing_stores = {"default": runner.pairing_store, "secondary": secondary}
        adapter.gateway_runner = runner
        adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.FEISHU))
        yield adapter, runner, secondary
    finally:
        secret_scope.set_multiplex_active(previous)


@pytest.mark.parametrize("route_by_user", [True, False], ids=["different-profile", "same-profile"])
def test_alias_authorization_stays_in_the_canonical_profile(mux, route_by_user):
    adapter, runner, secondary = mux
    route_match = {"user_id": "u_alice"} if route_by_user else {"chat_id": "oc_dm"}
    runner.config.profile_routes = [ProfileRoute(name="r", platform="feishu", profile="secondary", **route_match)]
    grant_store = runner.pairing_store if route_by_user else secondary
    assert grant_store.approve_code("feishu", grant_store.generate_code("feishu", "ou_alice"))

    def authorized():
        return adapter._is_interactive_operator_authorized("ou_alice", user_id="u_alice", chat_id="oc_dm")

    assert authorized() is (not route_by_user)
    if route_by_user:
        assert secondary.approve_code("feishu", secondary.generate_code("feishu", "u_alice"))
        assert authorized() is True
        assert secondary.revoke("feishu", "u_alice")
    else:
        assert secondary.revoke("feishu", "ou_alice")
    assert authorized() is False


@pytest.mark.parametrize("target_profile", ["default", "secondary"])
def test_secondary_bot_alias_grant_stays_with_its_callback_profile(mux, target_profile):
    adapter, runner, secondary = mux
    adapter.set_owner_profile("secondary")
    runner.adapters = {}
    runner._profile_adapters = {"secondary": {Platform.FEISHU: adapter}}
    runner.config.profile_routes = [ProfileRoute(
        name="r", platform="feishu", profile=target_profile, chat_id="oc_dm", bot_profile="secondary",
    )]
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.FEISHU, profile_name="secondary"))
    assert secondary.approve_code("feishu", secondary.generate_code("feishu", "ou_alice"))
    assert adapter._is_interactive_operator_authorized(
        "ou_alice", user_id="u_alice", chat_id="oc_dm",
    ) is (target_profile == "secondary")
