"""Routed sites whose store-mode behaviour no other suite reaches, and the LOCAL contracts the routing
must keep: child-env builders take any ``Mapping`` scope, ``.env`` publication survives a scope that
cannot resolve, and credential mirroring never claims a copy under a store that keeps credentials
elsewhere. Each store test names the bypass it kills (a direct file read, or a no-op refresh)."""

import os
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

import hermes_secret_routing as routing
from tests.conformance.secrets.reference_view import MemoryViewProvider


@pytest.fixture
def store(tmp_path, monkeypatch):
    """The in-memory store selected for a launch home laid out like a real one (``~/.hermes``)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY", "RVP_LOCAL"):
        monkeypatch.delenv(name, raising=False)
    provider = MemoryViewProvider()
    routing.reset_for_tests()
    monkeypatch.setitem(routing._PROVIDERS, "memview", provider)
    monkeypatch.setenv("HERMES_SECRET_STORE", "memview")
    yield provider, home
    routing.reset_for_tests()


class _BareMapping(Mapping):
    """``set_secret_scope``'s input contract: a Mapping, with no ``copy``."""

    def __getitem__(self, name):
        if name == "RV_KEY":
            return "scoped"
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return iter(("RV_KEY",))

    def __len__(self):
        return 1


def test_child_env_builders_take_any_mapping_scope_with_the_local_store(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_SECRET_STORE", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent import secret_scope
    from tools.environments import local
    monkeypatch.setattr(secret_scope, "is_multiplex_active", lambda: True)
    with secret_scope.set_secret_scope(_BareMapping(), profile_home=tmp_path):
        served = local.served_profile_child_env({"PATH": "/bin"}, inherit_credentials=True)
        built = local.build_subprocess_env({"PATH": "/bin"}, scrub_secrets=False, strip_launch_profile=True)
    assert served.get("RV_KEY") == "scoped"
    assert built.get("RV_KEY") == "scoped"


def test_a_live_scope_materialises_once_through_its_own_copy(tmp_path, monkeypatch):
    """A store's live view is copied atomically at a spawn boundary, never read key by key."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent import secret_scope
    from tools.environments import local

    class Spy(secret_scope.LiveSecretScope):
        copies = 0

        def copy(self):
            type(self).copies += 1
            return super().copy()

        def __getitem__(self, name):
            raise AssertionError(f"per-key read of {name} from a live view at a spawn boundary")
    monkeypatch.setattr(secret_scope, "is_multiplex_active", lambda: True)
    with secret_scope.set_secret_scope(Spy(({"RV_KEY": "live"},)), profile_home=tmp_path):
        served = local.served_profile_child_env({"PATH": "/bin"}, inherit_credentials=True)
        built = local.build_subprocess_env({"PATH": "/bin"}, scrub_secrets=False, strip_launch_profile=True)
    assert served.get("RV_KEY") == built.get("RV_KEY") == "live"
    assert Spy.copies == 2


def test_a_scope_that_cannot_resolve_still_publishes_the_write_to_the_process_env(monkeypatch):
    monkeypatch.delenv("HERMES_SECRET_STORE", raising=False)
    monkeypatch.delenv("RV_PUBLISHED", raising=False)
    from agent import secret_scope
    from hermes_cli.config_env_publication import publish_env_value

    def broken():
        raise RuntimeError("scope resolver down")
    monkeypatch.setattr(secret_scope, "current_secret_scope", broken)
    publish_env_value("RV_PUBLISHED", "v1")
    assert os.environ["RV_PUBLISHED"] == "v1"
    publish_env_value("RV_PUBLISHED", None)
    assert "RV_PUBLISHED" not in os.environ


def _provisioning(monkeypatch, home):
    from hermes_cli import profiles
    (home / "config.yaml").write_text("model:\n  default: m\n  provider: openrouter\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "seed_profile_skills", lambda path, quiet=True, problems=None: {"copied": []})
    monkeypatch.setattr(profiles, "check_alias_collision", lambda name: None)
    monkeypatch.setattr(profiles, "create_wrapper_script", lambda name: Path("/wrapper") / name)


def test_credential_mirroring_under_a_store_copies_nothing_and_says_so(store, monkeypatch):
    provider, home = store
    _provisioning(monkeypatch, home)
    (home / ".env").write_text("VIEW_STORE_TOKEN=boot-only\n", encoding="utf-8")  # a bootstrap line
    (home / "auth.json").write_text('{"version": 1, "providers": {}, "credential_pool": {}}', encoding="utf-8")
    provider.seed_external(home, {"OPENAI_API_KEY": "launch-key"})
    from hermes_cli.profile_provision import (
        CredentialMirrorRefused, ProfileCreateRequest, mirror_launch_credentials, provision_profile)
    result = provision_profile(ProfileCreateRequest(name="work", share_auth=False))
    profile_env = result.path / ".env"
    assert "boot-only" not in (profile_env.read_text(encoding="utf-8") if profile_env.exists() else "")
    assert not (result.path / "auth.json").exists()
    assert result.mirrored["env"] is False and result.mirrored["auth"] is not True
    assert [w for w in result.warnings if "secret store" in w and "no API keys or logins were copied" in w]
    with pytest.raises(CredentialMirrorRefused):
        mirror_launch_credentials(result.path, {})


_VOICE = "stt:\n  provider: local\ntts:\n  provider: edge\nvoice:\n  record_key: ctrl+b\n"


def _profile_config(path: Path) -> dict:
    from hermes_cli.config import read_user_config_raw
    from tui_gateway.launch_profile_policy import profile_config_scope
    with profile_config_scope(path):
        return read_user_config_raw() or {}


def test_a_store_refuses_only_the_credential_copy_not_the_config_sections(store, monkeypatch):
    """Kills the refusal that also skips the non-secret sections: the store copies the same config
    sections as the local store does, and no credential file."""
    _, home = store
    _provisioning(monkeypatch, home)
    with (home / "config.yaml").open("a", encoding="utf-8") as fh:
        fh.write(_VOICE)
    (home / ".env").write_text("VIEW_STORE_TOKEN=boot-only\n", encoding="utf-8")
    (home / "auth.json").write_text('{"version": 1, "providers": {}, "credential_pool": {}}', encoding="utf-8")
    from hermes_cli.profile_provision import ProfileCreateRequest, provision_profile
    stored = provision_profile(ProfileCreateRequest(name="work", share_auth=False))
    monkeypatch.delenv("HERMES_SECRET_STORE")
    routing.reset_for_tests()
    local = provision_profile(ProfileCreateRequest(name="home", share_auth=False))
    assert local.mirrored["voice"] is True and (local.path / "auth.json").exists()  # the local oracle
    assert _profile_config(local.path)["stt"] == {"provider": "local"}
    assert _profile_config(stored.path) == _profile_config(local.path)  # every non-secret section, as local
    assert stored.mirrored["voice"] is True
    assert stored.mirrored["env"] is False and stored.mirrored["auth"] is not True
    assert not (stored.path / "auth.json").exists()
    profile_env = stored.path / ".env"
    assert "boot-only" not in (profile_env.read_text(encoding="utf-8") if profile_env.exists() else "")
    assert [w for w in stored.warnings if "secret store" in w]


def test_a_profile_the_store_serves_auth_for_does_not_read_the_roots(store, monkeypatch):
    """Kills the bypass that tests ``auth.json`` on disk: the store's view is the profile's own."""
    _, home = store
    _provisioning(monkeypatch, home)
    from hermes_cli.profile_provision import ProfileCreateRequest, provision_profile
    result = provision_profile(ProfileCreateRequest(name="work", mirror_credentials=False))
    assert not (result.path / "auth.json").exists()
    assert routing.auth_json_exists(result.path / "auth.json")
    assert result.mirrored["auth"] is False


def test_restart_env_drops_the_launch_bot_token_the_store_supplies(store, monkeypatch):
    """Kills the bypass that re-parses the launch ``.env`` file instead of the store's view."""
    provider, home = store
    (home / ".env").write_text("RVP_LOCAL=1\n", encoding="utf-8")
    provider.seed_external(home, {"TELEGRAM_BOT_TOKEN": "launch-bot", "OPENAI_API_KEY": "launch-key"})
    import gateway.restart
    monkeypatch.setattr(gateway.restart, "is_supervised_gateway_launch", lambda: False)
    from hermes_cli.gateway_restart_env import _launch_bot_token_names
    assert "TELEGRAM_BOT_TOKEN" in _launch_bot_token_names()


def test_a_routed_child_sheds_the_launch_names_the_store_supplies(store, monkeypatch):
    """Kills the bypass that strips only the names in the launch ``.env`` file."""
    provider, home = store
    (home / ".env").write_text("RVP_LOCAL=1\n", encoding="utf-8")
    provider.seed_external(home, {"TELEGRAM_BOT_TOKEN": "launch-bot", "OPENAI_API_KEY": "launch-key"})
    target = home / "profiles" / "b"
    target.mkdir(parents=True)
    from tools.environments import local
    monkeypatch.setattr(local, "_is_routed_home", lambda t: Path(t) == target)
    env = {"PATH": "/bin", "TELEGRAM_BOT_TOKEN": "launch-bot", "OPENAI_API_KEY": "launch-key"}
    local.strip_launch_profile_env(env, target_home=str(target))
    assert "TELEGRAM_BOT_TOKEN" not in env and "OPENAI_API_KEY" not in env
    assert env["PATH"] == "/bin"


def test_a_bridge_reload_reaches_the_stores_bound_launch_scope(store, monkeypatch):
    """Kills a no-op ``follow_launch_bridged_env`` for a store's live launch scope."""
    provider, home = store
    provider.seed_external(home, {"OPENAI_API_KEY": "k"})
    from gateway.bridged_env import BRIDGED_ENV_MARKER_PREFIX as MARKER, bridged_env_digest
    from agent import secret_scope
    from tui_gateway import launch_profile_policy as policy
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "1,2")
    monkeypatch.setenv(MARKER + "TELEGRAM_ALLOWED_USERS", bridged_env_digest("1,2"))
    monkeypatch.setattr(policy, "_snapshot", None)
    monkeypatch.setattr(policy, "_source_writes", getattr(policy, "_source_writes", None))
    policy.capture_launch_env()
    scope = policy.launch_secret_scope(home)
    assert isinstance(scope, secret_scope.LiveSecretScope)
    assert scope.get("TELEGRAM_ALLOWED_USERS") == "1,2"
    token = secret_scope.set_secret_scope(scope)
    try:
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "1")  # the live YAML bridge removed user 2
        monkeypatch.setenv(MARKER + "TELEGRAM_ALLOWED_USERS", bridged_env_digest("1"))
        policy.follow_launch_bridged_env()
        assert secret_scope.current_secret_scope().get("TELEGRAM_ALLOWED_USERS") == "1"
    finally:
        secret_scope.reset_secret_scope(token)


def test_a_store_revocation_is_not_resurrected_from_the_frozen_launch_env(store, monkeypatch):
    """Kills keeping store-owned names in the frozen launch layer. The real loader publishes the
    store's view (registering the names it owns); a launch env frozen with a different, stale value
    for one of them must not answer once the store revokes it."""
    provider, home = store
    import hermes_constants
    from agent import secret_scope
    from hermes_cli.env_loader import load_hermes_dotenv
    from tui_gateway import launch_profile_policy as policy
    monkeypatch.setattr(hermes_constants, "_PINNED_PROCESS_HERMES_HOME", str(home))
    monkeypatch.setenv("RV_REVOKED", "ambient-stale")
    monkeypatch.setattr(policy, "_snapshot", None)
    monkeypatch.setattr(policy, "_source_writes", getattr(policy, "_source_writes", None))
    policy.capture_launch_env()
    provider.seed_external(home, {"RV_REVOKED": "store-current"})
    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    secret_scope.set_multiplex_active(True)
    try:
        scope = policy.launch_secret_scope(home)
        assert scope.get("RV_REVOKED") == "store-current"
        provider.write_view(home, env_unset=("RV_REVOKED",))
        assert scope.get("RV_REVOKED") is None
    finally:
        secret_scope.set_multiplex_active(False)
