"""Store-owned launch values cannot resurrect stale inherited or reserved entries."""

from agent import secret_scope as ss
import hermes_secret_routing as routing

pytest_plugins = ["tests.agent.test_secret_store_routing"]


def test_reserved_store_names_filtered_in_live_scope(store):
    provider, home = store
    provider.seed_external(home, {"VIEW_STORE_TOKEN": "forbidden", "HERMES_HOME": "forbidden", "PATH": "forbidden",
                                  "OPENAI_API_KEY": "valid-marker"})
    provider.refresh(home)
    scope = ss.build_profile_secret_scope(home)
    assert dict(scope) == {"OPENAI_API_KEY": "valid-marker"}
    assert scope.get("VIEW_STORE_TOKEN") is None
    assert scope.get("HERMES_HOME") is None
    assert scope.get("PATH") is None


def test_revoked_plane_name_never_falls_back_to_stale_launch(store, monkeypatch):
    provider, home = store
    monkeypatch.setenv("OPENAI_API_KEY", "launch-marker")
    provider.seed_external(home, {"OPENAI_API_KEY": "profile-marker"})
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    scope = ss.build_profile_secret_scope(home)
    provider._remote(home)["env"] = {}
    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    from hermes_scope import BoundSecrets, push_state
    with push_state(secrets=BoundSecrets(scope, str(home))):
        assert ss.get_secret("OPENAI_API_KEY") is None
    assert "OPENAI_API_KEY" in routing.published_env_names(home)
