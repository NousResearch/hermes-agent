"""Store-backed profiles use the same dotenv rules as file-backed profiles."""
import os
from collections.abc import Mapping
from types import SimpleNamespace

from tests.agent.test_secret_layer_precedence import profile_store as _profile_store

profile_store = _profile_store


def test_self_reference_uses_launch_baseline(profile_store, monkeypatch):
    home, seed = profile_store
    monkeypatch.setenv("PARITY_SELF_KEY", "launch")
    seed({"PARITY_SELF_KEY": "prefix:${PARITY_SELF_KEY}"})
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    assert os.environ["PARITY_SELF_KEY"] == "prefix:launch"


def test_forward_reference_uses_launch_baseline(profile_store, monkeypatch):
    home, seed = profile_store
    monkeypatch.setenv("PARITY_LATER_KEY", "launch")
    seed({"PARITY_EARLY_KEY": "${PARITY_LATER_KEY}", "PARITY_LATER_KEY": "profile"})
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    assert os.environ["PARITY_EARLY_KEY"] == "launch"


def test_same_view_reload_reasserts_profile_layer(profile_store, monkeypatch):
    home, seed = profile_store
    seed({"PARITY_RELOAD_KEY": "profile"})
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    monkeypatch.setenv("PARITY_RELOAD_KEY", "outside")
    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    assert os.environ["PARITY_RELOAD_KEY"] == "profile"


def test_empty_profile_still_makes_project_fallback(profile_store, tmp_path, monkeypatch):
    home, seed = profile_store
    seed({})
    monkeypatch.setenv("PARITY_PROJECT_KEY", "launch")
    project = tmp_path / "project.env"
    project.write_text("PARITY_PROJECT_KEY=project\n", encoding="utf-8")
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=home, project_env=project, load_external_secrets=False)
    assert os.environ["PARITY_PROJECT_KEY"] == "launch"


def test_launch_lookup_does_not_enumerate_snapshot(profile_store, monkeypatch):
    home, seed = profile_store
    seed({"PROFILE_KEY": "profile"})
    from hermes_cli.env_loader import load_hermes_dotenv
    from tui_gateway.launch_profile_policy import launch_secret_scope
    import hermes_secret_routing as routing
    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    monkeypatch.setenv("PARITY_ENV_ONLY_KEY", "launch")
    scope = launch_secret_scope(home)

    class NoIteration(Mapping):
        def __init__(self, values):
            self.entries = values

        def __getitem__(self, name):
            return self.entries[name]

        def __len__(self):
            return len(self.entries)

        def __iter__(self):
            raise AssertionError("one-key lookup enumerated the snapshot")

    provider = routing.routed_provider()
    if provider is not None:
        snap = provider.current_snapshot(home)
        proxy = SimpleNamespace(env=NoIteration(snap.env), version=snap.version)
        monkeypatch.setattr(provider, "current_snapshot", lambda home: proxy)
    assert scope["PARITY_ENV_ONLY_KEY"] == "launch"
