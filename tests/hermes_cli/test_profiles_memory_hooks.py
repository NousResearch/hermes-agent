"""Profile create/rename reach the memory provider the profile selects (agent.memory_provider's
on_profile_created / on_profile_renamed), best-effort, with that profile's home and secrets bound."""

from pathlib import Path
from unittest.mock import patch

import pytest

from agent.memory_provider import MemoryProvider


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return default_home


class _Stub(MemoryProvider):
    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []


def _select(home: Path, name: str) -> None:
    (home / "config.yaml").write_text(f"memory:\n  provider: {name}\n", encoding="utf-8")


def test_create_and_rename_call_the_selected_providers_hooks_with_its_home_bound(profile_env, monkeypatch):
    import plugins.memory as memory_pkg
    from hermes_cli.profiles import create_profile, rename_profile
    from hermes_constants import get_hermes_home

    calls = []

    class Recorder(_Stub):
        name = "recorder"

        def on_profile_created(self, profile_name, hermes_home):
            calls.append(("created", profile_name, hermes_home, str(get_hermes_home())))

        def on_profile_renamed(self, old_name, new_name, hermes_home):
            calls.append(("renamed", old_name, new_name, hermes_home, str(get_hermes_home())))

    monkeypatch.setattr(memory_pkg, "load_memory_provider",
                        lambda name, **kw: Recorder() if name == "recorder" else None)
    _select(profile_env, "recorder")
    create_profile("plain", no_alias=True)  # fresh: no memory.provider, so no hook
    created = create_profile("ssi_health", clone_config=True, no_alias=True)  # the clone carries it
    with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"):
        renamed = rename_profile("ssi_health", "heimdall")

    assert calls == [
        ("created", "ssi_health", str(created), str(created)),
        ("renamed", "ssi_health", "heimdall", str(renamed), str(renamed)),
    ]


def test_the_hook_reads_the_new_profiles_secrets_not_the_callers(profile_env, monkeypatch):
    import plugins.memory as memory_pkg
    from agent.secret_scope import get_secret
    from hermes_cli.profiles import create_profile

    seen = []

    class Reader(_Stub):
        name = "reader"

        def on_profile_created(self, profile_name, hermes_home):
            seen.append(get_secret("PROBE_PROFILE_KEY", "unset"))

    monkeypatch.setattr(memory_pkg, "load_memory_provider", lambda name, **kw: Reader())
    monkeypatch.setenv("PROBE_PROFILE_KEY", "caller-value")
    _select(profile_env, "reader")
    (profile_env / ".env").write_text("PROBE_PROFILE_KEY=clone-value\n", encoding="utf-8")
    create_profile("work", clone_config=True, no_alias=True)  # --clone copies the .env
    assert seen == ["clone-value"]


def test_the_hook_follows_the_effective_selection_including_env_refs(profile_env, monkeypatch):
    """``memory.provider: ${VAR}`` resolves through the new profile's own .env, like at runtime."""
    import plugins.memory as memory_pkg
    from hermes_cli.profiles import create_profile

    loaded = []
    monkeypatch.setattr(memory_pkg, "load_memory_provider", lambda name, **kw: loaded.append(name))
    (profile_env / "config.yaml").write_text("memory:\n  provider: ${PROBE_MEMORY_PROVIDER}\n", encoding="utf-8")
    (profile_env / ".env").write_text("PROBE_MEMORY_PROVIDER=recorder\n", encoding="utf-8")
    create_profile("work", clone_config=True, no_alias=True)
    assert loaded == ["recorder"]


def test_an_unreadable_config_backend_never_fails_create(profile_env, monkeypatch, caplog):
    from hermes_cli.config_backend import ConfigBackendUnavailable
    from hermes_cli.profiles import create_profile

    def boom(*_a, **_k):
        raise ConfigBackendUnavailable("backend down")

    import hermes_cli.config as config_mod
    monkeypatch.setattr(config_mod, "load_config_readonly", boom)
    with caplog.at_level("WARNING", logger="hermes_cli.profile_memory_config"):
        assert create_profile("work", no_alias=True).is_dir()
    assert "profile hook failed" in caplog.text


def test_a_failing_provider_hook_never_fails_create_or_rename(profile_env, monkeypatch):
    import plugins.memory as memory_pkg
    from hermes_cli.profiles import create_profile, rename_profile

    class Broken(_Stub):
        name = "broken"

        def on_profile_created(self, profile_name, hermes_home):
            raise RuntimeError("boom")

        def on_profile_renamed(self, old_name, new_name, hermes_home):
            raise RuntimeError("boom")

    monkeypatch.setattr(memory_pkg, "load_memory_provider", lambda name, **kw: Broken())
    _select(profile_env, "broken")
    create_profile("work", clone_config=True, no_alias=True)
    with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"):
        assert rename_profile("work", "play").is_dir()

