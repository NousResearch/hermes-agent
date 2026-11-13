"""Profile create/rename reach the memory provider the profile selects (hermes_agent.agent.memory_provider's
on_profile_created / on_profile_renamed), and a provider name is kept free as a profile alias."""

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_agent.agent.memory_provider import MemoryProvider


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
    from hermes_agent.hermes_cli.profiles import create_profile, rename_profile
    from hermes_agent.hermes_constants import get_hermes_home

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
    with patch("hermes_agent.hermes_cli.profiles.check_alias_collision", return_value="skip"):
        renamed = rename_profile("ssi_health", "heimdall")

    assert calls == [
        ("created", "ssi_health", str(created), str(created)),
        ("renamed", "ssi_health", "heimdall", str(renamed), str(renamed)),
    ]


def test_the_hook_reads_the_new_profiles_secrets_not_the_callers(profile_env, monkeypatch):
    import plugins.memory as memory_pkg
    from hermes_agent.agent.secret_scope import get_secret
    from hermes_agent.hermes_cli.profiles import create_profile

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
    from hermes_agent.hermes_cli.profiles import create_profile

    loaded = []
    monkeypatch.setattr(memory_pkg, "load_memory_provider", lambda name, **kw: loaded.append(name))
    (profile_env / "config.yaml").write_text("memory:\n  provider: ${PROBE_MEMORY_PROVIDER}\n", encoding="utf-8")
    (profile_env / ".env").write_text("PROBE_MEMORY_PROVIDER=recorder\n", encoding="utf-8")
    create_profile("work", clone_config=True, no_alias=True)
    assert loaded == ["recorder"]


def test_an_unreadable_config_backend_never_fails_create(profile_env, monkeypatch, caplog):
    from hermes_agent.hermes_cli.config_backend import ConfigBackendUnavailable
    from hermes_agent.hermes_cli.profiles import create_profile

    def boom(*_a, **_k):
        raise ConfigBackendUnavailable("backend down")

    import hermes_agent.hermes_cli.config as config_mod
    monkeypatch.setattr(config_mod, "load_config_readonly", boom)
    _select(profile_env, "recorder")  # an external provider, so the hook reaches the effective-config read
    with caplog.at_level("WARNING", logger="hermes_agent.hermes_cli.profile_memory_config"):
        assert create_profile("work", clone_config=True, no_alias=True).is_dir()
    assert "profile hook failed" in caplog.text


def test_a_missing_profile_secret_never_falls_back_to_the_callers_env(profile_env, monkeypatch):
    import plugins.memory as memory_pkg
    from hermes_agent.agent.secret_scope import get_secret
    from hermes_agent.hermes_cli.profiles import create_profile

    seen = []

    class Reader(_Stub):
        name = "reader"

        def on_profile_created(self, profile_name, hermes_home):
            seen.append(get_secret("PROBE_PROFILE_KEY", "unset"))

    monkeypatch.setattr(memory_pkg, "load_memory_provider", lambda name, **kw: Reader())
    monkeypatch.setenv("PROBE_PROFILE_KEY", "caller-value")
    _select(profile_env, "reader")
    create_profile("work", clone_config=True, no_alias=True)  # the cloned .env has no such key
    assert seen == ["unset"]


def test_a_provider_hook_that_exits_never_fails_create_or_rename(profile_env, monkeypatch):
    import plugins.memory as memory_pkg
    from hermes_agent.hermes_cli.profiles import create_profile, rename_profile
    from hermes_agent.hermes_constants import get_hermes_home

    class Exits(_Stub):
        name = "exits"

        def on_profile_created(self, profile_name, hermes_home):
            calls.append("created")
            raise SystemExit(3)

        def on_profile_renamed(self, old_name, new_name, hermes_home):
            calls.append("renamed")
            raise SystemExit(3)

    home = get_hermes_home()
    calls = []
    monkeypatch.setattr(memory_pkg, "load_memory_provider", lambda name, **kw: Exits())
    _select(profile_env, "exits")
    assert create_profile("work", clone_config=True, no_alias=True).is_dir()
    with patch("hermes_agent.hermes_cli.profiles.check_alias_collision", return_value="skip"):
        assert rename_profile("work", "play").is_dir()
    assert calls == ["created", "renamed"]

    assert get_hermes_home() == home  # the profile home override was reset


def test_a_failing_provider_hook_never_fails_create_or_rename(profile_env, monkeypatch):
    import plugins.memory as memory_pkg
    from hermes_agent.hermes_cli.profiles import create_profile, rename_profile

    calls = []

    class Broken(_Stub):
        name = "broken"

        def on_profile_created(self, profile_name, hermes_home):
            calls.append("created")
            raise RuntimeError("boom")

        def on_profile_renamed(self, old_name, new_name, hermes_home):
            calls.append("renamed")
            raise RuntimeError("boom")

    monkeypatch.setattr(memory_pkg, "load_memory_provider", lambda name, **kw: Broken())
    _select(profile_env, "broken")
    create_profile("work", clone_config=True, no_alias=True)
    with patch("hermes_agent.hermes_cli.profiles.check_alias_collision", return_value="skip"):
        assert rename_profile("work", "play").is_dir()
    assert calls == ["created", "renamed"]


def test_a_profile_without_an_external_provider_binds_no_secret_scope(profile_env, monkeypatch):
    """The common create: nothing selected, so no secret hydration, effective config or provider load."""
    from hermes_agent.agent import secret_scope
    from hermes_agent.hermes_cli.profiles import create_profile

    built = []
    monkeypatch.setattr(secret_scope, "build_profile_secret_scope", lambda *a, **k: built.append(a) or {})
    (profile_env / "config.yaml").write_text("memory:\n  provider: builtin\n", encoding="utf-8")
    assert create_profile("work", clone_config=True, no_alias=True).is_dir()
    assert create_profile("fresh", no_alias=True).is_dir()
    assert built == []  # recorded, not raised: the hook's guard would swallow a raise


def test_the_hook_sees_the_new_profiles_configured_secret_sources(profile_env, monkeypatch):
    import plugins.memory as memory_pkg
    from hermes_agent.hermes_cli import env_loader
    from hermes_agent.hermes_cli.profiles import create_profile

    hydrated = []
    monkeypatch.setattr(env_loader, "hydrate_profile_secret_sources", lambda home: hydrated.append(Path(home)) or {})
    monkeypatch.setattr(memory_pkg, "load_memory_provider", lambda name, **kw: None)
    _select(profile_env, "recorder")
    work = create_profile("work", clone_config=True, no_alias=True)
    assert hydrated == [work]


def test_an_installed_memory_provider_name_is_not_a_free_alias(profile_env, monkeypatch):
    """A memory provider registers ``hermes <name>``; core no longer hardcodes which names those are."""
    import plugins.memory as memory_pkg
    from hermes_agent.hermes_cli.profiles import check_alias_collision

    monkeypatch.setattr(memory_pkg, "list_memory_provider_names", lambda: ["somemem"])
    assert "memory provider" in check_alias_collision("somemem")
