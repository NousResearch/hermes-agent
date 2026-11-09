from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import profile_provision, profiles
from hermes_cli.profile_provision import ProfileCreateRequest, provision_profile


@pytest.fixture()
def launch(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("model:\n  default: m\n  provider: openrouter\n")
    (home / ".env").write_text("OPENROUTER_API_KEY=sk-or-x\nTELEGRAM_BOT_TOKEN=1:x\n")
    monkeypatch.setattr(profiles, "seed_profile_skills", lambda path, quiet=True: {"copied": ["s"]})
    monkeypatch.setattr(profiles, "check_alias_collision", lambda name: None)
    monkeypatch.setattr(profiles, "create_wrapper_script", lambda name: Path("/wrapper") / name)
    return home


def test_fresh_profile_gets_skills_and_alias_but_no_launch_keys_by_default(launch):
    result = provision_profile(ProfileCreateRequest(name="work"))
    assert result.path == launch / "profiles" / "work"
    assert result.skills_seeded == {"copied": ["s"]}
    assert result.alias_path == Path("/wrapper/work")
    assert "OPENROUTER_API_KEY" not in (result.path / ".env").read_text()
    assert result.warnings == []


def test_mirror_credentials_copies_provider_keys_and_strips_the_bot_token(launch):
    result = provision_profile(ProfileCreateRequest(name="work", mirror_credentials=True))
    env = (result.path / ".env").read_text()
    assert result.mirrored["env"] is True
    assert "OPENROUTER_API_KEY=sk-or-x" in env
    assert "TELEGRAM_BOT_TOKEN" not in env


def test_share_auth_mirrors_keys_but_not_auth_json(launch):
    (launch / "auth.json").write_text('{"version": 1, "providers": {}}')
    result = provision_profile(ProfileCreateRequest(name="work", mirror_credentials=True, share_auth=True))
    assert result.mirrored["env"] is True
    assert result.mirrored["auth"] == "shared"
    assert not (result.path / "auth.json").exists()


def test_clone_skips_bundled_skills_unless_asked(launch):
    cloned = provision_profile(ProfileCreateRequest(name="a", clone_from="default", clone_config=True))
    assert cloned.skills_seeded is None
    asked = provision_profile(ProfileCreateRequest(name="b", clone_from="default", clone_config=True,
                                                   seed_skills=True))
    assert asked.skills_seeded == {"copied": ["s"]}


def test_alias_collision_is_reported_not_written(launch, monkeypatch):
    monkeypatch.setattr(profiles, "check_alias_collision", lambda name: "taken")
    result = provision_profile(ProfileCreateRequest(name="work"))
    assert (result.alias_path, result.alias_collision) == (None, "taken")


def test_a_failing_step_is_a_warning_and_the_profile_stays(launch, monkeypatch):
    def boom(path, quiet=True):
        raise OSError("disk full")
    monkeypatch.setattr(profiles, "seed_profile_skills", boom)
    result = provision_profile(ProfileCreateRequest(name="work"))
    assert result.path.is_dir()
    assert result.warnings == ["bundled skills: disk full"]
    assert result.alias_path == Path("/wrapper/work")


def test_core_create_errors_propagate(launch):
    provision_profile(ProfileCreateRequest(name="work"))
    with pytest.raises(FileExistsError):
        provision_profile(ProfileCreateRequest(name="work"))


def test_every_surface_creates_through_provision_profile(launch, monkeypatch):
    seen = []
    real = profile_provision.provision_profile

    def spy(request):
        seen.append(request)
        return real(request)

    monkeypatch.setattr(profile_provision, "provision_profile", spy)
    from argparse import Namespace
    from hermes_cli import profile_cmd
    profile_cmd._profile_create(Namespace(
        profile_name="cli", clone=False, clone_all=False, no_alias=True, no_skills=False, clone_from=None,
        clone_channels=False, sync_imports=False, description=None))
    import asyncio
    from hermes_cli.web_models import ProfileCreate
    from hermes_cli.web_routers.profiles import create_profile_endpoint
    asyncio.run(create_profile_endpoint(ProfileCreate(name="rest")))
    from tui_gateway import server
    assert server._methods["profiles.create"](1, {"name": "rpc", "no_alias": True})["result"]["ok"]
    assert [r.name for r in seen] == ["cli", "rest", "rpc"]
    assert [r.mirror_credentials for r in seen] == [False, False, True]


def test_a_clone_reaches_the_selected_memory_providers_create_hook_once(launch, monkeypatch):
    import plugins.memory as memory_pkg
    from agent.memory_provider import MemoryProvider
    from hermes_constants import get_hermes_home

    calls = []

    class Recorder(MemoryProvider):
        name = "recorder"

        def is_available(self):
            return True

        def initialize(self, session_id, **kwargs):
            pass

        def get_tool_schemas(self):
            return []

        def on_profile_created(self, profile_name, hermes_home):
            calls.append((profile_name, hermes_home, str(get_hermes_home())))

    monkeypatch.setattr(memory_pkg, "load_memory_provider", lambda name, **kw: Recorder() if name == "recorder" else None)
    (launch / "config.yaml").write_text("memory:\n  provider: recorder\n")
    provision_profile(ProfileCreateRequest(name="fresh"))
    result = provision_profile(ProfileCreateRequest(name="work", clone_config=True, mirror_credentials=True))
    assert calls == [("work", str(result.path), str(result.path))]
    assert result.warnings == []
