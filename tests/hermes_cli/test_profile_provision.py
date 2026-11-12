from __future__ import annotations

from pathlib import Path

import pytest

from hermes_agent.hermes_cli import profile_provision, profiles
from hermes_agent.hermes_cli.profile_provision import ProfileCreateRequest, provision_profile


@pytest.fixture()
def launch(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("model:\n  default: m\n  provider: openrouter\n")
    (home / ".env").write_text("OPENROUTER_API_KEY=sk-or-x\nTELEGRAM_BOT_TOKEN=1:x\n")
    (home / "auth.json").write_text('{"version": 1, "providers": {}, "credential_pool": {}}')
    monkeypatch.setattr(profiles, "seed_profile_skills", lambda path, quiet=True: {"copied": ["s"]})
    monkeypatch.setattr(profiles, "check_alias_collision", lambda name: None)
    monkeypatch.setattr(profiles, "create_wrapper_script", lambda name: Path("/wrapper") / name)
    return home


def test_default_is_the_bot_page_default_keys_copied_auth_shared(launch):
    """The Desktop bot page's ticked "Share keys & accounts": provider keys copied (bot token
    stripped), auth.json shared not copied."""
    result = provision_profile(ProfileCreateRequest(name="work"))
    assert result.path == launch / "profiles" / "work"
    assert result.skills_seeded == {"copied": ["s"]}
    assert result.alias_path == Path("/wrapper/work")
    env = (result.path / ".env").read_text()
    assert "OPENROUTER_API_KEY=sk-or-x" in env
    assert "TELEGRAM_BOT_TOKEN" not in env
    assert result.mirrored["env"] is True
    assert result.mirrored["auth"] == "shared"
    assert not (result.path / "auth.json").exists()
    assert result.warnings == []


def test_isolated_auth_copies_auth_once(launch):
    """The bot page's box unticked: an isolated snapshot copy."""
    result = provision_profile(ProfileCreateRequest(name="work", share_auth=False))
    assert result.mirrored["auth"] is True
    assert (result.path / "auth.json").exists()


def test_no_credentials_copies_nothing_from_the_launch_profile(launch):
    # An inline endpoint key in the launch config must not ride in on core's model seed either.
    (launch / "config.yaml").write_text("model:\n  default: m\n  provider: lab\n"
                                        "providers:\n  lab:\n    base_url: http://lab\n    api_key: sk-inline\n")
    result = provision_profile(ProfileCreateRequest(name="work", mirror_credentials=False))
    assert "OPENROUTER_API_KEY" not in (result.path / ".env").read_text()
    config = result.path / "config.yaml"
    assert not config.exists() or "sk-inline" not in config.read_text()
    assert not (result.path / "auth.json").exists()
    # No auth.json of its own: it still reads the root's logins, and says so.
    assert result.mirrored == {"env": False, "auth": "shared", "model_inherited": False, "voice": False}


def test_clone_all_brings_its_own_auth_so_it_is_not_reported_shared(launch):
    result = provision_profile(ProfileCreateRequest(name="work", clone_all=True))
    assert (result.path / "auth.json").exists()
    assert result.mirrored["auth"] is False


def test_a_named_launch_profile_copies_its_logins_instead_of_sharing_the_roots(launch, monkeypatch):
    """Sharing means reading the root's auth.json; from launch profile ``b`` that would pair b's keys
    with the root's logins, so b's auth.json is copied."""
    b = launch / "profiles" / "b"
    b.mkdir(parents=True)
    (b / "config.yaml").write_text("model:\n  default: m\n  provider: openrouter\n")
    (b / ".env").write_text("OPENROUTER_API_KEY=sk-or-b\n")
    (b / "auth.json").write_text('{"version": 1, "providers": {"b-only": {}}, "credential_pool": {}}')
    monkeypatch.setenv("HERMES_HOME", str(b))
    result = provision_profile(ProfileCreateRequest(name="child"))
    assert result.path == launch / "profiles" / "child"
    assert "OPENROUTER_API_KEY=sk-or-b" in (result.path / ".env").read_text()
    assert result.mirrored["auth"] is True
    assert "b-only" in (result.path / "auth.json").read_text()
    assert result.warnings == []


def test_a_named_launch_profiles_subscription_login_is_not_copied_and_says_so(launch, monkeypatch):
    b = launch / "profiles" / "b"
    b.mkdir(parents=True)
    (b / "auth.json").write_text('{"version": 1, "providers": {}, "credential_pool": {"anthropic": '
                                 '[{"id": "x", "auth_type": "oauth", "refresh_token": "rt", "access_token": "at"}]}}')
    monkeypatch.setenv("HERMES_HOME", str(b))
    result = provision_profile(ProfileCreateRequest(name="child"))
    assert "anthropic" not in (result.path / "auth.json").read_text()
    assert any("subscription logins not carried over (anthropic)" in w for w in result.warnings)
    from hermes_agent.tui_gateway import server
    rpc = server._methods["profiles.create"](1, {"name": "child2", "no_alias": True})["result"]
    assert any("anthropic" in w for w in rpc["warnings"])


def test_shared_logins_with_no_launch_auth_json_still_report_shared(launch):
    (launch / "auth.json").unlink(missing_ok=True)
    result = provision_profile(ProfileCreateRequest(name="work", share_auth=False))
    assert not (result.path / "auth.json").exists()
    assert result.mirrored["auth"] == "shared"


def test_only_a_fresh_profile_gets_bundled_skills(launch):
    cloned = provision_profile(ProfileCreateRequest(name="a", clone_from="default", clone_config=True))
    full = provision_profile(ProfileCreateRequest(name="b", clone_all=True))
    assert (cloned.skills_seeded, full.skills_seeded) == (None, None)


def test_a_rejected_model_pick_is_reported_and_the_create_still_succeeds(launch, monkeypatch):
    from hermes_agent.hermes_cli.model_switch import ModelSwitchResult
    monkeypatch.setattr("hermes_agent.hermes_cli.model_switch.switch_model",
                        lambda **_kw: ModelSwitchResult(success=False, error_message="Unknown provider 'nope'."))
    result = provision_profile(ProfileCreateRequest(name="work", provider="nope", model="m"))
    assert (result.model_set, result.model_error, result.warnings) == (False, "Unknown provider 'nope'.", [])
    assert result.path.is_dir()


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
    from hermes_agent.hermes_cli import profile_cmd
    profile_cmd._profile_create(Namespace(
        profile_name="cli", clone=False, clone_all=False, no_alias=True, no_skills=False, clone_from=None,
        clone_channels=False, sync_imports=False, description=None))
    import asyncio
    from hermes_agent.hermes_cli.web_models import ProfileCreate
    from hermes_agent.hermes_cli.web_routers.profiles import create_profile_endpoint
    asyncio.run(create_profile_endpoint(ProfileCreate(name="rest")))
    from hermes_agent.tui_gateway import server
    assert server._methods["profiles.create"](1, {"name": "rpc", "no_alias": True})["result"]["ok"]
    assert [r.name for r in seen] == ["cli", "rest", "rpc"]
    # One default everywhere: the bot page's.
    assert [(r.mirror_credentials, r.share_auth) for r in seen] == [(True, True)] * 3


def _cli_args(*argv):
    """Parse ``hermes profile create ...`` with the real parser, so a misnamed flag dest fails here."""
    import argparse
    from hermes_agent.hermes_cli.subcommands.profile import build_profile_parser
    parser = argparse.ArgumentParser()
    build_profile_parser(parser.add_subparsers(), cmd_profile=lambda args: None)
    return parser.parse_args(["profile", "create", *argv])


def test_every_surface_has_both_opt_outs(launch, monkeypatch):
    """CLI flags, dashboard fields and RPC params map onto the same two request fields."""
    import asyncio
    from hermes_agent.hermes_cli import profile_cmd
    from hermes_agent.hermes_cli.web_models import ProfileCreate
    from hermes_agent.hermes_cli.web_routers.profiles import create_profile_endpoint
    from hermes_agent.tui_gateway import server
    seen = []
    real = profile_provision.provision_profile

    def spy(request):
        seen.append(request)
        return real(request)

    monkeypatch.setattr(profile_provision, "provision_profile", spy)
    profile_cmd._profile_create(_cli_args("c1", "--no-alias", "--isolated-auth"))
    profile_cmd._profile_create(_cli_args("c2", "--no-alias", "--no-credentials"))
    asyncio.run(create_profile_endpoint(ProfileCreate(name="r1", share_auth=False)))
    asyncio.run(create_profile_endpoint(ProfileCreate(name="r2", mirror_credentials=False)))
    server._methods["profiles.create"](1, {"name": "t1", "no_alias": True, "share_auth": False})
    server._methods["profiles.create"](1, {"name": "t2", "no_alias": True, "mirror_credentials": False})
    assert [(r.name, r.mirror_credentials, r.share_auth) for r in seen] == [
        ("c1", True, False), ("c2", False, True), ("r1", True, False), ("r2", False, True),
        ("t1", True, False), ("t2", False, True)]


def test_a_clone_reaches_the_selected_memory_providers_create_hook_once(launch, monkeypatch):
    import hermes_agent.plugins.memory as memory_pkg
    from hermes_agent.agent.memory_provider import MemoryProvider
    from hermes_agent.hermes_constants import get_hermes_home

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
    result = provision_profile(ProfileCreateRequest(name="work", clone_config=True))
    assert calls == [("work", str(result.path), str(result.path))]
    assert result.warnings == []


def test_the_cli_does_not_say_no_keys_when_it_copied_them(launch, capsys):
    from hermes_agent.hermes_cli import profile_cmd
    profile_cmd._profile_create(_cli_args("keys", "--no-alias"))
    out = capsys.readouterr().out
    assert "Copied API keys from 'default'" in out
    assert "shared with the default profile" in out
    assert "no API keys yet" not in out
    profile_cmd._profile_create(_cli_args("bare", "--no-alias", "--no-credentials"))
    out = capsys.readouterr().out
    assert "no API keys yet" in out
    assert "--isolated-auth copies" not in out  # the hint only where it would change something
    # --clone-from without --clone clones the source's .env, so it has keys too.
    profile_cmd._profile_create(_cli_args("cf", "--no-alias", "--clone-from", "keys"))
    assert "no API keys yet" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        profile_cmd._profile_create(_cli_args("cn", "--no-alias", "--clone", "--no-credentials"))
    assert "cannot be combined" in capsys.readouterr().err
    assert not (launch / "profiles" / "cn").exists()



def test_validating_in_another_home_runs_under_multiplex_semantics(launch, monkeypatch):
    from hermes_agent.agent.secret_scope import is_multiplex_active
    from hermes_agent.hermes_cli.model_switch import ModelSwitchResult
    other = launch / "profiles" / "other"
    other.mkdir(parents=True)
    (other / "config.yaml").write_text("{}\n", encoding="utf-8")
    seen: list[bool] = []

    def switch(**_kw):
        seen.append(is_multiplex_active())
        return ModelSwitchResult(success=False, error_message="no")

    monkeypatch.setattr("hermes_agent.hermes_cli.model_switch.switch_model", switch)
    provision_profile(ProfileCreateRequest(name="a", provider="p", model="m", model_validate_in=launch))
    provision_profile(ProfileCreateRequest(name="b", provider="p", model="m", model_validate_in=other))
    assert (seen, is_multiplex_active()) == ([False, True], False)
