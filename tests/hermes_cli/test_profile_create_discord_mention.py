"""``hermes profile create --discord-mention[=HANDLE]`` opts the new profile in to Discord
``$<profile>`` mentions in its OWN config.yaml (feature request #159776).

The contract with the reader: what the flag writes is exactly what the Discord plugin reads back
from a served profile's config through the gateway's real config loader.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig, load_gateway_config
from hermes_cli import profiles


@pytest.fixture()
def launch(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    (home / "config.yaml").write_text("model:\n  default: m\n  provider: openrouter\n")
    monkeypatch.setattr(profiles, "seed_profile_skills", lambda path, quiet=True, problems=None: {"copied": []})
    return home


def _create(*argv):
    """``hermes profile create ...`` through the real parser and handler."""
    import argparse

    from hermes_cli import profile_cmd
    from hermes_cli.subcommands.profile import build_profile_parser
    parser = argparse.ArgumentParser()
    build_profile_parser(parser.add_subparsers(), cmd_profile=lambda args: None)
    profile_cmd._profile_create(parser.parse_args(["profile", "create", *argv, "--no-alias"]))


def _handles_read_back(monkeypatch, name, home):
    """The ``{handle: profile}`` the Discord plugin computes when a gateway serves *name*."""
    import plugins.platforms.discord.adapter_profile_mentions as profile_mentions
    with monkeypatch.context() as m:
        m.setenv("HERMES_HOME", str(home))
        config = load_gateway_config()
    runner = SimpleNamespace(served_profile_names=lambda: ["default", name], _profile_configs={name: config},
                             _primary_profile_name="default", adapters={}, _profile_adapters={}, config=None)
    bot = SimpleNamespace(platform=Platform.DISCORD, config=PlatformConfig(extra={}), gateway_runner=runner)
    return profile_mentions.mentionable_handles(bot)


def test_the_flag_opts_the_new_profile_in_under_its_name(launch, monkeypatch, capsys):
    _create("agent-one", "--discord-mention")
    home = launch / "profiles" / "agent-one"
    assert "mentionable as $agent-one" in capsys.readouterr().out
    assert _handles_read_back(monkeypatch, "agent-one", home) == {"agent-one": "agent-one"}
    # Only the new profile's own config changed: the launch profile's is untouched.
    assert "discord" not in (launch / "config.yaml").read_text()


def test_a_custom_handle_is_written_with_it(launch, monkeypatch):
    _create("agent-two", "--discord-mention=$A2")
    assert _handles_read_back(monkeypatch, "agent-two", launch / "profiles" / "agent-two") == {"a2": "agent-two"}


def test_without_the_flag_the_profile_stays_unreachable(launch, monkeypatch):
    _create("plain")
    assert _handles_read_back(monkeypatch, "plain", launch / "profiles" / "plain") == {}


@pytest.mark.parametrize("argv, env", [
    (("--discord-mention=not a handle",), {}),
    (("--discord-mention",), {"HERMES_TUI_GATEWAY_URL": "ws://gateway.example:9119"}),
])
def test_a_refused_flag_creates_nothing(launch, monkeypatch, capsys, argv, env):
    """A bad handle, or a create that would land on another machine's gateway (whose config.yaml
    this terminal cannot write), is refused by the flag's own check before anything is made."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(SystemExit):
        _create("refused", *argv)
    assert "Error: --discord-mention" in capsys.readouterr().out
    assert not (launch / "profiles" / "refused").exists()
