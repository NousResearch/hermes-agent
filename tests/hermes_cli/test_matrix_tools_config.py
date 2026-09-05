"""Matrix opt-in through the real configuration and command owners."""
import argparse
import importlib

import pytest

from hermes_cli import tools_config as tc
from hermes_cli.main_agent_cmds import cmd_tools
from hermes_cli.subcommands.tools import build_tools_parser
from hermes_cli.toolset_scope import toolset_allowed_for_platform
from tools import matrix_tool

MATRIX_TOOLSETS = {"matrix", "matrix_admin"}


@pytest.mark.parametrize("config", [{}, {"platform_toolsets": {"matrix": []}}])
def test_matrix_tools_default_off(config):
    assert not MATRIX_TOOLSETS & tc._get_platform_tools(config, "matrix")


def test_matrix_picker_offers_both_toolsets():
    assert MATRIX_TOOLSETS <= tc._checklist_toolset_keys("matrix")


@pytest.mark.parametrize("platform", ["cli", "discord", "telegram", "cron"])
def test_matrix_tools_restricted_even_in_handwritten_config(platform):
    config = {"platform_toolsets": {platform: sorted(MATRIX_TOOLSETS)}}
    assert not MATRIX_TOOLSETS & tc._get_platform_tools(config, platform)
    assert not MATRIX_TOOLSETS & tc._checklist_toolset_keys(platform)
    assert all(not toolset_allowed_for_platform(t, platform) for t in MATRIX_TOOLSETS)


@pytest.mark.parametrize("toolset", sorted(MATRIX_TOOLSETS))
def test_matrix_enable_disable_persists_real_config(tmp_path, monkeypatch, capsys, toolset):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    token = set_hermes_home_override(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    try:
        tc.save_config({"platform_toolsets": {"matrix": [], "cli": ["terminal"]}})
        parser = argparse.ArgumentParser()
        build_tools_parser(parser.add_subparsers(), cmd_tools=cmd_tools)
        args = parser.parse_args(["tools", "enable", toolset, "--platform", "matrix"])
        args.func(args)
        assert "Unknown toolset" not in capsys.readouterr().out
        saved = tc.load_config()
        assert toolset in saved["platform_toolsets"]["matrix"]
        assert MATRIX_TOOLSETS & tc._get_platform_tools(saved, "matrix") == {toolset}
        assert saved["platform_toolsets"]["cli"] == ["terminal"]
        rejected = parser.parse_args(["tools", "enable", toolset])
        rejected.func(rejected)
        assert "not available on platform" in capsys.readouterr().out
        assert tc.load_config()["platform_toolsets"]["cli"] == ["terminal"]
        args.tools_action = "disable"
        args.func(args)
        assert not MATRIX_TOOLSETS & tc._get_platform_tools(tc.load_config(), "matrix")
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("platform", ["", "matrix", "discord"])
@pytest.mark.parametrize("credentials", [False, True])
def test_requirements_do_not_depend_on_session(monkeypatch, platform, credentials):
    monkeypatch.setattr(matrix_tool, "get_session_env", lambda key, default="": platform)
    secrets = {"MATRIX_ACCESS_TOKEN": "fixture", "MATRIX_HOMESERVER": "https://example.org"} if credentials else {}
    monkeypatch.setattr(matrix_tool, "get_secret", lambda key: secrets.get(key))
    assert matrix_tool.check_matrix_tool_requirements() is credentials


@pytest.mark.parametrize("credentials", [False, True])
@pytest.mark.parametrize("platforms", [("matrix", "cli"), ("cli", "matrix")])
def test_cached_requirements_survive_platform_switch(monkeypatch, credentials, platforms):
    owner = importlib.import_module("tools.registry")
    monkeypatch.setattr(owner, "_check_fn_cache", {})
    monkeypatch.setattr(owner, "_check_fn_last_good", {})
    monkeypatch.setattr(owner, "check_fn_cache_scope", lambda: None)
    monkeypatch.setattr(matrix_tool, "get_secret", lambda key: "fixture" if credentials else None)
    for platform in platforms:
        monkeypatch.setattr(matrix_tool, "get_session_env", lambda key, default="": platform)
        assert owner._check_fn_cached(matrix_tool.check_matrix_tool_requirements) is credentials
