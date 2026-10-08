"""Manual fallback policy stays explicit across config, CLI setup and bootstrap."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import hermes_yaml as yaml

from hermes_cli.auth import AuthError
from hermes_cli.config import get_config_path, load_config
from hermes_cli.config_effective import load_user_config_effective
from hermes_cli.runtime_provider import resolve_runtime_with_fallback

CHAIN = [{"provider": "anthropic", "model": "backup-model"}]


def write_config(config):
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_policy_reader_is_strict_through_both_surface_loaders():
    from hermes_cli.fallback_config import get_fallback_auto_activate

    for value in (False, "false", "true", 1, None, []):
        write_config({"fallback": {"auto_activate": value, "min_switch_reset_seconds": 120}})
        for loader in (load_config, load_user_config_effective):
            cfg = loader()
            assert get_fallback_auto_activate(cfg) is False
            assert cfg["fallback"]["min_switch_reset_seconds"] == 120
    write_config({"fallback": {"min_switch_reset_seconds": 120}})
    assert get_fallback_auto_activate(load_config()) is True
    assert get_fallback_auto_activate(load_user_config_effective()) is True


def test_shared_bootstrap_refuses_manual_chain_without_resolving_backup(monkeypatch):
    write_config({"fallback_providers": CHAIN, "fallback": {"auto_activate": False}})
    error = AuthError("primary credentials expired")
    resolver = Mock(side_effect=error)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolver)

    with pytest.raises(AuthError) as raised:
        resolve_runtime_with_fallback(load_user_config_effective(), requested="openai-codex")

    assert raised.value is error
    assert resolver.call_count == 1


def test_first_added_chain_defaults_manual_despite_merged_defaults(monkeypatch):
    from hermes_cli import fallback_cmd
    from hermes_cli.config import save_config

    write_config({"model": {"provider": "openai", "default": "primary"},
                  "fallback": {"min_switch_reset_seconds": 120}})

    def pick(args=None):
        cfg = load_config()
        cfg["model"] = {"provider": "anthropic", "default": "backup-model"}
        save_config(cfg)

    monkeypatch.setattr("hermes_cli.main._require_tty", lambda *args: None)
    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", pick)
    fallback_cmd.cmd_fallback_add(SimpleNamespace())

    cfg = load_user_config_effective()
    assert cfg["fallback_providers"] == CHAIN
    assert cfg["fallback"]["auto_activate"] is False
    assert cfg["fallback"]["min_switch_reset_seconds"] == 120
    assert cfg["model"]["provider"] == "openai"


def test_cli_bootstrap_refuses_manual_chain_without_changing_route(monkeypatch):
    import cli

    cfg = {"fallback_providers": CHAIN, "fallback": {"auto_activate": False}}
    write_config(cfg)
    monkeypatch.setattr(cli, "_hermes_home", get_config_path().parent)
    monkeypatch.setattr(cli, "CLI_CONFIG", cli.load_cli_config())
    shell = cli.HermesCLI(model="primary", provider="openai-codex", compact=True, max_turns=1)
    resolver = Mock(side_effect=AuthError("primary credentials expired"))
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolver)

    assert shell._ensure_runtime_credentials() is False
    assert resolver.call_count == 1
    assert (shell.requested_provider, shell.model) == ("openai-codex", "primary")


@pytest.mark.parametrize("settings", [None, [], "automatic", {"auto_activate": "false"}, {"auto_activate": 1}])
def test_invalid_policy_is_reported_and_never_allows_automatic_fallback(settings):
    from hermes_cli.config import validate_config_structure
    from hermes_cli.fallback_config import get_fallback_auto_activate

    cfg = {"fallback": settings}
    assert get_fallback_auto_activate(cfg) is False
    issues = validate_config_structure(cfg)
    assert any("fallback" in issue.message for issue in issues)


@pytest.mark.parametrize("mode", ["on", "off"])
def test_auto_command_persists_explicit_policy_and_preserves_chain_and_wait(mode, capsys):
    import argparse
    from hermes_cli.subcommands.fallback import build_fallback_parser

    write_config({"fallback_providers": CHAIN, "fallback": {"min_switch_reset_seconds": 120}})
    parser = argparse.ArgumentParser()
    build_fallback_parser(parser.add_subparsers())
    args = parser.parse_args(["fallback", "auto", mode])
    args.func(args)
    cfg = load_user_config_effective()
    assert cfg["fallback"]["auto_activate"] is (mode == "on")
    assert cfg["fallback"]["min_switch_reset_seconds"] == 120
    assert cfg["fallback_providers"] == CHAIN
    args = parser.parse_args(["fallback", "list"])
    args.func(args)
    assert ("Mode: automatic" if mode == "on" else "Mode: manual") in capsys.readouterr().out


@pytest.mark.parametrize("initial,expected", [
    ({"fallback": {"auto_activate": True}}, True),
    ({"fallback": {"auto_activate": False}}, False),
    ({"fallback_providers": [{"provider": "zai", "model": "older-backup"}]}, True),
    ({"fallback_model": {"provider": "zai", "model": "older-backup"}}, True),
])
def test_adding_chain_respects_existing_or_explicit_mode(monkeypatch, initial, expected):
    from hermes_cli import fallback_cmd
    from hermes_cli.config import save_config
    from hermes_cli.fallback_config import get_fallback_auto_activate

    write_config({"model": {"provider": "openai", "default": "primary"}, **initial})

    def pick(args=None):
        cfg = load_config()
        cfg["model"] = {"provider": "anthropic", "default": "backup-model"}
        save_config(cfg)

    monkeypatch.setattr("hermes_cli.main._require_tty", lambda *args: None)
    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", pick)
    fallback_cmd.cmd_fallback_add(SimpleNamespace())
    cfg = load_user_config_effective()
    assert get_fallback_auto_activate(cfg) is expected
    assert cfg["fallback_providers"][-1] == CHAIN[0]
    assert "fallback_model" not in cfg


def test_oneshot_bootstrap_refuses_manual_chain_before_agent_construction(monkeypatch):
    from hermes_cli import oneshot

    write_config({"model": {"provider": "openai-codex", "default": "primary"},
                  "fallback_providers": CHAIN, "fallback": {"auto_activate": False}})
    resolver = Mock(side_effect=AuthError("primary credentials expired"))
    constructor = Mock()
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolver)
    monkeypatch.setattr("run_agent.AIAgent", constructor)
    with pytest.raises(AuthError, match="primary credentials expired"):
        oneshot._run_agent("hello")
    assert resolver.call_count == 1
    constructor.assert_not_called()


@pytest.mark.parametrize("headless", [False, True])
def test_cli_constructor_declares_interactivity_even_with_callable_headless_callback(monkeypatch, headless):
    import cli

    write_config({"fallback": {"auto_activate": False}, "fallback_providers": CHAIN})
    monkeypatch.setattr(cli, "_hermes_home", get_config_path().parent)
    monkeypatch.setattr(cli, "CLI_CONFIG", cli.load_cli_config())
    shell = cli.HermesCLI(model="primary", provider="openai", compact=True, max_turns=1)
    shell._single_query_mode = headless
    shell._session_db = SimpleNamespace()
    monkeypatch.setattr(shell, "_install_tool_callbacks", lambda: None)
    monkeypatch.setattr(cli, "_prepare_deferred_agent_startup", lambda: None)
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **kw: None)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "provider": "openai", "api_key": "test-key", "api_mode": "chat_completions",
        "base_url": "https://api.example.invalid/v1"})
    captured = {}

    def construct(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("run_agent.AIAgent", construct)
    assert shell._init_agent()
    assert callable(captured["clarify_callback"])
    assert captured["fallback_auto_activate"] is False
    assert captured["fallback_selection_interactive"] is (not headless)


def _bootstrap_cli(monkeypatch):
    import cli
    write_config({"fallback_providers": CHAIN, "fallback": {"auto_activate": True}})
    monkeypatch.setattr(cli, "_hermes_home", get_config_path().parent)
    monkeypatch.setattr(cli, "CLI_CONFIG", cli.load_cli_config())
    shell = cli.HermesCLI(model="primary", provider="openai", compact=True, max_turns=1)
    monkeypatch.setattr(shell, "_maybe_print_free_tier_available_notice", lambda: None)
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs["requested"])
        if kwargs["requested"] == "openai":
            raise AuthError("primary unavailable")
        return {"provider": kwargs["requested"], "api_key": "test-key", "api_mode": "chat_completions",
                "base_url": "https://backup.example.invalid/v1"}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    assert shell._ensure_runtime_credentials()
    assert shell.requested_provider == "anthropic"
    return shell, calls


def test_manual_edit_expires_pre_agent_automatic_switch(monkeypatch):
    shell, calls = _bootstrap_cli(monkeypatch)
    old_agent = SimpleNamespace(release_clients=Mock(), _fallback_chain=CHAIN,
                               _fallback_activated=False, _fallback_auto_activate=True)
    shell.agent = old_agent
    write_config({"fallback_providers": CHAIN, "fallback": {"auto_activate": False}})
    calls.clear()
    assert shell.chat("next turn") is None
    assert calls == ["openai"]
    assert (shell.requested_provider, shell.model) == ("openai", "primary")
    old_agent.release_clients.assert_called_once()
    assert shell.agent is None


@pytest.mark.parametrize("once", [False, True])
def test_explicit_model_choice_clears_bootstrap_provenance_and_once_restores_it(monkeypatch, once):
    shell, calls = _bootstrap_cli(monkeypatch)
    snapshot = shell._snapshot_model_runtime() if once else None
    choice = SimpleNamespace(new_model="chosen-model", target_provider="zai", api_key="chosen-key",
                             base_url="https://chosen.example.invalid/v1", api_mode="chat_completions")
    assert shell._stage_and_swap_model(choice, shell.model)
    assert shell._fallback_bootstrap_primary is None
    if once:
        shell._restore_model_runtime_snapshot(snapshot)
        assert shell._fallback_bootstrap_primary is not None
    shell._fallback_auto_activate = False
    calls.clear()
    assert shell._ensure_runtime_credentials() is (not once)
    assert calls == (["openai"] if once else ["zai"])


def test_successful_new_session_route_reset_clears_bootstrap_provenance(monkeypatch):
    import cli
    from hermes_cli.cli_session_mixin import _reset_model_to_config_default

    shell, calls = _bootstrap_cli(monkeypatch)
    monkeypatch.setitem(cli.CLI_CONFIG, "model", {"default": "new-default", "provider": "zai"})
    choice = SimpleNamespace(success=True, new_model="new-default", target_provider="zai",
                             api_key="chosen-key", base_url="https://chosen.example.invalid/v1",
                             api_mode="chat_completions")
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: choice)
    _reset_model_to_config_default(shell, silent=True)
    assert shell._fallback_bootstrap_primary is None
    shell._fallback_auto_activate = False
    calls.clear()
    assert shell._ensure_runtime_credentials()
    assert calls == ["zai"]
    assert shell.model == "new-default"
