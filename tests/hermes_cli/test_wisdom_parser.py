import argparse
from unittest.mock import patch

from hermes_cli.subcommands.wisdom import build_wisdom_parser


def parser():
    value = argparse.ArgumentParser()
    with patch("hermes_wisdom.entitlement.is_entitled", return_value=True):
        build_wisdom_parser(value.add_subparsers(dest="command"))
    return value


def test_real_cli_parser_registers_wisdom_without_plugin_discovery(monkeypatch):
    from hermes_cli import main, plugins
    from hermes_cli.subcommands.wisdom import cmd_wisdom

    def unexpected_discovery():
        raise AssertionError("A built-in Wisdom command must not discover plugins")

    monkeypatch.setattr(main.sys, "argv", ["hermes", "wisdom", "sync", "--json"])
    monkeypatch.setattr("hermes_wisdom.entitlement.is_entitled", lambda: True)
    monkeypatch.setattr(plugins, "discover_plugins", unexpected_discovery)
    value, _subparsers = main._build_cli_parser()
    args = value.parse_args(main.sys.argv[1:])
    assert args.func is cmd_wisdom
    assert args.wisdom_command == "sync"
    assert args.action == "status"


def test_unentitled_parser_help_omits_wisdom(monkeypatch):
    monkeypatch.setattr("hermes_wisdom.entitlement.is_entitled", lambda: False)
    value = argparse.ArgumentParser()
    build_wisdom_parser(value.add_subparsers(dest="command"))
    assert "wisdom" not in value.format_help().lower()


def test_all_foundation_commands_are_registered():
    value = parser()
    commands = {
        "setup": [],
        "status": [],
        "scan": [],
        "suggest": [],
        "candidates": [],
        "review": ["draft"],
        "approve": ["draft"],
        "decline": ["draft"],
        "list": [],
        "show": ["skill"],
        "install": ["skill"],
        "versions": ["skill"],
        "check": [],
        "update": ["skill"],
        "uninstall": ["skill"],
        "notifications": [],
    }
    for command, trailing in commands.items():
        args = value.parse_args(["wisdom", command, *trailing])
        assert args.wisdom_command == command


def test_install_plan_apply_arguments_are_stable():
    value = parser()
    plan = value.parse_args(["wisdom", "install", "skill-1@v2", "--plan", "--json"])
    assert plan.reference == "skill-1@v2"
    assert plan.plan is True
    apply = value.parse_args([
        "wisdom",
        "install",
        "--apply-receipt",
        "wip_123",
        "--accept-partial",
    ])
    assert apply.apply_receipt == "wip_123"


def test_mute_supports_shared_status_and_unmute_without_local_skill_scope():
    value = parser()
    assert value.parse_args(["wisdom", "mute"]).duration == "status"
    for choice in ["status", "1d", "1w", "30d", "forever", "off"]:
        assert value.parse_args(["wisdom", "mute", choice]).duration == choice


def test_sync_defaults_to_read_only_and_requires_explicit_retry(monkeypatch):
    from unittest.mock import Mock
    from hermes_cli.subcommands.wisdom import cmd_wisdom
    from hermes_wisdom import service as module
    service = Mock()
    service.sync_status.return_value = {'can_retry': True}
    service.retry_sync.return_value = {'can_retry': False}
    monkeypatch.setattr(module, 'WisdomService', lambda: service)
    monkeypatch.setattr('hermes_wisdom.entitlement.require_entitlement', lambda: None)
    args = parser().parse_args(['wisdom', 'sync', '--json'])
    assert args.action == 'status'
    assert cmd_wisdom(args) == 0
    service.sync_status.assert_called_once()
    service.retry_sync.assert_not_called()
    assert cmd_wisdom(parser().parse_args(['wisdom', 'sync', 'retry', '--json'])) == 0
    service.retry_sync.assert_called_once()


def test_cli_subcommand_denies_before_service_construction(monkeypatch, capsys):
    from hermes_cli.subcommands.wisdom import cmd_wisdom
    from hermes_wisdom.package import PackagePolicyError

    def denied():
        raise PackagePolicyError("Collective Wisdom unavailable")

    monkeypatch.setattr("hermes_wisdom.entitlement.require_entitlement", denied)
    monkeypatch.setattr(
        "hermes_wisdom.service.WisdomService",
        lambda: (_ for _ in ()).throw(AssertionError("service constructed")),
    )
    assert cmd_wisdom(parser().parse_args(["wisdom", "status", "--json"])) == 6
    assert "Collective Wisdom unavailable" in capsys.readouterr().out


def test_setup_requires_an_explicit_disclosure_switch_for_automation():
    value = parser()
    setup = value.parse_args(["wisdom", "setup", "--accept-disclosure", "--json"])
    assert setup.accept_disclosure is True


def test_update_and_consent_arguments_are_stable():
    value = parser()
    update = value.parse_args([
        "wisdom",
        "update",
        "skill-1",
        "--plan",
        "--accept-sensitive",
        "--preserve-modified",
    ])
    assert update.skill_id == "skill-1"
    assert update.accept_sensitive is True
    assert update.preserve_modified is True
    all_updates = value.parse_args(["wisdom", "update", "--all", "--json"])
    assert all_updates.all is True
    suggest = value.parse_args([
        "wisdom",
        "suggest",
        "skill-1",
        "--description",
        "Owner copy",
        "--system-specification-json",
        '{"hermes":{"minimum_version":"0.17.0"}}',
    ])
    assert suggest.system_specification.startswith("{")
