import argparse

from hermes_cli.subcommands.wisdom import build_wisdom_parser


def parser():
    value = argparse.ArgumentParser()
    build_wisdom_parser(value.add_subparsers(dest="command"))
    return value


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
