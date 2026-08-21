"""Focused tests for the read-only profile composition doctor report."""

import json
from argparse import ArgumentParser

import pytest

from hermes_cli.subcommands.doctor import build_doctor_parser


def _parser():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_doctor_parser(subparsers, cmd_doctor=lambda args: args)
    return parser


def test_doctor_parser_accepts_profile_report_flags():
    args = _parser().parse_args(
        ["doctor", "--profile", "coder", "--json"]
    )
    assert args.profile == "coder"
    assert args.all_profiles is False
    assert args.json is True


def test_profile_report_is_json_safe_deterministic_and_redacted(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles" / "coder" / "memories").mkdir(parents=True)
    (home / "profiles" / "coder" / "plugins").mkdir()
    (home / "config.yaml").write_text("model:\n  default: root-model\n")
    (home / "profiles" / "coder" / "config.yaml").write_text(
        "model:\n  default: coder-model\n  provider: local\n"
    )
    (home / "profiles" / "coder" / ".env").write_text(
        "OPENAI_API_KEY=super-secret\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.profile_doctor import build_profile_doctor_report

    report = build_profile_doctor_report(all_profiles=True)
    encoded = json.dumps(report, sort_keys=True)
    assert json.loads(encoded) == report
    assert [item["name"] for item in report["profiles"]] == ["default", "coder"]
    coder = report["profiles"][1]
    assert coder["config_present"] is True
    assert coder["model"] == "coder-model"
    assert coder["memory_present"] is True
    assert coder["plugins_present"] is True
    assert coder["gateway_present"] is False
    assert "super-secret" not in encoded
    assert ".env" not in encoded


def test_profile_report_selects_named_profile_and_rejects_unknown(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles" / "coder").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.profile_doctor import build_profile_doctor_report

    report = build_profile_doctor_report(profile="coder")
    assert [item["name"] for item in report["profiles"]] == ["coder"]
    with pytest.raises(ValueError, match="does not exist"):
        build_profile_doctor_report(profile="missing")


def test_profile_report_rejects_ambiguous_scope():
    from hermes_cli.profile_doctor import build_profile_doctor_report

    with pytest.raises(ValueError, match="cannot be used together"):
        build_profile_doctor_report(profile="coder", all_profiles=True)
