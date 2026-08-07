from __future__ import annotations

import argparse
import json

import pytest

import hermes_cli.doctor as doctor_mod
from hermes_cli.subcommands.doctor import build_doctor_parser
from hermes_cli.reliability_doctor import DiagnosticResult


def test_targeted_doctor_human_mode_returns_zero_without_failures(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor_mod,
        "diagnose_skill",
        lambda target: [
            DiagnosticResult(
                "skill", target, "env-present", "GH_TOKEN", "pass", "env_present"
            )
        ],
    )

    rc = doctor_mod.run_doctor(
        argparse.Namespace(
            doctor_target="skill", target="github", all=False, smoke=False, json=False
        )
    )

    assert rc == 0
    assert (
        "PASS skill:github env-present:GH_TOKEN env_present" in capsys.readouterr().out
    )


def test_targeted_doctor_human_mode_returns_one_for_failures(monkeypatch):
    monkeypatch.setattr(
        doctor_mod,
        "diagnose_skill",
        lambda target: [
            DiagnosticResult(
                "skill", target, "env-present", "GH_TOKEN", "fail", "env_missing"
            )
        ],
    )

    rc = doctor_mod.run_doctor(
        argparse.Namespace(
            doctor_target="skill", target="github", all=False, smoke=False, json=False
        )
    )

    assert rc == 1


def test_targeted_doctor_json_mode_is_valid_and_stable(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor_mod,
        "diagnose_all_skills",
        lambda: {
            "github": [
                DiagnosticResult(
                    "skill", "github", "command-exists", "gh", "pass", "command_found"
                )
            ]
        },
    )

    rc = doctor_mod.run_doctor(
        argparse.Namespace(
            doctor_target="skill", target=None, all=True, smoke=False, json=True
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["github"][0]["reason"] == "command_found"
    assert "stdout" not in payload["github"][0]


def test_targeted_doctor_all_json_bounds_and_disambiguates_stored_ids(
    monkeypatch, capsys
):
    first = "x" * 600 + "a"
    second = "x" * 600 + "b"
    monkeypatch.setattr(
        doctor_mod,
        "diagnose_all_crons",
        lambda: {
            first: [DiagnosticResult("cron", first, "cron", first, "pass", "ok")],
            second: [DiagnosticResult("cron", second, "cron", second, "pass", "ok")],
        },
    )

    rc = doctor_mod.run_doctor(
        argparse.Namespace(
            doctor_target="cron", target=None, all=True, smoke=False, json=True
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2
    assert all(len(key) <= 512 for key in payload)
    assert first not in payload
    assert second not in payload


def test_targeted_doctor_ambiguous_cron_names_return_exit_2(monkeypatch):
    monkeypatch.setattr(
        doctor_mod,
        "diagnose_cron",
        lambda target: [
            DiagnosticResult(
                "cron", target, "cron", target, "fail", "cron_ref_ambiguous"
            )
        ],
    )

    rc = doctor_mod.run_doctor(
        argparse.Namespace(
            doctor_target="cron", target="daily", all=False, smoke=False, json=False
        )
    )

    assert rc == 2


def test_targeted_doctor_parser_rejects_removed_smoke_flag():
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_doctor_parser(subparsers, cmd_doctor=lambda args: args)

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["doctor", "cron", "daily", "--smoke"])

    assert exc.value.code == 2


@pytest.mark.parametrize(
    "args",
    [
        argparse.Namespace(
            doctor_target="skill", target=None, all=False, smoke=False, json=False
        ),
        argparse.Namespace(
            doctor_target="cron", target=None, all=False, smoke=False, json=False
        ),
    ],
)
def test_targeted_doctor_requires_target_or_all(args):
    assert doctor_mod.run_doctor(args) == 2
