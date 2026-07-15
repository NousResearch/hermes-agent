from __future__ import annotations

import argparse

import pytest

from hermes_cli.subcommands.providers import build_providers_parser


def _handler(args):  # pragma: no cover - identity only
    return "providers-handler"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_providers_parser(subparsers, cmd_providers=_handler)
    return parser


def test_validate_parser_uses_current_dispatch_shape():
    namespace = _build_parser().parse_args([
        "providers",
        "validate",
        "--provider",
        "custom:local",
        "--model",
        "model-a",
        "--toolsets",
        "file",
        "--suite",
        "agent-readiness",
        "--out",
        "/tmp/receipts",
        "--timeout",
        "7",
    ])

    assert namespace.command == "providers"
    assert namespace.providers_command == "validate"
    assert namespace.provider == "custom:local"
    assert namespace.model == "model-a"
    assert namespace.toolsets == "file"
    assert namespace.timeout == 7.0
    assert namespace.func is _handler


def test_provider_help_labels_compatibility_and_not_qualification(capsys):
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["providers", "validate", "--help"])

    output = capsys.readouterr().out
    assert "tier-0 compatibility smoke" in output
    assert "not qualification or" in output
    assert "replacement evidence" in output
    assert "SessionDB" in output


def test_provider_parser_exposes_paired_evaluation_surface():
    namespace = _build_parser().parse_args([
        "providers",
        "evaluate",
        "--candidate-manifest",
        "candidate.json",
        "--incumbent-manifest",
        "incumbent.json",
        "--evaluation-config",
        "evaluation.yaml",
        "--out",
        "/tmp/run",
        "--execute",
        "--seed",
        "7",
    ])
    assert namespace.providers_command == "evaluate"
    assert namespace.candidate_manifest == "candidate.json"
    assert namespace.incumbent_manifest == "incumbent.json"
    assert namespace.evaluation_config == "evaluation.yaml"
    assert namespace.execute is True
    assert namespace.seed == 7


def test_provider_parser_exposes_offline_score_and_suite_listing():
    score = _build_parser().parse_args(["providers", "score", "--run-dir", "/tmp/run"])
    assert score.providers_command == "score"
    assert score.run_dir == "/tmp/run"
    suites = _build_parser().parse_args(["providers", "suites", "list"])
    assert suites.providers_command == "suites"
    assert suites.suites_command == "list"


def test_provider_parser_restricts_tier0_to_file_toolset():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["providers", "validate", "--toolsets", "terminal"])
