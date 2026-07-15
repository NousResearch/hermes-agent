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
    namespace = _build_parser().parse_args(
        [
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
        ]
    )

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


def test_provider_parser_does_not_expose_future_evaluation_commands():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["providers", "evaluate"])


def test_provider_parser_restricts_tier0_to_file_toolset():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["providers", "validate", "--toolsets", "terminal"])
