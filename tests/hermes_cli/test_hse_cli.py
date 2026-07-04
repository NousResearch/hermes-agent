from __future__ import annotations

import json
import os
import subprocess
from argparse import Namespace
from pathlib import Path

from hermes_cli import hse
from hermes_cli.subcommands.hse import build_hse_parser


def test_hse_status_payload_is_read_only(tmp_path):
    hse_repo = tmp_path / "hse"
    active_repo = tmp_path / "hermes"
    hse_repo.mkdir()
    active_repo.mkdir()
    py = hse_repo / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/usr/bin/env python\n")
    py.chmod(0o755)

    payload = hse.build_status_payload(hse_repo=hse_repo, active_repo=active_repo)

    assert payload["status"] == "ready"
    assert payload["hse_repo"]["path"] == str(hse_repo.resolve())
    assert payload["active_hermes_repo"]["path"] == str(active_repo.resolve())
    assert payload["hse_python"] == str(py.resolve())
    assert payload["boundary"] == {
        "default_read_only": True,
        "active_apply_performed": False,
        "github_write_performed": False,
        "cron_or_gateway_mutation_performed": False,
        "provider_or_model_spend_performed": False,
        "deploy_or_publication_performed": False,
        "arbitrary_module_execution_allowed": False,
    }


def test_hse_module_refuses_non_evolution_module(capsys):
    rc = hse.run_evolution_module(Namespace(module="os", module_args=[]))

    captured = capsys.readouterr()
    assert rc == 2
    assert "Refusing non-evolution module" in captured.err


def test_hse_module_runs_explicit_evolution_module_with_shell_disabled(monkeypatch, tmp_path):
    hse_repo = tmp_path / "hse"
    active_repo = tmp_path / "active"
    hse_repo.mkdir()
    active_repo.mkdir()
    py = hse_repo / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/usr/bin/env python\n")
    py.chmod(0o755)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(hse.subprocess, "run", fake_run)

    rc = hse.run_evolution_module(
        Namespace(
            module="evolution.monitor.strict_unattended_loop",
            module_args=["--", "--help"],
            hse_repo=str(hse_repo),
            active_hermes_repo=str(active_repo),
        )
    )

    assert rc == 0
    command, kwargs = calls[0]
    assert command == [str(py.resolve()), "-m", "evolution.monitor.strict_unattended_loop", "--help"]
    assert kwargs["cwd"] == str(hse_repo.resolve())
    assert kwargs["shell"] is False
    assert kwargs["env"]["HERMES_SELF_EVOLUTION_REPO"] == str(hse_repo.resolve())
    assert kwargs["env"]["HERMES_AGENT_REPO"] == str(active_repo.resolve())
    assert str(hse_repo.resolve()) in kwargs["env"]["PYTHONPATH"].split(os.pathsep)


def test_hse_and_evolve_parsers_share_handler():
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_hse_parser(subparsers, cmd_hse=lambda args: 0)

    hse_args = parser.parse_args(["hse", "status", "--json"])
    evolve_args = parser.parse_args(["evolve", "module", "evolution.foo", "--", "--help"])

    assert hse_args.command == "hse"
    assert hse_args.hse_command == "status"
    assert hse_args.json is True
    assert evolve_args.command == "evolve"
    assert evolve_args.hse_command == "module"
    assert evolve_args.module == "evolution.foo"
    assert evolve_args.module_args == ["--help"]


def test_hse_status_command_outputs_json(tmp_path, capsys):
    hse_repo = tmp_path / "hse"
    active_repo = tmp_path / "active"
    hse_repo.mkdir()
    active_repo.mkdir()
    py = hse_repo / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/usr/bin/env python\n")
    py.chmod(0o755)

    rc = hse.hse_command(
        Namespace(
            hse_command="status",
            hse_repo=str(hse_repo),
            active_hermes_repo=str(active_repo),
            json=True,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ready"
    assert payload["boundary"]["default_read_only"] is True
