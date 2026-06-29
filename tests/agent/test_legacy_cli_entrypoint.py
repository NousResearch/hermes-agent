from __future__ import annotations

import tomllib
from pathlib import Path

import run_agent


def _raise_if_agent_starts(**_kwargs):
    raise AssertionError("legacy metadata commands must not start the agent")


def test_legacy_cli_version_does_not_start_agent(monkeypatch, capsys):
    monkeypatch.setattr(run_agent, "main", _raise_if_agent_starts)

    assert run_agent.legacy_cli_main(["--version"]) == 0

    output = capsys.readouterr().out
    assert "Hermes Agent v" in output


def test_legacy_cli_help_does_not_start_agent(monkeypatch, capsys):
    monkeypatch.setattr(run_agent, "main", _raise_if_agent_starts)

    assert run_agent.legacy_cli_main(["--help"]) == 0

    output = capsys.readouterr().out
    assert "usage: hermes-agent" in output


def test_legacy_cli_no_args_is_side_effect_free(monkeypatch, capsys):
    monkeypatch.setattr(run_agent, "main", _raise_if_agent_starts)

    assert run_agent.legacy_cli_main([]) == 0

    output = capsys.readouterr().out
    assert "usage: hermes-agent" in output
    assert "No query provided" in output


def test_legacy_cli_query_dispatches_to_existing_runner(monkeypatch):
    calls = {}

    def fake_main(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(run_agent, "main", fake_main)

    assert run_agent.legacy_cli_main(
        [
            "--query",
            "hello from the legacy entry point",
            "--model",
            "provider/model",
            "--max-turns",
            "3",
            "--enabled-toolsets",
            "web,terminal",
            "--verbose",
        ]
    ) == 0

    assert calls == {
        "query": "hello from the legacy entry point",
        "model": "provider/model",
        "api_key": None,
        "base_url": "",
        "max_turns": 3,
        "enabled_toolsets": "web,terminal",
        "disabled_toolsets": None,
        "list_tools": False,
        "save_trajectories": False,
        "save_sample": False,
        "verbose": True,
        "log_prefix_chars": 20,
    }


def test_packaged_hermes_agent_script_uses_safe_wrapper():
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["hermes-agent"] == "run_agent:legacy_cli_main"
