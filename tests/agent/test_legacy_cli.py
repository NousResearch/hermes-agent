"""The packaged ``hermes-agent`` console script honours argv (#54648).

A console script calls its target with no arguments; these tests go through the
target named in pyproject ``[project.scripts]`` exactly the way pip's wrapper does.
"""

from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

import pytest

import run_agent


def _run_console_script(monkeypatch, *argv: str):
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    module, func = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]["hermes-agent"].split(":")
    monkeypatch.setattr(sys, "argv", ["hermes-agent", *argv])
    try:
        return getattr(importlib.import_module(module), func)()
    except SystemExit as exc:
        return exc.code


@pytest.mark.parametrize("argv", [("--help",), ("-h",), ("--version",), ()])
def test_metadata_invocations_never_start_an_agent(argv, monkeypatch, capsys):
    def _no_agent(**_kwargs):
        raise AssertionError("a metadata invocation built an agent")

    monkeypatch.setattr(run_agent, "AIAgent", _no_agent)

    assert _run_console_script(monkeypatch, *argv) in (0, None)
    out = capsys.readouterr().out
    assert "usage: hermes-agent" in out or out.startswith("Hermes Agent v")


def test_query_and_runner_options_reach_the_agent(monkeypatch, capsys):
    seen = {}

    class _Agent:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def run_conversation(self, query):
            seen["query"] = query
            return {"completed": True, "api_calls": 1, "messages": [], "final_response": "ok"}

    monkeypatch.setattr(run_agent, "AIAgent", _Agent)

    assert _run_console_script(monkeypatch, "--query", "hello there", "--max-turns", "3",
                               "--disabled-toolsets", "web") in (0, None)
    assert seen["query"] == "hello there"
    assert seen["init"]["max_iterations"] == 3
    assert seen["init"]["disabled_toolsets"] == ["web"]
