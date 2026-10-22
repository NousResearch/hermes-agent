"""The bundled ``project-handover`` skill teaches commands the real ``hermes project`` CLI accepts.

The skill has no helper script; its contract is the commands it tells the agent to run through
``terminal``. Each one is parsed by the real argparse tree (so a renamed flag fails here, not in a
user's session), the resume path can reach a project's folders, every write is attributed
``--by agent``, and resume reads use ``--json``.
"""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

import hermes_yaml as yaml
from hermes_cli import projects_cmd

SKILL = Path(__file__).resolve().parents[2] / "skills" / "software-development" / "project-handover" / "SKILL.md"
_FENCE = re.compile(r"^```(?:bash|sh|shell)\n(.*?)^```", re.M | re.S)
_PLACEHOLDER = re.compile(r"<[a-z_-]+>")


def _text() -> str:
    return SKILL.read_text(encoding="utf-8")


def _project_commands() -> list[list[str]]:
    """Every ``hermes project …`` line inside a shell fence, tokenised, placeholders filled."""
    commands = []
    for block in _FENCE.findall(_text()):
        for line in block.splitlines():
            line = line.split(" #", 1)[0].strip()
            if line.startswith("hermes project"):
                commands.append(shlex.split(_PLACEHOLDER.sub("demo", line)))
    return commands


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes")
    projects_cmd.build_parser(parser.add_subparsers(dest="command"))
    return parser


def _parsed(action: str) -> list[argparse.Namespace]:
    parser = _parser()
    return [ns for ns in (parser.parse_args(argv[1:]) for argv in _project_commands())
            if ns.project_action == action]


def test_frontmatter_names_the_skill_and_credits_the_human():
    fm = yaml.safe_load(_text().split("---", 2)[1])
    assert fm["name"] == "project-handover"
    assert str(fm["author"]).startswith("Praggy (praggybuilds)")


def test_every_documented_project_command_parses_with_the_real_cli():
    parser = _parser()
    actions = {parser.parse_args(argv[1:]).project_action for argv in _project_commands()}
    # Resolving an unnamed project needs `list` for candidates and `show` for their folders.
    assert {"list", "show", "state"} <= actions


def test_resume_reads_are_json_and_writes_are_attributed_to_the_agent():
    states = _parsed("state")
    assert any(ns.json and not ns.set_state for ns in states)
    writes = [ns for ns in states if ns.set_state]
    assert writes
    assert all(ns.by == "agent" for ns in writes)
