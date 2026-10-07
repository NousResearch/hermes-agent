"""The bundled ``project-handover`` skill teaches commands the real ``hermes project state`` CLI accepts.

The skill has no helper script; its contract is the commands it tells the agent to run through
``terminal``. Each one is parsed by the real argparse tree (so a renamed flag fails here, not in a
user's session), every write is attributed ``--by agent``, and resume reads use ``--json``.
"""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

import pytest

import hermes_yaml as yaml
from hermes_cli import projects_cmd

SKILL = Path(__file__).resolve().parents[2] / "skills" / "software-development" / "project-handover" / "SKILL.md"
_FENCE = re.compile(r"^```(?:bash|sh|shell)\n(.*?)^```", re.M | re.S)
_PLACEHOLDER = re.compile(r"<[a-z_-]+>")


def _text() -> str:
    return SKILL.read_text(encoding="utf-8")


def _state_commands() -> list[list[str]]:
    """Every ``hermes project state …`` line inside a shell fence, tokenised, placeholders filled."""
    commands = []
    for block in _FENCE.findall(_text()):
        for line in block.splitlines():
            line = line.split(" #", 1)[0].strip()
            if line.startswith("hermes project state"):
                commands.append(shlex.split(_PLACEHOLDER.sub("demo", line)))
    return commands


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes")
    projects_cmd.build_parser(parser.add_subparsers(dest="command"))
    return parser


def test_frontmatter_names_the_skill_and_credits_the_human():
    fm = yaml.safe_load(_text().split("---", 2)[1])
    assert fm["name"] == "project-handover"
    assert str(fm["author"]).startswith("Praggy (praggybuilds)")


def test_documented_commands_parse_with_the_real_cli():
    commands = _state_commands()
    assert any("--set" in c for c in commands) and any("--json" in c for c in commands)
    parser = _parser()
    for argv in commands:
        args = parser.parse_args(argv[1:])
        assert args.project_action == "state", argv


def test_every_documented_write_is_attributed_to_the_agent():
    writes = [c for c in _state_commands() if "--set" in c]
    assert writes
    for argv in writes:
        assert _parser().parse_args(argv[1:]).by == "agent", " ".join(argv)


@pytest.mark.parametrize("phrase", ["--json", "--by agent", "terminal"])
def test_resume_and_write_guidance_is_present(phrase):
    assert phrase in _text()
