"""The sandboxed-prototyping skill only tells the agent to run what `hermes sandbox run` accepts.

The skill is the agent's whole instruction for running untrusted code. A command in it that the
CLI rejects (a flag or `--setup-network` value that does not exist, a renamed option) makes the
agent improvise, and the easiest improvisation is running the code on the host.
"""

import argparse
import re
import shlex
from pathlib import Path

import pytest

from hermes_cli.sandbox_cmd import NO_RUNTIME_EXIT
from hermes_cli.subcommands.sandbox import build_sandbox_parser

SKILL = Path(__file__).resolve().parents[2] / "skills" / "software-development" / "sandboxed-prototyping" / "SKILL.md"
# Inline code spans and terminal("...") examples; the bracketed synopsis line is grammar, not a command.
_COMMAND_RE = re.compile(r"`(hermes sandbox run [^`]+)`|terminal\(\"(hermes sandbox run [^\"]+)\"\)")


def _skill_commands():
    text = SKILL.read_text(encoding="utf-8")
    found = [a or b for a, b in _COMMAND_RE.findall(text)]
    return [c for c in found if "[" not in c]


def test_every_sandbox_command_in_the_skill_parses_with_the_real_cli():
    commands = _skill_commands()
    assert len(commands) >= 3, "the skill no longer shows runnable `hermes sandbox run` commands"
    parser = argparse.ArgumentParser(exit_on_error=False)
    build_sandbox_parser(parser.add_subparsers(dest="command"))
    for command in commands:
        argv = shlex.split(command)[1:]
        try:
            args = parser.parse_args(argv)
        except (SystemExit, argparse.ArgumentError) as exc:
            pytest.fail(f"`hermes sandbox run` rejects a command the skill teaches: {command!r} ({exc})")
        run_command = [token for token in args.run_command if token != "--"]
        assert run_command, f"no command after -- in {command!r}"


def test_the_no_runtime_exit_status_the_skill_names_is_the_cli_s():
    text = SKILL.read_text(encoding="utf-8")
    named = set(re.findall(r"`(\d+)` no container runtime|exits `(\d+)`", text))
    statuses = {int(code) for pair in named for code in pair if code}
    assert statuses == {NO_RUNTIME_EXIT}, f"skill names {statuses}, CLI fails closed with {NO_RUNTIME_EXIT}"
