"""Bash completion keeps its command context after a leading profile selector."""

import os
import shlex
import subprocess

import pytest

from hermes_cli._parser import build_top_level_parser
from hermes_cli.completion import generate_bash
from hermes_cli.subcommands.profile import build_profile_parser

pytestmark = pytest.mark.linux_only


def _complete(words, tmp_path):
    parser, subparsers, _ = build_top_level_parser()
    build_profile_parser(subparsers, cmd_profile=lambda args: None)
    script = (
        generate_bash(parser)
        + "\nCOMP_WORDS=(" + shlex.join(["hermes", *words]) + ")\n"
        + f"COMP_CWORD={len(words)}\n"
        + '_hermes_completion\nprintf "%s\\n" "${COMPREPLY[@]}"\n'
    )
    result = subprocess.run(
        ["bash", "--noprofile", "--norc"],
        input=script,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "HOME": str(tmp_path)},
        timeout=10,
        check=True,
    )
    assert not result.stderr
    return [line for line in result.stdout.splitlines() if line]


@pytest.mark.parametrize("profile_words", [
    ["-p", "demo"],
    ["--profile", "demo"],
    ["--profile=demo"],
    ["--profile", "=", "demo"],  # Readline splits '=' at the default word break.
])
@pytest.mark.parametrize("words", [
    [""],
    ["profile", ""],
    ["chat", "--"],
])
def test_profile_selector_preserves_command_completions(tmp_path, profile_words, words):
    expected = _complete(words, tmp_path)
    assert expected, "The live parser must offer candidates for the control invocation."
    assert _complete([*profile_words, *words], tmp_path) == expected


def test_profile_selector_preserves_profile_name_completions(tmp_path):
    (tmp_path / ".hermes" / "profiles" / "demo").mkdir(parents=True)
    expected = _complete(["profile", "use", "de"], tmp_path)
    assert "demo" in expected
    assert _complete(["-p", "default", "profile", "use", "de"], tmp_path) == expected
    assert _complete(["--profile", "de"], tmp_path) == expected
