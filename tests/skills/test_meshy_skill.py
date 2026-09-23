"""Tests for the meshy optional skill.

Structural + contract checks only (stdlib + pytest, no network). The commands
were validated against the published meshy-cli 0.4.0 (flags via --help, JSON
envelope and exit codes via its loopback contract tests upstream in
meshy-dev/meshy-3d-agent).
"""

import re
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "creative" / "meshy"
SKILL_MD = SKILL_DIR / "SKILL.md"
JSON_FLAGS = ("--format json", "--no-update-check")


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _meshy_commands(text: str):
    """Every inline-code or fenced command that invokes the CLI."""
    inline = re.findall(r"`(meshy [^`]+)`", text)
    fenced = [
        line.strip()
        for block in re.findall(r"```bash\n(.*?)```", text, re.S)
        for line in block.splitlines()
        if line.strip().startswith("meshy ")
    ]
    return inline + fenced


def test_required_sections_present(skill_text: str):
    for heading in (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ):
        assert heading in skill_text, f"missing section: {heading}"


def test_login_uses_the_device_flow_never_the_secret_mode(skill_text: str):
    logins = [c for c in _meshy_commands(skill_text) if c.startswith("meshy auth login")]
    assert logins, "the skill must document how to sign in"
    for cmd in logins:
        assert "--device" in cmd, cmd
        assert "--no-wait" not in cmd, "--no-wait prints the device code, a bearer secret"
    # The stderr line the agent relays to the user, as the CLI prints it.
    assert "Enter code XXXX-XXXX at https://www.meshy.ai/device" in skill_text


def test_full_commands_request_machine_output(skill_text: str):
    """Commands written out in full (not Quick Reference fragments) carry the JSON flags."""
    full = [c for c in _meshy_commands(skill_text) if "--format" in c]
    assert full, "expected fully written commands"
    for cmd in full:
        for flag in JSON_FLAGS:
            assert flag in cmd, f"{cmd!r} lacks {flag}"
    assert "append `--output-schema v1 --format json --no-update-check`" in skill_text


def test_writes_are_confined_to_a_workspace(skill_text: str):
    for cmd in _meshy_commands(skill_text):
        if " --output " in f" {cmd} " or cmd.startswith("meshy mesh prepare-print"):
            assert "--workspace" in cmd, f"write without --workspace: {cmd!r}"


def test_async_create_then_wait(skill_text: str):
    creates = [c for c in _meshy_commands(skill_text) if re.search(r"\bcreate\b", c) and "--help" not in c]
    assert creates
    for cmd in creates:
        assert "--async" in cmd, f"create must submit once and return: {cmd!r}"
    assert "result.submission.task_id" in skill_text
    assert "never create a second task" in skill_text


def test_pinned_fallback_runner_matches_the_documented_cli(skill_text: str):
    assert "npm exec --yes --package=meshy-cli@0.4.0 --" in skill_text
    assert "Node.js 22.12" in skill_text


def test_platforms_declared(skill_text: str):
    m = re.search(r"^platforms: (.*)$", skill_text, re.MULTILINE)
    assert m, "platforms field required"
    for os_name in ("linux", "macos", "windows"):
        assert os_name in m.group(1)
