"""Contract tests for the optional runpod skill.

The skill's value is cost discipline around paid GPU pods, so these tests pin
the load-bearing pieces: the API key is declared as a secret, it never appears
as a literal on a command line, and the procedure always ends in a verified
terminate.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import hermes_yaml as yaml

SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "mlops"
    / "runpod"
    / "SKILL.md"
)


@pytest.fixture(scope="module")
def skill() -> tuple[dict, str]:
    text = SKILL_MD.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _, fm, body = text.split("---\n", 2)
    return yaml.safe_load(fm), body


def test_api_key_declared_as_required_secret(skill) -> None:
    fm, _ = skill
    names = [v["name"] for v in fm.get("required_environment_variables", [])]
    assert names == ["RUNPOD_API_KEY"]


def test_api_key_never_passed_as_command_line_argument(skill) -> None:
    _, body = skill
    # A literal key or an -H "Authorization: Bearer $KEY" argument would leak
    # through the process list and terminal logs; the skill uses curl --config -.
    assert not re.search(r"rpa_[A-Za-z0-9]{8,}", body)
    assert not re.search(r"-H\s+['\"]Authorization", body)
    assert "--config -" in body


def test_procedure_ends_with_verified_terminate(skill) -> None:
    _, body = skill
    procedure = body.split("## Procedure", 1)[1].split("\n## ", 1)[0]
    steps = re.findall(r"^\d+\. \*\*(.+?)\*\*", procedure, flags=re.MULTILINE)
    assert steps, "procedure has no numbered steps"
    assert "terminate" in steps[-1].lower()
    last_step = re.split(r"\n(?=\d+\. )", procedure.strip())[-1]
    assert "returns `null`" in last_step


def test_every_procedure_step_has_completion_criterion(skill) -> None:
    _, body = skill
    procedure = body.split("## Procedure", 1)[1].split("\n## ", 1)[0]
    steps = re.split(r"\n(?=\d+\. )", procedure.strip())
    steps = [s for s in steps if re.match(r"\d+\. ", s)]
    for step in steps:
        assert "*Done when:*" in step, f"step without completion criterion: {step[:60]}"


def test_pitfalls_numbered_sequentially(skill) -> None:
    _, body = skill
    pitfalls = body.split("## Pitfalls", 1)[1].split("\n## ", 1)[0]
    numbers = [int(n) for n in re.findall(r"^(\d+)\. ", pitfalls, flags=re.MULTILINE)]
    assert numbers == list(range(1, len(numbers) + 1))


def test_passes_skills_guard_as_community_install() -> None:
    # The skill is also distributed through a community tap; the hub scanner
    # blocks community installs on caution verdicts, so it must scan clean.
    from tools.skills_guard import scan_skill, should_allow_install

    result = scan_skill(SKILL_MD.parent, source="community")
    allowed, reason = should_allow_install(result)
    assert allowed, f"{reason}: {[(f.pattern_id, f.line) for f in result.findings]}"
