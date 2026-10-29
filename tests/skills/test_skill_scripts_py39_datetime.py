"""Skill scripts must import cleanly on the interpreter that runs them.

SKILL.md tells the agent to run these scripts with the user's ``python3`` —
on a default macOS install that is /usr/bin/python3 3.9, not the Hermes
managed interpreter. A repo-wide ruff UP017 pass once rewrote every
``timezone.utc`` under optional-skills/ to the 3.11-only ``datetime.UTC``
alias and broke all of them at import (#138239; UP017 is now disabled for
``optional-skills/**`` in pyproject.toml). This guard keeps that class of
regression out even if the ruff exclusion is ever lifted.
"""
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKILL_SCRIPTS = sorted((REPO / "optional-skills").glob("**/scripts/*.py"))


def test_skill_scripts_were_found():
    # The glob itself is load-bearing: if it ever matches nothing the guard
    # below would pass vacuously.
    assert len(SKILL_SCRIPTS) > 50


def test_skill_scripts_avoid_py311_only_datetime_utc():
    offenders = []
    for path in SKILL_SCRIPTS:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "datetime":
                if any(alias.name == "UTC" for alias in node.names):
                    offenders.append(f"{path.relative_to(REPO)}: `from datetime import UTC` (3.11+)")
            elif isinstance(node, ast.Attribute) and node.attr == "UTC":
                offenders.append(f"{path.relative_to(REPO)}: `<datetime>.UTC` attribute (3.11+)")
    assert not offenders, (
        "skill scripts run on the user's system python3 (>=3.9) and cannot use "
        "the 3.11-only datetime.UTC alias; use timezone.utc instead:\n  "
        + "\n  ".join(offenders)
    )
