"""Platform gates of optional skills whose bodies are bound to one OS family.

Front-matter ``platforms`` feeds skill selection, so a declared platform the
body cannot run on gets the skill offered and failing at step one. Asserts on
the declared data through the real frontmatter parser (``agent.skill_utils``)
instead of faking the host OS (tests/conftest.py host-marker policy).
"""
from pathlib import Path

import pytest

from agent.skill_utils import PLATFORM_MAP, parse_frontmatter

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("skill", "platform", "why"),
    [
        ("optional-skills/creative/heartmula", "windows", ". .venv/bin/activate"),
        ("optional-skills/mlops/tensorrt-llm", "macos", "Requires CUDA"),
    ],
)
def test_os_bound_skill_is_not_offered_on_the_wrong_platform(skill, platform, why):
    fm, body = parse_frontmatter((REPO / skill / "SKILL.md").read_text(encoding="utf-8"))
    assert why in body, f"{skill}: body no longer carries the OS-binding step; re-evaluate its platforms"
    declared = fm.get("platforms")
    assert declared, f"{skill}: an OS-bound body needs an explicit platforms list"
    declared = {PLATFORM_MAP.get(str(p).lower().strip(), str(p).lower().strip()) for p in declared}
    assert PLATFORM_MAP.get(platform, platform) not in declared, (
        f"{skill} declares a platform its body cannot run on: {platform}"
    )
