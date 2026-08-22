from __future__ import annotations

import re
from pathlib import Path

from tools.skills_tool import (
    _get_required_environment_variables,
    _parse_frontmatter,
)


SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "productivity"
    / "ticktick"
    / "SKILL.md"
)


def _load_skill() -> tuple[dict, str]:
    content = SKILL_PATH.read_text(encoding="utf-8")
    return _parse_frontmatter(content)


def test_ticktick_token_is_optional() -> None:
    frontmatter, _ = _load_skill()

    variables = _get_required_environment_variables(frontmatter)

    assert variables == [
        {
            "name": "TICKTICK_API_TOKEN",
            "prompt": "TickTick API token (Settings > Account > API Token)",
            "help": (
                "Optional. Only for the headless or gateway Bearer path. "
                "Skip it if you use the browser OAuth flow."
            ),
            "required_for": "headless Bearer authentication",
            "optional": True,
        }
    ]


def test_ticktick_skill_uses_modern_structure() -> None:
    _, body = _load_skill()
    headings = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]

    assert body.lstrip().startswith("# TickTick Skill\n")
    assert all(heading in body for heading in headings)
    assert [body.index(heading) for heading in headings] == sorted(
        body.index(heading) for heading in headings
    )

    introduction = body.split("## When to Use", 1)[0]
    introduction = introduction.split("# TickTick Skill", 1)[1].strip()
    sentences = re.findall(r"[^.!?]+[.!?]", introduction)
    assert 2 <= len(sentences) <= 3


def test_headless_setup_removes_oauth_marker() -> None:
    _, body = _load_skill()

    assert "Remove that line before adding the `headers` block" in body
    assert "Do not keep `auth: oauth` alongside the Authorization header" in body
    assert 'Authorization: "Bearer ${TICKTICK_API_TOKEN}"' in body
