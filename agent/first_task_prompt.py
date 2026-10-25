"""The ``first-task`` skill for the task chat a setup handoff opens.

``start_chat`` sends it in that chat's first user message, the way ``/initiate-setup`` carries its skill, so
no system prompt changes. It rides at the tail, after the visible ask and the "What setup learned" block:
session previews read the ask, and the history projection cuts the message at ``MARKER`` so no surface
shows the skill.
"""

from __future__ import annotations

import json
from pathlib import Path

from hermes_constants import get_optional_skills_dir

MARKER = "\n\n[/first-task]\n\n"


def first_task_tail(connect: list, install: list) -> str:
    """``connect``: connector ids to show on one connect card; ``install``: catalog plugin ids to install."""
    skill_dir = get_optional_skills_dir(Path(__file__).resolve().parent.parent / "optional-skills")
    skill = (skill_dir / "productivity" / "first-task" / "SKILL.md").read_text(encoding="utf-8-sig").strip()
    facts = json.dumps({"connect": connect, "install": install}, ensure_ascii=False)
    return f"{MARKER}{skill}\n\n```json\n{facts}\n```"


def visible_text(text: str) -> str:
    """The part of a user message a surface shows: everything before the first-task skill."""
    return text.partition(MARKER)[0]
