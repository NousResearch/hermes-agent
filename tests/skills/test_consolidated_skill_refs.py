"""Guard the skill-consolidation rename (#126931).

The github-* -> github and ocr-and-documents/nano-pdf -> pdf consolidation left
code, help text and shipped docs pointing at skills that no longer ship, so
fresh installs dead-ended (e.g. the Desktop GitHub pill inserting /github-auth,
which comes back as an unknown command). These tests pin the renamed call
sites and resolve the names they reference against the shipped catalog, so a
future consolidation cannot silently reintroduce dead references.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent

# Built-ins retired by the consolidation. Their names must not appear in the
# reference surfaces below; the webhook subscription label in
# skills/autonomous-ai-agents/hermes-agent/references/webhooks.md and the
# "(merged from ...)" provenance headers in skills/productivity/pdf/references/
# are intentional exceptions and are not in this list's checked files.
RETIRED_BUILTINS = (
    "github-auth",
    "github-code-review",
    "github-issue-to-pr",
    "github-issues",
    "github-pr-workflow",
    "github-repo-management",
    "ocr-and-documents",
    "nano-pdf",
)

# User- or agent-facing surfaces that must only name skills that ship. Plain
# code files plus the localized copies of the strings they render.
REFERENCE_FILES = (
    "apps/desktop/src/store/suggestion-providers/github.ts",
    "apps/desktop/src/store/suggestion-providers/skill.ts",
    "apps/desktop/src/i18n/de.ts",
    "apps/desktop/src/i18n/en.ts",
    "apps/desktop/src/i18n/es.ts",
    "apps/desktop/src/i18n/fr.ts",
    "apps/desktop/src/i18n/ru.ts",
    "apps/desktop/src/i18n/zh.ts",
    "cli.py",
    "hermes_cli/_parser.py",
    "hermes_cli/kanban_db.py",
    "hermes_cli/kanban_parser.py",
    "hermes_cli/web_routers/git.py",
    "gateway/run.py",
    "tools/kanban_tools_schemas.py",
    "tools/read_extract.py",
    "locales/en.yaml",
    "web/src/i18n/en.ts",
    "plugins/kanban/dashboard/dist/index.js",
    "CONTRIBUTING.md",
    "CONTRIBUTING.es.md",
    "skills/research/arxiv/SKILL.md",
    "skills/productivity/meeting-action-items/SKILL.md",
)


def _shipped_skill_names() -> set[str]:
    names: set[str] = set()
    for base in ("skills", "optional-skills"):
        for skill_md in (REPO / base).glob("*/**/SKILL.md"):
            text = skill_md.read_text(encoding="utf-8")
            match = re.search(r"^name:\s*(\S+)", text, re.MULTILINE)
            if match:
                names.add(match.group(1))
    assert names, "skill catalog scan found no SKILL.md frontmatter names"
    return names


@pytest.mark.parametrize("rel_path", REFERENCE_FILES)
def test_reference_surfaces_do_not_name_retired_builtins(rel_path: str) -> None:
    path = REPO / rel_path
    assert path.is_file(), f"{rel_path} no longer exists; update REFERENCE_FILES"
    text = path.read_text(encoding="utf-8")
    hits = sorted(name for name in RETIRED_BUILTINS if name in text)
    assert not hits, (
        f"{rel_path} references retired built-in skill(s) {hits}; "
        "point them at the consolidated skill instead"
    )


def test_desktop_github_pill_skill_ships() -> None:
    """The pill's /<skill> invoke must resolve to a skill that actually ships."""
    source = (REPO / "apps/desktop/src/store/suggestion-providers/github.ts").read_text(
        encoding="utf-8"
    )
    match = re.search(r"const SKILL_NAME = '([^']+)'", source)
    assert match, "GitHub suggestion pill no longer declares a SKILL_NAME constant"
    assert match.group(1) in _shipped_skill_names(), (
        f"pill SKILL_NAME {match.group(1)!r} is not in the shipped skill catalog"
    )
