"""Contract: skill names referenced in code/help/docs must exist in the shipped catalog.

#98539 merged six ``github-*`` skills into one ``github`` skill and folded
``ocr-and-documents``/``nano-pdf`` into ``pdf``, but call sites kept pointing
at the retired names (Desktop pill ``/github-auth``, ``-s`` examples, tips,
kanban help, bundled docs, and the consolidated skill's own references).
A referenced name that is not in the catalog dead-ends at runtime
(unknown slash command / unresolvable preload).

Reads source text and the skill tree only — never imports product modules.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# (relative path, line filter or None, [regexes]) — group(1) is a referenced name.
# Curated to real call sites; generic prose and test fixtures are out of scope.
SOURCES: list[tuple[str, str | None, list[str]]] = [
    ("apps/desktop/src/store/suggestion-providers/github.ts", None, [r"SKILL_NAME\s*=\s*'([^']+)'"]),
    ("hermes_cli/_parser.py", None, [r"-s\s+([A-Za-z0-9_,-]+)"]),
    ("locales/*.yaml", "hermes chat -s ", [r"-s\s+([A-Za-z0-9_,-]+)"]),
    ("cli.py", None, [r"python cli\.py --skills\s+([A-Za-z0-9_,-]+)"]),
    ("hermes_cli/kanban_parser.py", None, [r"--skill\s+([A-Za-z0-9-]+)"]),
    ("hermes_cli/kanban_db.py", None, [r"e\.g\.\s+`blogwatcher`,\s+`([a-z0-9-]+)`"]),
    ("hermes_cli/web_routers/git.py", None, [r"pill offers\s+`/([a-z0-9-]+)`"]),
    ("gateway/run.py", None, [r"or the ([a-z0-9-]+) skill"]),
    (
        "skills/autonomous-ai-agents/hermes-agent/references/webhooks.md",
        None,
        [r'--skills\s+"([^"]*github[^"]*)"'],
    ),
    ("skills/productivity/meeting-action-items/SKILL.md", "tracker connector", [r"`([a-z0-9-]+)`"]),
    ("skills/research/arxiv/SKILL.md", None, [r"see the `([a-z0-9-]+)` skill"]),
    (
        "skills/software-development/github/references/auth.md",
        None,
        [r"see `([a-z0-9-]+)` skill"],
    ),
    (
        "skills/software-development/github/references/code-review.md",
        None,
        [r"see `([a-z0-9-]+)` skill"],
    ),
    (
        "skills/software-development/github/references/issues.md",
        None,
        [r"see `([a-z0-9-]+)` skill"],
    ),
    (
        "skills/software-development/github/references/pr-workflow.md",
        None,
        [r"see `([a-z0-9-]+)` skill"],
    ),
    (
        "skills/software-development/github/references/repo-management.md",
        None,
        [r"see `([a-z0-9-]+)` skill"],
    ),
    ("skills/software-development/github/references/issue-to-pr.md", None, [r"Load `([a-z0-9-]+)`"]),
    ("skills/productivity/pdf/references/nano-pdf-editing.md", None, [r"see the `([a-z0-9-]+)` skill"]),
    ("apps/desktop/src/i18n/en.ts", None, [r"done: 'Added /([a-z0-9-]+)'"]),
    ("apps/desktop/src/i18n/de.ts", None, [r"done: '/([a-z0-9-]+) hinzugef"]),
    ("apps/desktop/src/i18n/es.ts", None, [r"done: 'Se a\u00f1adi\u00f3 /([a-z0-9-]+)'"]),
    ("apps/desktop/src/i18n/fr.ts", None, [r"done: 'Ajout de /([a-z0-9-]+)'"]),
    ("apps/desktop/src/i18n/ru.ts", None, [r"done: '\u0414\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u043e /([a-z0-9-]+)'"]),
    ("apps/desktop/src/i18n/zh.ts", None, [r"done: '\u5df2\u6dfb\u52a0 /([a-z0-9-]+)'"]),
    # The web dashboard's kanban placeholder names a skill to force-load.
    ("web/src/i18n/*.ts", None, [r"[,、]\s*([a-z][a-z0-9-]+)\"\s*,?$"]),
    ("tools/kanban_tools_schemas.py", None, [r"\['([a-z0-9-]+)'\] for a reviewer"]),
    ("tools/read_extract.py", None, [r"([a-z0-9-]+) skill \(marker-pdf\)"]),
]

# Not this issue's subject: pre-consolidation example token, not a retired builtin.
IGNORE = {"hermes-agent-dev"}

_NAME_RE = re.compile(r"name\s*:\s*([A-Za-z0-9_-]+)")


def _catalog() -> set[str]:
    names = set()
    for base in (REPO_ROOT / "skills", REPO_ROOT / "optional-skills"):
        for md in base.rglob("SKILL.md"):
            text = md.read_text(encoding="utf-8", errors="replace")
            head = text[:500] if text.startswith("---") else ""
            match = _NAME_RE.search(head)
            names.add(match.group(1) if match else md.parent.name)
    return names


def _referenced() -> dict[str, list[str]]:
    token = re.compile(r"[A-Za-z0-9_-]+")
    found: dict[str, list[str]] = {}
    for glob, line_filter, patterns in SOURCES:
        paths = sorted(REPO_ROOT.glob(glob)) if "*" in glob else [REPO_ROOT / glob]
        for path in paths:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line_filter:
                lines = [line for line in lines if line_filter in line]
            rel = path.relative_to(REPO_ROOT).as_posix()
            for line in lines:
                for pattern in patterns:
                    for match in re.finditer(pattern, line):
                        chunk = next(g for g in match.groups() if g)
                        found.setdefault(rel, []).extend(token.findall(chunk))
    return found


def test_referenced_skill_names_exist_in_catalog():
    """Every skill name a call site points at must ship under that name."""
    catalog = _catalog()
    missing = {
        f"{rel}: {name}"
        for rel, names in _referenced().items()
        for name in names
        if name not in catalog and name not in IGNORE
    }
    assert not missing, (
        "call sites reference skills that no longer ship "
        "(renamed/merged skill left these behind — repoint at the consolidated name):\n"
        + "\n".join(sorted(missing))
    )


def test_no_stale_nested_github_auth_paths():
    """The merged-away ``skills/github/github-auth`` install path is gone."""
    stale = [
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "skills" / "software-development" / "github").rglob("*.md")
        if "skills/github/github-auth" in path.read_text(encoding="utf-8", errors="replace")
    ]
    gh_env = REPO_ROOT / "skills/software-development/github/scripts/gh-env.sh"
    if "skills/github/github-auth" in gh_env.read_text(encoding="utf-8", errors="replace"):
        stale.append("skills/software-development/github/scripts/gh-env.sh")
    assert not stale, (
        "consolidated github skill still points at the retired nested install path "
        "(installed layout is skills/software-development/github/...):\n" + "\n".join(stale)
    )
