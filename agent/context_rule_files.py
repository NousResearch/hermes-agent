"""Claude Code rule modules (``.claude/rules/**/*.md``) for the project-context prompt.

Claude Code loads every ``.md`` under ``.claude/rules/`` (nested directories included) alongside
``CLAUDE.md``, and other coding agents have adopted the same directory. A rule whose YAML frontmatter names ``paths``
globs is scoped to those files. Hermes has no per-read trigger inside the prompt builder, so scoped
rules load at launch with their scope stated in the section heading: the model sees which files a
rule governs instead of the rule staying invisible until a matching read happens.
"""

from __future__ import annotations

import re
from pathlib import Path

RULES_DIR = Path(".claude") / "rules"
RULE_LABEL_PREFIX = ".claude/rules/"

_FRONTMATTER_RE = re.compile(r"\A\ufeff?---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_PATHS_KEY_RE = re.compile(r"^paths[ \t]*:[ \t]*(.*)$", re.MULTILINE)
_LIST_ITEM_RE = re.compile(r"^[ \t]+-[ \t]*(.+?)[ \t]*$", re.MULTILINE)


def discover_rule_files(cwd_path: Path) -> list[tuple[str, Path]]:
    """``(label, path)`` for every ``.md`` under ``<cwd>/.claude/rules`` in sorted order.

    Symlinks that resolve outside the project are skipped (same containment subdirectory hints apply)."""
    rules_dir = cwd_path / RULES_DIR
    try:
        if not rules_dir.is_dir():
            return []
        root = cwd_path.resolve()
        found = []
        for path in sorted(rules_dir.rglob("*.md")):
            if not path.is_file() or (path.is_symlink() and not path.resolve().is_relative_to(root)):
                continue
            found.append((RULE_LABEL_PREFIX + path.relative_to(rules_dir).as_posix(), path))
        return found
    except OSError:
        return []


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def rule_scope(content: str) -> list[str]:
    """The ``paths`` globs from a rule's YAML frontmatter; ``[]`` for an unconditional rule.

    Only ``paths`` is read (Claude Code ignores every other key). Accepts the block-list, inline-list
    and single-scalar spellings without a YAML dependency; anything unparseable reads as unconditional,
    which errs on loading the rule rather than dropping it."""
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return []
    block = match.group(1)
    key = _PATHS_KEY_RE.search(block)
    if not key:
        return []
    inline = key.group(1).strip()
    if inline.startswith("["):
        return [g for g in (_unquote(p) for p in inline.strip("[]").split(",")) if g]
    if inline:
        return [_unquote(inline)]
    rest = block[key.end():]
    globs: list[str] = []
    for line in rest.split("\n"):
        if not line.strip():
            continue
        item = _LIST_ITEM_RE.match(line)
        if item is None:
            break  # next top-level key
        globs.append(_unquote(item.group(1)))
    return [g for g in globs if g]


def rule_heading(label: str, scope: list[str]) -> str:
    """Section heading for a rule: the label, plus its ``paths`` scope when it has one."""
    return f"{label} (applies to files matching: {', '.join(scope)})" if scope else label
