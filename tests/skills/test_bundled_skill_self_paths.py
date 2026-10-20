"""Contract: a bundled skill's self-referencing install paths match its own location.

Bundled skills are installed to ``$HERMES_HOME/skills/<category>/<name>``,
preserving the category path. When a skill is moved between categories — e.g.
the six-way fold that moved the former ``skills/github/*`` skills to
``skills/software-development/github`` — the install-path strings embedded in
its own docs and scripts keep pointing at the pre-move directory, so every
copy-pasteable snippet raises FileNotFoundError. Nothing else catches that
drift today; this test does, for every bundled skill, at CI time.

Same contract as ``test_optional_skill_self_paths.py``, applied to the
bundled tree.

Only *self*-references are checked (the reference names the skill itself),
so cross-skill references and foreign-repo paths in port notes are immune.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUNDLED = REPO / "skills"

# Text formats that carry copy-pasteable install paths; data files can be
# huge, so cap the scan well above any real doc/script.
_SCAN_SUFFIXES = {".md", ".py", ".txt", ".sh", ".yaml", ".yml", ".toml", ".json"}
_MAX_SCAN_BYTES = 2 * 1024 * 1024


def _skill_roots():
    roots = {skill_md.parent for skill_md in BUNDLED.rglob("SKILL.md")}
    return sorted(roots)


def _scan_files(root):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SCAN_SUFFIXES:
            continue
        if path.stat().st_size > _MAX_SCAN_BYTES:
            continue
        yield path


def _stale_self_references(root):
    """Yield (file, line_no, snippet) for self-install paths that cannot resolve."""
    name = re.escape(root.name)
    # skills/<category>/<name>/... — joined-string form
    joined = re.compile(rf"skills/([\w.-]+)/{name}\b")
    # Path(...) / "skills" / "<category>" / "<name>" — segmented form
    segmented = re.compile(
        rf'["\']skills["\']\s*/\s*["\']([\w.-]+)["\']\s*/\s*["\']{name}["\']'
    )
    stale = []
    for path in _scan_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for match in joined.finditer(line):
                category = match.group(1)
                if not (BUNDLED / category / root.name).is_dir():
                    stale.append((path.relative_to(REPO), line_no, match.group(0)))
            for match in segmented.finditer(line):
                category = match.group(1)
                if not (BUNDLED / category / root.name).is_dir():
                    stale.append((path.relative_to(REPO), line_no, match.group(0)))
    return stale


def test_bundled_skills_self_install_paths_resolve_to_their_own_location():
    offenders = []
    for root in _skill_roots():
        offenders.extend(_stale_self_references(root))
    assert not offenders, (
        "Bundled skills reference their own install path under a category "
        "that does not match their location (the skill moved, the embedded "
        "paths did not):\n"
        + "\n".join(f"  {f}:{line}: {snippet}" for f, line, snippet in offenders)
    )
