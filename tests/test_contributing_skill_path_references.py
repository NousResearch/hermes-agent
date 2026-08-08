"""Regression test for the skill-path example references in CONTRIBUTING.md.

CONTRIBUTING.md points new skill authors at concrete bundled skills as
copyable examples:

    See `skills/gifs/gif-search/` and `skills/email/himalaya/` for examples.

The gif example pointed at ``skills/gifs/gif-search/``, which does not exist:
``skills/gifs/`` is only a category folder holding a ``DESCRIPTION.md``. The
actual skill lives at ``skills/media/gif-search/`` (the path used everywhere
else in the docs, e.g. the skills catalog). A contributor following the
CONTRIBUTING guide would ``ls`` a missing directory.

This test extracts every backtick-quoted ``skills/...`` path referenced in
CONTRIBUTING.md and asserts each one exists on disk, so a stale example path
fails here instead of confusing a contributor.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRIBUTING = REPO_ROOT / "CONTRIBUTING.md"

# Matches `skills/...` (optionally trailing slash) inside backticks.
_SKILL_PATH_RE = re.compile(r"`(skills/[^`]+?)/?`")


def _referenced_skill_paths():
    text = CONTRIBUTING.read_text(encoding="utf-8")
    # De-dupe while preserving order for a readable failure message.
    seen = {}
    for match in _SKILL_PATH_RE.finditer(text):
        seen.setdefault(match.group(1).rstrip("/"), None)
    return list(seen)


def test_contributing_has_skill_path_references():
    """Guard against the regex silently matching nothing (e.g. a refactor)."""
    assert _referenced_skill_paths(), (
        "Expected at least one `skills/...` example path in CONTRIBUTING.md"
    )


def test_contributing_skill_example_paths_exist_on_disk():
    missing = [
        rel
        for rel in _referenced_skill_paths()
        if not (REPO_ROOT / rel).exists()
    ]
    assert not missing, (
        "CONTRIBUTING.md references skill example paths that do not exist on "
        f"disk: {missing}. Update the reference to the skill's real location "
        "(e.g. skills/media/gif-search)."
    )
