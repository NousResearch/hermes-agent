"""Tests for scripts/ci/lockfile_diff.py.

The differ's job is semantic comparison: reordering and integrity-hash
churn in the lockfile text must produce an empty diff, while actual
version movement must show up as added/removed/updated regardless of
where in the file it appears.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "lockfile_diff.py"
_spec = importlib.util.spec_from_file_location("lockfile_diff", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load lockfile_diff.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _lock(packages: dict[str, dict]) -> str:
    return json.dumps(
        {
            "name": "hermes",
            "lockfileVersion": 3,
            "packages": {"": {"name": "hermes"}, **packages},
        }
    )


BASE = _lock(
    {
        "node_modules/react": {"version": "18.2.0", "integrity": "sha512-aaa"},
        "node_modules/left-pad": {"version": "1.3.0", "integrity": "sha512-bbb"},
        "node_modules/foo/node_modules/react": {"version": "17.0.2"},
    }
)


def test_parse_skips_root_and_versionless():
    text = _lock({"node_modules/linked": {"link": True}})
    parsed = _mod.parse_lockfile(text)
    assert parsed == {}  # root entry and versionless link both skipped


def test_reorder_and_hash_churn_is_empty_diff():
    # Same packages, reordered, different integrity hashes — the exact noise
    # that makes textual lockfile diffs unreadable.
    reordered = _lock(
        {
            "node_modules/foo/node_modules/react": {"version": "17.0.2"},
            "node_modules/left-pad": {"version": "1.3.0", "integrity": "sha512-XYZ"},
            "node_modules/react": {"version": "18.2.0", "integrity": "sha512-ZZZ"},
        }
    )
    d = _mod.diff_locks(_mod.parse_lockfile(BASE), _mod.parse_lockfile(reordered))
    assert d == {"added": [], "removed": [], "updated": []}
    assert _mod.render_markdown({"package-lock.json": d}) == ""


def test_add_remove_update_all_detected():
    head = _lock(
        {
            "node_modules/react": {"version": "18.3.1"},  # updated
            "node_modules/is-even": {"version": "1.0.0"},  # added
            "node_modules/foo/node_modules/react": {"version": "17.0.2"},  # unchanged
            # left-pad removed
        }
    )
    d = _mod.diff_locks(_mod.parse_lockfile(BASE), _mod.parse_lockfile(head))
    assert d["added"] == [("node_modules/is-even", "1.0.0")]
    assert d["removed"] == [("node_modules/left-pad", "1.3.0")]
    assert d["updated"] == [("node_modules/react", "18.2.0", "18.3.1")]


def test_nested_dedup_is_distinct_entry():
    # The same package at two nesting levels must be tracked separately —
    # bumping only the nested copy must not look like a top-level change.
    head = _lock(
        {
            "node_modules/react": {"version": "18.2.0"},
            "node_modules/left-pad": {"version": "1.3.0"},
            "node_modules/foo/node_modules/react": {"version": "17.0.3"},
        }
    )
    d = _mod.diff_locks(_mod.parse_lockfile(BASE), _mod.parse_lockfile(head))
    assert d["updated"] == [("node_modules/foo/node_modules/react", "17.0.2", "17.0.3")]


def test_render_markdown_contains_marker_and_versions():
    d = _mod.diff_locks(
        _mod.parse_lockfile(BASE),
        _mod.parse_lockfile(_lock({"node_modules/react": {"version": "19.0.0"}})),
    )
    md = _mod.render_markdown({"apps/desktop/package-lock.json": d})
    assert md.startswith(_mod.COMMENT_MARKER)  # workflow finds its comment by prefix
    assert "⚠️" in md
    assert "`apps/desktop/package-lock.json`" in md
    assert "`18.2.0`" in md and "`19.0.0`" in md
    # nested display name keeps the parent chain visible
    assert "nested under foo" in md


def test_render_markdown_omits_unchanged_lockfiles():
    changed = _mod.diff_locks({}, {"node_modules/x": "1.0.0"})
    unchanged = _mod.diff_locks({}, {})
    md = _mod.render_markdown({"a/package-lock.json": changed, "b/package-lock.json": unchanged})
    assert "a/package-lock.json" in md
    assert "b/package-lock.json" not in md
