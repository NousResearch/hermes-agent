"""Tests for scripts/ci/lockfile_diff.py.

The differ's job is semantic comparison: reordering and integrity-hash
churn in the lockfile text must produce an empty diff, while actual
version movement must show up as added/removed/updated regardless of
where in the file it appears.

tests/fixtures/pnpm_lock/ holds two real pnpm 12 lockfiles (``pnpm install
--lockfile-only``) of a two-project workspace, single-document as the repo's is.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_PATH = _ROOT / "scripts" / "ci" / "lockfile_diff.py"
_spec = importlib.util.spec_from_file_location("lockfile_diff", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load lockfile_diff.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

PNPM_BASE = (_ROOT / "tests/fixtures/pnpm_lock/base.yaml").read_text(encoding="utf-8")
PNPM_HEAD = (_ROOT / "tests/fixtures/pnpm_lock/head.yaml").read_text(encoding="utf-8")


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


def test_render_markdown_omits_unchanged_lockfiles():
    changed = _mod.diff_locks({}, {"node_modules/x": "1.0.0"})
    unchanged = _mod.diff_locks({}, {})
    md = _mod.render_markdown({"a/package-lock.json": changed, "b/package-lock.json": unchanged})
    assert "a/package-lock.json" in md
    assert "b/package-lock.json" not in md


def test_pnpm_lock_reduces_to_package_versions():
    # is-number is locked twice (root and pkg).
    assert _mod.parse_pnpm_lockfile(PNPM_BASE) == {
        "@types/ms": "0.7.31",
        "is-number": "6.0.0, 7.0.0",
        "left-pad": "1.3.0",
        "ms": "2.1.2",
    }


def test_pnpm_add_remove_update_all_detected():
    d = _mod.diff_locks(_mod.parse_pnpm_lockfile(PNPM_BASE), _mod.parse_pnpm_lockfile(PNPM_HEAD))
    assert d["added"] == [("escape-string-regexp", "4.0.0")]
    assert d["removed"] == [("left-pad", "1.3.0")]
    # Moving one of two locked copies (pkg's is-number 6 -> 5) updates that package.
    assert d["updated"] == [
        ("@types/ms", "0.7.31", "0.7.34"),
        ("is-number", "6.0.0, 7.0.0", "5.0.0, 7.0.0"),
        ("ms", "2.1.2", "2.1.3"),
    ]


def test_pnpm_lock_refuses_what_it_cannot_read():
    with pytest.raises(ValueError, match="lockfileVersion: '6.0'"):
        _mod.parse_pnpm_lockfile(PNPM_BASE.replace("lockfileVersion: '9.0'", "lockfileVersion: '6.0'"))
    # A pnpm pin recorded in the lock arrives as a leading document; scanners read only
    # the first document, so the lock must stay single-document.
    pin = "---\nlockfileVersion: '9.0'\n\nimporters:\n\n  .:\n    configDependencies: {}\n\n---\n"
    with pytest.raises(ValueError, match="one YAML document, found 2"):
        _mod.parse_pnpm_lockfile(pin + PNPM_BASE)


def test_diff_refs_reports_the_pnpm_lock_and_nested_npm_locks(tmp_path):
    def commit(pnpm_lock: str, npm_lock: str) -> None:
        (tmp_path / "pnpm-lock.yaml").write_text(pnpm_lock, encoding="utf-8")
        (tmp_path / "website").mkdir(exist_ok=True)
        (tmp_path / "website/package-lock.json").write_text(npm_lock, encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, timeout=30)
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                        "commit", "-q", "--no-verify", "-m", "lock"], cwd=tmp_path, check=True, timeout=30)

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, timeout=30)
    commit(PNPM_BASE, BASE)
    commit(PNPM_HEAD, _lock({"node_modules/react": {"version": "18.3.1"}}))

    diffs = _mod.diff_refs("HEAD~1", "HEAD", str(tmp_path))
    assert set(diffs) == {"pnpm-lock.yaml", "website/package-lock.json"}
    md = _mod.render_markdown(diffs)
    assert "#### `pnpm-lock.yaml`" in md
    assert "| ms | `2.1.2` | `2.1.3` |" in md
    assert "| react | `18.2.0` | `18.3.1` |" in md
