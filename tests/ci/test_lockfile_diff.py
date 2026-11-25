"""Tests for scripts/ci/lockfile_diff.py.

The differ's job is semantic comparison: reordering and integrity-hash
churn in the lockfile text must produce an empty diff, while actual
version movement must show up as added/removed/updated regardless of
where in the file it appears.

tests/fixtures/pnpm_lock/ holds real pnpm 12 lockfiles (``pnpm install
--lockfile-only``) of two-project workspaces, single-document as the repo's is.
base/head bump, add and remove packages; repin-base/repin-head only move
dependencies between versions that both stay locked; so do alias-base/alias-head,
through ``npm:`` aliases (a project's ``my-ms``, and ``@isaacs/cliui``'s
``string-width-cjs`` via an override) and a packageExtensions edge added to debug.
extra-key keeps a foreign top-level ``bit:`` block, which pnpm writes between
``packages`` and ``snapshots``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
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
REPIN_BASE = (_ROOT / "tests/fixtures/pnpm_lock/repin-base.yaml").read_text(encoding="utf-8")
REPIN_HEAD = (_ROOT / "tests/fixtures/pnpm_lock/repin-head.yaml").read_text(encoding="utf-8")
ALIAS_BASE = (_ROOT / "tests/fixtures/pnpm_lock/alias-base.yaml").read_text(encoding="utf-8")
ALIAS_HEAD = (_ROOT / "tests/fixtures/pnpm_lock/alias-head.yaml").read_text(encoding="utf-8")
EXTRA_KEY = (_ROOT / "tests/fixtures/pnpm_lock/extra-key.yaml").read_text(encoding="utf-8")


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


def test_pnpm_add_remove_update_all_detected(tmp_path):
    # Every move here shows in the package rows, so no project or edge row repeats it.
    d = _mod.diff_refs("HEAD~1", "HEAD", _commit_pnpm_locks(tmp_path, PNPM_BASE, PNPM_HEAD))
    d = d["pnpm-lock.yaml"]
    assert d["added"] == [("escape-string-regexp", "4.0.0")]
    assert d["removed"] == [("left-pad", "1.3.0")]
    # Moving one of two locked copies (pkg's is-number 6 -> 5) updates that package.
    assert d["updated"] == [
        ("@types/ms", "0.7.31", "0.7.34"),
        ("is-number", "6.0.0, 7.0.0", "5.0.0, 7.0.0"),
        ("ms", "2.1.2", "2.1.3"),
    ]


def _commit_pnpm_locks(repo: Path, *locks: str) -> str:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, timeout=30)
    for lock in locks:
        (repo / "pnpm-lock.yaml").write_text(lock, encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, timeout=30)
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                        "commit", "-q", "--no-verify", "-m", "lock"], cwd=repo, check=True, timeout=30)
    return str(repo)


def test_pnpm_dependency_moving_between_locked_versions_is_reported(tmp_path):
    # The root and pkg swap ms 2.1.2 and 2.1.3, and an override moves debug's ms:
    # both versions stay locked, so the packages section is identical on both sides.
    assert _mod.parse_pnpm_lockfile(REPIN_BASE) == _mod.parse_pnpm_lockfile(REPIN_HEAD)
    diffs = _mod.diff_refs("HEAD~1", "HEAD", _commit_pnpm_locks(tmp_path, REPIN_BASE, REPIN_HEAD))
    assert diffs["pnpm-lock.yaml"] == {
        "added": [],
        "removed": [],
        "updated": [
            ("ms *(pkg)*", "2.1.3", "2.1.2"),
            ("ms *(root)*", "2.1.2", "2.1.3"),
            ("ms *(via debug@4.3.4)*", "2.1.2", "2.1.3"),
        ],
    }
    assert "| ms *(root)* | `2.1.2` | `2.1.3` |" in _mod.render_markdown(diffs)


def test_pnpm_alias_and_new_edge_between_locked_versions_are_reported():
    # Every version stays locked, so only the importer and snapshot edges move.
    assert _mod.parse_pnpm_lockfile(ALIAS_BASE) == _mod.parse_pnpm_lockfile(ALIAS_HEAD)
    assert _mod.diff_pnpm_locks(ALIAS_BASE, ALIAS_HEAD) == {
        "added": [("string-width *(via debug@4.3.4)*", "4.2.3")],
        "removed": [],
        "updated": [
            ("my-ms *(pkg)*", "ms@2.1.3", "ms@2.1.2"),
            ("string-width-cjs *(via @isaacs/cliui@8.0.2)*", "string-width@4.2.3", "string-width@5.1.2"),
        ],
    }


def test_pnpm_repin_through_peer_suffixes_and_other_dependency_fields_is_reported():
    # The repo lock's references mostly carry a peer suffix, its snapshots list
    # transitivePeerDependencies, and moves also land in devDependencies and
    # optionalDependencies. Here the consumer's own peer moves too (8 -> 9).
    def variant(lock: str, peer: str) -> str:
        for old, new in (("\n  pkg:\n    dependencies:\n", "\n  pkg:\n    devDependencies:\n"),
                         ("\n  .:\n    dependencies:\n", "\n  .:\n    optionalDependencies:\n"),
                         ("  debug@4.3.4:\n    dependencies:\n", f"  debug@4.3.4(supports-color@{peer}):\n"
                                                                "    optionalDependencies:\n")):
            assert lock.count(old) == 1
            lock = lock.replace(old, new)
        lock = re.sub(r"(?m)^( +(?:version|ms): 2\.1\.[23])$", rf"\1(supports-color@{peer})", lock)
        return re.sub(r"(\n    optionalDependencies:\n      ms: \S+\n)",
                      r"\1    transitivePeerDependencies:\n      - supports-color\n", lock)

    assert _mod.diff_pnpm_locks(variant(REPIN_BASE, "8.1.1"), variant(REPIN_HEAD, "9.4.0")) == {
        "added": [],
        "removed": [],
        "updated": [
            ("ms *(pkg)*", "2.1.3", "2.1.2"),
            ("ms *(root)*", "2.1.2", "2.1.3"),
            ("ms *(via debug@4.3.4)*", "2.1.2", "2.1.3"),
        ],
    }


def _edit(lock: str, *edits: tuple[str, str, int]) -> str:
    for pattern, replacement, count in edits:
        lock, found = re.subn(pattern, replacement, lock)
        assert found == count, pattern
    return lock


_ROOT_MS = r"specifier: 2\.1\.2\n        version: 2\.1\.2\n"  # the root's ms in repin-base


@pytest.mark.parametrize(("edits", "row"), [
    # The root moves from 2.1.2, still locked, to 2.1.4, which only the head locks.
    ([(_ROOT_MS, "specifier: 2.1.4\n        version: 2.1.4\n", 1),
      (r"\nsnapshots:\n", "\n  ms@2.1.4:\n    resolution: {integrity: sha512-x}\n\nsnapshots:\n", 1),
      (r"\Z", "\n  ms@2.1.4: {}\n", 1)],
     ("ms", "2.1.2, 2.1.3", "2.1.2, 2.1.3, 2.1.4")),
    # The root and debug move from 2.1.2, which only the base locks, to 2.1.3, still locked.
    ([(_ROOT_MS, "specifier: 2.1.3\n        version: 2.1.3\n", 1),
      (r"      ms: 2\.1\.2\n", "      ms: 2.1.3\n", 1),
      (r"\n  ms@2\.1\.2:(?: \{\}|\n    resolution: .*)\n", "\n", 2)],
     ("ms", "2.1.2, 2.1.3", "2.1.3")),
])
def test_pnpm_move_to_or_from_a_version_locked_on_one_side_stays_one_package_row(edits, row):
    # The package row already shows such a move, so no project or edge row repeats it.
    assert _mod.diff_pnpm_locks(REPIN_BASE, _edit(REPIN_BASE, *edits)) == {
        "added": [], "removed": [], "updated": [row]}


def test_pnpm_project_that_appears_or_disappears_is_one_row():
    renamed = REPIN_BASE.replace("\n  pkg:\n", "\n  packages/pkg:\n")
    assert _mod.diff_pnpm_locks(REPIN_BASE, renamed) == {
        "added": [("project packages/pkg", "1 locked dependency")],
        "removed": [("project pkg", "1 locked dependency")],
        "updated": [],
    }


@pytest.mark.parametrize("stray", ["# a note", "<<<<<<< HEAD", "=======", "\tdependencies:", "- item"])
def test_pnpm_section_refuses_a_column_0_line_that_is_not_a_key(stray):
    # A column-0 line that is not a lock key would otherwise end importers early,
    # and every project after it would be reported as removed.
    broken = REPIN_BASE.replace("\n  pkg:\n", f"\n{stray}\n  pkg:\n")
    with pytest.raises(ValueError, match="unreadable importers line"):
        _mod.diff_pnpm_locks(REPIN_BASE, broken)


def test_pnpm_section_ends_at_a_top_level_key_pnpm_does_not_define():
    # pnpm round-trips root keys it does not define (Bit's ``bit:``) and sorts them
    # after packages, so such a key must end a section like any pnpm key.
    from scripts.ci.pnpm_lock import section, section_keys

    assert "\nbit:\n  approved:\n    - ms\n\nsnapshots:\n" in EXTRA_KEY
    assert section_keys(EXTRA_KEY, "importers") == [".", "packages/a"]
    assert section_keys(EXTRA_KEY, "packages")[-1] == "yallist@4.0.0"
    assert section(EXTRA_KEY, "snapshots")["semver@7.6.0"] == {"dependencies": {"lru-cache": "6.0.0"}}
    without_bit = EXTRA_KEY.replace("bit:\n  approved:\n    - ms\n\n", "")
    assert _mod.diff_pnpm_locks(without_bit, EXTRA_KEY) == {"added": [], "removed": [], "updated": []}
    # An empty workspace: the foreign key directly follows importers.
    empty = "lockfileVersion: '9.0'\n\nimporters:\n\n  .: {}\n\nbit:\n  approved: []\n"
    assert section_keys(empty, "importers") == ["."]


def test_pnpm_section_keys_refuses_a_key_line_not_indented_two_spaces():
    # ``ab@1`` one column short would otherwise read as the key ``b@1``.
    from scripts.ci.pnpm_lock import section_keys

    with pytest.raises(ValueError, match="unreadable packages line"):
        section_keys(REPIN_HEAD.replace("\n  ms@2.1.3:", "\n ms@2.1.3:"), "packages")


def test_pnpm_section_reads_an_explicit_key():
    # pnpm writes a key over 1024 characters (a long peer suffix on a snapshot) as
    # ``? key`` / ``: value``; the value's first line follows the ``: ``.
    from scripts.ci.pnpm_lock import section, section_keys

    explicit = REPIN_HEAD
    for old, new in (("  ms@2.1.2:\n    resolution:", "  ? ms@2.1.2\n  : resolution:"),
                     ("  debug@4.3.4:\n    dependencies:", "  ? debug@4.3.4\n  : dependencies:"),
                     ("  ms@2.1.3: {}", "  ? ms@2.1.3\n  : {}")):
        assert explicit.count(old) == 1
        explicit = explicit.replace(old, new)
    for name in ("snapshots", "packages"):
        assert section(explicit, name) == section(REPIN_HEAD, name)
        assert section_keys(explicit, name) == section_keys(REPIN_HEAD, name)
    assert _mod.diff_pnpm_locks(REPIN_BASE, explicit) == _mod.diff_pnpm_locks(REPIN_BASE, REPIN_HEAD)
    with pytest.raises(ValueError, match="unreadable snapshots line"):
        section(explicit.replace("  : dependencies:", "    dependencies:"), "snapshots")
    with pytest.raises(ValueError, match="unreadable snapshots line"):  # a value line with no key line
        section(REPIN_HEAD.replace("  ms@2.1.3: {}", "  ms@2.1.3:\n  : {}"), "snapshots")


def test_pnpm_explicit_quoted_file_key_preserves_embedded_colon_space():
    from scripts.ci.pnpm_lock import section, section_keys

    # POSIX file: paths can contain ": ", so js-yaml quotes their keys.
    # Real pnpm switches to an explicit mapping key above 1024 characters.
    path = "/".join(["a" * 150] * 7) + "/package: with space"
    key = f"ms@file:vendor/{path}"
    assert len(key) > 1024
    old = "  ms@2.1.2:\\n    resolution:".replace("\\n", "\n")
    assert REPIN_HEAD.count(old) == 1
    explicit = REPIN_HEAD.replace(old, f"  ? '{key}'\\n  : resolution:".replace("\\n", "\n"), 1)

    assert key in section(explicit, "packages")
    assert key in section_keys(explicit, "packages")
    # Below the threshold pnpm writes the same quoted key as an ordinary ``'key':`` line.
    short = REPIN_HEAD.replace(old, "  'ms@file:vendor/package: with space':\n    resolution:", 1)
    assert "ms@file:vendor/package: with space" in section_keys(short, "packages")


@pytest.mark.parametrize(("old", "new"), [
    ("    dependencies:\n      ms:\n        specifier: 2.1.3\n        version: 2.1.3\n",
     "    dependencies: {ms: {specifier: 2.1.3, version: 2.1.3}}\n"),  # a flow map
    ("        specifier: 2.1.2\n        version: 2.1.2\n", "        specifier: 2.1.2\n"),  # no version
    ("  debug@4.3.4:\n    dependencies:\n      ms: 2.1.2\n", "  debug@4.3.4:\n    dependencies: {ms: 2.1.2}\n"),
])
def test_pnpm_entry_of_unknown_shape_is_refused(old, new):
    assert old in REPIN_BASE
    with pytest.raises(ValueError, match="pnpm-lock.yaml: "):
        _mod.diff_pnpm_locks(REPIN_BASE, REPIN_BASE.replace(old, new, 1))


def test_pnpm_lock_refuses_what_it_cannot_read():
    with pytest.raises(ValueError, match="lockfileVersion: '6.0'"):
        _mod.parse_pnpm_lockfile(PNPM_BASE.replace("lockfileVersion: '9.0'", "lockfileVersion: '6.0'"))
    # A pnpm pin recorded in the lock arrives as a leading document; scanners read only
    # the first document, so the lock must stay single-document.
    pin = "---\nlockfileVersion: '9.0'\n\nimporters:\n\n  .:\n    configDependencies: {}\n\n---\n"
    with pytest.raises(ValueError, match="one YAML document, found 2"):
        _mod.parse_pnpm_lockfile(pin + PNPM_BASE)
    # The project and snapshot rows read each entry's fields: a shape pnpm does not
    # write fails rather than being skipped.
    with pytest.raises(ValueError, match="unreadable importers line"):
        _mod.diff_pnpm_locks(PNPM_BASE.replace("        version: 6.0.0", "         version: 6.0.0"), PNPM_HEAD)


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
