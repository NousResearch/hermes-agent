"""scripts/check_compat_pointers.py prunes an excluded tree instead of walking then discarding it.

The scanner used one ``ROOT.rglob("*.py")`` and filtered afterwards (``parts[0] in SKIP_DIRS``), so
an excluded dependency tree was fully enumerated for zero output — and a *nested* one,
``pkg/node_modules/…``, was not filtered at all and got read, AST-parsed and reported as in-tree
code. The contract pinned here: an excluded directory is never entered (proved with an ``os.scandir``
spy — the call the walk would make to descend), so it contributes zero scanned files, while every
non-excluded sibling is still walked and reported exactly as before.
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_compat_pointers.py"

# A file that imports through a plugin-compat pointer, and one that does not.
POINTER = "from hermes_cli.kanban_db import connect  # plugin-compat pointer\n\n\nCONNECT = connect\n"
CLEAN = "import json\n\n\ndef helper():\n    return json.dumps({})\n"
MANIFEST_ENTRIES = {"entries": [{"facade": "hermes_cli.kanban_db", "name": "connect", "kind": "moved-lazy"}]}

_NESTED_DEP = Path("pkg") / "deep" / "nested" / "node_modules"


def _can_symlink() -> bool:
    """Check if we can create symlinks (needs admin/dev-mode on Windows)."""
    try:
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            src.write_text("x", encoding="utf-8")
            (Path(d) / "lnk").symlink_to(src)
            return True
    except OSError:
        return False


def _load():
    """Import the script as a module (it is a script, not a package member)."""
    spec = importlib.util.spec_from_file_location("check_compat_pointers", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write(path: Path, text: str = CLEAN) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _scanned(mod, root: Path) -> set[str]:
    """The scanner's own file stream, as root-relative posix paths."""
    return {p.relative_to(root).as_posix() for p in mod._py_files(root)}


def _hits(out: str) -> set[str]:
    """The reported ``file:line: …`` lines of ``main()``'s stdout."""
    return {ln.strip().replace("\\", "/") for ln in out.splitlines() if ln.startswith("  ")}


class _ScandirSpy:
    """Record every directory ``os.scandir`` is asked to open — i.e. every dir the walk enters."""

    def __init__(self):
        self.opened: list[str] = []
        self._orig = os.scandir

    def __enter__(self):
        spy = self

        def counting_scandir(path=".", *a, **kw):
            spy.opened.append(os.fspath(path))
            return spy._orig(path, *a, **kw)

        os.scandir = counting_scandir
        return self

    def __exit__(self, *exc):
        os.scandir = self._orig
        return False

    def under(self, path: Path) -> list[str]:
        prefix = str(path) + os.sep
        return [p for p in self.opened if p == str(path) or p.startswith(prefix)]


def test_excluded_directories_are_pruned_at_any_depth_and_never_entered(tmp_path):
    mod = _load()
    _write(tmp_path / "first_party" / "a.py")
    _write(tmp_path / "first_party" / "node_modules" / "dep" / "dep.py", POINTER)
    _write(tmp_path / "node_modules" / "top" / "top.py", POINTER)
    _write(tmp_path / _NESTED_DEP / "x" / "y.py", POINTER)
    _write(tmp_path / "pkg" / "deep" / "nested" / "keep.py")
    # `vendor` is a first-party directory here, not a dependency tree: it stays in scope.
    _write(tmp_path / "pkg" / "deep" / "vendor" / "vendored.py")

    assert _scanned(mod, tmp_path) == {
        "first_party/a.py",
        "pkg/deep/nested/keep.py",
        "pkg/deep/vendor/vendored.py",
    }

    with _ScandirSpy() as spy:
        _scanned(mod, tmp_path)

    for excluded in (Path("node_modules"), Path("first_party") / "node_modules", _NESTED_DEP):
        assert spy.under(tmp_path / excluded) == [], f"excluded tree {excluded} was entered"
    # …and the walk does still reach the directories that *contain* an excluded tree, so the prune
    # happens at the excluded directory rather than by skipping its parent.
    assert spy.under(tmp_path / "pkg" / "deep" / "nested")
    assert spy.under(tmp_path / "first_party")


def test_exclusion_predicate_keeps_root_layout_refs_in_scope(tmp_path):
    """Root-only layout exclusions must not leak down: `tests/skills/` is first-party code."""
    mod = _load()
    assert mod._excluded_dir(Path("skills"))
    assert mod._excluded_dir(Path("evals"))
    assert mod._excluded_dir(Path(".git"))
    assert mod._excluded_dir(Path("node_modules"))
    assert mod._excluded_dir(Path("a") / "b" / "node_modules")
    assert not mod._excluded_dir(Path("tests") / "skills")
    assert not mod._excluded_dir(Path("tests") / "evals")
    assert not mod._excluded_dir(Path("pkg") / "skills")
    assert not mod._excluded_dir(Path("pkg") / "vendor")

    _write(tmp_path / "skills" / "s.py")
    _write(tmp_path / "evals" / "e.py")
    _write(tmp_path / "website" / "w.py")
    _write(tmp_path / "tests" / "skills" / "test_s.py")
    _write(tmp_path / "tests" / "evals" / "test_e.py")
    _write(tmp_path / "pkg" / "skills" / "s.py")

    assert _scanned(mod, tmp_path) == {
        "tests/skills/test_s.py",
        "tests/evals/test_e.py",
        "pkg/skills/s.py",
    }


def test_pointer_in_a_nested_dependency_tree_is_never_scanned_or_reported(tmp_path, monkeypatch, capsys):
    mod = _load()
    _write(tmp_path / "compat_manifest.json", json.dumps(MANIFEST_ENTRIES))
    # The regression: this file lives in a dependency tree vendored inside a first-party package,
    # and used to be reported as an in-tree violation.
    _write(tmp_path / "hermes_cli" / "node_modules" / "dependency.py", POINTER)
    _write(tmp_path / "hermes_cli" / "clean.py")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "MANIFEST", tmp_path / "compat_manifest.json")

    assert mod.main() == 0
    out = capsys.readouterr().out
    assert _hits(out) == set()
    assert "node_modules" not in out


def test_first_party_pointer_is_still_reported_and_reported_once(tmp_path, monkeypatch, capsys):
    mod = _load()
    _write(tmp_path / "compat_manifest.json", json.dumps(MANIFEST_ENTRIES))
    _write(tmp_path / "hermes_cli" / "uses_pointer.py", POINTER)
    _write(tmp_path / "hermes_cli" / "node_modules" / "dependency.py", POINTER)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "MANIFEST", tmp_path / "compat_manifest.json")

    assert mod.main() == 1
    out = capsys.readouterr().out
    assert _hits(out) == {"hermes_cli/uses_pointer.py:1: from hermes_cli.kanban_db import connect"}
    assert "node_modules" not in out


def test_path_patterns_prune_by_location_or_by_name(tmp_path, monkeypatch):
    mod = _load()
    _write(tmp_path / "pkg" / "a" / "vendor" / "v.py")
    _write(tmp_path / "pkg" / "a" / "b" / "vendor" / "v.py")
    _write(tmp_path / "other" / "vendor" / "v.py")
    _write(tmp_path / "pkg" / "a" / "keep.py")

    # A location pattern: fnmatch's `*` crosses `/`, so this prunes at any depth under pkg/.
    monkeypatch.setattr(mod, "SKIP_PATH_PATTERNS", ("pkg/*/vendor",))
    assert _scanned(mod, tmp_path) == {"other/vendor/v.py", "pkg/a/keep.py"}

    # A bare name pattern applies at every depth.
    monkeypatch.setattr(mod, "SKIP_PATH_PATTERNS", ("vendor",))
    assert _scanned(mod, tmp_path) == {"pkg/a/keep.py"}


def test_scanner_still_skips_itself_and_the_compat_contract_test(tmp_path):
    mod = _load()
    _write(tmp_path / "scripts" / "check_compat_pointers.py", POINTER)
    _write(tmp_path / "tests" / "hermes_cli" / "test_compat_manifest_targets.py", POINTER)
    _write(tmp_path / "keep.py")
    assert _scanned(mod, tmp_path) == {"keep.py"}


@pytest.mark.skipif(not _can_symlink(), reason="Symlinks need elevated privileges")
def test_linked_trees_are_not_followed_and_a_symlink_loop_terminates(tmp_path):
    mod = _load()
    outside = tmp_path / "outside"
    _write(outside / "dep.py", POINTER)
    tree = tmp_path / "tree"
    _write(tree / "src" / "a.py")
    (tree / "src" / "linked").symlink_to(outside, target_is_directory=True)
    (tree / "src" / "loop").symlink_to(tree, target_is_directory=True)

    assert _scanned(mod, tree) == {"src/a.py"}
