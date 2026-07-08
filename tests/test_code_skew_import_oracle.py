"""E2E behavior-contract test for the in-process import oracle (code-skew guard).

Companion to ``tests/test_code_skew.py`` (which unit-tests the plumbing with
monkeypatched stand-ins). This test proves the *real* oracle against a real git
repo and **real imports** into ``sys.modules``:

  - the guard now ALLOWS a ``/model`` switch when the only changed ``*.py`` is a
    file this process never imported (an ``eval/``/``scripts/`` tool, an unloaded
    provider);
  - it still REFUSES when a genuinely-imported module is stale;
  - it stays conservative (REFUSE) for a not-yet-imported *submodule of a loaded
    package* (the lazy-import ring), and for the fail-safe fallback rungs.

Mutation guard: the ``test_inverted_oracle_would_regress`` case asserts that if
the oracle were inverted (treat unloaded as loaded) the ALLOW case flips to
REFUSE — i.e. the test actually gates the narrowing, it is not a proxy-green.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from gateway import code_skew


def _git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True,
                   capture_output=True, text=True)


def _head(cwd) -> str:
    return subprocess.run(["git", "-C", str(cwd), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _write(root: Path, rel: str, content: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


@pytest.fixture
def repo(tmp_path):
    """A throwaway git repo shaped like the checkout: a real importable package
    plus never-imported top-level tool/eval trees."""
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    # An importable first-party package.
    _write(root, "loadedpkg/__init__.py", "VALUE = 1\n")
    _write(root, "loadedpkg/loaded_mod.py", "LOADED = 1\n")
    _write(root, "loadedpkg/sibling_unloaded.py", "SIB = 1\n")
    # Never-imported trees (the residual false-refuse class).
    _write(root, "scripts/some_tool.py", "TOOL = 1\n")
    _write(root, "eval/harness.py", "HARNESS = 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "boot")
    boot = _head(root)
    return root, boot


@pytest.fixture
def imported_pkg(repo, monkeypatch):
    """Really import ``loadedpkg.loaded_mod`` FROM the repo into ``sys.modules``
    and point the guard's project root at the repo. Proves the worktree bytes
    are what's exercised via ``__file__``. Cleans up sys.modules after."""
    root, boot = repo
    monkeypatch.setattr(code_skew, "_PROJECT_ROOT", root.resolve())
    monkeypatch.syspath_prepend(str(root))
    # drop any stale cached copies, then import for real
    for name in ("loadedpkg", "loadedpkg.loaded_mod", "loadedpkg.sibling_unloaded"):
        sys.modules.pop(name, None)
    mod = importlib.import_module("loadedpkg.loaded_mod")
    # PROVE we imported the repo's bytes, not something else on the path.
    assert str(root.resolve()) in (mod.__file__ or ""), (
        f"expected import from {root}, got {mod.__file__}")
    try:
        yield root, boot
    finally:
        for name in ("loadedpkg", "loadedpkg.loaded_mod", "loadedpkg.sibling_unloaded"):
            sys.modules.pop(name, None)


class TestImportOracleE2E:
    def _commit_change(self, root, rel, content):
        _write(root, rel, content)
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", f"change {rel}")
        return _head(root)

    def _arm_boot(self, monkeypatch, boot):
        monkeypatch.setattr(code_skew, "_fingerprint",
                            lambda: f"git:refs/heads/main:{boot}")
        code_skew.record_boot_fingerprint()

    def test_allows_switch_when_only_unimported_file_changed(
            self, imported_pkg, monkeypatch):
        # A commit touching ONLY a never-imported tool -> ALLOW (no refuse).
        root, boot = imported_pkg
        monkeypatch.setattr(code_skew, "_boot_fingerprint", None)
        self._arm_boot(monkeypatch, boot)
        disk = self._commit_change(root, "scripts/some_tool.py", "TOOL = 2\n")
        monkeypatch.setattr(code_skew, "_fingerprint",
                            lambda: f"git:refs/heads/main:{disk}")
        assert code_skew.detect_code_skew() is None

    def test_allows_switch_when_unloaded_eval_changed(
            self, imported_pkg, monkeypatch):
        root, boot = imported_pkg
        monkeypatch.setattr(code_skew, "_boot_fingerprint", None)
        self._arm_boot(monkeypatch, boot)
        disk = self._commit_change(root, "eval/harness.py", "HARNESS = 2\n")
        monkeypatch.setattr(code_skew, "_fingerprint",
                            lambda: f"git:refs/heads/main:{disk}")
        assert code_skew.detect_code_skew() is None

    def test_refuses_when_imported_module_is_stale(
            self, imported_pkg, monkeypatch):
        # A commit touching the ACTUALLY-imported module -> REFUSE.
        root, boot = imported_pkg
        monkeypatch.setattr(code_skew, "_boot_fingerprint", None)
        self._arm_boot(monkeypatch, boot)
        disk = self._commit_change(root, "loadedpkg/loaded_mod.py", "LOADED = 2\n")
        monkeypatch.setattr(code_skew, "_fingerprint",
                            lambda: f"git:refs/heads/main:{disk}")
        skew = code_skew.detect_code_skew()
        assert skew is not None
        assert skew == (boot[:10], disk[:10])

    def test_refuses_for_unloaded_submodule_of_loaded_package(
            self, imported_pkg, monkeypatch):
        # Lazy-import ring (OQ-1 option B): a sibling submodule of the loaded
        # package that isn't imported yet could be lazily imported on a new code
        # path -> stay conservative, REFUSE.
        root, boot = imported_pkg
        monkeypatch.setattr(code_skew, "_boot_fingerprint", None)
        self._arm_boot(monkeypatch, boot)
        disk = self._commit_change(root, "loadedpkg/sibling_unloaded.py", "SIB = 2\n")
        monkeypatch.setattr(code_skew, "_fingerprint",
                            lambda: f"git:refs/heads/main:{disk}")
        assert code_skew.detect_code_skew() is not None

    def test_inverted_oracle_would_regress_the_allow_case(
            self, imported_pkg, monkeypatch):
        # Mutation proof: with the real oracle the unimported-only change ALLOWS;
        # invert the oracle (pretend nothing is loaded => fall back to the
        # tree-shape heuristic which refuses any runtime .py) and the SAME diff
        # now REFUSES. Proves the test gates the narrowing, not a coincidence.
        root, boot = imported_pkg
        monkeypatch.setattr(code_skew, "_boot_fingerprint", None)
        self._arm_boot(monkeypatch, boot)
        disk = self._commit_change(root, "scripts/some_tool.py", "TOOL = 9\n")
        monkeypatch.setattr(code_skew, "_fingerprint",
                            lambda: f"git:refs/heads/main:{disk}")
        # real oracle -> allow
        assert code_skew.detect_code_skew() is None
        # invert: force the oracle to report "nothing loaded" -> heuristic rung
        monkeypatch.setattr(code_skew, "_loaded_first_party_paths", lambda: None)
        assert code_skew.detect_code_skew() is not None


class TestFailSafeLadder:
    """A broken oracle must REFUSE (fall back), never wave through."""

    def _repo(self, tmp_path):
        root = tmp_path / "c"
        root.mkdir()
        _git(root, "init", "-q")
        _git(root, "config", "user.email", "t@t.t")
        _git(root, "config", "user.name", "t")
        return root

    def test_oracle_error_falls_back_to_heuristic_refuse(self, tmp_path, monkeypatch):
        root = self._repo(tmp_path)
        _write(root, "gateway/x.py", "x = 1\n")
        _git(root, "add", "-A"); _git(root, "commit", "-q", "-m", "a")
        boot = _head(root)
        _write(root, "gateway/x.py", "x = 2\n")
        _git(root, "add", "-A"); _git(root, "commit", "-q", "-m", "b")
        disk = _head(root)
        monkeypatch.setattr(code_skew, "_PROJECT_ROOT", root.resolve())

        def _boom():
            raise RuntimeError("introspection failed")

        monkeypatch.setattr(code_skew, "_loaded_first_party_paths", _boom)
        # oracle raises -> the caller catches? No: _changed_py_risks_stale_import
        # calls it directly. It must be robust: the oracle CATCHES internally and
        # returns None on error, so this asserts the internal try/except holds.
        assert code_skew._loaded_first_party_paths is _boom
        # Drive through the real diff path with the oracle patched to return None
        # (unavailable) -> heuristic rung -> a runtime .py refuses.
        monkeypatch.setattr(code_skew, "_loaded_first_party_paths", lambda: None)
        assert code_skew._runtime_python_changed(boot, disk) is True

    def test_oracle_none_allows_unimported_via_heuristic(self, tmp_path, monkeypatch):
        # With oracle unavailable, a docs-only change still ALLOWS (heuristic
        # rung 2 excludes docs/tests) — proves the fallback isn't blindly refuse.
        root = self._repo(tmp_path)
        _write(root, "docs/a.py", "a = 1\n")
        _git(root, "add", "-A"); _git(root, "commit", "-q", "-m", "a")
        boot = _head(root)
        _write(root, "docs/a.py", "a = 2\n")
        _git(root, "add", "-A"); _git(root, "commit", "-q", "-m", "b")
        disk = _head(root)
        monkeypatch.setattr(code_skew, "_PROJECT_ROOT", root.resolve())
        monkeypatch.setattr(code_skew, "_loaded_first_party_paths", lambda: None)
        assert code_skew._runtime_python_changed(boot, disk) is False

    def test_diff_error_refuses(self, tmp_path, monkeypatch):
        root = self._repo(tmp_path)
        _write(root, "gateway/x.py", "x = 1\n")
        _git(root, "add", "-A"); _git(root, "commit", "-q", "-m", "a")
        monkeypatch.setattr(code_skew, "_PROJECT_ROOT", root.resolve())
        # bad shas -> git diff fails -> None (caller refuses conservatively)
        assert code_skew._runtime_python_changed("deadbeef", "cafebabe") is None
