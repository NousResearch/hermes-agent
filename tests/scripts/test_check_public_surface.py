"""scripts/ci/check_public_surface.py detects silently dropped public names, methods and test defs.

Replayed against the Sep 2026 refactor PR at open it reports 1,703 public names dropped in 341 modules
(the figure reviewers had to find ~30 of by hand); here the contract is pinned on a throwaway repo.
"""
import importlib.util
import subprocess
import textwrap
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check_public_surface.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_public_surface", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin:/usr/local/bin"})


def test_dropped_public_names_methods_and_test_defs_are_reported_and_moves_are_not(tmp_path, monkeypatch):
    repo = tmp_path
    (repo / "agent").mkdir(); (repo / "tests").mkdir(); (repo / "docs").mkdir()
    (repo / "agent" / "mod.py").write_text(textwrap.dedent('''
        import os
        from json import loads as parse

        LIMIT = 3
        def keep(): ...
        def gone(): ...
        def _private(): ...
        class K:
            def pub(self): ...
            def __len__(self): return 0
            def _hidden(self): ...
        class Big:
            def stays(self): ...
            def extracted(self): ...
    '''), encoding="utf-8")
    (repo / "agent" / "moved.py").write_text("def relocated(): ...\n", encoding="utf-8")
    (repo / "tests" / "test_x.py").write_text("def test_a(): ...\ndef test_b(): ...\nasync def test_c(): ...\n", encoding="utf-8")
    (repo / "docs" / "notes.py").write_text("def doc_helper(): ...\n", encoding="utf-8")
    _git(repo, "init", "-q", "-b", "main"); _git(repo, "add", "."); _git(repo, "commit", "-qm", "base")

    # HEAD: drop `gone`, `parse`, `K.pub`, `K.__len__`; drop `_private`/`K._hidden` (private: fine);
    # move `relocated` into mod with a re-export left behind (fine); lose one test def; change a
    # non-source module (ignored).
    (repo / "agent" / "mod.py").write_text(textwrap.dedent('''
        import os
        from agent.moved import relocated  # re-export

        LIMIT = 3
        def keep(): ...
        class K: ...
        class _BigMixin:
            def extracted(self): ...   # moved into an in-module base: still reachable on Big
        class Big(_BigMixin):
            def stays(self): ...
    '''), encoding="utf-8")
    (repo / "agent" / "moved.py").write_text("def relocated(): ...\ndef relocated2(): ...\n", encoding="utf-8")
    (repo / "tests" / "test_x.py").write_text("def test_a(): ...\nasync def test_c(): ...\n", encoding="utf-8")
    (repo / "docs" / "notes.py").write_text("def other(): ...\n", encoding="utf-8")
    _git(repo, "add", "."); _git(repo, "commit", "-qm", "head")

    monkeypatch.chdir(repo)
    mod = _load()
    report = mod.diff_surface("HEAD~1", "HEAD")
    assert report["dropped_names"] == {"agent/mod.py": ["gone", "parse"]}
    assert report["dropped_methods"] == {"agent/mod.py": ["K.__len__", "K.pub"]}
    assert report["test_drops"] == {"tests/test_x.py": (3, 2)}
    assert mod.main(["--base", "HEAD~1", "--head", "HEAD"]) == 0          # advisory
    assert mod.main(["--base", "HEAD~1", "--head", "HEAD", "--strict"]) == 1
    assert mod.main(["--base", "HEAD", "--head", "HEAD", "--strict"]) == 0  # nothing dropped


def test_unresolvable_refs_are_an_error_not_a_clean_report(tmp_path, monkeypatch):
    """Independent-review witness: a nonexistent base ref under --strict reported zero drops and exited 0,
    which would make a mis-fetched CI job look clean."""
    repo = tmp_path
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "init", "-q", "-b", "main"); _git(repo, "add", "."); _git(repo, "commit", "-qm", "base")
    monkeypatch.chdir(repo)
    mod = _load()
    assert mod.main(["--base", "origin/does-not-exist", "--head", "HEAD", "--strict"]) == 2
    assert mod.main(["--base", "origin/does-not-exist", "--head", "HEAD"]) == 2
