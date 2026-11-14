"""scripts/ci/check_public_surface.py detects silently dropped public names, methods and test defs.

Replayed against the Sep 2026 refactor PR at open it reports 1,703 public names dropped in 341 modules
(the figure reviewers had to find ~30 of by hand); here the contract is pinned on a throwaway repo.
"""
import importlib.util
import os
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
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": os.environ["PATH"]})


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


def test_a_moved_file_is_compared_with_its_old_self(tmp_path, monkeypatch):
    """Git reports a moved file as a rename, not a modification; a public name dropped during the move
    (into ``src/hermes_agent/`` or between packages) and a moved test file losing defs are both drops."""
    repo = tmp_path
    (repo / "agent").mkdir(); (repo / "tools").mkdir(); (repo / "tests").mkdir()
    (repo / "agent" / "mod.py").write_text("def keep(): ...\ndef gone(): ...\n" + "X = 1\n" * 20, encoding="utf-8")
    (repo / "tools" / "util.py").write_text("def stays(): ...\ndef lost(): ...\n" + "Y = 1\n" * 20, encoding="utf-8")
    (repo / "tests" / "test_mod.py").write_text("def test_a(): ...\ndef test_b(): ...\n" + "Z = 1\n" * 20, encoding="utf-8")
    (repo / "agent" / "out.py").write_text("def kept(): ...\ndef dropped(): ...\n" + "W = 1\n" * 20, encoding="utf-8")
    (repo / "agent" / "plain.py").write_text("def kept(): ...\ndef gone_too(): ...\n" + "V = 1\n" * 20, encoding="utf-8")
    _git(repo, "init", "-q", "-b", "main"); _git(repo, "add", "."); _git(repo, "commit", "-qm", "base")

    (repo / "src" / "hermes_agent" / "agent").mkdir(parents=True); (repo / "gateway").mkdir()
    (repo / "agent" / "mod.py").rename(repo / "src" / "hermes_agent" / "agent" / "mod.py")
    (repo / "src" / "hermes_agent" / "agent" / "mod.py").write_text("def keep(): ...\n" + "X = 1\n" * 20, encoding="utf-8")
    (repo / "tools" / "util.py").rename(repo / "gateway" / "util.py")
    (repo / "gateway" / "util.py").write_text("def stays(): ...\n" + "Y = 1\n" * 20, encoding="utf-8")
    (repo / "providers").mkdir()
    (repo / "agent" / "out.py").rename(repo / "providers" / "out.py")
    (repo / "providers" / "out.py").write_text("def kept(): ...\n" + "W = 1\n" * 20, encoding="utf-8")
    (repo / "agent" / "plain.py").rename(repo / "agent" / "caf\u00e9.py")
    (repo / "agent" / "caf\u00e9.py").write_text("def kept(): ...\n" + "V = 1\n" * 20, encoding="utf-8")
    (repo / "tests" / "test_mod.py").rename(repo / "tests" / "test_moved.py")
    (repo / "tests" / "test_moved.py").write_text("def test_a(): ...\n" + "Z = 1\n" * 20, encoding="utf-8")
    _git(repo, "add", "-A"); _git(repo, "commit", "-qm", "move")

    monkeypatch.chdir(repo)
    report = _load().diff_surface("HEAD~1", "HEAD")
    assert report["dropped_names"] == {"src/hermes_agent/agent/mod.py": ["gone"], "gateway/util.py": ["lost"],
                                       "providers/out.py": ["dropped"], "agent/caf\u00e9.py": ["gone_too"]}
    assert report["test_drops"] == {"tests/test_moved.py": (2, 1)}

    # A source module moved under tests/ and a test file moved out of tests/ keep their kind.
    (repo / "agent" / "helper.py").write_text("def kept(): ...\ndef lost_too(): ...\n" + "U = 1\n" * 20, encoding="utf-8")
    (repo / "tests" / "test_out.py").write_text("def test_a(): ...\ndef test_b(): ...\n" + "T = 1\n" * 20, encoding="utf-8")
    _git(repo, "add", "."); _git(repo, "commit", "-qm", "more files")
    (repo / "agent" / "helper.py").rename(repo / "tests" / "helper.py")
    (repo / "tests" / "helper.py").write_text("def kept(): ...\n" + "U = 1\n" * 20, encoding="utf-8")
    (repo / "scripts").mkdir()
    (repo / "tests" / "test_out.py").rename(repo / "scripts" / "test_out.py")
    (repo / "scripts" / "test_out.py").write_text("def test_a(): ...\n" + "T = 1\n" * 20, encoding="utf-8")
    _git(repo, "add", "-A"); _git(repo, "commit", "-qm", "kind-crossing moves")
    crossing = _load().diff_surface("HEAD~1", "HEAD")
    assert crossing["dropped_names"] == {"tests/helper.py": ["lost_too"]}
    assert crossing["test_drops"] == {"scripts/test_out.py": (2, 1)}

    # Once the move lands, later edits have no old top-level source path.
    (repo / "src" / "hermes_agent" / "agent" / "mod.py").write_text("X = 1\n" * 20, encoding="utf-8")
    _git(repo, "add", "."); _git(repo, "commit", "-qm", "drop after move")
    assert _load().diff_surface("HEAD~1", "HEAD")["dropped_names"] == {
        "src/hermes_agent/agent/mod.py": ["keep"]}
