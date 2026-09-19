"""scripts/ci/check_silent_revert.py flags a branch whose file matches a version the target already moved past.

Contract pinned on a throwaway repo: a branch that writes back the pre-landing blob of a file is a
finding (the landed commit would be undone without a conflict); ordinary forward work on the same file
and a brand-new file are not. Ported from gastownhall/gastown#4840.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check_silent_revert.py"
_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin:/usr/local/bin"}


def _load():
    spec = importlib.util.spec_from_file_location("check_silent_revert", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass field resolution needs the module registered
    spec.loader.exec_module(mod)
    return mod


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=_ENV)


def _commit(repo, path, text, msg):
    (repo / path).write_text(text, encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-qm", msg)


def test_stale_branch_carrying_pre_landing_blob_is_flagged_and_forward_work_is_not(tmp_path):
    repo = tmp_path
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "a.py", "v1\n", "base")
    _commit(repo, "b.py", "x\n", "add b")
    _commit(repo, "a.py", "v2 (landed fix)\n", "fix: land v2 of a")
    # The observed class: the branch is cut AFTER v2 landed (the landing commit is its direct parent, so
    # there is nothing to three-way against), and the agent writes back the stale v1 copy it was holding.
    # To git that is an ordinary edit; to the repo it undoes the landed fix.
    _git(repo, "checkout", "-q", "-b", "stale")
    _commit(repo, "a.py", "v1\n", "sync a.py from my copy")
    _commit(repo, "b.py", "x\ny\n", "forward work on b")  # forward edit: not a revert
    _commit(repo, "c.py", "new\n", "brand new file")  # new file: cannot revert anything
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "--no-edit", "stale")  # merges cleanly: nothing conflicts, v2 is gone

    mod = _load()
    findings = mod.check_for_reverts("main~1", "stale", cwd=str(repo))
    assert [f.path for f in findings] == ["a.py"]
    assert findings[0].superseded_subject == "base"
    assert "would undo" in str(findings[0])

    # --strict is the hard gate; default stays advisory (exit 0) so a deliberate revert is a caller decision.
    strict = subprocess.run(["python3", str(SCRIPT), "--base", "main~1", "--head", "stale", "--strict"],
                            cwd=repo, capture_output=True, text=True, env=_ENV)
    assert strict.returncode == 1 and "a.py" in strict.stdout and "b.py" not in strict.stdout
    advisory = subprocess.run(["python3", str(SCRIPT), "--base", "main~1", "--head", "stale"],
                              cwd=repo, capture_output=True, text=True, env=_ENV)
    assert advisory.returncode == 0
