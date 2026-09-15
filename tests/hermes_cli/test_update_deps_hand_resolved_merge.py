"""Regression: a hand-resolved merge made the dependency sync skip in silence.

`_editable_install_is_current` decides whether `uv pip install -e .` can be skipped by asking
whether the pulled commits touched any install-defining file: it diffs
``pre_pull_sha..HEAD`` over pyproject.toml / setup.py / setup.cfg / MANIFEST.in / uv.lock.

The updater captures ``pre_pull_sha`` with ``_capture_head_sha`` at the START of a run. When a
pull conflicts, the run aborts and tells the user to resolve it by hand -- and the user then
re-runs. On that second run HEAD already CONTAINS the hand-resolved merge, so ``pre_pull_sha``
and HEAD are the same commit, the diff is empty *by construction*, and a whole pull's worth of
dependency changes installs nothing. There is no "Python dependencies" line in the log.

Measured 2026-09-13 after exactly that sequence: ``snowballstemmer==3.1.1`` and
``pillow-heif==1.5.0`` were declared in pyproject.toml and absent from the venv.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hermes_cli.update_cmd_deps import _editable_install_is_current  # noqa: E402


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout


def _base_repo(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / "pyproject.toml").write_text('dependencies = ["httpx"]\n')
    (tmp_path / "module.py").write_text("x = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")


def _hand_resolved_merge(tmp_path: Path, incoming_file: str, incoming_body: str) -> str:
    """Return the SHA of a merge commit that brought `incoming_file` in -- the shape the updater
    sees on the re-run after a hand-resolved conflict."""
    _git(tmp_path, "checkout", "-q", "-b", "incoming")
    (tmp_path / incoming_file).write_text(incoming_body)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "incoming change")
    _git(tmp_path, "checkout", "-q", "main")
    (tmp_path / "unrelated.txt").write_text("local work\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "local work")
    _git(tmp_path, "merge", "--no-ff", "-q", "-m", "Merge incoming", "incoming")
    return _git(tmp_path, "rev-parse", "HEAD").strip()


def test_merge_that_changed_dependencies_is_not_skipped(tmp_path):
    """The invariant: if the merged-in commits touched an install-defining file, the install is
    NOT current -- regardless of pre_pull_sha happening to equal HEAD."""
    _base_repo(tmp_path)
    merge_sha = _hand_resolved_merge(tmp_path, "pyproject.toml",
                                    'dependencies = ["httpx", "snowballstemmer"]\n')

    # This is what the updater passes on the re-run after a hand-resolved conflict.
    assert _editable_install_is_current(["git"], tmp_path, merge_sha) is False


def test_merge_that_touched_only_unrelated_files_is_still_skipped(tmp_path):
    """The fast path must survive the fix: a merge bringing in no install-defining change is
    still current, so the reinstall (and its Windows hermes.exe quarantine race) is skipped."""
    _base_repo(tmp_path)
    merge_sha = _hand_resolved_merge(tmp_path, "module.py", "x = 2\n")

    assert _editable_install_is_current(["git"], tmp_path, merge_sha) is True


def test_no_parent_to_compare_against_fails_closed(tmp_path):
    """A root commit has no parent; the function must report 'not current' (install) rather
    than skip on a diff it could not take."""
    _base_repo(tmp_path)
    root_sha = _git(tmp_path, "rev-list", "--max-parents=0", "HEAD").strip()

    assert _editable_install_is_current(["git"], tmp_path, root_sha) is False
