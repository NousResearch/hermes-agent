"""Every job of the main CI lanes ends by checking that it left the checkout clean.

The step (.github/actions/check-clean-worktree) runs ``git status --porcelain=v1
--untracked-files=normal`` and reports tracked changes and untracked, non-ignored
files. Stray test output reached main more than once (log.txt, default.tar.gz,
MagicMock/<name>/<id> SQLite stores, a ``%SystemDrive%`` tree); .gitignore hides
the rest. Its mode is set in ONE place, the action's ``mode`` input default.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]
ACTION_DIR = ".github/actions/check-clean-worktree"
ACTION = ROOT / ACTION_DIR / "action.yml"
USES = f"./{ACTION_DIR}"
WORKFLOWS = ("tests.yml", "tests-os.yml", "lint.yml", "js-tests.yml")
MODES = ("blocking", "warning", "off")


def _load(path: Path):
    return YAML(typ="base").load(path.read_text(encoding="utf-8"))


def _checkout_jobs():
    for filename in WORKFLOWS:
        for name, job in (_load(ROOT / ".github/workflows" / filename)["jobs"] or {}).items():
            steps = job.get("steps") or []  # reusable-workflow calls have none
            if any(str(step.get("uses", "")).startswith("actions/checkout@") for step in steps):
                yield f"{filename}:{name}", job, steps


def test_every_checkout_job_ends_with_the_clean_worktree_step():
    jobs = list(_checkout_jobs())
    assert len(jobs) >= 10, [label for label, *_ in jobs]  # tests 4, tests-os 2, lint 4, js 1
    problems = []
    for label, job, steps in jobs:
        last = steps[-1]
        if last.get("uses") != USES:
            problems.append(f"{label}: last step is {last.get('name') or last.get('uses')!r}")
            continue
        if "if" in last:
            # Default success(): after a failed step the job is red already, and that step may
            # have left temp files its own later cleanup would remove (false positives).
            problems.append(f"{label}: runs with `if: {last['if']}`; leave `if:` off")
        if "mode" in (last.get("with") or {}):
            problems.append(f"{label}: passes `mode`; the action's default is the only switch")
        checkout = next(s for s in steps if str(s.get("uses", "")).startswith("actions/checkout@"))
        sparse = str((checkout.get("with") or {}).get("sparse-checkout", ""))
        if sparse and ACTION_DIR not in sparse.split():
            problems.append(f"{label}: sparse checkout omits {ACTION_DIR}, so the step cannot load")
    assert not problems, "\n".join(problems)


def test_no_workflow_overrides_the_master_switch():
    offenders = []
    for path in sorted((ROOT / ".github/workflows").glob("*.y*ml")):
        for name, job in (_load(path).get("jobs") or {}).items():
            for step in job.get("steps") or []:
                if step.get("uses") == USES and "mode" in (step.get("with") or {}):
                    offenders.append(f"{path.name}:{name}")
    assert not offenders, offenders


def test_master_switch_is_a_known_mode():
    assert _load(ACTION)["inputs"]["mode"]["default"] in MODES


# ── behaviour: run the action's script against a real repository ──────────────────────


@pytest.fixture
def repo(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not installed")
    work = tmp_path / "repo"
    work.mkdir()
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    for args in (["init", "-q"], ["config", "user.email", "t@example.invalid"], ["config", "user.name", "t"]):
        subprocess.run([git, *args], cwd=work, env=env, check=True)
    (work / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (work / "tracked.txt").write_text("one\n", encoding="utf-8")
    subprocess.run([git, "add", "."], cwd=work, env=env, check=True)
    subprocess.run([git, "commit", "-qm", "init"], cwd=work, env=env, check=True)
    return work, env


def _run(cwd: Path, env: dict, mode: str, tmp_path: Path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not installed")
    script = _load(ACTION)["runs"]["steps"][0]["run"]
    summary = tmp_path / "summary.md"
    result = subprocess.run(
        [bash, "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
        cwd=cwd, env={**env, "_MODE": mode, "GITHUB_STEP_SUMMARY": str(summary)},
        capture_output=True, text=True, encoding="utf-8", timeout=60, check=False,
    )
    return result.returncode, result.stdout, summary


@pytest.mark.platforms("posix")
def test_clean_checkout_passes_and_ignored_files_do_not_count(repo, tmp_path):
    work, env = repo
    (work / "ignored").mkdir()
    (work / "ignored" / "out.db").write_text("x", encoding="utf-8")
    code, out, _ = _run(work, env, "blocking", tmp_path)
    assert code == 0, out
    assert "clean" in out and "::error" not in out


@pytest.mark.platforms("posix")
def test_blocking_fails_and_names_untracked_and_modified_files(repo, tmp_path):
    work, env = repo
    (work / "log.txt").write_text("stray", encoding="utf-8")
    (work / "tracked.txt").write_text("two\n", encoding="utf-8")
    code, out, summary = _run(work, env, "blocking", tmp_path)
    assert code == 1, out
    assert "::error title=Dirty checkout::log.txt: untracked file left in the checkout" in out
    assert "::error title=Dirty checkout::tracked.txt: tracked file changed" in out
    assert "tracked.txt | 2 +-" in out  # git diff --stat
    assert "log.txt" in summary.read_text(encoding="utf-8")


@pytest.mark.platforms("posix")
def test_warning_mode_reports_without_failing(repo, tmp_path):
    work, env = repo
    (work / "default.tar.gz.part").write_text("stray", encoding="utf-8")
    code, out, _ = _run(work, env, "warning", tmp_path)
    assert code == 0, out
    assert "::warning title=Dirty checkout::default.tar.gz.part" in out
    assert "::error" not in out


@pytest.mark.platforms("posix")
def test_off_skips_and_unknown_mode_fails_closed(repo, tmp_path):
    work, env = repo
    (work / "stray").write_text("x", encoding="utf-8")
    assert _run(work, env, "off", tmp_path)[0] == 0
    code, out, _ = _run(work, env, "blockng", tmp_path)
    assert code == 1 and "::error" in out


@pytest.mark.platforms("posix")
def test_annotations_cap_at_ten_with_the_full_list_in_the_log(repo, tmp_path):
    work, env = repo
    for i in range(25):
        (work / f"stray{i:02}").write_text("x", encoding="utf-8")
    code, out, _ = _run(work, env, "blocking", tmp_path)
    assert code == 1
    assert out.count("::error ") == 10  # GitHub keeps 10 per step: 9 files + the summary line
    assert "left 25 changed or untracked path(s)" in out
    assert all(f"?? stray{i:02}" in out for i in range(25))


@pytest.mark.platforms("posix")
def test_missing_checkout_is_reported(tmp_path):
    bare = tmp_path / "nogit"
    bare.mkdir()
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    code, out, _ = _run(bare, env, "blocking", tmp_path)
    assert code == 1 and "no git checkout" in out
    assert _run(bare, env, "warning", tmp_path)[0] == 0
