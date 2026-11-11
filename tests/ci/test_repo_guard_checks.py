"""The repository guards in scripts/ci, run as their workflow steps run them, on temp git repos.

Each guard is a step of .github/workflows/repo-guards.yml: it reads the checkout, prints its
verdict, exits non-zero to fail the step, and publishes a review_status line that the live PR
comment (scripts/ci/live_comment.py) parses.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "ci"


def _git_env(name: str = "Main", email: str = "teknium@example.com") -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME=name,
               GIT_AUTHOR_EMAIL=email, GIT_COMMITTER_NAME=name, GIT_COMMITTER_EMAIL=email)
    return env


def git(repo: Path, *args: str, author: tuple[str, str] = ("Main", "teknium@example.com")) -> str:
    return subprocess.run(["git", *args], cwd=repo, env=_git_env(*author), capture_output=True,
                          text=True, check=True, stdin=subprocess.DEVNULL, timeout=60).stdout


def commit(repo: Path, files: dict[str, str | bytes], author=("Main", "teknium@example.com")) -> str:
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", "c", author=author)
    return git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo whose origin/main is its root commit."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "update-ref", "refs/remotes/origin/main", commit(root, {"README": "x\n"}))
    return root


class Step:
    def __init__(self, tmp_path: Path):
        self.runner_temp = tmp_path / "runner-temp"
        self.runner_temp.mkdir()
        self.output = tmp_path / "github-output"
        self.summary = tmp_path / "step-summary"

    def run(self, repo: Path, script: str, *args: str, **env: str) -> subprocess.CompletedProcess:
        full_env = dict(_git_env(), RUNNER_TEMP=str(self.runner_temp), GITHUB_OUTPUT=str(self.output),
                        GITHUB_STEP_SUMMARY=str(self.summary), **env)
        return subprocess.run([sys.executable, str(SCRIPTS / script), *args], cwd=repo, env=full_env,
                              capture_output=True, text=True, encoding="utf-8",
                              stdin=subprocess.DEVNULL, timeout=120, check=False)

    def outputs(self) -> dict[str, str]:
        lines = self.output.read_text(encoding="utf-8").splitlines() if self.output.exists() else []
        return dict(line.split("=", 1) for line in lines)

    def review_status(self) -> list[dict]:
        return json.loads(self.outputs()["review_status"])

    def artifact(self, name: str) -> list[dict]:
        text = (self.runner_temp / "review-status" / name / "review-status.json").read_text(encoding="utf-8")
        return json.loads(text.removeprefix("review_status="))


@pytest.fixture
def step(tmp_path: Path) -> Step:
    return Step(tmp_path)


def test_infographic_images_fail_in_any_spelling_and_other_images_pass(repo, step):
    commit(repo, {"docs/assets/infographic.png": b"product", "infographic/notes.md": "text"})
    passed = step.run(repo, "check_infographic_images.py")
    assert passed.returncode == 0, passed.stdout
    assert step.review_status() == []

    commit(repo, {"pr/infograficos/a.PNG": b"img", "x/Infographic-v2/b.jpeg": b"img"})
    step.output.unlink()
    failed = step.run(repo, "check_infographic_images.py")
    assert failed.returncode == 1
    assert "::error::2 PR-infographic image(s) are tracked in git." in failed.stdout
    assert "  pr/infograficos/a.PNG\n" in failed.stdout
    assert step.review_status()[0]["results"][0]["title"] == "PR infographic committed to the repo"


def test_unrelated_history_fails_and_publishes_the_history_artifact(repo, step):
    commit(repo, {"a": "1"})
    assert step.run(repo, "check_common_ancestor.py").returncode == 0
    assert step.artifact("history-check") == []

    git(repo, "checkout", "-q", "--orphan", "orphan")
    commit(repo, {"b": "2"})
    failed = step.run(repo, "check_common_ancestor.py")
    assert failed.returncode == 1
    assert "::error::This PR has no common ancestor with main." in failed.stdout
    assert step.artifact("history-check")[0]["source"] == "unrelated histories"


def test_only_unmapped_human_emails_fail_the_contributor_check(repo, step):
    commit(repo, {"contributors/emails/mapped@x.org": "alice\n",
                  "scripts/releases/authors_legacy.py": 'AUTHOR_MAP = {"legacy@y.org": "bob"}\n'})
    for email in ("mapped@x.org", "legacy@y.org", "12+c@users.noreply.github.com",
                  "noreply@anthropic.com"):
        commit(repo, {email: "x"}, author=("Known", email))
    assert step.run(repo, "check_contributor_emails.py").returncode == 0
    assert step.artifact("contributor-check") == []

    commit(repo, {"u": "1"}, author=("Zoë New", "zoe@example.net"))
    failed = step.run(repo, "check_contributor_emails.py")
    assert failed.returncode == 1
    assert "python3 scripts/add_contributor.py zoe@example.net <github-username>" in failed.stdout
    [status] = step.artifact("contributor-check")
    assert status["results"][0]["detail"] == "zoe@example.net (Zoë New)"


def test_supply_chain_scan_flags_payload_lines_but_not_lockfiles(repo, step):
    base = git(repo, "rev-parse", "HEAD").strip()
    # Built from pieces: the scanner reads this PR's own added lines, so no line here may match it.
    decode, run = "base64." + "b64decode", "subprocess." + "run("
    payload = f"exec({decode}(blob))\n"
    head = commit(repo, {"uv.lock": payload, "ok.py": "print(1)\n"})
    shas = {"_HERMES_PR_BASE_SHA": base, "_HERMES_PR_HEAD_SHA": head}
    assert step.run(repo, "scan_supply_chain.py", "patterns", **shas).returncode == 0
    assert step.outputs()["found"] == "false"

    head = commit(repo, {"evil.py": payload + run + "['sh', '-c', " + "chr" + "(65)])\n"})
    step.output.unlink()
    assert step.run(repo, "scan_supply_chain.py", "patterns", _HERMES_PR_BASE_SHA=base,
                    _HERMES_PR_HEAD_SHA=head).returncode == 0
    assert step.outputs()["found"] == "true"
    report = (step.runner_temp / "supply-chain-findings.md").read_text(encoding="utf-8")
    assert "base64 decode + exec/eval combo" in report
    assert "subprocess with encoded/obfuscated command" in report


def _scan_pr(repo: Path, step: Step, base: str, files: dict[str, str]) -> str:
    """Commit ``files`` on a fresh branch off ``base`` and return the patterns scan's verdict."""
    git(repo, "checkout", "-q", "--detach", base)
    head = commit(repo, files)
    step.output.unlink(missing_ok=True)
    assert step.run(repo, "scan_supply_chain.py", "patterns", _HERMES_PR_BASE_SHA=base,
                    _HERMES_PR_HEAD_SHA=head).returncode == 0
    return step.outputs()["found"]


def _module_with_payload() -> tuple[str, str]:
    """A module that already holds a payload-shaped line, and a different payload line."""
    decode = "base64." + "b64decode"
    body = "".join(f"def f{i}():\n    return {i}\n\n" for i in range(20))
    return body + f"code = exec({decode}(blob))\n", f"x = eval({decode}(other))\n"


FORWARD = "from new.pkg.mod import *  # moved to new/pkg/mod.py\n"


def test_supply_chain_scan_passes_a_moved_file_whose_old_path_keeps_a_forwarding_file(repo, step):
    module, _ = _module_with_payload()
    base = commit(repo, {"pkg/mod.py": module})
    assert _scan_pr(repo, step, base, {"new/pkg/mod.py": module, "pkg/mod.py": FORWARD}) == "false"


def test_supply_chain_scan_passes_a_move_into_a_folder_git_quotes(repo, step):
    module, _ = _module_with_payload()
    base = commit(repo, {"pkg/mod.py": module})
    assert _scan_pr(repo, step, base, {"n\u00e9w/pkg/mod.py": module, "pkg/mod.py": FORWARD}) == "false"


def test_supply_chain_scan_flags_a_payload_line_added_to_a_moved_or_new_file(repo, step):
    module, new_payload = _module_with_payload()
    base = commit(repo, {"pkg/mod.py": module})
    assert _scan_pr(repo, step, base, {"new/pkg/mod.py": module + new_payload,
                                       "pkg/mod.py": FORWARD}) == "true"
    assert "x = eval(" in (step.runner_temp / "supply-chain-findings.md").read_text(encoding="utf-8")
    assert _scan_pr(repo, step, base, {"fresh.py": new_payload}) == "true"


def test_supply_chain_scan_flags_a_payload_line_copied_from_a_file_that_keeps_it(repo, step):
    module, _ = _module_with_payload()
    base = commit(repo, {"pkg/mod.py": module})
    copied = {"elsewhere/run.py": module, "pkg/mod.py": module + "# touched\n"}
    assert _scan_pr(repo, step, base, copied) == "true"
    assert "code = exec(" in (step.runner_temp / "supply-chain-findings.md").read_text(encoding="utf-8")


def test_supply_chain_scan_flags_the_second_copy_of_a_moved_file(repo, step):
    module, _ = _module_with_payload()
    base = commit(repo, {"pkg/mod.py": module})
    twice = {"new/a/mod.py": module, "new/b/mod.py": module, "pkg/mod.py": FORWARD}
    assert _scan_pr(repo, step, base, twice) == "true"


def test_supply_chain_scan_flags_a_payload_line_taken_from_a_file_of_another_name(repo, step):
    module, _ = _module_with_payload()
    payload = module.splitlines(keepends=True)[-1]
    base = commit(repo, {"tests/fixture.py": module, "pkg/run.py": "x = 1\n"})
    taken = {"tests/fixture.py": module.removesuffix(payload), "pkg/run.py": "x = 1\n" + payload}
    assert _scan_pr(repo, step, base, taken) == "true"


def test_dependency_bounds_report_lists_only_floor_only_specs(repo, step):
    base = commit(repo, {"pyproject.toml": 'deps = [\n  "old>=1",\n]\n'})
    head = commit(repo, {"pyproject.toml": 'deps = [\n  "old>=1",\n  "bounded>=2,<3", "loose>=2.1",\n]\n'})
    assert step.run(repo, "scan_supply_chain.py", "dependency-bounds", _HERMES_PR_BASE_SHA=base,
                    _HERMES_PR_HEAD_SHA=head).returncode == 0
    assert step.outputs()["found"] == "true"
    report = (step.runner_temp / "supply-chain-unbounded.txt").read_text(encoding="utf-8")
    assert report == '"loose>=2.1"\n'


def test_emit_publishes_the_report_only_when_the_flag_is_true(repo, step):
    (step.runner_temp / "supply-chain-unbounded.txt").write_text('"loose>=2.1"\n', encoding="utf-8")
    step.run(repo, "review_status.py", "emit", "dependency-bounds", FOUND="false")
    assert step.review_status() == []

    step.output.unlink()
    step.run(repo, "review_status.py", "emit", "dependency-bounds", FOUND="true")
    [status] = step.review_status()
    assert status["results"][0]["detail"] == '"loose>=2.1"\n'


def test_merge_concatenates_the_supply_chain_statuses_and_skips_malformed_ones(repo, step):
    one = json.dumps([{"source": "supply chain", "results": []}])
    step.run(repo, "review_status.py", "merge", "supply-chain", "SCAN", "DEPS", "BAD",
             SCAN=one, DEPS=one, BAD="{not json")
    assert step.artifact("supply-chain") == [{"source": "supply chain", "results": []}] * 2
    assert not step.output.exists() or step.output.read_text() == ""


FAKE_CHECK = """#!/bin/sh
plan=$(cat "$PLAN"); printf %s "${plan#?}" > "$PLAN"
case "$plan" in
  o*) echo "Resolved 300 packages"; exit 0;;
  s*) echo "error: The lockfile at uv.lock needs to be updated"; exit 1;;
  n*) echo "error: Request failed after 3 retries"; exit 2;;
  h*) echo "Resolving dependencies"; exec sleep 30;;
esac
"""


def _fake_check(tmp_path: Path, plan: str) -> dict[str, str]:
    """A ``python`` on PATH whose successive ``--check-lock`` runs follow ``plan`` (o/s/n)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text(FAKE_CHECK, encoding="utf-8")
    (bin_dir / "python").chmod(0o755)
    (tmp_path / "plan").write_text(plan, encoding="utf-8")
    return {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "PLAN": str(tmp_path / "plan")}


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("plan,code,title", [
    ("o", 0, None),
    ("s", 1, "uv.lock out of sync"),
])
def test_uv_lock_verdict_and_review_status(repo, step, tmp_path, plan, code, title):
    result = step.run(repo, "verify_uv_lock.py", **_fake_check(tmp_path, plan))
    assert result.returncode == code, result.stdout
    statuses = step.artifact("uv-lockfile")
    assert [s["results"][0]["title"] for s in statuses] == ([title] if title else [])
    stale_summary = step.summary.exists() and "out of sync" in step.summary.read_text(encoding="utf-8")
    assert stale_summary == (plan == "s")


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("plan,verdict,sleeps", [
    ("no", "in-sync", [5]),
    ("ns", "stale", [5]),
    ("nnn", "unreachable", [5, 15]),
    ("nh", "timed-out", [5]),
])
def test_uv_lock_retries_only_registry_failures(tmp_path, monkeypatch, capsys, plan, verdict, sleeps):
    sys.path.insert(0, str(SCRIPTS.parents[1]))
    from scripts.ci import verify_uv_lock

    for key, value in _fake_check(tmp_path, plan).items():
        monkeypatch.setenv(key, value)
    slept: list[float] = []
    assert verify_uv_lock.verify(tmp_path / "log", sleep=slept.append, timeout=2) == verdict
    assert slept == sleeps
    out = capsys.readouterr().out
    assert out.count("::warning::") == len(sleeps)
    if verdict == "timed-out":
        assert "Resolving dependencies" in out  # the stopped attempt's log is still shown
