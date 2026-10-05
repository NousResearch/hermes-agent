"""Execute fork CI workflow logic against local Git and the real suite planner."""

import json
import os
import shlex
from pathlib import Path
import subprocess
import sys

import yaml

from scripts.ci.classify_changes import classify
from scripts.run_tests_parallel import _discover_files, _read_files_from

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"


def workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_aggregate_rejects_cancelled_and_unknown_results(tmp_path):
    jobs = workflow("ci.yaml")["jobs"]
    step = next(s for job in jobs.values() for s in job.get("steps", [])
                if s.get("id") == "evaluate")
    for result, expected in [("success", 0), ("skipped", 0), ("failure", 1),
                             ("cancelled", 1), ("unknown", 1)]:
        output = tmp_path / result
        run = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
            env={**os.environ, "NEEDS": json.dumps({"tests": {"result": result}}),
                 "GITHUB_OUTPUT": str(output)},
            capture_output=True, text=True, check=False,
        )
        assert run.returncode == expected, run.stdout + run.stderr
        assert json.loads(output.read_text().split("=", 1)[1]) == {"tests": result}


def test_attribution_checks_only_actual_target_and_still_rejects_unmapped(tmp_path):
    step = next(
        s for s in workflow("contributor-check.yml")["jobs"]["check-attribution"]["steps"]
        if s.get("id") == "check-emails"
    )
    env = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_AUTHOR_NAME": "Fixture",
        "GIT_COMMITTER_NAME": "Fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
        "GIT_AUTHOR_EMAIL": "historical@example.invalid",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        # Valid Git ref containing shell syntax: it must remain literal data.
        "PR_BASE_REF": "deploy/$(false)",
    }

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, env=env, text=True).strip()

    git("init", "-q")
    git("commit", "--allow-empty", "-qm", "old main")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    git("commit", "--allow-empty", "-qm", "deployment history")
    git("update-ref", f"refs/remotes/origin/{env['PR_BASE_REF']}", "HEAD")
    (tmp_path / "contributors/emails").mkdir(parents=True)
    (tmp_path / "contributors/emails/current@example.invalid").write_text("fixture\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/release.py").write_text("", encoding="utf-8")
    env["GIT_AUTHOR_EMAIL"] = "current@example.invalid"
    git("commit", "--allow-empty", "-qm", "mapped PR author")

    def check():
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
            cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
        )

    passed = check()
    assert passed.returncode == 0, passed.stdout + passed.stderr
    assert step["env"]["PR_BASE_REF"] == "${{ github.base_ref }}"
    assert step["if"] == "github.event_name == 'pull_request'"
    assert (tmp_path / "review-status.json").read_text(encoding="utf-8").strip() == "review_status=[]"
    env["GIT_AUTHOR_EMAIL"] = "unmapped@example.invalid"
    git("commit", "--allow-empty", "-qm", "unmapped PR author")
    rejected = check()
    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    status = (tmp_path / "review-status.json").read_text(encoding="utf-8").split("=", 1)[1]
    detail = json.loads(status)[0]["results"][0]["detail"]
    assert "unmapped@example.invalid" in detail
    assert "historical@example.invalid" not in detail
    env["PR_BASE_REF"] = ""
    assert check().returncode != 0


def test_active_fork_lanes_are_hosted_and_python_plan_covers_every_file(tmp_path):
    lanes = classify([".github/workflows/tests.yml"])
    assert all(lanes[k] for k in ("python", "frontend", "rust", "nix", "installer", "ci_review"))
    # Inspect all reusable workflows called by CI, plus independently triggered
    # Nix. Docker's upstream-only build/publish and disabled Desktop E2E retain
    # their existing gates; neither allocates runners in this fork PR.
    orchestrator = workflow("ci.yaml")
    active = [
        job["uses"].rsplit("/", 1)[-1]
        for job in orchestrator["jobs"].values()
        if "uses" in job and job.get("if") is not False
    ]
    hosted = {"ubuntu-latest", "windows-latest", "macos-latest"}
    for name in [*active, "nix.yml"]:
        for job in workflow(name)["jobs"].values():
            if "runs-on" not in job:
                continue
            runner = job["runs-on"]
            if runner == "${{ matrix.runner }}":
                assert all(row["runner"] in hosted for row in job["strategy"]["matrix"]["include"])
            else:
                assert runner in hosted, (name, runner)

    jobs = workflow("tests.yml")["jobs"]
    plan_step = next(s for s in jobs["plan"]["steps"] if s.get("id") == "plan")
    # Execute the actual YAML step with real imports/discovery, substituting
    # only the checkout path so its output stays in the temporary directory.
    command = plan_step["run"].replace(
        "python3 scripts/run_tests_parallel.py",
        f"{shlex.quote(sys.executable)} {shlex.quote(str(ROOT / 'scripts/run_tests_parallel.py'))}",
    )
    output = tmp_path / "output"
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", command], cwd=tmp_path,
        env={**os.environ, "GITHUB_OUTPUT": str(output)}, check=True,
    )
    matrix = json.loads(output.read_text(encoding="utf-8").split("=", 1)[1])
    assert jobs["test"]["strategy"]["matrix"] == "${{ fromJSON(needs.plan.outputs.matrix) }}"
    assert jobs["test"]["strategy"]["fail-fast"] is False
    test_step = next(s for s in jobs["test"]["steps"] if s.get("name") == "Run tests")
    assert test_step["env"]["TEST_FILES"] == "${{ matrix.slice.files }}"
    # Drive the shell's real files-from handoff, replacing only the expensive
    # pytest invocation with the real manifest reader. No test source is read.
    handoff = tmp_path / "handoff"
    handoff.write_text(
        f'#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(ROOT)!r})\n'
        'from scripts.run_tests_parallel import _read_files_from\n'
        'assert sys.argv[1] == "--files-from"\n'
        'assert _read_files_from(sys.argv[2])\n', encoding="utf-8",
    )
    handoff.chmod(0o755)
    command = test_step["run"].replace("source .venv/bin/activate", ":").replace("scripts/run_tests.sh", str(handoff))
    assigned = []
    for shard in matrix["slice"]:
        subprocess.run(["bash", "-e", "-o", "pipefail", "-c", command], cwd=tmp_path,
                       env={**os.environ, "TEST_FILES": shard["files"]}, check=True)
        assigned.extend(_read_files_from(str(tmp_path / "test-files.txt")))
    expected = [str(p.relative_to(ROOT)) for p in _discover_files([ROOT / "tests"])]
    assert sorted(assigned) == sorted(expected)
    assert len(assigned) == len(set(assigned))
    os_rows = workflow("tests-os.yml")["jobs"]["os-tests"]["strategy"]["matrix"]["include"]
    assert {(r["runner"], r["marker"]) for r in os_rows} == {
        ("macos-latest", "macos_only"), ("windows-latest", "windows_only"),
    }


def test_js_workflow_runs_every_discovered_check_even_after_failure(tmp_path):
    step = next(s for s in workflow("js-tests.yml")["jobs"]["check"]["steps"]
                if s.get("name") == "Run all workspace checks")
    # Execute the real scheduler against npm fixtures, not source-text matches.
    # Each child takes an exclusive directory lock, so overlapping checks fail.
    npm = tmp_path / "npm"
    npm.write_text(
        f'#!{sys.executable}\n'
        'import json, pathlib, sys, time\n'
        'root = pathlib.Path(__file__).parent\n'
        'if sys.argv[1] == "query":\n'
        '    print(json.dumps([{"location":"one", "scripts":{"check:a":"a", "check:b":"b", "check":"all"}},\n'
        '                      {"location":"two", "scripts":{"check":"all"}}]))\n'
        '    sys.exit(0)\n'
        'assert sys.argv[1:3] == ["run", "--prefix"]\n'
        'lock = root / "active"\n'
        'lock.mkdir()\n'
        'with (root / "ran").open("a") as f: f.write(" ".join(sys.argv[3:]) + "\\n")\n'
        'time.sleep(0.1)\n'
        'lock.rmdir()\n'
        'sys.exit(1 if sys.argv[-1] == "check:a" else 0)\n', encoding="utf-8",
    )
    npm.chmod(0o755)
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]], cwd=ROOT,
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert (tmp_path / "ran").read_text(encoding="utf-8").splitlines() == [
        "one check:a", "one check:b", "two check",
    ]
    assert "1 of 3 checks failed" in result.stderr
    assert "up to 1 at a time" in result.stdout
