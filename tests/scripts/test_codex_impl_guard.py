import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / "scripts" / "runtime" / "codex_impl_guard.py"


def _load_guard_module():
    spec = importlib.util.spec_from_file_location("codex_impl_guard_under_test", GUARD)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _init_repo(repo: Path) -> None:
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    target = repo / "demo.py"
    target.write_text("VALUE = 'old'\n", encoding="utf-8")
    _git(repo, "add", "demo.py")
    _git(repo, "commit", "-m", "init")


def _write_fake_codex(path: Path, body: str) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys, time\n"
        + body,
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _run_guard(repo: Path, fake_codex: Path, *extra: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["HERMES_CODEX_IMPL_GUARD_ALLOW_FAKE_CODEX"] = "1"
    return subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "--codex-bin",
            str(fake_codex),
            "--workdir",
            str(repo),
            "--prompt",
            "Change the allowed file only.",
            "--raw-log",
            str(repo.parent / "impl.raw.log"),
            "--final-file",
            str(repo.parent / "impl.final.json"),
            *extra,
        ],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        env=env,
    )


def test_arbitrary_codex_bin_is_rejected_without_test_override(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(tmp_path / "fake_not_codex.py", "print('should not run')\n")

    proc = subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "--codex-bin",
            str(fake),
            "--workdir",
            str(repo),
            "--prompt",
            "Change the allowed file only.",
            "--allowed-file",
            "demo.py",
            "--raw-log",
            str(repo.parent / "impl.raw.log"),
            "--final-file",
            str(repo.parent / "impl.final.json"),
        ],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )

    assert proc.returncode == 3, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "unusable"
    assert result["reason"] == "unsupported_codex_bin"


def test_dash_prefixed_prompt_is_passed_after_end_of_options_separator(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    argv_file = tmp_path / "argv.json"
    fake = _write_fake_codex(
        tmp_path / "codex-yuna",
        f"open({str(argv_file)!r}, 'w', encoding='utf-8').write(json.dumps(sys.argv[1:]))\n",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "--codex-bin",
            str(fake),
            "--workdir",
            str(repo),
            "--prompt",
            "--sandbox read-only should stay prompt text",
            "--allowed-file",
            "demo.py",
            "--raw-log",
            str(repo.parent / "impl.raw.log"),
            "--final-file",
            str(repo.parent / "impl.final.json"),
        ],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        env={**os.environ, "HERMES_CODEX_IMPL_GUARD_ALLOW_FAKE_CODEX": "1"},
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    fake_argv = json.loads(argv_file.read_text(encoding="utf-8"))
    assert "--" in fake_argv
    assert fake_argv[-1].endswith("--sandbox read-only should stay prompt text")


def test_supported_codex_bin_requires_verified_sandbox_before_launch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    proc = subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "--codex-bin",
            "codex-yuna",
            "--workdir",
            str(repo),
            "--prompt",
            "Change the allowed file only.",
            "--allowed-file",
            "demo.py",
            "--raw-log",
            str(repo.parent / "impl.raw.log"),
            "--final-file",
            str(repo.parent / "impl.final.json"),
        ],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )

    assert proc.returncode == 3, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "unusable"
    assert result["reason"] == "sandbox_not_verified"


def test_dirty_require_clean_blocks_before_codex_runs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "demo.py").write_text("VALUE = 'dirty'\n", encoding="utf-8")
    marker = repo / "codex-ran.txt"
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"open({str(marker)!r}, 'w', encoding='utf-8').write('ran')\n",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py", "--dirty-baseline-policy", "require-clean")

    assert proc.returncode == 3, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "unusable"
    assert result["reason"] == "dirty_baseline_not_clean"
    assert "demo.py" in result["dirty_baseline"]
    assert not marker.exists()


def test_allow_listed_owned_policy_allows_dirty_baseline_inside_allowlist(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "demo.py").write_text("VALUE = 'dirty before codex'\n", encoding="utf-8")
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"open(os.path.join({str(repo)!r}, 'demo.py'), 'w', encoding='utf-8').write(\"VALUE = 'codex change'\\n\")\n",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py", "--dirty-baseline-policy", "allow-listed-owned")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "review_needed"
    assert result["reason"] == "codex_exit_zero_safe_diff"
    assert result["dirty_baseline_policy"] == "allow-listed-owned"
    assert result["dirty_baseline"] == ["demo.py"]
    assert result["allowlist_violations"] == []
    assert "demo.py" in result["changed_files"]


def test_allow_listed_owned_policy_reports_unchanged_dirty_baseline(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "demo.py").write_text("VALUE = 'dirty before codex'\n", encoding="utf-8")
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        "# Codex exits without changing the already dirty allowed file.\n",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py", "--dirty-baseline-policy", "allow-listed-owned")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "review_needed"
    assert result["dirty_baseline_policy"] == "allow-listed-owned"
    assert result["dirty_baseline"] == ["demo.py"]
    assert result["allowlist_violations"] == []
    assert result["changed_files"] == ["demo.py"]


def test_allow_listed_owned_policy_rejects_dirty_baseline_outside_allowlist(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "notes.txt").write_text("preexisting dirty\n", encoding="utf-8")
    marker = repo / "codex-ran.txt"
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"open({str(marker)!r}, 'w', encoding='utf-8').write('ran')\n",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py", "--dirty-baseline-policy", "allow-listed-owned")

    assert proc.returncode == 3, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "unusable"
    assert result["reason"] == "dirty_baseline_outside_allowlist"
    assert result["dirty_baseline_policy"] == "allow-listed-owned"
    assert result["dirty_baseline_violations"] == ["notes.txt"]
    assert not marker.exists()


def test_dirty_path_overlap_detects_directory_and_child_allowlist_relationships():
    guard = _load_guard_module()

    assert guard._dirty_path_overlaps_allowlist("scratch", ["scratch/keep.txt"], []) is True
    assert guard._dirty_path_overlaps_allowlist("scratch/keep.txt", ["scratch"], []) is True
    assert guard._dirty_path_overlaps_allowlist("scratch", [], ["scratch/**"]) is True
    assert guard._dirty_path_overlaps_allowlist("scratch/keep.txt", [], ["scratch/**"]) is True
    assert guard._dirty_path_overlaps_allowlist("scratch", ["demo.py"], ["tests/**"]) is False


def test_policy_candidate_paths_keeps_changed_dirty_directory_as_candidate(tmp_path):
    guard = _load_guard_module()
    repo = tmp_path / "repo"
    repo.mkdir()
    scratch = repo / "scratch"
    scratch.mkdir()
    (scratch / "keep.txt").write_text("before\n", encoding="utf-8")
    baseline = guard._dirty_baseline_fingerprints(repo, ["scratch"])

    (scratch / "keep.txt").write_text("after\n", encoding="utf-8")
    candidates = guard._policy_candidate_paths(
        ["scratch", "demo.py"],
        policy="fail-on-overlap",
        dirty_baseline=["scratch"],
        baseline_fingerprints=baseline,
        workdir=repo,
    )

    assert candidates == ["demo.py", "scratch"]


def test_fail_on_overlap_policy_allows_unrelated_dirty_baseline(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "notes.txt").write_text("preexisting dirty\n", encoding="utf-8")
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"open(os.path.join({str(repo)!r}, 'demo.py'), 'w', encoding='utf-8').write(\"VALUE = 'codex change'\\n\")\n",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py", "--dirty-baseline-policy", "fail-on-overlap")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "review_needed"
    assert result["dirty_baseline_policy"] == "fail-on-overlap"
    assert result["dirty_baseline"] == ["notes.txt"]
    assert result["allowlist_violations"] == []
    assert result["untracked_files"] == ["notes.txt"]


def test_fail_on_overlap_policy_blocks_mutation_inside_unrelated_dirty_directory(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    scratch = repo / "scratch"
    scratch.mkdir()
    (scratch / "keep.txt").write_text("preexisting dirty\n", encoding="utf-8")
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"open(os.path.join({str(repo)!r}, 'demo.py'), 'w', encoding='utf-8').write(\"VALUE = 'codex change'\\n\")\n"
        f"open(os.path.join({str(repo)!r}, 'scratch', 'keep.txt'), 'a', encoding='utf-8').write('codex touched outside allowlist\\n')\n",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py", "--dirty-baseline-policy", "fail-on-overlap")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "blocked_by_allowlist"
    assert result["reason"] == "allowlist_violation"
    assert "scratch/keep.txt" in result["allowlist_violations"]


def test_fail_on_overlap_policy_rejects_dirty_baseline_inside_allowlist(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "demo.py").write_text("VALUE = 'dirty before codex'\n", encoding="utf-8")
    marker = repo / "codex-ran.txt"
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"open({str(marker)!r}, 'w', encoding='utf-8').write('ran')\n",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py", "--dirty-baseline-policy", "fail-on-overlap")

    assert proc.returncode == 3, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "unusable"
    assert result["reason"] == "dirty_baseline_overlaps_allowlist"
    assert result["dirty_baseline_policy"] == "fail-on-overlap"
    assert result["dirty_baseline_overlap"] == ["demo.py"]
    assert not marker.exists()


def test_allowed_change_returns_review_needed_with_bounded_stdout(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"""
args = sys.argv[1:]
final_path = args[args.index('--output-last-message') + 1]
open(os.path.join({str(repo)!r}, 'demo.py'), 'w', encoding='utf-8').write("VALUE = 'new'\\n")
open(final_path, 'w', encoding='utf-8').write(json.dumps({{'summary': 'changed demo'}}))
print('PROGRESS_SENTINEL_SHOULD_STAY_IN_RAW_LOG')
""",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "review_needed"
    assert result["trusted_completion"] is True
    assert result["changed_files"] == ["demo.py"]
    assert result["allowlist_violations"] == []
    assert "PROGRESS_SENTINEL" not in proc.stdout
    assert (repo.parent / "impl.raw.log").read_text(encoding="utf-8").strip() == "PROGRESS_SENTINEL_SHOULD_STAY_IN_RAW_LOG"


def test_untracked_allowlist_violation_blocks_without_reverting(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"open(os.path.join({str(repo)!r}, 'outside.py'), 'w', encoding='utf-8').write('bad = True\\n')\n",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "blocked_by_allowlist"
    assert "outside.py" in result["untracked_files"]
    assert result["allowlist_violations"] == ["outside.py"]
    assert (repo / "outside.py").exists()


def test_nonzero_with_safe_diff_returns_takeover_candidate(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"""
open(os.path.join({str(repo)!r}, 'demo.py'), 'w', encoding='utf-8').write("VALUE = 'candidate'\\n")
sys.exit(7)
""",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py")

    assert proc.returncode == 2, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "takeover_candidate"
    assert result["reason"] == "codex_nonzero_with_safe_diff"
    assert result["trusted_completion"] is False
    assert result["changed_files"] == ["demo.py"]


def test_absolute_allowed_glob_is_rejected_before_codex_runs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    marker = repo / "codex-ran.txt"
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"open({str(marker)!r}, 'w', encoding='utf-8').write('ran')\n",
    )

    proc = _run_guard(repo, fake, "--allowed-glob", "/tmp/*.py")

    assert proc.returncode == 3, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "unusable"
    assert result["reason"] == "invalid_allowlist"
    assert not marker.exists()


def test_parent_escape_allowed_file_is_rejected_before_codex_runs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    marker = repo / "codex-ran.txt"
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"open({str(marker)!r}, 'w', encoding='utf-8').write('ran')\n",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "../demo.py")

    assert proc.returncode == 3, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "unusable"
    assert result["reason"] == "invalid_allowlist"
    assert not marker.exists()


def test_parent_escape_inside_allowed_glob_is_rejected_before_codex_runs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    marker = repo / "codex-ran.txt"
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"open({str(marker)!r}, 'w', encoding='utf-8').write('ran')\n",
    )

    proc = _run_guard(repo, fake, "--allowed-glob", "src/../*.py")

    assert proc.returncode == 3, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "unusable"
    assert result["reason"] == "invalid_allowlist"
    assert result["invalid_allowlist"] == ["src/../*.py"]
    assert not marker.exists()


def test_untracked_file_matching_allowed_glob_is_allowed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"open(os.path.join({str(repo)!r}, 'new_feature.py'), 'w', encoding='utf-8').write('ok = True\\n')\n",
    )

    proc = _run_guard(repo, fake, "--allowed-glob", "*.py")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "review_needed"
    assert result["untracked_files"] == ["new_feature.py"]
    assert result["allowlist_violations"] == []


def test_untracked_symlink_escape_is_blocked(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    outside = tmp_path / "outside-target.py"
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"os.symlink({str(outside)!r}, os.path.join({str(repo)!r}, 'escape.py'))\n",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "escape.py")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "blocked_by_allowlist"
    assert result["allowlist_violations"] == ["escape.py"]


def test_deleted_allowed_file_is_reported(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"os.remove(os.path.join({str(repo)!r}, 'demo.py'))\n",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "review_needed"
    assert result["deleted_files"] == ["demo.py"]


def test_submodule_gitlink_change_is_blocked_even_when_allowlisted(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"""
import subprocess
os.makedirs(os.path.join({str(repo)!r}, 'vendor'), exist_ok=True)
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd={str(repo)!r}, text=True).strip()
subprocess.run(['git', 'update-index', '--add', '--cacheinfo', '160000,' + head + ',vendor/sub'], cwd={str(repo)!r}, check=True)
""",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "vendor/sub")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "blocked_by_allowlist"
    assert result["submodule_changes"] == ["vendor/sub"]
    assert result["allowlist_violations"] == [".git/index", "vendor/sub"]


def test_ignored_artifact_created_by_codex_is_blocked(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / ".gitignore").write_text(".env\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore env")
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"open(os.path.join({str(repo)!r}, '.env'), 'w', encoding='utf-8').write('SECRET=bad\\n')\n",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "blocked_by_allowlist"
    assert result["ignored_artifacts"] == [".env"]
    assert ".env" in result["allowlist_violations"]


def test_ignored_artifact_is_blocked_even_when_allowlisted_by_glob(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / ".gitignore").write_text("cache.tmp\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore cache")
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        "open('cache.tmp', 'w', encoding='utf-8').write('residue')\n",
    )

    proc = _run_guard(repo, fake, "--allowed-glob", "**")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "blocked_by_allowlist"
    assert result["ignored_artifacts"] == ["cache.tmp"]
    assert result["allowlist_violations"] == ["cache.tmp"]


def test_git_metadata_mutation_is_blocked(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        "open('.git/config', 'a', encoding='utf-8').write('\\n[alias]\\nunsafe = !echo bad\\n')\n",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "blocked_by_allowlist"
    assert ".git/config" in result["git_metadata_changes"]
    assert ".git/config" in result["allowlist_violations"]


def test_oversized_ignored_artifact_is_blocked_without_full_read(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / ".gitignore").write_text("big.bin\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore big")
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        "open('big.bin', 'wb').write(b'x' * 1000001)\n",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "blocked_by_allowlist"
    assert result["ignored_artifacts"] == ["big.bin"]
    assert result["allowlist_violations"] == ["big.bin"]


def test_fake_codex_bypass_requires_pytest_context(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(tmp_path / "fake_codex.py", "print('should not run')\n")
    env = os.environ.copy()
    env["HERMES_CODEX_IMPL_GUARD_ALLOW_FAKE_CODEX"] = "1"
    env.pop("PYTEST_CURRENT_TEST", None)

    proc = subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "--codex-bin",
            str(fake),
            "--workdir",
            str(repo),
            "--prompt",
            "Change the allowed file only.",
            "--allowed-file",
            "demo.py",
            "--raw-log",
            str(repo.parent / "impl.raw.log"),
            "--final-file",
            str(repo.parent / "impl.final.json"),
        ],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        env=env,
    )

    assert proc.returncode == 3, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "unusable"
    assert result["reason"] == "unsupported_codex_bin"


def test_nested_git_repo_under_allowlisted_glob_is_blocked(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"""
os.makedirs(os.path.join({str(repo)!r}, 'vendor', 'sub', '.git'), exist_ok=True)
open(os.path.join({str(repo)!r}, 'vendor', 'sub', '.git', 'config'), 'w', encoding='utf-8').write('[core]\\n')
""",
    )

    proc = _run_guard(repo, fake, "--allowed-glob", "vendor/**")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "blocked_by_allowlist"
    assert result["submodule_changes"] == ["vendor/sub"]
    assert "vendor/sub" in result["allowlist_violations"]


def test_nonzero_without_diff_returns_failed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(tmp_path / "fake_codex.py", "sys.exit(9)\n")

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "failed"
    assert result["reason"] == "codex_nonzero_without_diff"
    assert result["trusted_completion"] is False


def test_unsupported_verify_id_fails_closed_before_codex_runs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    marker = repo / "codex-ran.txt"
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"open({str(marker)!r}, 'w', encoding='utf-8').write('ran')\n",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py", "--verify-cmd-id", "pytest-file:tests/test_demo.py")

    assert proc.returncode == 3, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "unusable"
    assert result["reason"] == "unsupported_verify_cmd_id"
    assert result["unsupported_verify_cmd_ids"] == ["pytest-file:tests/test_demo.py"]
    assert not marker.exists()


def test_explicit_output_paths_inside_workdir_fail_closed_before_codex_runs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    marker = repo / "codex-ran.txt"
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"open({str(marker)!r}, 'w', encoding='utf-8').write('ran')\n",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "--codex-bin",
            str(fake),
            "--workdir",
            str(repo),
            "--prompt",
            "Change the allowed file only.",
            "--allowed-file",
            "demo.py",
            "--raw-log",
            str(repo / "impl.raw.log"),
        ],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        env={**os.environ, "HERMES_CODEX_IMPL_GUARD_ALLOW_FAKE_CODEX": "1"},
    )

    assert proc.returncode == 3, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "unusable"
    assert result["reason"] == "unsafe_output_path"
    assert result["unsafe_output_path_label"] == "raw_log_path"
    assert not marker.exists()


def test_diff_check_includes_staged_changes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        "import subprocess\n"
        "open('demo.py', 'w', encoding='utf-8').write('VALUE = 1  ' + chr(10))\n"
        "subprocess.run(['git', 'add', 'demo.py'], check=True)\n",
    )
    proc = _run_guard(repo, fake, "--allowed-file", "demo.py", "--verify-cmd-id", "diff-check")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "blocked_by_allowlist"
    assert result["reason"] == "allowlist_violation"
    assert ".git/index" in result["allowlist_violations"]


def test_unknown_verify_id_fails_closed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"open(os.path.join({str(repo)!r}, 'demo.py'), 'w', encoding='utf-8').write(\"VALUE = 'new'\\\\n\")\n",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py", "--verify-cmd-id", "unknown")

    assert proc.returncode == 3, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "unusable"
    assert result["reason"] == "unsupported_verify_cmd_id"
    assert result["unsupported_verify_cmd_ids"] == ["unknown"]


def test_diff_flood_with_safe_diff_returns_takeover_candidate(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"""
open(os.path.join({str(repo)!r}, 'demo.py'), 'w', encoding='utf-8').write("VALUE = 'flood'\\n")
for i in range(100):
    print('+diff flood line', i)
    sys.stdout.flush()
    time.sleep(0.001)
""",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py", "--diff-line-threshold", "10", "--kill-grace-seconds", "0.1")

    assert proc.returncode == 2, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "takeover_candidate"
    assert result["reason"] == "codex_terminated_with_safe_diff"
    assert result["diff_flood_detected"] is True
    assert result["changed_files"] == ["demo.py"]


def test_source_flood_with_safe_diff_returns_takeover_candidate(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"""
open(os.path.join({str(repo)!r}, 'demo.py'), 'w', encoding='utf-8').write("VALUE = 'source flood'\\n")
for i in range(100):
    print('def generated_function_%03d():' % i)
    sys.stdout.flush()
    time.sleep(0.001)
""",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py", "--source-line-threshold", "10", "--kill-grace-seconds", "0.1")

    assert proc.returncode == 2, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "takeover_candidate"
    assert result["reason"] == "codex_terminated_with_safe_diff"
    assert result["source_flood_detected"] is True
    assert result["changed_files"] == ["demo.py"]


def test_json_field_flood_with_safe_diff_returns_takeover_candidate(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    source_blob = "\n".join(f"def generated_function_{i:03d}():" for i in range(20))
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"""
open(os.path.join({str(repo)!r}, 'demo.py'), 'w', encoding='utf-8').write("VALUE = 'json flood'\\n")
print(json.dumps({{'output': {source_blob!r}}}))
sys.stdout.flush()
time.sleep(5)
""",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py", "--source-line-threshold", "10", "--kill-grace-seconds", "0.1")

    assert proc.returncode == 2, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "takeover_candidate"
    assert result["reason"] == "codex_terminated_with_safe_diff"
    assert result["codex_reason"] == "json_field_flood"
    assert result["json_field_flood_detected"] is True
    assert result["json_flood_field"] == "output"
    assert "generated_function" not in proc.stdout


def test_json_field_char_threshold_flood_with_safe_diff_returns_takeover_candidate(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    large_blob = "x" * 200
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"""
open(os.path.join({str(repo)!r}, 'demo.py'), 'w', encoding='utf-8').write("VALUE = 'json char flood'\\n")
print(json.dumps({{'content': {large_blob!r}}}))
sys.stdout.flush()
time.sleep(5)
""",
    )

    proc = _run_guard(
        repo,
        fake,
        "--allowed-file",
        "demo.py",
        "--json-field-char-threshold",
        "100",
        "--kill-grace-seconds",
        "0.1",
    )

    assert proc.returncode == 2, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "takeover_candidate"
    assert result["reason"] == "codex_terminated_with_safe_diff"
    assert result["codex_reason"] == "json_field_flood"
    assert result["json_field_flood_detected"] is True
    assert result["json_flood_field"] == "content"
    assert result["json_flood_chars"] == len(large_blob)
    assert large_blob not in proc.stdout


def test_diff_flood_with_allowlist_violation_blocks(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"""
open(os.path.join({str(repo)!r}, 'outside.py'), 'w', encoding='utf-8').write('bad = True\\n')
for i in range(100):
    print('+diff flood line', i)
    sys.stdout.flush()
    time.sleep(0.001)
""",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py", "--diff-line-threshold", "10", "--kill-grace-seconds", "0.1")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "blocked_by_allowlist"
    assert result["diff_flood_detected"] is True
    assert result["allowlist_violations"] == ["outside.py"]


def test_timeout_with_safe_diff_returns_takeover_candidate(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"""
open(os.path.join({str(repo)!r}, 'demo.py'), 'w', encoding='utf-8').write("VALUE = 'timeout'\\n")
sys.stdout.flush()
time.sleep(5)
""",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py", "--timeout-seconds", "0.2", "--kill-grace-seconds", "0.1")

    assert proc.returncode == 2, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "takeover_candidate"
    assert result["reason"] == "codex_terminated_with_safe_diff"
    assert result["codex_reason"] == "timeout"


def test_git_mv_rename_evidence_is_reported(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"""
import subprocess
subprocess.run(['git', 'mv', 'demo.py', 'renamed.py'], cwd={str(repo)!r}, check=True)
""",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py", "--allowed-file", "renamed.py")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "blocked_by_allowlist"
    assert ".git/index" in result["allowlist_violations"]
    assert result["renamed_files"] == ["demo.py", "renamed.py"]


def test_diff_check_verify_passes_and_records_result(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    fake = _write_fake_codex(
        tmp_path / "fake_codex.py",
        f"open(os.path.join({str(repo)!r}, 'demo.py'), 'w', encoding='utf-8').write(\"VALUE = 'new'\\\\n\")\n",
    )

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py", "--verify-cmd-id", "diff-check")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "passed"
    assert result["reason"] == "verification_passed"
    assert result["verification"][0]["id"] == "diff-check"
    assert result["verification"][0]["status"] == "passed"
    assert len(proc.stdout) < 8000


def test_non_git_workdir_is_unusable(tmp_path):
    repo = tmp_path / "not-git"
    repo.mkdir()
    fake = _write_fake_codex(tmp_path / "fake_codex.py", "print('should not run')\n")

    proc = _run_guard(repo, fake, "--allowed-file", "demo.py")

    assert proc.returncode == 3, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "unusable"
    assert result["reason"] == "not_git_repo"
