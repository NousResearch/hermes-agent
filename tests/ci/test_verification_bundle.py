"""Tests for the local verification bundle suggester scripts.

The helper is intentionally script-only: it suggests commands and never executes
checks, reads secrets, or mutates Hermes runtime state.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = ROOT / "scripts" / "ci" / "verification_bundle.py"
CLI_PATH = ROOT / "scripts" / "suggest_verification_bundle.py"


def _load_bundle():
    assert BUNDLE_PATH.exists(), "scripts/ci/verification_bundle.py should exist"
    spec = importlib.util.spec_from_file_location("verification_bundle", BUNDLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cli():
    assert CLI_PATH.exists(), "scripts/suggest_verification_bundle.py should exist"
    spec = importlib.util.spec_from_file_location("suggest_verification_bundle", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ids(bundle) -> set[str]:
    return {cmd.id for cmd in bundle.commands}


def _init_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(
        ["git", "config", "user.email", "tester@example.invalid"],
        cwd=path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test Runner"],
        cwd=path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _bash_or_skip() -> str:
    bash = shutil.which("bash")
    if not bash or not shutil.which("git") or not shutil.which("python"):
        pytest.skip("bash, git, and python are required to smoke-test generated shell commands")
    probe = subprocess.run(
        [bash, "-lc", "printf ok"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0 or probe.stdout != "ok":
        pytest.skip("bash is present but unavailable in this environment")
    return bash


def test_gateway_and_desktop_store_paths_get_focused_checks():
    mod = _load_bundle()

    bundle = mod.suggest_bundle([
        "gateway/run.py",
        "apps/desktop/src/store/layout.ts",
    ])

    ids = _ids(bundle)
    assert bundle.changed_paths == (
        "gateway/run.py",
        "apps/desktop/src/store/layout.ts",
    )
    assert "gateway-restart-drain-pytest" in ids
    assert "desktop-store-vitest" in ids
    assert "desktop-typecheck" in ids
    assert "git-diff-check" in ids
    assert "conflict-marker-scan" in ids
    assert "added-line-security-scan" in ids
    assert bundle.risk_level == "medium"


def test_traversal_like_changed_path_is_canonicalized_before_classification():
    mod = _load_bundle()

    bundle = mod.suggest_bundle(["docs/../gateway/run.py"])

    assert bundle.changed_paths == ("gateway/run.py",)
    assert "gateway-restart-drain-pytest" in _ids(bundle)
    assert bundle.risk_level == "medium"


def test_unknown_and_empty_inputs_fail_open():
    mod = _load_bundle()

    unknown = mod.suggest_bundle(["Makefile"])
    unknown_ids = _ids(unknown)
    assert "python-hygiene-pytest" in unknown_ids
    assert "git-diff-check" in unknown_ids
    assert any("fail-open" in note.lower() for note in unknown.notes)

    empty = mod.suggest_bundle([])
    empty_ids = _ids(empty)
    assert "full-python-tests" in empty_ids
    assert "desktop-typecheck" in empty_ids
    assert empty.risk_level == "high"
    assert any("empty" in note.lower() for note in empty.notes)


def test_json_and_markdown_format_are_machine_and_human_readable():
    mod = _load_bundle()
    bundle = mod.suggest_bundle(["gateway/run.py"])

    data = json.loads(mod.format_json(bundle))
    assert data["changed_paths"] == ["gateway/run.py"]
    assert any(cmd["id"] == "gateway-restart-drain-pytest" for cmd in data["commands"])

    markdown = mod.format_markdown(bundle)
    assert markdown.startswith("# Verification Bundle")
    assert "`gateway/run.py`" in markdown
    assert "gateway-restart-drain-pytest" in markdown


def test_cli_json_stdout_is_parseable_and_stderr_is_clean_for_path_mode():
    assert CLI_PATH.exists(), "scripts/suggest_verification_bundle.py should exist"

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--paths",
            "gateway/run.py",
            "--format",
            "json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""
    data = json.loads(proc.stdout)
    assert data["changed_paths"] == ["gateway/run.py"]
    assert any(cmd["id"] == "gateway-restart-drain-pytest" for cmd in data["commands"])


def test_cli_from_git_collects_changed_paths_from_base_ref(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git is required for --from-git positive coverage")

    gateway_dir = tmp_path / "gateway"
    gateway_dir.mkdir()
    target = gateway_dir / "run.py"
    target.write_text("print('initial')\n", encoding="utf-8")

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(
        ["git", "config", "user.email", "tester@example.invalid"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test Runner"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(["git", "add", "gateway/run.py"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(["git", "branch", "base"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    target.write_text("print('changed')\n", encoding="utf-8")
    subprocess.run(["git", "add", "gateway/run.py"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(
        ["git", "commit", "-m", "change gateway"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--from-git",
            "--base",
            "base",
            "--format",
            "json",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""
    data = json.loads(proc.stdout)
    assert data["changed_paths"] == ["gateway/run.py"]
    assert any(cmd["id"] == "gateway-restart-drain-pytest" for cmd in data["commands"])


def test_cli_from_git_decodes_utf8_paths_when_platform_utf8_mode_is_disabled(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git is required for --from-git Unicode coverage")
    _init_git_repo(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("baseline\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", readme.name],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "branch", "base"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "config", "core.quotepath", "false"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    gateway = tmp_path / "gateway"
    gateway.mkdir()
    unicode_path = gateway / "😀.py"
    unicode_path.write_text("safe = True\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", str(unicode_path.relative_to(tmp_path))],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "unicode path"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    env = os.environ.copy()
    env["PYTHONUTF8"] = "0"

    proc = subprocess.run(
        [sys.executable, str(CLI_PATH), "--from-git", "--base", "base", "--format", "json"],
        cwd=tmp_path,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""
    assert json.loads(proc.stdout)["changed_paths"] == ["gateway/😀.py"]


def test_cli_from_git_ignores_repository_selection_environment(tmp_path):
    candidate = tmp_path / "candidate"
    shadow = tmp_path / "shadow"
    _init_git_repo(candidate)
    _init_git_repo(shadow)
    for repo in (candidate, shadow):
        readme = repo / "README.md"
        readme.write_text("baseline\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "--", readme.name],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "baseline"],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            ["git", "branch", "base"],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    gateway = candidate / "gateway"
    gateway.mkdir()
    (gateway / "run.py").write_text("safe = True\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", "gateway/run.py"],
        cwd=candidate,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "candidate change"],
        cwd=candidate,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    env = os.environ.copy()
    env["GIT_DIR"] = str(shadow / ".git")
    env["GIT_WORK_TREE"] = str(shadow)

    proc = subprocess.run(
        [sys.executable, str(CLI_PATH), "--from-git", "--base", "base", "--format", "json"],
        cwd=candidate,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["changed_paths"] == ["gateway/run.py"]


def test_suggested_shell_commands_quote_changed_paths():
    mod = _load_bundle()
    py_path = "gateway/weird name;$(touch nope)'quote.py"
    node_path = "apps/desktop/electron/-bad name;$(touch nope).cjs"

    bundle = mod.suggest_bundle([py_path, node_path])
    commands = {cmd.id: cmd.command for cmd in bundle.commands}

    py_parts = shlex.split(commands["py-compile-changed"], posix=True)
    assert py_parts[-1] == py_path

    node_parts = shlex.split(commands["node-check-electron-cjs"], posix=True)
    assert "--" in node_parts
    assert node_parts[-1] == node_path


def test_added_line_security_scan_ignores_pattern_text_and_flags_lhs_assignments():
    mod = _load_bundle()

    harmless_diff = "\n".join(
        [
            '+pattern = r"api_key|secret|password|token|passwd\\\\s*="',
            '+terms = ("api_key", "secret", "password", "token", "passwd")',
            '+note = "mention token = in documentation without assigning a token variable"',
        ]
    )
    assert mod.scan_added_line_security_hits(harmless_diff) == []

    term_a = "api" + "_key"
    term_b = "client" + "_secret"
    term_c = "to" + "ken"
    risky_diff = "\n".join(
        [
            f'+{term_a} = "dummy-value"',
            f'+{term_b} = "dummy-value"',
            f'+config["{term_c}"] = "dummy-value"',
        ]
    )
    assert mod.scan_added_line_security_hits(risky_diff) == [
        "redacted-hit",
        "redacted-hit",
        "redacted-hit",
    ]


def test_generated_added_line_security_scan_command_redacts_and_uses_lhs(tmp_path):
    mod = _load_bundle()
    bash = shutil.which("bash")
    if not bash or not shutil.which("git") or not shutil.which("python"):
        pytest.skip("bash, git, and python are required to smoke-test the generated shell command")
    bash_probe = subprocess.run(
        [bash, "-lc", "printf ok"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if bash_probe.returncode != 0 or bash_probe.stdout != "ok":
        pytest.skip("bash is present but unavailable in this environment")

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(
        ["git", "config", "user.email", "tester@example.invalid"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test Runner"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    sample = tmp_path / "sample.py"
    sample.write_text("safe = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "sample.py"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    command = mod._added_line_security_scan_command()
    term_a = "api" + "_key"

    sample.write_text('pattern = r"api_key|secret|password|token|passwd\\\\s*="\n', encoding="utf-8")
    harmless = subprocess.run(
        [bash, "-lc", command],
        cwd=tmp_path,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    assert harmless.returncode == 0, harmless.stdout + harmless.stderr
    assert "redacted-hit" not in harmless.stdout

    sample.write_text(f'{term_a} = "dummy-value"\n', encoding="utf-8")
    risky = subprocess.run(
        [bash, "-lc", command],
        cwd=tmp_path,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    assert risky.returncode == 1
    assert "redacted-hit" in risky.stdout
    assert "dummy-value" not in risky.stdout
    assert "dummy-value" not in risky.stderr

    subprocess.run(
        ["git", "checkout", "--", "sample.py"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    untracked = tmp_path / "untracked.py"
    untracked.write_text(f'{term_a} = "untracked-dummy-value"\n', encoding="utf-8")
    untracked_risky = subprocess.run(
        [bash, "-lc", command],
        cwd=tmp_path,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    assert untracked_risky.returncode == 1
    assert "redacted-hit" in untracked_risky.stdout
    assert "untracked-dummy-value" not in untracked_risky.stdout
    assert "untracked-dummy-value" not in untracked_risky.stderr


def test_generated_security_scan_rejects_untracked_windows_junction_ancestor(tmp_path):
    if os.name != "nt":
        pytest.skip("Windows junction coverage only applies on Windows")
    mod = _load_bundle()
    bash = _bash_or_skip()
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    _init_git_repo(repo)
    outside.mkdir()
    (outside / "safe.py").write_text("safe = True\n", encoding="utf-8")
    junction = repo / "junction"
    proc = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink /J junction ..\\outside"],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).decode("cp949", errors="replace")
        pytest.skip(f"junction creation unavailable: {detail}")

    try:
        assert junction.is_symlink() is False
        command = mod._added_line_security_scan_command()
        scanned = subprocess.run(
            [bash, "-lc", command],
            cwd=repo,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        assert scanned.returncode == 1
        assert "redacted-hit" in scanned.stdout
        assert "safe = True" not in scanned.stdout
        assert "safe = True" not in scanned.stderr
    finally:
        subprocess.run(
            ["cmd.exe", "/d", "/c", "rmdir junction"],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )


def test_generated_security_scan_fails_closed_for_staged_and_untracked_utf16(tmp_path):
    mod = _load_bundle()
    bash = _bash_or_skip()
    command = mod._added_line_security_scan_command()
    term = "api" + "_key"
    marker = "utf16-dummy-value"

    staged_repo = tmp_path / "staged"
    _init_git_repo(staged_repo)
    staged_file = staged_repo / "staged.py"
    staged_file.write_text(f'{term} = "{marker}"\n', encoding="utf-16")
    subprocess.run(
        ["git", "add", "--", staged_file.name],
        cwd=staged_repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    staged = subprocess.run(
        [bash, "-lc", command],
        cwd=staged_repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    assert staged.returncode == 1
    assert "redacted-hit" in staged.stdout
    assert marker not in staged.stdout
    assert marker not in staged.stderr

    untracked_repo = tmp_path / "untracked"
    _init_git_repo(untracked_repo)
    (untracked_repo / "untracked.py").write_text(f'{term} = "{marker}"\n', encoding="utf-16")
    untracked = subprocess.run(
        [bash, "-lc", command],
        cwd=untracked_repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    assert untracked.returncode == 1
    assert "redacted-hit" in untracked.stdout
    assert marker not in untracked.stdout
    assert marker not in untracked.stderr


def test_generated_security_scan_redacts_staged_and_unstaged_undecodable_bytes(tmp_path):
    mod = _load_bundle()
    bash = _bash_or_skip()
    command = mod._added_line_security_scan_command()
    term = "api" + "_key"
    marker = "undecodable-dummy-value"
    payload = f'{term} = "{marker}"'.encode("utf-8") + b"\xff\n"

    staged_repo = tmp_path / "staged"
    _init_git_repo(staged_repo)
    (staged_repo / "staged.py").write_bytes(payload)
    subprocess.run(
        ["git", "add", "--", "staged.py"],
        cwd=staged_repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    unstaged_repo = tmp_path / "unstaged"
    _init_git_repo(unstaged_repo)
    unstaged_file = unstaged_repo / "unstaged.py"
    unstaged_file.write_text("safe = True\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", "unstaged.py"],
        cwd=unstaged_repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=unstaged_repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    unstaged_file.write_bytes(payload)

    for repo in (staged_repo, unstaged_repo):
        scanned = subprocess.run(
            [bash, "-lc", command],
            cwd=repo,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        assert scanned.returncode == 1
        assert "redacted-hit" in scanned.stdout
        assert marker not in scanned.stdout
        assert marker not in scanned.stderr
        assert "traceback" not in scanned.stderr.lower()


def test_generated_security_scan_disables_external_diff_bypass(tmp_path):
    mod = _load_bundle()
    bash = _bash_or_skip()
    command = mod._added_line_security_scan_command()
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    sample = repo / "sample.py"
    sample.write_text("safe = True\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", sample.name],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    (repo / "empty-diff.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    subprocess.run(
        ["git", "config", "diff.external", "sh empty-diff.sh"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    term = "api" + "_key"
    marker = "external-diff-dummy-value"
    sample.write_text(f'{term} = "{marker}"\n', encoding="utf-8")

    scanned = subprocess.run(
        [bash, "-lc", command],
        cwd=repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    assert scanned.returncode == 1
    assert "redacted-hit" in scanned.stdout
    assert marker not in scanned.stdout
    assert marker not in scanned.stderr


def test_generated_security_scan_disables_textconv_bypass(tmp_path):
    mod = _load_bundle()
    bash = _bash_or_skip()
    command = mod._added_line_security_scan_command()
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "empty-textconv.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (repo / ".gitattributes").write_text("*.bin diff=empty\n", encoding="utf-8")
    subprocess.run(
        ["git", "config", "diff.empty.textconv", "sh empty-textconv.sh"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    term = "api" + "_key"
    marker = "textconv-dummy-value"
    (repo / "payload.bin").write_text(f'{term} = "{marker}"\n', encoding="utf-16")
    subprocess.run(
        ["git", "add", "--", ".gitattributes", "payload.bin"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    scanned = subprocess.run(
        [bash, "-lc", command],
        cwd=repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    assert scanned.returncode == 1
    assert "redacted-hit" in scanned.stdout
    assert marker not in scanned.stdout
    assert marker not in scanned.stderr


def test_generated_security_scan_ignores_repository_selection_environment(tmp_path):
    mod = _load_bundle()
    bash = _bash_or_skip()
    command = mod._added_line_security_scan_command()
    candidate_repo = tmp_path / "candidate"
    shadow_repo = tmp_path / "shadow"
    _init_git_repo(candidate_repo)
    _init_git_repo(shadow_repo)
    sample = candidate_repo / "sample.py"
    sample.write_text("safe = True\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", sample.name],
        cwd=candidate_repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=candidate_repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    term = "api" + "_key"
    marker = "shadow-repo-dummy-value"
    sample.write_text(f'{term} = "{marker}"\n', encoding="utf-8")
    env = os.environ.copy()
    env["GIT_DIR"] = str(shadow_repo / ".git")
    env["GIT_WORK_TREE"] = str(shadow_repo)

    scanned = subprocess.run(
        [bash, "-lc", command],
        cwd=candidate_repo,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert scanned.returncode == 1
    assert "redacted-hit" in scanned.stdout
    assert marker not in scanned.stdout
    assert marker not in scanned.stderr


def test_generated_security_scan_rejects_active_clean_filter_without_executing_it(tmp_path):
    mod = _load_bundle()
    bash = _bash_or_skip()
    command = mod._added_line_security_scan_command()
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    sample = repo / "sample.py"
    sample.write_text("safe = True\n", encoding="utf-8")
    (repo / ".gitattributes").write_text("sample.py filter=hide\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", ".gitattributes", sample.name],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    (repo / "hide-filter.sh").write_text(
        "#!/bin/sh\nprintf ran > filter-ran.txt\nprintf 'safe = True\\n'\n", encoding="utf-8"
    )
    subprocess.run(
        ["git", "config", "filter.hide.clean", "sh hide-filter.sh"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    term = "api" + "_key"
    marker = "clean-filter-dummy-value"
    sample.write_text(f'{term} = "{marker}"\n', encoding="utf-8")

    scanned = subprocess.run(
        [bash, "-lc", command],
        cwd=repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert scanned.returncode == 1
    assert "redacted-hit" in scanned.stdout
    assert marker not in scanned.stdout
    assert marker not in scanned.stderr
    assert not (repo / "filter-ran.txt").exists()


def test_generated_security_scan_rejects_ident_attribute_transform(tmp_path):
    mod = _load_bundle()
    bash = _bash_or_skip()
    command = mod._added_line_security_scan_command()
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    term = "api" + "_key"
    sample = repo / "sample.py"
    sample.write_text(f'{term} = "$Id$"\n', encoding="utf-8")
    (repo / ".gitattributes").write_text("sample.py ident\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", ".gitattributes", sample.name],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    marker = "SYNTHETIC_IDENT_VALUE"
    sample.write_text(f'{term} = "$Id: {marker} $"\n', encoding="utf-8")

    scanned = subprocess.run(
        [bash, "-lc", command],
        cwd=repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert scanned.returncode == 1
    assert "redacted-hit" in scanned.stdout
    assert marker not in scanned.stdout
    assert marker not in scanned.stderr


@pytest.mark.parametrize("attribute_line", ["sample.py text eol=lf", "sample.py -diff"])
def test_generated_security_scan_handles_eol_and_binary_diff_attributes(tmp_path, attribute_line):
    mod = _load_bundle()
    bash = _bash_or_skip()
    command = mod._added_line_security_scan_command()
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    sample = repo / "sample.py"
    sample.write_text("safe = True\n", encoding="utf-8")
    (repo / ".gitattributes").write_text(attribute_line + "\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", ".gitattributes", sample.name],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    term = "api" + "_key"
    marker = "attribute-dummy-value"
    sample.write_text(f'{term} = "{marker}"\n', encoding="utf-8")

    scanned = subprocess.run(
        [bash, "-lc", command],
        cwd=repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert scanned.returncode == 1
    assert "redacted-hit" in scanned.stdout
    assert marker not in scanned.stdout
    assert marker not in scanned.stderr


def test_generated_security_scan_rejects_gitlink_entries(tmp_path):
    mod = _load_bundle()
    bash = _bash_or_skip()
    command = mod._added_line_security_scan_command()
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    readme = repo / "README.md"
    readme.write_text("baseline\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", readme.name],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "update-index", "--add", "--cacheinfo", f"160000,{head},vendor/sub"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    scanned = subprocess.run(
        [bash, "-lc", command],
        cwd=repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert scanned.returncode == 1
    assert "redacted-hit" in scanned.stdout
    assert scanned.stderr == ""


def test_generated_security_scan_rejects_untracked_nested_repository(tmp_path):
    mod = _load_bundle()
    bash = _bash_or_skip()
    command = mod._added_line_security_scan_command()
    parent = tmp_path / "parent"
    nested = parent / "vendor" / "nested"
    _init_git_repo(parent)
    readme = parent / "README.md"
    readme.write_text("baseline\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", readme.name],
        cwd=parent,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "parent baseline"],
        cwd=parent,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _init_git_repo(nested)
    secret_file = nested / "secret.py"
    secret_file.write_text("safe = True\n", encoding="utf-8")
    (nested / ".gitattributes").write_text("secret.py filter=hide\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", ".gitattributes", secret_file.name],
        cwd=nested,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "nested baseline"],
        cwd=nested,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    (nested / "hide-filter.sh").write_text(
        "#!/bin/sh\nprintf ran > filter-ran.txt\nprintf 'safe = True\\n'\n", encoding="utf-8"
    )
    subprocess.run(
        ["git", "config", "filter.hide.clean", "sh hide-filter.sh"],
        cwd=nested,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    term = "api" + "_key"
    marker = "nested-repo-dummy-value"
    secret_file.write_text(f'{term} = "{marker}"\n', encoding="utf-8")

    scanned = subprocess.run(
        [bash, "-lc", command],
        cwd=parent,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert scanned.returncode == 1
    assert "redacted-hit" in scanned.stdout
    assert marker not in scanned.stdout
    assert marker not in scanned.stderr
    assert not (nested / "filter-ran.txt").exists()


@pytest.mark.parametrize("index_flag", ["--assume-unchanged", "--skip-worktree"])
def test_generated_security_scan_rejects_hidden_index_flags(tmp_path, index_flag):
    mod = _load_bundle()
    bash = _bash_or_skip()
    command = mod._added_line_security_scan_command()
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    sample = repo / "sample.py"
    sample.write_text("safe = True\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", sample.name],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "update-index", index_flag, sample.name],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    term = "api" + "_key"
    marker = f"hidden-index-{index_flag[2:]}-dummy-value"
    sample.write_text(f'{term} = "{marker}"\n', encoding="utf-8")

    scanned = subprocess.run(
        [bash, "-lc", command],
        cwd=repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert scanned.returncode == 1
    assert "redacted-hit" in scanned.stdout
    assert marker not in scanned.stdout
    assert marker not in scanned.stderr


def test_generated_security_scan_disables_fsmonitor_hook(tmp_path):
    mod = _load_bundle()
    bash = _bash_or_skip()
    command = mod._added_line_security_scan_command()
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    sample = repo / "sample.py"
    sample.write_text("safe = True\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", sample.name],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    (repo / "fsmonitor.sh").write_text("#!/bin/sh\nprintf ran > fsmonitor-ran.txt\nexit 0\n", encoding="utf-8")
    subprocess.run(
        ["git", "config", "core.fsmonitor", "sh fsmonitor.sh"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    term = "api" + "_key"
    marker = "fsmonitor-dummy-value"
    sample.write_text(f'{term} = "{marker}"\n', encoding="utf-8")

    scanned = subprocess.run(
        [bash, "-lc", command],
        cwd=repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert scanned.returncode == 1
    assert "redacted-hit" in scanned.stdout
    assert marker not in scanned.stdout
    assert marker not in scanned.stderr
    assert not (repo / "fsmonitor-ran.txt").exists()


def test_generated_security_scan_converts_git_timeout_to_redacted_exit(monkeypatch, capsys):
    mod = _load_bundle()
    command = mod._added_line_security_scan_command()
    source = command.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["git"], timeout=30)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(SystemExit) as exc_info:
        exec(compile(source, "<generated-security-scan>", "exec"), {})

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "redacted-hit" in captured.out
    assert captured.err == ""


def test_stdout_only_cli_does_not_write_import_bytecode(tmp_path):
    scripts_dir = tmp_path / "scripts"
    ci_dir = scripts_dir / "ci"
    ci_dir.mkdir(parents=True)
    copied_cli = shutil.copy2(CLI_PATH, scripts_dir / CLI_PATH.name)
    shutil.copy2(BUNDLE_PATH, ci_dir / BUNDLE_PATH.name)
    env = os.environ.copy()
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env.pop("PYTHONPYCACHEPREFIX", None)

    proc = subprocess.run(
        [sys.executable, str(copied_cli), "--paths", "gateway/run.py", "--format", "json"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["changed_paths"] == ["gateway/run.py"]
    assert not (ci_dir / "__pycache__").exists()


def test_scripts_paths_recommend_scope_c_focused_tests():
    mod = _load_bundle()

    bundle = mod.suggest_bundle([
        "scripts/ci/verification_bundle.py",
        "scripts/scaffold_ulw_ledger.py",
    ])

    ids = _ids(bundle)
    assert "verification-bundle-pytest" in ids
    assert "ulw-ledger-scaffold-pytest" in ids


def test_cli_rejects_output_path_to_preserve_read_only_contract(tmp_path):
    output_path = tmp_path / "bundle.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--paths",
            "gateway/run.py",
            "--format",
            "json",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert not output_path.exists()


def test_from_git_rejects_dash_prefixed_base_ref():
    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--from-git",
            "--base=-bad-ref",
            "--format",
            "json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "base" in proc.stderr.lower()


def test_from_git_missing_executable_returns_clean_public_error():
    env = os.environ.copy()
    env["PATH"] = ""

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--from-git",
            "--base",
            "HEAD",
            "--format",
            "json",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "git" in proc.stderr.lower()
    assert "traceback" not in proc.stderr.lower()


def test_from_git_rejects_nul_bearing_git_path_output(monkeypatch, tmp_path):
    mod = _load_cli()
    outputs = iter([f"{tmp_path}\n".encode("utf-8"), b"docs/\x00safe.md\n"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "_run_git_bytes", lambda *_args, **_kwargs: next(outputs))

    with pytest.raises(RuntimeError, match="NUL|invalid"):
        mod._paths_from_git("HEAD")


def test_from_git_failure_does_not_expose_raw_git_output(monkeypatch, tmp_path):
    mod = _load_cli()
    marker = "sensitive-git-stderr-marker"

    def fail(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=["git"], returncode=1, stdout=b"", stderr=marker.encode("utf-8"))

    monkeypatch.setattr(mod.subprocess, "run", fail)
    with pytest.raises(RuntimeError) as exc_info:
        mod._run_git_bytes(["git", "status"], cwd=tmp_path, env={})

    assert marker not in str(exc_info.value)


def test_from_git_oserror_does_not_expose_exception_text(monkeypatch, capsys):
    mod = _load_cli()
    marker = "sensitive-oserror-marker"

    def fail(*_args, **_kwargs):
        raise FileNotFoundError(marker)

    monkeypatch.setattr(mod.subprocess, "run", fail)
    exit_code = mod.main(["--from-git", "--base", "HEAD", "--format", "json"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "git" in captured.err.lower()
    assert marker not in captured.err
    assert "traceback" not in captured.err.lower()
