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
