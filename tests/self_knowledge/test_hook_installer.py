import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "install-self-knowledge-hook.sh"


def _git_init(path: Path):
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)


def test_hook_installer_is_idempotent(tmp_path):
    _git_init(tmp_path)

    subprocess.run([str(SCRIPT)], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run([str(SCRIPT)], cwd=tmp_path, check=True, capture_output=True, text=True)

    hook = tmp_path / ".git" / "hooks" / "pre-commit"
    text = hook.read_text()
    assert text.count(">>> hermes self-knowledge hook >>>") == 1
    assert "hermes self-knowledge --refresh" in text


def test_hook_installer_refuses_foreign_hook_without_force(tmp_path):
    _git_init(tmp_path)
    hook = tmp_path / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/usr/bin/env bash\necho foreign\n")

    result = subprocess.run([str(SCRIPT)], cwd=tmp_path, capture_output=True, text=True)

    assert result.returncode == 2
    assert "foreign pre-commit hook exists" in result.stderr


def test_hook_installer_force_appends_to_foreign_hook(tmp_path):
    _git_init(tmp_path)
    hook = tmp_path / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/usr/bin/env bash\necho foreign\n")

    subprocess.run([str(SCRIPT), "--force"], cwd=tmp_path, check=True, capture_output=True, text=True)

    text = hook.read_text()
    assert "echo foreign" in text
    assert "hermes self-knowledge --refresh" in text
