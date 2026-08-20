import os
from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO_ROOT / "setup-hermes.sh"


def _run(
    *args: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def _init_repo(path: Path) -> Path:
    path.mkdir()
    _run("git", "-c", "init.defaultBranch=main", "init", "-q", str(path))
    _run("git", "config", "user.name", "Hermes Test", cwd=path)
    _run("git", "config", "user.email", "hermes-test@example.invalid", cwd=path)
    (path / "tracked").write_text("test\n", encoding="utf-8")
    _run("git", "add", "tracked", cwd=path)
    _run("git", "commit", "-qm", "initial", cwd=path)
    return path


def _make_checkout(tmp_path: Path, *, linked_worktree: bool) -> Path:
    primary = _init_repo(tmp_path / "primary")
    if not linked_worktree:
        return primary

    checkout = tmp_path / "linked"
    _run("git", "worktree", "add", "--detach", str(checkout), cwd=primary)
    return checkout


def _extract_function(name: str) -> str:
    content = SETUP_SCRIPT.read_text(encoding="utf-8")
    match = re.search(rf"(?ms)^{re.escape(name)}\(\) \{{\n.*?^\}}\n", content)
    assert match is not None, f"could not locate {name} in setup-hermes.sh"
    return match.group(0)


def _command_link_block() -> str:
    content = SETUP_SCRIPT.read_text(encoding="utf-8")
    start = content.index('HERMES_BIN="$SCRIPT_DIR/venv/bin/hermes"')
    end = content.index("\n\nif is_termux; then", start)
    return content[start:end]


def _run_command_link_block(
    tmp_path: Path,
    *,
    linked_worktree: bool,
    existing_launcher: bool,
    broken_launcher: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    checkout = _make_checkout(tmp_path, linked_worktree=linked_worktree)
    hermes_bin = checkout / "venv" / "bin" / "hermes"
    hermes_bin.parent.mkdir(parents=True)
    hermes_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    launcher = command_dir / "hermes"
    if broken_launcher:
        launcher.symlink_to(tmp_path / "missing-launcher-target")
    elif existing_launcher:
        launcher.write_text("canonical launcher\n", encoding="utf-8")

    harness = "\n".join(
        [
            "set -euo pipefail",
            "GREEN=''",
            "YELLOW=''",
            "NC=''",
            "get_command_link_dir() { printf '%s' \"$COMMAND_DIR\"; }",
            "get_command_link_display_dir() { printf '%s' \"$COMMAND_DIR\"; }",
            _extract_function("is_linked_worktree"),
            _command_link_block(),
        ]
    )
    env = os.environ | {"SCRIPT_DIR": str(checkout), "COMMAND_DIR": str(command_dir)}
    result = subprocess.run(
        ["bash", "-c", harness],
        capture_output=True,
        text=True,
        env=env,
    )
    return result, launcher, hermes_bin


def _classify_checkout(checkout: Path) -> str:
    harness = "\n".join(
        [
            "set -euo pipefail",
            _extract_function("is_linked_worktree"),
            "if is_linked_worktree; then printf linked; else printf ordinary; fi",
        ]
    )
    env = os.environ | {"SCRIPT_DIR": str(checkout)}
    return _run("bash", "-c", harness, env=env).stdout


def test_setup_hermes_script_is_valid_shell():
    result = subprocess.run(["bash", "-n", str(SETUP_SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_setup_hermes_script_has_termux_path():
    content = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "is_termux()" in content
    assert ".[termux]" in content
    assert "constraints-termux.txt" in content
    assert "$PREFIX/bin" in content


def test_setup_preserves_existing_launcher_from_linked_worktree(tmp_path: Path):
    result, launcher, _ = _run_command_link_block(
        tmp_path,
        linked_worktree=True,
        existing_launcher=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Linked Git worktree detected" in result.stdout
    assert not launcher.is_symlink()
    assert launcher.read_text(encoding="utf-8") == "canonical launcher\n"


def test_setup_links_from_linked_worktree_when_launcher_is_absent(tmp_path: Path):
    result, launcher, hermes_bin = _run_command_link_block(
        tmp_path,
        linked_worktree=True,
        existing_launcher=False,
    )

    assert result.returncode == 0, result.stderr
    assert launcher.is_symlink()
    assert launcher.resolve() == hermes_bin


def test_setup_preserves_broken_launcher_from_linked_worktree(tmp_path: Path):
    result, launcher, _ = _run_command_link_block(
        tmp_path,
        linked_worktree=True,
        existing_launcher=False,
        broken_launcher=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Linked Git worktree detected" in result.stdout
    assert launcher.is_symlink()
    assert launcher.readlink() == tmp_path / "missing-launcher-target"


def test_setup_replaces_launcher_from_primary_checkout(tmp_path: Path):
    result, launcher, hermes_bin = _run_command_link_block(
        tmp_path,
        linked_worktree=False,
        existing_launcher=True,
    )

    assert result.returncode == 0, result.stderr
    assert launcher.is_symlink()
    assert launcher.resolve() == hermes_bin


def test_worktree_detection_does_not_classify_submodule_as_linked_worktree(tmp_path: Path):
    source = _init_repo(tmp_path / "source")
    superproject = _init_repo(tmp_path / "superproject")
    _run(
        "git",
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(source),
        "submodule",
        cwd=superproject,
    )

    assert (superproject / "submodule" / ".git").is_file()
    assert _classify_checkout(superproject / "submodule") == "ordinary"
