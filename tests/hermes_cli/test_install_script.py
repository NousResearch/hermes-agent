from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install.sh"


def _write_fake_git(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "git.log"
    fake_git = bin_dir / "git"
    fake_git.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail

            printf '%s\\n' "$*" >> "$GIT_CALL_LOG"

            if [ "${1:-}" != "clone" ]; then
                echo "unexpected git command: $*" >&2
                exit 1
            fi

            target="${@: -1}"
            url=""
            for arg in "$@"; do
                case "$arg" in
                    https://github.com/NousResearch/hermes-agent.git|git@github.com:NousResearch/hermes-agent.git)
                        url="$arg"
                        ;;
                esac
            done

            case "$url" in
                https://github.com/NousResearch/hermes-agent.git)
                    if [ "${FAIL_HTTPS_CLONE:-0}" = "1" ]; then
                        exit 1
                    fi
                    ;;
                git@github.com:NousResearch/hermes-agent.git)
                    if [ "${FAIL_SSH_CLONE:-0}" = "1" ]; then
                        exit 1
                    fi
                    ;;
                *)
                    echo "unexpected clone url: $url" >&2
                    exit 1
                    ;;
            esac

            mkdir -p "$target/.git"
            """
        ),
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    return bin_dir, log_path


def _run_clone_repo(tmp_path: Path, *, fail_https: bool = False, fail_ssh: bool = False) -> subprocess.CompletedProcess[str]:
    bin_dir, log_path = _write_fake_git(tmp_path)
    install_dir = tmp_path / "install"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["GIT_CALL_LOG"] = str(log_path)
    env["TEST_INSTALL_DIR"] = str(install_dir)
    if fail_https:
        env["FAIL_HTTPS_CLONE"] = "1"
    if fail_ssh:
        env["FAIL_SSH_CLONE"] = "1"

    return subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; set --; source "$INSTALL_SCRIPT"; INSTALL_DIR="$TEST_INSTALL_DIR"; BRANCH="main"; clone_repo',
        ],
        env={**env, "INSTALL_SCRIPT": str(INSTALL_SCRIPT)},
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


def test_install_script_is_valid_shell():
    result = subprocess.run(["bash", "-n", str(INSTALL_SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_clone_repo_prefers_https_for_fresh_install(tmp_path: Path):
    result = _run_clone_repo(tmp_path)

    assert result.returncode == 0, result.stderr
    call_log = (tmp_path / "git.log").read_text(encoding="utf-8").splitlines()
    assert len(call_log) == 1
    assert "https://github.com/NousResearch/hermes-agent.git" in call_log[0]
    assert "git@github.com:NousResearch/hermes-agent.git" not in call_log[0]
    assert "Cloned via HTTPS" in result.stdout


def test_clone_repo_falls_back_to_ssh_when_https_fails(tmp_path: Path):
    result = _run_clone_repo(tmp_path, fail_https=True)

    assert result.returncode == 0, result.stderr
    call_log = (tmp_path / "git.log").read_text(encoding="utf-8").splitlines()
    assert len(call_log) == 2
    assert "https://github.com/NousResearch/hermes-agent.git" in call_log[0]
    assert "git@github.com:NousResearch/hermes-agent.git" in call_log[1]
    assert "HTTPS failed, trying SSH..." in result.stdout
    assert "Cloned via SSH" in result.stdout
