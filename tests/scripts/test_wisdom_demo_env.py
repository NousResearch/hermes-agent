from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_SCRIPT = REPO_ROOT / "scripts" / "wisdom-demo-env.sh"
DEMO_HERMES = REPO_ROOT / "scripts" / "wisdom-demo-bin" / "hermes"


def demo_env(tmp_path: Path) -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "HOME": str(tmp_path),
        "HERMES_WISDOM_PYTHON": sys.executable,
        "HERMES_WISDOM_QUIET": "1",
    })
    value.pop("HERMES_WISDOM_REPO", None)
    value.pop("HERMES_DESKTOP_HERMES_ROOT", None)
    value.pop("HERMES_DESKTOP_PYTHON", None)
    return value


def test_sourcing_demo_environment_pins_every_surface_to_worktree(tmp_path: Path):
    command = r"""
set +e
source "$1"
source "$1"
case "$-" in *e*) echo 'source unexpectedly enabled errexit' >&2; exit 91;; esac
printf 'command=%s\n' "$(command -v hermes)"
printf 'repo=%s\n' "$HERMES_WISDOM_REPO"
printf 'desktop_root=%s\n' "$HERMES_DESKTOP_HERMES_ROOT"
printf 'desktop_python=%s\n' "$HERMES_DESKTOP_PYTHON"
printf 'path_count=%s\n' "$(printf '%s' "$PATH" | awk -F: -v p="$HERMES_WISDOM_REPO/scripts/wisdom-demo-bin" '{n=0; for(i=1;i<=NF;i++) if($i==p)n++; print n}')"
hermes --version
"""
    result = subprocess.run(
        ["bash", "-c", command, "wisdom-demo-test", str(ENV_SCRIPT)],
        cwd=REPO_ROOT,
        env=demo_env(tmp_path),
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert f"command={DEMO_HERMES}" in result.stdout
    assert f"repo={REPO_ROOT}" in result.stdout
    assert f"desktop_root={REPO_ROOT}" in result.stdout
    assert f"desktop_python={sys.executable}" in result.stdout
    assert "path_count=1" in result.stdout
    assert f"Install directory: {REPO_ROOT}" in result.stdout


def test_executable_demo_environment_forwards_to_worktree_cli(tmp_path: Path):
    result = subprocess.run(
        [str(ENV_SCRIPT), "--", "hermes", "wisdom", "--help"],
        cwd=REPO_ROOT,
        env=demo_env(tmp_path),
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert "usage: hermes wisdom" in result.stdout
    assert "suggest" in result.stdout
    assert "approve" in result.stdout
    assert "install" in result.stdout


def test_invalid_explicit_python_fails_closed(tmp_path: Path):
    value = demo_env(tmp_path)
    value["HERMES_WISDOM_PYTHON"] = str(tmp_path / "missing-python")

    result = subprocess.run(
        [str(ENV_SCRIPT), "--", "hermes", "wisdom", "--help"],
        cwd=REPO_ROOT,
        env=value,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode != 0
    assert "HERMES_WISDOM_PYTHON is not executable" in result.stderr
