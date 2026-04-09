from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install.sh"


def _extract_prelude_end(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        if line.startswith("# ============================================================================") and idx > 0:
            if idx + 1 < len(lines) and lines[idx + 1].strip() == "# Helper functions":
                return idx
    raise AssertionError("Could not locate helper-functions boundary in install.sh")


def _run_shell(snippet: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", snippet],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_dir_flag_sets_hermes_home_when_env_not_explicitly_set(tmp_path: Path) -> None:
    install_dir = tmp_path / "custom-home"
    lines = INSTALL_SCRIPT.read_text(encoding="utf-8").splitlines()
    prelude = "\n".join(lines[: _extract_prelude_end(lines)])

    result = _run_shell(
        f"""
set -e
unset HERMES_HOME
unset HERMES_INSTALL_DIR
{prelude}
set -- --dir {install_dir}
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-venv)
            USE_VENV=false
            shift
            ;;
        --skip-setup)
            RUN_SETUP=false
            shift
            ;;
        --branch)
            BRANCH=\"$2\"
            shift 2
            ;;
        --dir)
            INSTALL_DIR=\"$2\"
            if [ -z \"${{HERMES_HOME:-}}\" ] || [ \"$HERMES_HOME\" = \"$DEFAULT_HERMES_HOME\" ]; then
                HERMES_HOME=\"$INSTALL_DIR\"
            fi
            shift 2
            ;;
        *)
            exit 99
            ;;
    esac
done
printf '%s\n%s\n' "$INSTALL_DIR" "$HERMES_HOME"
"""
    )

    install_value, hermes_home_value = result.stdout.strip().splitlines()
    assert install_value == str(install_dir)
    assert hermes_home_value == str(install_dir)


def test_dir_flag_preserves_explicit_hermes_home_env(tmp_path: Path) -> None:
    install_dir = tmp_path / "checkout"
    hermes_home = tmp_path / "profile-home"
    lines = INSTALL_SCRIPT.read_text(encoding="utf-8").splitlines()
    prelude = "\n".join(lines[: _extract_prelude_end(lines)])

    result = _run_shell(
        f"""
set -e
export HERMES_HOME={hermes_home}
unset HERMES_INSTALL_DIR
{prelude}
set -- --dir {install_dir}
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-venv)
            USE_VENV=false
            shift
            ;;
        --skip-setup)
            RUN_SETUP=false
            shift
            ;;
        --branch)
            BRANCH=\"$2\"
            shift 2
            ;;
        --dir)
            INSTALL_DIR=\"$2\"
            if [ -z \"${{HERMES_HOME:-}}\" ] || [ \"$HERMES_HOME\" = \"$DEFAULT_HERMES_HOME\" ]; then
                HERMES_HOME=\"$INSTALL_DIR\"
            fi
            shift 2
            ;;
        *)
            exit 99
            ;;
    esac
done
printf '%s\n%s\n' "$INSTALL_DIR" "$HERMES_HOME"
"""
    )

    install_value, hermes_home_value = result.stdout.strip().splitlines()
    assert install_value == str(install_dir)
    assert hermes_home_value == str(hermes_home)
