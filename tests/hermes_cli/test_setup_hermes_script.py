from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO_ROOT / "setup-hermes.sh"


def test_setup_hermes_script_is_valid_shell():
    result = subprocess.run(["bash", "-n", str(SETUP_SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_setup_hermes_script_has_termux_path():
    content = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "is_termux()" in content
    assert ".[termux]" in content
    assert "constraints-termux.txt" in content
    assert "$PREFIX/bin" in content


def test_setup_hermes_script_fails_closed_on_unverified_fallback_by_default():
    content = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "HERMES_SETUP_ALLOW_UNVERIFIED_INSTALL" in content
    assert "Default posture is fail-closed" in content
    assert "Refusing to continue with an unverified PyPI install." in content
    assert "break glass temporarily" in content
