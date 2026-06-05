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


def test_setup_hermes_script_has_linux_i686_python_gate():
    content = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "is_linux_i686()" in content
    assert "configure_linux_i686_tempdir()" in content
    assert 'export TMPDIR="$SCRIPT_DIR/.tmp"' in content
    assert "find_compatible_python()" in content
    assert "uv-managed CPython does not publish Linux i686 builds" in content
    assert '"$PYTHON_PATH" -m venv venv' in content
