"""Regression coverage for installer-managed uv paths with spaces."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
SETUP_HERMES_SH = REPO_ROOT / "setup-hermes.sh"


def _active_lines(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_install_script_quotes_uv_cmd_when_executing() -> None:
    text = INSTALL_SH.read_text()

    assert 'UV_VERSION=$("$UV_CMD" --version 2>/dev/null)' in text
    assert '"$UV_CMD" venv venv --python "$PYTHON_VERSION"' in text
    assert 'UV_PROJECT_ENVIRONMENT="$INSTALL_DIR/venv" "$UV_CMD" sync --extra all --locked' in text
    assert 'if "$UV_CMD" pip install -e "$spec" 2>"$ALL_INSTALL_LOG"; then' in text

    active = "\n".join(_active_lines(INSTALL_SH))
    assert "$($UV_CMD " not in active
    assert "\n    $UV_CMD " not in active
    assert " $UV_CMD sync" not in active
    assert "if $UV_CMD " not in active


def test_setup_helper_quotes_uv_cmd_when_executing() -> None:
    text = SETUP_HERMES_SH.read_text()

    assert 'UV_VERSION=$("$UV_CMD" --version 2>/dev/null)' in text
    assert 'if "$UV_CMD" python find "$PYTHON_VERSION" &> /dev/null; then' in text
    assert 'PYTHON_PATH=$("$UV_CMD" python find "$PYTHON_VERSION")' in text
    assert '"$UV_CMD" venv venv --python "$PYTHON_VERSION"' in text
    assert 'UV_PROJECT_ENVIRONMENT="$SCRIPT_DIR/venv" "$UV_CMD" sync --extra all --locked' in text

    active = "\n".join(_active_lines(SETUP_HERMES_SH))
    assert "$($UV_CMD " not in active
    assert "\n    $UV_CMD " not in active
    assert " $UV_CMD sync" not in active
    assert "if $UV_CMD " not in active
