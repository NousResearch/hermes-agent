"""Regression for install.sh when the uv binary path contains spaces.

If uv is installed under ``$HOME/.local/bin`` and the user's home directory
contains a space, every command-position use of ``$UV_CMD`` must stay quoted.
Otherwise shell word splitting turns the uv path into multiple tokens and the
installer fails in the ``venv`` and ``python-deps`` stages. See #53086.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_uv_command_invocations_stay_quoted_for_spacey_home_paths() -> None:
    text = INSTALL_SH.read_text(encoding="utf-8")

    expected_fragments = [
        'UV_VERSION=$("$UV_CMD" --version 2>/dev/null)',
        '"$UV_CMD" venv venv --python "$PYTHON_VERSION"',
        'if UV_PROJECT_ENVIRONMENT="$INSTALL_DIR/venv" "$UV_CMD" sync --extra all --locked; then',
        'if "$UV_CMD" pip install -e "$spec" 2>"$ALL_INSTALL_LOG"; then',
    ]

    for fragment in expected_fragments:
        assert fragment in text, f"Missing quoted uv invocation: {fragment}"

    forbidden_fragments = [
        'UV_VERSION=$($UV_CMD --version 2>/dev/null)',
        '$UV_CMD venv venv --python "$PYTHON_VERSION"',
        'if UV_PROJECT_ENVIRONMENT="$INSTALL_DIR/venv" $UV_CMD sync --extra all --locked; then',
        'if $UV_CMD pip install -e "$spec" 2>"$ALL_INSTALL_LOG"; then',
    ]

    for fragment in forbidden_fragments:
        assert fragment not in text, f"Found unquoted uv invocation: {fragment}"
