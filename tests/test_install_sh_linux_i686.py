"""Regression tests for Linux i686 installer guards.

uv ships Linux i686 binaries, but uv-managed CPython does not publish Linux
i686 builds. The POSIX installer must therefore keep that path gated to
32-bit x86 Linux and use an already-installed Python instead of trying
``uv python install``.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _function_body(name: str) -> str:
    text = INSTALL_SH.read_text(encoding="utf-8")
    _, _, rest = text.partition(f"{name}() {{\n")
    assert rest, f"Could not find {name}() in scripts/install.sh"
    body, _, _ = rest.partition("\n}\n")
    assert body, f"Could not find {name}() body"
    return body


def test_linux_i686_detection_is_narrow() -> None:
    text = INSTALL_SH.read_text(encoding="utf-8")

    assert "is_linux_i686()" in text
    assert 'i386|i486|i586|i686)' in text
    assert 'getconf LONG_BIT' in text
    assert '[ "${OS:-}" = "linux" ]' in text


def test_linux_i686_tempdir_defaults_under_hermes_home() -> None:
    body = _function_body("configure_linux_i686_tempdir")

    assert 'if ! is_linux_i686 || [ -n "${TMPDIR:-}" ]; then' in body
    assert 'export TMPDIR="$HERMES_HOME/tmp"' in body
    assert 'mkdir -p "$TMPDIR"' in body


def test_linux_i686_check_python_avoids_uv_python_install() -> None:
    body = _function_body("check_python")

    i686_idx = body.find("if is_linux_i686; then")
    uv_install_idx = body.find('"$UV_CMD" python install "$PYTHON_VERSION"')
    assert i686_idx != -1, "check_python must have a Linux i686 branch"
    assert uv_install_idx != -1, "test expected the regular uv python install path"
    assert i686_idx < uv_install_idx, "Linux i686 must branch before uv python install"
    assert "find_compatible_python" in body
    assert "uv-managed CPython does not publish Linux i686 builds" in body
    assert "HERMES_PYTHON=/path/to/python" in body


def test_linux_i686_venv_uses_system_python() -> None:
    body = _function_body("setup_venv")

    i686_idx = body.find("if is_linux_i686; then")
    uv_venv_idx = body.find('$UV_CMD venv venv --python "$PYTHON_VERSION"')
    assert i686_idx != -1, "setup_venv must have a Linux i686 branch"
    assert uv_venv_idx != -1, "test expected the regular uv venv path"
    assert i686_idx < uv_venv_idx, "Linux i686 must branch before uv venv"
    assert '"$PYTHON_PATH" -m venv venv' in body
    assert 'export UV_PYTHON="$INSTALL_DIR/venv/bin/python"' in body
