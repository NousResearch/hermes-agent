"""Regression guard for uv-managed Python installs in shell installers."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
SETUP_HERMES_SH = REPO_ROOT / "setup-hermes.sh"


def test_install_sh_does_not_publish_uv_python_to_global_bin() -> None:
    text = INSTALL_SH.read_text(encoding="utf-8")

    assert '"$UV_CMD" python install --no-bin "$PYTHON_VERSION"' in text
    assert '"$UV_CMD" python install "$PYTHON_VERSION"' not in text


def test_setup_hermes_does_not_publish_uv_python_to_global_bin() -> None:
    text = SETUP_HERMES_SH.read_text(encoding="utf-8")

    assert '$UV_CMD python install --no-bin "$PYTHON_VERSION"' in text
    assert '$UV_CMD python install "$PYTHON_VERSION"' not in text
