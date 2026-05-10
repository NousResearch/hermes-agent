"""Regression tests for install.sh optional-extras fallback visibility."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
SETUP_HERMES_SH = REPO_ROOT / "setup-hermes.sh"


def test_install_sh_warns_that_base_fallback_disables_optional_extras() -> None:
    text = INSTALL_SH.read_text()
    assert "Base install completed, but optional extras were not installed" in text
    assert "Messaging gateways may be missing adapter dependencies" in text
    assert "uv pip install -e '.[all]'" in text


def test_setup_hermes_warns_that_base_fallback_disables_optional_extras() -> None:
    text = SETUP_HERMES_SH.read_text()
    assert "Base install completed, but optional extras were not installed" in text
    assert "Messaging gateways may be missing adapter dependencies" in text
    assert "uv pip install -e '.[all]'" in text
