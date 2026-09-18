"""Regression tests for dashboard dependency verification in installers."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def test_posix_installer_repairs_corrupt_uvicorn_subpackage_with_pip() -> None:
    text = INSTALL_SH.read_text(encoding="utf-8")

    assert "from uvicorn.supervisors import ChangeReload" in text
    assert "--force-reinstall" in text
    assert "--no-cache-dir" in text
    assert "uvicorn[standard]==0.41.0" in text


def test_windows_installer_repairs_corrupt_uvicorn_subpackage_with_pip() -> None:
    text = INSTALL_PS1.read_text(encoding="utf-8")

    assert "from uvicorn.supervisors import ChangeReload" in text
    assert "--force-reinstall" in text
    assert "--no-cache-dir" in text
    assert "uvicorn[standard]==0.41.0" in text
