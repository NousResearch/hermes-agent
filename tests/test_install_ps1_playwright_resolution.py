"""Regression coverage for Windows Playwright command resolution (#70787)."""

from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"
PS_TEST = REPO_ROOT / "scripts" / "tests" / "test-install-ps1-playwright-resolution.ps1"


def test_installer_uses_explicit_playwright_package_fallback() -> None:
    source = INSTALL_PS1.read_text(encoding="ascii")

    assert "Resolve-PlaywrightInvocation -InstallDir $InstallDir -NpxExe $npxExe" in source
    assert '"apps/desktop/node_modules/.bin"' in source
    assert '"--package=playwright"' in source
    assert "& $pwCommand @pwArgs" in source
    assert "& $npxExe --yes playwright install chromium" not in source


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="PowerShell is unavailable")
def test_playwright_resolver_behavior() -> None:
    subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(PS_TEST)],
        cwd=REPO_ROOT,
        check=True,
    )
