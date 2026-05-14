"""Regression tests for install.sh browser setup.

Browser automation is optional. The installer should not leave Hermes
half-installed just because Playwright's managed Chromium download hangs on an
unsupported distribution.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_install_script_skips_playwright_download_when_system_browser_exists() -> None:
    text = INSTALL_SH.read_text()

    assert "find_system_browser()" in text
    assert "google-chrome google-chrome-stable chromium chromium-browser chrome" in text
    assert "Skipping Playwright browser download; Hermes will use the system browser." in text


def test_install_script_persists_system_browser_for_agent_browser() -> None:
    text = INSTALL_SH.read_text()

    assert "configure_browser_env_from_system_browser()" in text
    assert "AGENT_BROWSER_EXECUTABLE_PATH=$browser_path" in text


def test_playwright_installs_are_timeout_guarded() -> None:
    text = INSTALL_SH.read_text()

    assert "run_browser_install_with_timeout()" in text
    assert "run_browser_install_with_timeout 600 npx playwright install chromium" in text
    assert "run_browser_install_with_timeout 600 npx playwright install --with-deps chromium" in text
