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
    # --with-deps is still invoked on apt-based systems, but only when sudo
    # is available non-interactively (root or passwordless sudo). Non-sudo
    # service users fall back to the browser-only install — see
    # install_node_deps() in install.sh.
    assert "run_browser_install_with_timeout 600 npx playwright install --with-deps chromium" in text


def test_install_script_supports_skip_browser_flag() -> None:
    """--skip-browser (and --no-playwright alias) skips the Playwright install."""
    text = INSTALL_SH.read_text()

    assert "--skip-browser|--no-playwright)" in text
    assert "SKIP_BROWSER=true" in text
    assert 'if [ "$SKIP_BROWSER" = true ]; then' in text
    assert "--skip-browser Skip Playwright/Chromium install" in text


def test_install_script_skips_with_deps_when_no_sudo() -> None:
    """Non-sudo users on apt distros must not block on an interactive sudo prompt."""
    text = INSTALL_SH.read_text()

    # The apt branch must gate --with-deps behind a sudo capability check
    # (root or non-interactive sudo), otherwise the installer hangs for
    # service-user installs (systemd accounts, operator users, etc.).
    assert 'if [ "$(id -u)" -eq 0 ] || (command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null); then' in text
    assert "sudo npx playwright install-deps chromium" in text
