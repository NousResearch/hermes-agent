"""Regression tests for install.sh browser setup.

Browser automation is optional. The installer should not leave Hermes
half-installed just because Playwright's managed Chromium download hangs on an
unsupported distribution.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_install_script_does_not_autodetect_system_browser_on_path() -> None:
    """The installer must not scan PATH/well-known locations for a browser.

    Auto-detection silently bound the install to whatever ``command -v
    chromium`` resolved to — most damagingly a Snap Chromium, whose sandbox
    blocks agent-browser's control socket and hangs every browser_navigate. The
    fallback was dropped in favor of always using the bundled Playwright
    Chromium, so the old PATH-scan and "use the system browser" path are gone.
    """
    text = INSTALL_SH.read_text()

    assert "find_system_browser()" in text
    assert "google-chrome google-chrome-stable chromium chromium-browser chrome" not in text
    assert "Skipping Playwright browser download; Hermes will use the system browser." not in text


def test_install_script_honors_explicit_browser_override_only() -> None:
    """find_system_browser consults only an explicit AGENT_BROWSER_EXECUTABLE_PATH."""
    text = INSTALL_SH.read_text()

    assert 'override="${AGENT_BROWSER_EXECUTABLE_PATH:-}"' in text
    # An explicit override still skips the bundled download (override, not fallback).
    assert "Skipping bundled Chromium download" in text


def test_install_script_strips_stale_snap_browser_override() -> None:
    """Already-affected installs must auto-recover.

    A pre-existing AGENT_BROWSER_EXECUTABLE_PATH pointing at a Snap Chromium is
    the exact value that hangs the browser tool, and the runtime reads it from
    .env — so the installer strips it (and a Snap override is rejected even when
    set explicitly) so the bundled Chromium download runs on update.
    """
    text = INSTALL_SH.read_text()

    assert "strip_snap_browser_override()" in text
    assert "^AGENT_BROWSER_EXECUTABLE_PATH=/snap/" in text
    # Both install paths invoke the migration before resolving a browser.
    assert text.count("strip_snap_browser_override") >= 3
    # A snap path is rejected by find_system_browser itself.
    assert "/snap/*) return 1 ;;" in text


def test_playwright_installs_are_timeout_guarded() -> None:
    text = INSTALL_SH.read_text()

    # The timeout wrapper still exists and is used internally by the install
    # wrapper, so every Playwright download remains bounded.
    assert "run_browser_install_with_timeout()" in text
    # Playwright installs now go through run_playwright_install(), which wraps
    # run_browser_install_with_timeout (timeout-guarded) and adds an
    # unrecognized-platform fallback retry.
    assert "run_playwright_install 600 npx playwright install chromium" in text
    # --with-deps is still invoked on apt-based systems, but only when sudo
    # is available non-interactively (root or passwordless sudo). Non-sudo
    # service users fall back to the browser-only install — see
    # install_node_deps() in install.sh.
    assert "run_playwright_install 600 npx playwright install --with-deps chromium" in text
    # The wrapper still bounds the download with the timeout helper.
    assert 'run_browser_install_with_timeout "$timeout_seconds" "$@"' in text



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


def test_playwright_install_retries_with_platform_override_on_failure() -> None:
    """Installer must self-correct when Playwright doesn't recognize the host.

    On apt releases newer than Playwright knows (Ubuntu 26.04, Debian 14, future
    distros) `playwright install` hangs/fails (#35166). run_playwright_install
    must retry ONCE with PLAYWRIGHT_HOST_PLATFORM_OVERRIDE pinned to the newest
    known build — but only on failure (try-native-first, so a host Playwright
    already supports keeps its native build and avoids the glibc mismatch in
    microsoft/playwright#35114), and never when the operator pinned the value.
    """
    text = INSTALL_SH.read_text()

    assert "run_playwright_install()" in text
    assert "playwright_fallback_platform()" in text
    # Fallback target is the newest known build, arch-aware.
    assert 'echo "ubuntu24.04-x64"' in text
    assert 'echo "ubuntu24.04-arm64"' in text
    # Try native first: only retry after the first attempt fails.
    assert 'if run_browser_install_with_timeout "$timeout_seconds" "$@" 2>/dev/null; then' in text
    # Operator-pinned override is respected (retry skipped).
    assert 'if [ -n "${PLAYWRIGHT_HOST_PLATFORM_OVERRIDE:-}" ]; then' in text
    # The retry actually sets the override for the child process.
    assert 'PLAYWRIGHT_HOST_PLATFORM_OVERRIDE="$fallback" \\' in text

