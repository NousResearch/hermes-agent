"""Regression tests for install.sh conditional sudo/dependency handling.

The installer must not invoke sudo to install a system package that is already
present, and when sudo is unavailable it must tell the user exactly which
dependencies are missing so they can install them as root (without granting
sudo to the Hermes service account).
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_libatomic1_install_is_gated_on_package_presence() -> None:
    """libatomic1 is installed only when it is not already present.

    The managed Node's linux-x64 builds link libatomic.so.1, which minimal
    Debian/Ubuntu images may not ship (#87460). The installer used to run
    `apt-get install libatomic1` whenever apt was present, invoking sudo even
    when the package was already there. The apt branch must first check
    `dpkg -s libatomic1` so a satisfied dependency never triggers a sudo
    prompt, and must only attempt the install non-interactively (root or
    passwordless sudo) so a password-prompting sudo cannot block the installer.
    """
    text = INSTALL_SH.read_text()

    # The apt install of libatomic1 must be gated behind a presence check.
    assert "command -v apt-get >/dev/null 2>&1 && ! dpkg -s libatomic1 >/dev/null 2>&1; then" in text
    # Root or passwordless sudo only — no bare `$sudo_cmd` that could prompt.
    assert "sudo -n true 2>/dev/null" in text
    # The install command itself is still present inside the gated branch.
    assert "apt-get install -y -qq libatomic1" in text


def test_build_tools_check_matches_installed_package() -> None:
    """The build-tools presence check must match what is actually installed.

    install_deps() installs `build-essential python3-dev libffi-dev` but used to
    check for `gcc` instead, so the sudo path could fire even when build-essential
    was already present. The check must probe the real package set.
    """
    text = INSTALL_SH.read_text()

    assert "for pkg in build-essential python3-dev libffi-dev; do" in text
    assert "for pkg in gcc python3-dev libffi-dev; do" not in text


def test_non_sudo_playwright_branch_reports_missing_packages() -> None:
    """The non-sudo Playwright branch lists missing packages before the sudo hint.

    When sudo is unavailable, the installer used to print only the admin command
    `sudo npx playwright install-deps chromium` without saying which packages were
    actually missing. It must now run `install-deps --dry-run` (apt-get install -s,
    no root needed) so the user sees the exact dependency list to install via root.
    """
    text = INSTALL_SH.read_text()

    # The dry-run must appear in the non-sudo fallback, before the admin hint.
    assert "npx playwright install-deps --dry-run chromium || true" in text
    # The admin command is still surfaced afterwards.
    assert "sudo npx playwright install-deps chromium" in text
