"""Regression tests: installers provision cua-driver (Computer Use).

Policy: choosing Computer Use should be a config flip, not a surprise
multi-minute binary fetch. The installers pre-install cua-driver
(best-effort, skippable), and the dashboard toggle auto-installs when the
binary is still missing (see test_web_routers_tools_install_on_enable.py).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


class TestInstallSh:
    def test_has_skip_flag(self) -> None:
        text = INSTALL_SH.read_text()
        assert "--skip-computer-use)" in text
        assert "SKIP_COMPUTER_USE=true" in text
        assert "SKIP_COMPUTER_USE=false" in text  # default off
        assert "--skip-computer-use  Skip the cua-driver" in text

    def test_install_function_wired_into_main_and_stage(self) -> None:
        text = INSTALL_SH.read_text()
        assert "install_computer_use_driver() {" in text
        # main() flow and the node-deps stage both run it.
        assert text.count("install_computer_use_driver\n") >= 2

    def test_install_is_timeboxed_above_upstream_lock_window(self) -> None:
        """The upstream installer serializes on a lock with a 600s stale
        window; a ceiling below that reintroduces the self-perpetuating
        wedge (#58762). Must stay >= 660."""
        text = INSTALL_SH.read_text()
        assert "run_with_timeout 660 /bin/bash -c" in text

    def test_install_is_best_effort(self) -> None:
        text = INSTALL_SH.read_text()
        assert "Computer Use driver install failed" in text
        assert "hermes computer-use install" in text

    def test_skips_unwritable_applications_dir(self) -> None:
        """Non-admin macOS accounts can't receive CuaDriver.app (#47865
        class) — skip cleanly instead of failing every install."""
        text = INSTALL_SH.read_text()
        assert '[ -d /Applications ] && [ ! -w /Applications ]' in text


class TestInstallPs1:
    def test_has_skip_switch(self) -> None:
        text = INSTALL_PS1.read_text()
        assert "[switch]$SkipComputerUse," in text
        assert "if ($SkipComputerUse)" in text

    def test_install_function_wired_into_node_deps(self) -> None:
        text = INSTALL_PS1.read_text()
        assert "function Install-CuaDriver {" in text
        assert "    Install-CuaDriver\n" in text

    def test_install_is_timeboxed_above_upstream_lock_window(self) -> None:
        text = INSTALL_PS1.read_text()
        assert "Wait-Job $job -Timeout 660" in text

    def test_install_is_best_effort(self) -> None:
        text = INSTALL_PS1.read_text()
        assert "Computer Use driver install timed out" in text
        assert "hermes computer-use install" in text
