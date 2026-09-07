"""Regression tests: installers provision cua-driver (Computer Use).

Policy: choosing Computer Use should be a config flip, not a surprise
multi-minute binary fetch. The installers pre-install cua-driver
(best-effort, skippable), and the dashboard toggle auto-installs when the
binary is still missing (see test_web_routers_tools_install_on_enable.py).
"""

from pathlib import Path
import json
import shutil
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def _run_native_cua(tmp_path, host, mode):
    executable = shutil.which(host)
    assert executable, f"Native installer matrix requires {host}"
    result = subprocess.run(
        [
            executable,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(REPO_ROOT / "scripts/tests/test-install-ps1-cua-contract.ps1"),
            "-WorkRoot",
            str(tmp_path),
            "-Mode",
            mode,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    receipt = next(
        line.split("=", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("CUA_FIXTURE_RESULT=")
    )
    return json.loads(receipt), output


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
        assert "[ -d /Applications ] && [ ! -w /Applications ]" in text


class TestInstallPs1:
    def test_has_skip_switch(self) -> None:
        text = INSTALL_PS1.read_text()
        assert "[switch]$SkipComputerUse," in text
        assert "if ($SkipComputerUse)" in text

    def test_install_function_wired_into_node_deps(self) -> None:
        text = INSTALL_PS1.read_text()
        assert "function Install-CuaDriver {" in text
        assert "    Install-CuaDriver\n" in text

    @pytest.mark.windows_only
    @pytest.mark.parametrize("host", ["powershell.exe", "pwsh.exe"])
    def test_install_is_timeboxed_above_upstream_lock_window(
        self, tmp_path, host
    ) -> None:
        receipt, _ = _run_native_cua(tmp_path, host, "failure")
        assert receipt["timeout"] >= 660

    @pytest.mark.windows_only
    @pytest.mark.parametrize("host", ["powershell.exe", "pwsh.exe"])
    @pytest.mark.parametrize("mode", ["failure", "timeout", "missing"])
    def test_install_is_best_effort(self, tmp_path, host, mode) -> None:
        receipt, output = _run_native_cua(tmp_path, host, mode)
        assert receipt["completed"]
        assert "hermes computer-use install" in output
        expected = {
            "failure": "failed (exit 7)",
            "timeout": "timed out",
            "missing": "did not produce a compatible runtime",
        }
        assert expected[mode] in output

    def test_install_rechecks_runtime_contract_before_success(self) -> None:
        text = INSTALL_PS1.read_text()
        assert "$installedCuaDriver = Get-Command cua-driver" in text
        assert (
            "Test-CuaDriverRuntimeContract -DriverPath $installedCuaDriver.Source"
            in text
        )
        assert "did not produce a compatible runtime" in text
