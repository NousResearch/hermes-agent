"""Regression tests for install.sh browser install modes."""

from __future__ import annotations

import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _extract_function_body(name: str) -> str:
    text = INSTALL_SH.read_text()
    match = re.search(
        rf"^{re.escape(name)}\(\)\s*\{{\s*\n(?P<body>.*?)^\}}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"{name}() not found in scripts/install.sh"
    return match.group(0)


def _run_install_node_deps(*, distro: str, browser_install_mode: str) -> str:
    func = _extract_function_body("install_node_deps")

    with tempfile.TemporaryDirectory(prefix="hermes-install-sh-") as tmp:
        tmp_path = Path(tmp)
        bin_dir = tmp_path / "bin"
        install_dir = tmp_path / "install"
        bin_dir.mkdir()
        install_dir.mkdir()
        (install_dir / "package.json").write_text("{}")

        for name, content in {
            "npm": "#!/bin/bash\nexit 0\n",
            "npx": "#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"$TMP_NPX_LOG\"\nexit 0\n",
        }.items():
            path = bin_dir / name
            path.write_text(content)
            path.chmod(path.stat().st_mode | stat.S_IXUSR)

        driver = tmp_path / "driver.sh"
        driver.write_text(
            "\n".join(
                [
                    "#!/bin/bash",
                    "set -e",
                    func,
                    "log_info(){ :; }",
                    "log_warn(){ :; }",
                    "log_success(){ :; }",
                    "HAS_NODE=true",
                    f'DISTRO="{distro}"',
                    f'BROWSER_INSTALL_MODE="{browser_install_mode}"',
                    f'INSTALL_DIR="{install_dir}"',
                    "install_node_deps",
                ]
            )
            + "\n"
        )
        driver.chmod(driver.stat().st_mode | stat.S_IXUSR)

        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
        env["TMP_NPX_LOG"] = str(tmp_path / "npx.log")

        subprocess.run(["bash", str(driver)], check=True, env=env)

        log_path = tmp_path / "npx.log"
        return log_path.read_text() if log_path.exists() else ""


def test_install_script_help_mentions_browser_install_mode() -> None:
    text = INSTALL_SH.read_text()
    assert "--browser-install-mode MODE" in text
    assert "auto = install browser + system deps when supported" in text
    assert "user = install browser only, never use sudo/system pkg mgr" in text
    assert "skip = skip Playwright browser install entirely" in text


def test_ubuntu_auto_mode_uses_with_deps() -> None:
    log = _run_install_node_deps(distro="ubuntu", browser_install_mode="auto")
    assert "playwright install --with-deps chromium" in log


def test_ubuntu_user_mode_skips_system_dependency_install() -> None:
    log = _run_install_node_deps(distro="ubuntu", browser_install_mode="user")
    assert "playwright install chromium" in log
    assert "--with-deps" not in log


def test_skip_mode_omits_playwright_install_entirely() -> None:
    log = _run_install_node_deps(distro="ubuntu", browser_install_mode="skip")
    assert log == ""
