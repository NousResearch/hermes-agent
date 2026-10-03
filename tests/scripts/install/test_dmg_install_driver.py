"""The DMG driver must wait for native bootstrap completion, not install.sh's marker."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


DRIVER = Path(__file__).resolve().parents[2] / "install/e2e-assets/drive-dmg-install.sh"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("missing,expected", [
    ("", True),
    ("legacy", True),
    ("checkout", False),
    ("launcher", False),
    ("app", False),
    ("completion", False),
    ("historical", True),
    ("marker-only", False),
    ("wrong-root-log", False),
])
def test_dmg_driver_requires_complete_pm_source_install(tmp_path, missing, expected):
    root = tmp_path / "installed source"
    root.mkdir()
    if missing != "legacy":
        (root / "pm").mkdir(parents=True)
        (root / "pm/lock.json").write_text("{}", encoding="utf-8")
    if missing != "checkout":
        (root / ".git").mkdir()
    launcher = root / ".hermes/bin/hermes"
    if missing not in {"launcher", "legacy"}:
        launcher.parent.mkdir(parents=True)
        launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        launcher.chmod(0o755)
    if missing != "app":
        (root / "apps/desktop/release/mac-arm64/Hermes.app").mkdir(parents=True)
    if missing not in {"completion", "historical", "wrong-root-log"}:
        (root / ".hermes-bootstrap-complete").write_text("completed", encoding="utf-8")
    # The legacy file must not mask a missing PM publication.
    legacy = root / "venv/bin/hermes"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    legacy.chmod(0o755)

    mocks = tmp_path / "mock-bin"
    mocks.mkdir()
    for name, script in {
        "osascript": "#!/bin/sh\nprintf 'no-window\\n'\n",
        "cliclick": "#!/bin/sh\nexit 0\n",
        "screencapture": "#!/bin/sh\nexit 0\n",
        "sleep": f"#!/bin/sh\n{shutil.which('sleep')} 0.05\n",
    }.items():
        command = mocks / name
        command.write_text(script, encoding="utf-8")
        command.chmod(0o755)
    app_bin = tmp_path / "Hermes-Setup"
    app_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    app_bin.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    if missing not in {"completion", "marker-only"}:
        logs = home / ".hermes/logs"
        logs.mkdir(parents=True)
        logged_root = tmp_path / "other-install" if missing == "wrong-root-log" else root
        (logs / "bootstrap-installer.log").write_text(
            f"INFO hermes_bootstrap_lib::bootstrap: bootstrap complete install_root={logged_root}\n",
            encoding="utf-8",
        )
    result = subprocess.run(
        ["bash", str(DRIVER), "--app-bin", str(app_bin), "--install-dir", str(root),
         "--install-timeout-secs", "2", "--proof-dir", str(tmp_path / "proof")],
        env=dict(os.environ, HOME=str(home), PATH=f"{mocks}:{os.environ['PATH']}"),
        capture_output=True, text=True, timeout=15,
    )
    assert (result.returncode == 0) is expected, result.stdout + result.stderr
    assert ("install landed:" in result.stdout) is expected, result.stdout
