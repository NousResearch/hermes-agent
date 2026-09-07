"""The update hand-off cleans only the browser profile it launched."""
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.linux_only
@pytest.mark.parametrize("failed", [False, True])
def test_shim_removes_owned_profile_on_exit(tmp_path, failed):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    browser = bin_dir / "google-chrome"
    browser.write_text(
        '#!/bin/bash\ntrap "exit 0" TERM\n'
        'for arg in "$@"; do\n'
        'case "$arg" in --user-data-dir=*) dir="${arg#--user-data-dir=}" ;; esac\n'
        'done\nmkdir -p "$dir"\nprintf "%s" "$dir" > "$HOME/launched-profile"\n'
        'while :; do sleep 0.1; done\n', encoding="utf-8",
    )
    browser.chmod(0o755)
    config = tmp_path / ".config"
    config.mkdir()
    (config / "mimeapps.list").write_text(
        '[Default Applications]\nx-scheme-handler/http=google-chrome.desktop\n'
        'x-scheme-handler/https=google-chrome.desktop\ntext/html=google-chrome.desktop\n',
        encoding="utf-8",
    )
    sibling = tmp_path / "hermes-update-ui-unrelated"
    sibling.mkdir()
    (sibling / "keep").write_text("keep", encoding="utf-8")
    install = tmp_path / "hermes-agent"
    install.mkdir()
    env = {**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(tmp_path),
           "XDG_CONFIG_HOME": str(config), "TMPDIR": str(tmp_path),
           "PATH": f"{bin_dir}:/usr/bin:/bin", "HERMES_SELFTEST_HOLD_SECONDS": "1",
           "HERMES_UPDATE_SHIM_GRACE_SECONDS": "1", "HERMES_SELFTEST_FAIL": "1" if failed else "0"}
    result = subprocess.run(
        ["bash", str(Path(__file__).resolve().parents[1] / "scripts/desktop-update/posix.sh"),
         "--install-root", str(install), "--self-test-ui"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == (1 if failed else 0), result.stdout + result.stderr
    owned = Path((tmp_path / "launched-profile").read_text(encoding="utf-8"))
    assert not owned.exists()
    assert (sibling / "keep").read_text(encoding="utf-8") == "keep"
