"""The update hand-off cleans only an atomically claimed browser profile."""
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.linux_only
@pytest.mark.parametrize("outcome", ["success", "error", "allocation-failed"])
def test_shim_removes_only_its_owned_profile(tmp_path, outcome):
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
    if outcome == "allocation-failed":
        allocator = bin_dir / "mktemp"
        allocator.write_text('#!/bin/sh\nprintf "%s\\n" "$HOME"\nexit 1\n', encoding="utf-8")
        allocator.chmod(0o755)
    config = tmp_path / ".config"
    config.mkdir()
    (config / "mimeapps.list").write_text(
        '[Default Applications]\nx-scheme-handler/http=google-chrome.desktop\n'
        'x-scheme-handler/https=google-chrome.desktop\ntext/html=google-chrome.desktop\n',
        encoding="utf-8",
    )
    install = tmp_path / "hermes-agent"
    install.mkdir()
    env = {**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(tmp_path),
           "XDG_CONFIG_HOME": str(config), "TMPDIR": str(tmp_path),
           "PATH": f"{bin_dir}:/usr/bin:/bin", "HERMES_SELFTEST_HOLD_SECONDS": "1",
           "HERMES_UPDATE_SHIM_GRACE_SECONDS": "1", "HERMES_SELFTEST_FAIL": "1" if outcome == "error" else ""}
    process = subprocess.Popen(
        ["bash", "-c", 'while [ ! -f "$HOME/start" ]; do sleep .05; done; exec bash "$@"', "probe",
         str(Path(__file__).resolve().parents[1] / "scripts/desktop-update/posix.sh"),
         "--install-root", str(install), "--self-test-ui"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    collision = tmp_path / f"hermes-update-ui-{process.pid}"
    collision.mkdir()
    (collision / "keep").write_text("keep", encoding="utf-8")
    (tmp_path / "start").touch()
    stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == (1 if outcome == "error" else 0), stdout + stderr
    launched = tmp_path / "launched-profile"
    if outcome == "allocation-failed":
        assert not launched.exists()
    else:
        owned = Path(launched.read_text(encoding="utf-8"))
        assert owned != collision
        assert not owned.exists()
    assert (collision / "keep").read_text(encoding="utf-8") == "keep"
