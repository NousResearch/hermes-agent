"""A fork-history refusal is reported as a refusal, not as a bug to share.

``hermes_cli/update_history.guard_fork_history`` stops a non-fast-forward fork
update on purpose: it prints ``HERMES_UPDATE_HISTORY_REVIEW_REQUIRED`` and
exits 2. Both desktop hand-offs must turn that into a plain "refused" result,
keep exit 2, and never send the user to ``hermes debug share``. An exit 2
without the banner still reads as the generic failure.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from tests.installation_launcher_fixture import publish_fixture_launcher

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SHIM_DIR = ROOT / "scripts" / "desktop-update"
MARKER = "HERMES_UPDATE_HISTORY_REVIEW_REQUIRED"


def _assert_receipt(receipt: dict, refused: bool) -> None:
    assert receipt["ok"] is False
    assert receipt["exit_code"] == 2
    message = receipt["message"]
    if refused:
        assert message.startswith("Update refused:"), message
        assert "fast-forward" in message and "--yes does not override" in message
        assert "debug share" not in message
    else:
        assert not message.startswith("Update refused"), message


WINDOWS_CLI = """
import json, os, sys
from pathlib import Path
def main():
    if '--version' in sys.argv:
        print('Install directory: ' + str(Path(__file__).resolve().parents[1])); return 0
    if '--help' in sys.argv:
        print('--keep-stash'); return 0
    with Path(os.environ['HANDOFF_CALLS']).open('a') as stream:
        stream.write(json.dumps(sys.argv[1:]) + '\\n')
    if os.environ.get('HANDOFF_BANNER'):
        print(os.environ['HANDOFF_BANNER'])
    return 2
if __name__ == '__main__':
    sys.exit(main())
"""


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("refused", [True, False])
def test_windows_handoff_names_a_history_refusal(tmp_path: Path, refused: bool) -> None:
    install = tmp_path / "checkout"
    publish_fixture_launcher(install, WINDOWS_CLI)
    home = tmp_path / "profile"; home.mkdir()
    calls = tmp_path / "calls.jsonl"
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(SHIM_DIR / "windows.ps1"), "-InstallRoot", str(install), "-NoUi"],
        cwd=tmp_path,
        env={**os.environ, "HERMES_HOME": str(home),
             "HERMES_RUNTIME_DIR": str(tmp_path / "empty-store"),
             "HANDOFF_CALLS": str(calls), "HANDOFF_BANNER": MARKER if refused else ""},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert len(calls.read_text().splitlines()) == 1, "a refusal must not be retried"
    receipt = json.loads((home / ".hermes-update-result.json").read_text(encoding="utf-8-sig"))
    _assert_receipt(receipt, refused)
    assert ("debug share" in receipt["message"]) is not refused


POSIX_HERMES = """#!/usr/bin/env bash
case "$*" in *--help*) echo "--keep-stash"; exit 0 ;; esac
printf '%s\\n' "$*" >> "$HERMES_TEST_ARGV"
[ -n "$HANDOFF_BANNER" ] && echo "$HANDOFF_BANNER"
exit 2
"""


@pytest.mark.skipif(
    not (os.path.exists("/bin/bash") and os.path.exists("/usr/bin/python3")),
    reason="posix.sh detaches through /bin/bash and /usr/bin/python3",
)
@pytest.mark.parametrize("refused", [True, False])
def test_posix_handoff_names_a_history_refusal(tmp_path: Path, refused: bool) -> None:
    install_root = tmp_path / "hermes-agent"
    (install_root / "venv" / "bin").mkdir(parents=True)
    hermes = install_root / "venv" / "bin" / "hermes"
    hermes.write_text(POSIX_HERMES)
    hermes.chmod(0o755)
    argv_log = tmp_path / "argv.txt"
    env = {**os.environ, "TMPDIR": str(tmp_path), "HERMES_HOME": str(tmp_path),
           "HERMES_TEST_ARGV": str(argv_log), "HANDOFF_BANNER": MARKER if refused else ""}
    subprocess.run(["/bin/bash", str(SHIM_DIR / "posix.sh"), "--install-root", str(install_root), "--no-ui"],
                   env=env, timeout=60, check=True)
    result = tmp_path / ".hermes-update-result.json"
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline and not result.exists():
        time.sleep(0.1)
    assert result.exists(), "hand-off never wrote its result file"
    update_calls = [c for c in argv_log.read_text().splitlines() if " update " in f" {c} "]
    assert len(update_calls) == 1, "a refusal must not be retried"
    _assert_receipt(json.loads(result.read_text()), refused)
