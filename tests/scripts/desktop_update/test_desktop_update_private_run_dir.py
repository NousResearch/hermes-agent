"""The posix Desktop hand-off keeps its run files out of fixed names in the shared temp root.

The status JSON the progress window renders, its write-then-rename temp file, the UI port file
and the custody go-file used to sit at ``${TMPDIR:-/tmp}/hermes-update-*.<pid>``, created through
shell ``>`` redirects that follow a symlink another local user planted there. They now live in
one fresh ``mktemp -d`` dir (0700, unpredictable), which the run removes when it finishes. Drives
the real ``posix.sh`` end to end with a stand-in ``hermes`` that reports what it saw mid-run.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.waiting import wait_until

SHIM = Path(__file__).resolve().parents[3] / "scripts" / "desktop-update" / "posix.sh"

pytestmark = [
    pytest.mark.platforms("posix"),
    pytest.mark.skipif(not (os.path.exists("/bin/bash") and os.path.exists("/usr/bin/python3")),
                       reason="posix.sh detaches through /bin/bash and /usr/bin/python3"),
]

# Stands in for `hermes update`: records where the hand-off pointed its children and what sat in
# the temp root while the update ran.
FAKE_HERMES = """#!/usr/bin/env bash
case "$*" in *--help*) echo "--keep-stash"; exit 0 ;; esac
/usr/bin/python3 - "$HERMES_UPDATE_STATUS_FILE" "$TMPDIR" "$HERMES_TEST_CAPTURE" <<'PY'
import json, os, sys
status, tmp, out = sys.argv[1:]
st = os.lstat(os.path.dirname(status))
with open(out, "w", encoding="utf-8") as fh:
    json.dump({"status": status, "dir_mode": st.st_mode & 0o777, "dir_uid": st.st_uid,
               "tmp_entries": sorted(os.listdir(tmp))}, fh)
PY
"""


def test_run_files_live_in_a_private_dir_removed_at_the_end(tmp_path):
    shared_tmp = tmp_path / "shared-tmp"
    shared_tmp.mkdir()
    shared_tmp.chmod(0o1777)
    install_root = tmp_path / "hermes-agent"
    (install_root / "venv" / "bin").mkdir(parents=True)
    hermes = install_root / "venv" / "bin" / "hermes"
    hermes.write_text(FAKE_HERMES, encoding="utf-8")
    hermes.chmod(0o755)
    capture = tmp_path / "seen.json"
    env = {**os.environ, "TMPDIR": str(shared_tmp), "HERMES_TEST_CAPTURE": str(capture)}
    env.pop("HERMES_HOME", None)  # legacy install-parent home: tmp_path

    subprocess.run(["/bin/bash", str(SHIM), "--install-root", str(install_root), "--no-ui"],
                   env=env, timeout=60, check=True, stdin=subprocess.DEVNULL)
    result = tmp_path / ".hermes-update-result.json"
    wait_until(lambda: result.exists() and not any(shared_tmp.iterdir()), timeout=45,
               what="the detached update to finish and remove its run dir")

    seen = json.loads(capture.read_text(encoding="utf-8-sig"))
    run_dir = Path(seen["status"]).parent
    assert run_dir.parent == shared_tmp and run_dir.name.startswith("hermes-update-run-")
    assert seen["dir_mode"] == 0o700 and seen["dir_uid"] == os.getuid()  # windows-footgun: ok — posix-only
    assert [e for e in seen["tmp_entries"] if not e.startswith("hermes-update-run-")] == []
    assert json.loads(result.read_text(encoding="utf-8-sig"))["ok"] is True
    assert list(shared_tmp.iterdir()) == []
