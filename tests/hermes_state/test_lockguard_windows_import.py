"""Windows fcntl compatibility shims must not prevent SessionDB startup (#118026)."""

import os
import subprocess
import sys

import pytest


@pytest.mark.windows_only
@pytest.mark.parametrize("shim", ["absent", "partial", "ofd_constants"])
def test_session_db_works_without_posix_lockguard(tmp_path, shim):
    code = r'''
import os
import sys
import types
from pathlib import Path

if sys.argv[1] == "absent":
    sys.modules["fcntl"] = None
else:
    shim = types.ModuleType("fcntl")
    if sys.argv[1] == "ofd_constants":
        shim.F_OFD_SETLK = 37
        shim.F_RDLCK = 0
        shim.F_UNLCK = 2
    sys.modules["fcntl"] = shim

from hermes_state import SessionDB
import hermes_state_lockguard as guard
assert not guard.supported(), "POSIX OFD locks must remain disabled on Windows"
assert guard.hold(Path(os.environ["HERMES_HOME"]) / "missing.db") == {}
guard.release({})
db = SessionDB(Path(os.environ["HERMES_HOME"]) / "state.db")
try:
    db.create_session("windows-import", source="cli")
    db.append_message("windows-import", role="user", content="still writable")
    assert db.get_messages("windows-import")[0]["content"] == "still writable"
finally:
    db.close()
'''
    env = os.environ.copy()
    for key in ("HOME", "USERPROFILE", "HERMES_HOME", "APPDATA", "LOCALAPPDATA"):
        env[key] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", code, shim], env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
