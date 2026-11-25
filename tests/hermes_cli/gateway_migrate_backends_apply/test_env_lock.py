"""A rollback writes ``.env`` back under the same ``.env`` write lock Hermes' own writers take (and the apply's
write takes): while another process holds it, the write-back waits; once it is released the file is put back."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from hermes_cli import gateway_migrate_backends_apply as engine
from tests.hermes_cli.test_gateway_migrate_backends_apply import _PASSWORD, FakeHost, _args, _config_files
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("linux", "macos")  # the lock is flock between two real processes

_REPO = Path(__file__).resolve().parents[3]
_HOLD = ("import sys, pathlib; from hermes_cli.config import _env_write_lock; env = pathlib.Path(sys.argv[1]); "
         "from tests.waiting import wait_until\n"
         "with _env_write_lock(env):\n"
         "    (env.parent / 'lock-held').write_text('')\n"
         "    wait_until((env.parent / 'lock-release').exists, timeout=60)\n")


def test_the_rollback_writes_env_back_only_under_the_env_write_lock(home, tmp_path):
    host = FakeHost(home)
    host.gateway_lies = True  # verify fails: rollback
    before = _config_files(home)
    env = home / ".env"
    real_status, holder = host.gateway_status, []

    def hold_the_lock_during_verify(dial, port):
        if not holder:  # another Hermes writer takes the .env lock while the move is being checked
            holder.append(subprocess.Popen([sys.executable, "-c", _HOLD, str(env)], cwd=_REPO,
                                           env={**os.environ, "PYTHONPATH": str(_REPO)}))
            wait_until((home / "lock-held").exists, timeout=60, what="the other writer holding the .env lock")
        return real_status(dial, port)

    host.gateway_status = hold_the_lock_during_verify
    result: list[int] = []
    run = threading.Thread(target=lambda: result.append(engine.cmd_apply(host, _args())), daemon=True)
    run.start()
    try:
        wait_until(lambda: holder and (home / "lock-held").exists(), timeout=60, what="the lock held")
        wait_until(lambda: (home / "config.yaml").read_bytes() == before["config.yaml"], timeout=60,
                   what="the rollback reaching .env (config.yaml, put back first, is back)")
        run.join(timeout=1.0)
        assert run.is_alive() and _PASSWORD in env.read_text(encoding="utf-8")  # it waits: nothing written yet
    finally:
        (home / "lock-release").write_text("", encoding="utf-8")
        holder[0].wait(timeout=60) if holder else None
    run.join(timeout=60)
    assert result == [engine.EXIT_ROLLED_BACK] and _config_files(home) == before
