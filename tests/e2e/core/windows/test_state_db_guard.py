"""The state.db write-guard sees a live foreign holder on native Windows.

``hermes sessions optimize-storage`` rewrites the store (FTS rebuild + VACUUM). Run under a
live writer, it is how every agent ends up refusing turns (#110054), so it refuses while
another process holds ``state.db`` or its WAL sidecars. The holder here is the realistic
one: a real ``hermes gateway run`` for the same profile. Control: the same command succeeds
once the gateway is stopped, so a refusal is about the holder and nothing else.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from tests.e2e.core.windows._helpers import (
    expect,
    hermes,
    hermes_argv,
    kill_tree,
    make_home,
    process_tree,
    wait_until,
)
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [pytest.mark.platforms("windows"), pytest.mark.integration, pytest.mark.live_system_guard_bypass]

def _running(home) -> bool:
    try:
        state = json.loads((home.hermes_home / "gateway_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return state.get("gateway_state") == "running"


def test_optimize_storage_refuses_while_gateway_holds_store(tmp_path: Path) -> None:
    with FakeLLMServer() as srv:
        home = make_home(tmp_path, srv.base_url)
        log = tmp_path / "gateway.log"
        with log.open("wb") as fh:
            gw = subprocess.Popen(hermes_argv("gateway", "run"), cwd=home.project, env=home.env(),
                                  stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT)
        tree: list = []
        try:
            wait_until(lambda: _running(home) or gw.poll() is not None, 120, "the gateway to report running")
            assert gw.poll() is None, f"gateway exited during boot:\n{log.read_text(errors='replace')}"
            tree = process_tree(gw.pid)
            assert home.db_path.exists(), "the running gateway never opened state.db"

            held = hermes(home, "sessions", "optimize-storage")
            refused = held.returncode != 0 and "Refusing" in held.stdout
            stop = hermes(home, "gateway", "stop")
            assert stop.returncode == 0, stop.tail()
            gw.wait(timeout=60)
            quiet = hermes(home, "sessions", "optimize-storage")
            assert quiet.returncode == 0, f"optimize-storage fails even with no holder:\n{quiet.tail()}"
            # venv launcher OR its interpreter child; whole-number match (pid 12 is not in 4123).
            named = any(re.search(rf"\b{p.pid}\b", held.stdout) for p in tree)
            expect(refused and named,
                   f"optimize-storage ran under a live gateway (pid {gw.pid}) instead of refusing:\n{held.tail()}")
        finally:
            kill_tree(tree or process_tree(gw.pid))
