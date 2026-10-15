"""Same-named gateways under distinct roots retain independent process ownership."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import dashboard_procs, gateway


@pytest.mark.platforms("macos")
@pytest.mark.spawns_gateway_lookalike
def test_named_profile_scan_and_reap_follow_real_child_homes(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    homes = [tmp_path / root / "profiles" / "ops" for root in ("owned", "foreign")]
    for home in homes:
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    script = tmp_path / "hermes"
    script.write_text(
        "import os, pathlib, time\n"
        "pathlib.Path(os.environ['STUB_READY']).touch()\n"
        "parent, deadline = os.getppid(), time.monotonic() + 120\n"
        "while os.getppid() == parent and time.monotonic() < deadline:\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    children = []
    try:
        for i, home in enumerate(homes):
            ready = tmp_path / f"ready-{i}"
            child = subprocess.Popen(
                [sys.executable, str(script), "--profile", "ops", "gateway", "run"],
                env={**os.environ, "HERMES_HOME": str(home.parent.parent), "STUB_READY": str(ready)},
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            children.append(child)
            deadline = time.monotonic() + 15
            while not ready.exists():
                assert child.poll() is None and time.monotonic() < deadline, "stub failed to start"
                time.sleep(0.05)

        owned, foreign = children
        real_home_for_pid = dashboard_procs._hermes_home_for_pid
        # Resolve real argv/environment only for test children. Other processes cannot
        # trigger reads of the operator's configuration or become signal targets.
        monkeypatch.setattr(dashboard_procs, "_hermes_home_for_pid", lambda pid:
                            real_home_for_pid(pid) if pid in {owned.pid, foreign.pid} else None)
        assert real_home_for_pid(owned.pid) == str(homes[0])
        assert real_home_for_pid(foreign.pid) == str(homes[1])
        assert gateway._scan_gateway_pids(set()) == [owned.pid]
        real_find = gateway.find_gateway_pids

        def only_test_child(*args, **kwargs):
            found = real_find(*args, **kwargs)
            assert set(found) <= {owned.pid}, "refusing to signal any unowned process"
            return found

        monkeypatch.setattr(gateway, "find_gateway_pids", only_test_child)
        monkeypatch.setattr(gateway, "_get_service_pids", lambda **kwargs: set())
        assert gateway._reap_unsupervised_gateway_orphans()
        owned.wait(timeout=10)
        assert foreign.poll() is None
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=10)
