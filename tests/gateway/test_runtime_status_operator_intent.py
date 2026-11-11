"""An operator's ``hermes gateway stop`` in the s6 container survives the next container start.

The gateway owns ``gateway_state`` in ``gateway_state.json``; the s6 lifecycle CLI owns
``desired_state``. The container boot reconciler starts the root slot from ``desired_state``, so a
stop is durable only if no later gateway status write puts the gateway's boot-time value back.
"""
from __future__ import annotations

import json
import os
import subprocess

from gateway import status
from hermes_cli import container_boot, service_manager


class _S6Tools:
    """Stands in for the ``/command/s6-*`` binaries: answers ``s6-svstat`` with a pid, accepts
    ``s6-svc``, and records each invocation."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, cmd, *args, timeout=5, check=False):
        self.calls.append((cmd, *args))
        stdout = f"{self.pid}\n" if cmd == "s6-svstat" else ""
        return subprocess.CompletedProcess([cmd, *args], 0, stdout=stdout, stderr="")


def test_operator_stop_under_s6_keeps_the_gateway_down_on_the_next_container_start(tmp_path, monkeypatch):
    home = tmp_path / "data"
    home.mkdir()
    scandir = tmp_path / "hermes-service"
    scandir.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    s6 = _S6Tools(os.getpid())
    monkeypatch.setattr(service_manager, "_s6_run", s6)
    mgr = service_manager.S6ServiceManager(scandir=scandir)

    # First container start: no state yet, the root slot is registered down.
    container_boot.reconcile_profile_gateways(hermes_home=home, scandir=scandir, container_argv=())
    # `hermes gateway start`, then the supervised gateway boots and serves.
    mgr.start("gateway-default")
    assert status.write_runtime_status(
        gateway_state="starting", exit_reason=None, clear_profile_platforms=True,
        reload_existing=True, wait_timeout=5.0)
    status.publish_runtime_status(gateway_state="running")
    assert status.flush_runtime_status(timeout=5.0)

    # `hermes gateway stop`: s6-svc -d returns at once; the gateway drains and stops afterwards.
    mgr.stop("gateway-default")
    assert ("s6-svc", "-d", str(scandir / "gateway-default")) in s6.calls
    status.publish_runtime_status(gateway_state="draining")
    status.publish_runtime_status(gateway_state="stopped", exit_reason="stopped by operator")
    assert status.flush_runtime_status(timeout=5.0)

    # Next container start.
    actions = container_boot.reconcile_profile_gateways(hermes_home=home, scandir=scandir, container_argv=())
    root = next(a for a in actions if a.profile == "default")
    assert (root.prior_state, root.action) == ("stopped", "registered")
    assert (scandir / "gateway-default" / "down").exists()
    record = json.loads((home / "gateway_state.json").read_text(encoding="utf-8"))
    assert (record["gateway_state"], record["desired_state"]) == ("stopped", "stopped")
