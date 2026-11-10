"""Regression for #152451: a lock winner may not have bound its control socket yet."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.waiting import wait_until


_OWNER = r'''
import asyncio
import os
from pathlib import Path
import sys
from gateway import host_rendezvous as hr
from gateway.control_socket import GatewayControlServer

home, ready = Path(sys.argv[1]), Path(sys.argv[2])
multiplex = sys.argv[3] == "true"
outcome, error = hr.claim_host_lock(hr.ROLE_GATEWAY)
assert outcome is hr.HostLockOutcome.ACQUIRED, error
hr.publish_record(hr.ROLE_GATEWAY, profiles=(), home=str(home))
ready.with_suffix(".claimed").touch()

async def serve():
    if not sys.stdin.readline():
        return
    server = GatewayControlServer(home, verb_handlers={
        "identify": lambda: {"pid": os.getpid(), "hermes_home": str(home),
                            "served_profiles": ["worker"], "multiplex": multiplex},
        "rescan-profiles": lambda: {"multiplex": multiplex, "served_profiles": ["worker"]},
    })
    assert await server.start()
    ready.touch()
    try:
        await asyncio.to_thread(sys.stdin.readline)
    finally:
        await server.stop()

try:
    asyncio.run(serve())
finally:
    hr.release_host_lock(hr.ROLE_GATEWAY)
'''


@pytest.mark.platforms("posix")  # Real host flock and Unix control socket.
@pytest.mark.parametrize("multiplex", [False, True])
def test_lock_loser_waits_for_owner_channel_before_deciding(tmp_path, monkeypatch, multiplex):
    from gateway import control_socket, host_attach, host_rendezvous as hr
    from gateway.run import _claim_host_gateway_role

    root = tmp_path / "root"
    owner_home = root / "profiles" / "worker"
    owner_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    host_attach.invalidate_host_gateway_cache()
    ready = tmp_path / "ready"
    env = {**os.environ, "HERMES_HOME": str(owner_home)}
    child = subprocess.Popen(
        [sys.executable, "-c", _OWNER, str(owner_home), str(ready), str(multiplex).lower()],
        env=env, stdin=subprocess.PIPE, text=True,
    )
    identify = control_socket.identify_gateway
    probes = 0

    def identify_after_bind(home, **kwargs):
        nonlocal probes
        probes += 1
        # Keep the socket absent for the first real probe. Bind only when the
        # caller retries, so this boot ordering is deterministic, not a sleep race.
        if probes == 2:
            child.stdin.write("bind\n")
            child.stdin.flush()
            wait_until(ready.exists, what="owner control socket bound")
        return identify(home, **kwargs)

    monkeypatch.setattr(control_socket, "identify_gateway", identify_after_bind)
    try:
        wait_until(ready.with_suffix(".claimed").exists, what="owner claimed host lock")
        if multiplex:
            with pytest.raises(SystemExit) as error:
                _claim_host_gateway_role()
            assert error.value.code == 75
        else:
            _claim_host_gateway_role()
        assert child.poll() is None
    finally:
        child.stdin.close()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)
        hr.release_host_lock(hr.ROLE_GATEWAY)
        host_attach.invalidate_host_gateway_cache()


def test_standalone_waits_for_booting_multiplexer_peer(tmp_path, monkeypatch):
    """Mirror race: the multiplexer won the lock and is live, but its socket binds a moment later."""
    from gateway import host_attach, status
    from hermes_cli import profiles

    root = tmp_path / "hermes"
    home = root / "profiles" / "solo"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("gateway:\n  standalone: true\n")
    (root / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: root)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: root / "profiles")
    host_pid = os.getpid() + 1
    monkeypatch.setattr(status, "live_gateway_pid_for_home",
                        lambda h: host_pid if Path(h) == root else None)
    identity = {"pid": host_pid, "hermes_home": str(root), "profile": "default",
                "multiplex": True, "served_profiles": ["default"]}
    probes = []

    def identify(h):
        probes.append(h)
        # Unbound on the first probe; answers once the caller retries within its window.
        return identity if len(probes) > 1 else None

    monkeypatch.setattr(host_attach, "_identify", identify)
    decision = host_attach.standalone_attach_decision(home, None)
    assert decision is not None and decision.outcome == host_attach.START
