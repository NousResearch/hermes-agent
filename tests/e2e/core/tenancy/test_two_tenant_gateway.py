"""C7 two-tenant canary: ONE multiplexed messaging gateway serving three profiles.

Class: multiplex / tenancy isolation (issue_classes.md C7). Users hit it as "profile B's cron job ran
with profile A's terminal settings", "my secondary profile answered with the launch profile's key",
"deleting a profile killed the shared gateway", "a restart re-pinned one profile's env for all"
(#89315, #101719, #105396, #111151, #102769, #107692).

Harness: a real ``hermes gateway run`` child process with ``gateway.multiplex_profiles: true`` serving
the launch profile ``default`` plus ``alpha`` and ``beta``. Each profile owns its own loopback provider
(accepting only its own key), and a distinct value for EVERY per-profile knob: provider key (same env
var NAME, different value), API-server key, a non-secret .env marker, model id, MEMORY.md, SOUL.md,
terminal cwd, cron prompt, and a shell variable its tool call exports. Turns are driven through the
real api_server platform (``/v1`` for the launch profile, ``/p/<name>/v1`` for the others); every
turn runs ``env | sort; pwd`` in the terminal tool, so the provider receives an env snapshot taken
INSIDE the tool subprocess.

Phases (same invariant after each): interleaved turns + a cron tick firing one job per profile ->
cross-profile API keys -> create/attach/delete a fourth profile (host PID must not change) ->
SIGTERM restart + a second cron tick. Invariant (``check_isolation``): every request a provider
recorded carries only its own tenant's key/model and no other tenant's canary; every env snapshot ran
in its own cwd and shows no other tenant's canary; no file in any profile home (state.db + WAL, logs,
sessions, cron output, config, .env) contains another tenant's canary; the gateway's own output
never contains any tenant's secret.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from . import _helpers as H

# The gateway child is spawned and reaped by this module (fake HOME, no systemd bus in its env).
pytestmark = pytest.mark.spawns_gateway_lookalike

NAMES = ("default", "alpha", "beta")


class MultiplexGateway:
    def __init__(self, root: Path, port: int) -> None:
        self.root, self.home, self.port = root, root / "home", port
        self.log_path = root / "gateway.log"
        self.proc: subprocess.Popen | None = None
        self.pids: list[int] = []

    def start(self) -> None:
        log = open(self.log_path, "a", encoding="utf-8")  # noqa: SIM115 - handed to the child
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "hermes_cli.main", "gateway", "run"], cwd=str(self.home),
            env=H.hermetic_env(self.home), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        self.pids.append(self.proc.pid)
        H.poll(lambda: self.healthy() or self._died(), 120, "multiplexed gateway /health")
        assert self.proc.poll() is None, f"gateway exited rc={self.proc.returncode}: {self.tail()}"

    def _died(self) -> bool:
        return self.proc is not None and self.proc.poll() is not None

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=3) as r:
                return r.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def chat(self, prefix: str, api_key: str, text: str) -> tuple[int, Any]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{prefix}/v1/chat/completions",
            data=json.dumps({"model": "hermes", "messages": [{"role": "user", "content": text}]}).encode(),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, e.read()[:500]

    def stop(self) -> None:
        if self.proc is None:
            return
        H.kill_group(self.proc, signal.SIGTERM)
        try:
            self.proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            H.kill_group(self.proc)
            self.proc.wait(timeout=10)
        H.kill_group(self.proc)

    def tail(self, n: int = 3000) -> str:
        return self.log_path.read_text(encoding="utf-8", errors="replace")[-n:] if self.log_path.exists() else ""


def prefix(name: str) -> str:
    return "" if name == "default" else f"/p/{name}"


@pytest.fixture(scope="module")
def fleet(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("c7-gateway")
    home = root / "home"
    tenants = H.make_tenants(root, NAMES)
    port = H.free_port()
    for t in tenants.values():
        assert t.srv is not None
        t.srv.start()
    # The profile root is HOME-anchored: prove it is inside the sandbox before writing any profile.
    home.mkdir(parents=True)
    H.assert_profiles_root_under(root, home)
    for t in tenants.values():
        extra = {} if t.name != "default" else {
            "gateway": {"multiplex_profiles": True},
            "platforms": {"api_server": {"enabled": True, "extra": {"host": "127.0.0.1", "port": port}}},
        }
        H.write_tenant_home(t, extra)
    gw = MultiplexGateway(root, port)
    try:
        yield root, tenants, gw
    finally:
        gw.stop()
        for pid in gw.pids:
            assert not H.pid_alive(pid), f"gateway pid {pid} survived teardown"
        for t in tenants.values():
            assert t.srv is not None
            t.srv.stop()


def _turns(gw: MultiplexGateway, tenants: dict[str, H.Tenant], order: list[str]) -> None:
    for name in order:
        t = tenants[name]
        before = len(t.srv.requests)  # type: ignore[union-attr]
        status, body = gw.chat(prefix(name), t.api_server_key, f"turn for {name}")
        assert status == 200, f"{name} turn via api_server failed: {status} {body!r}\n{gw.tail()}"
        answer = body["choices"][0]["message"]["content"]
        assert f"probe done for {name}" in answer, f"{name} was answered by someone else: {answer!r}"
        assert len(t.srv.requests) >= before + 2, f"{name}'s turn never reached its own provider"  # type: ignore[union-attr]


def _await_cron(tenants: dict[str, H.Tenant], at_least: int) -> None:
    # tool call + follow-up per fire; the ticker's first tick runs right after boot.
    H.poll(lambda: all(H.cron_requests(t) >= 2 * at_least for t in tenants.values()), 150,
           f"cron fire #{at_least} in every profile")


def test_multiplexed_gateway_never_crosses_tenants(fleet) -> None:
    root, tenants, gw = fleet
    home = root / "home"

    # Phase 1: boot with one due cron job per profile; interleave turns across all three.
    for t in tenants.values():
        H.seed_due_cron_job(t, home)
    gw.start()
    host_pid = gw.proc.pid  # type: ignore[union-attr]
    _turns(gw, tenants, ["alpha", "default", "beta", "alpha", "beta", "default"])
    _await_cron(tenants, 1)
    H.check_isolation(tenants, min_snapshots=3)

    # Phase 2: each profile's API-server key is honoured only on its own route.
    for owner in tenants.values():
        for route in tenants.values():
            if route is owner:
                continue
            seen = len(route.srv.requests)  # type: ignore[union-attr]
            status, _ = gw.chat(prefix(route.name), owner.api_server_key, "cross-tenant key")
            assert status in (401, 403), f"{owner.name}'s API key opened {route.name}'s route (HTTP {status})"
            assert len(route.srv.requests) == seen, f"{owner.name}'s key drove {route.name}'s provider"  # type: ignore[union-attr]

    # Phase 3: profile churn next to the live host never restarts or kills it.
    created = H.run_hermes(["profile", "create", "gamma"], home)
    assert created.returncode == 0, created.stderr[-2000:]
    attach = H.run_hermes(["-p", "gamma", "gateway", "run"], home, timeout=150)
    assert "[harness] killed" not in attach.stderr, "`-p gamma gateway run` started a second gateway"
    assert gw.proc.pid == host_pid and gw.proc.poll() is None, "profile create/attach replaced the host"  # type: ignore[union-attr]
    deleted = H.run_hermes(["profile", "delete", "gamma", "--yes"], home)
    assert deleted.returncode == 0, deleted.stderr[-2000:]
    assert gw.proc.pid == host_pid and gw.proc.poll() is None, f"profile delete killed the host\n{gw.tail()}"  # type: ignore[union-attr]
    _turns(gw, tenants, ["alpha", "beta"])
    H.check_isolation(tenants, min_snapshots=3)

    # Phase 4: SIGTERM restart; the new host re-serves all three and ticks each profile's cron again.
    gw.stop()
    assert not H.pid_alive(host_pid), "old gateway still alive after SIGTERM"
    for t in tenants.values():
        H.seed_due_cron_job(t, home)
    gw.start()
    _turns(gw, tenants, ["beta", "default", "alpha"])
    _await_cron(tenants, 2)
    H.check_isolation(tenants, min_snapshots=5)

    gw.stop()
    leaks = H.text_leaks("gateway output", gw.tail(10**9), tenants)
    assert not leaks, "\n".join(leaks)
