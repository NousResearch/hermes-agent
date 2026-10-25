"""An unmapped gateway's restart debt is retired only by ITS runtime (#132338 review C).

The pause records an unmapped gateway (no profile PID file, e.g. a Scheduled Task) by the argv it
replays. Two homes can run identical selectorless argv, their environment picking the home, so the
readiness predicate must also match the home the paused runtime ran on: otherwise home B's gateway,
already back after a partial recovery, retires home A's debt while A stays down.

Real processes (argv shaped like ``hermes gateway run``, each under its own ``HERMES_HOME``) and the
real predicate; the fleet discovery that hands it candidate PIDs is the only thing bypassed. The
replay restores that home too: the respawn's ``HERMES_HOME`` is the paused runtime's, not the updater's.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from gateway.status import get_process_start_time
from hermes_cli.update_cmd_windows import _unmapped_ready_filter


@pytest.fixture
def spawn(tmp_path):
    entry = tmp_path / "bin" / "hermes"  # a ``hermes`` entry token: the cmdline reads as a gateway
    entry.parent.mkdir()
    entry.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    argv = [sys.executable, str(entry), "gateway", "run"]
    procs: list[subprocess.Popen] = []

    def _spawn(home) -> subprocess.Popen:
        proc = subprocess.Popen(argv, env={**os.environ, "HERMES_HOME": str(home)})
        procs.append(proc)  # Popen returns after exec: the gateway argv and environment are in place
        return proc

    yield _spawn, argv
    for proc in procs:
        proc.kill()
        proc.wait(timeout=10)


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike  # stub children that sleep; the fixture reaps them
def test_an_equal_argv_gateway_of_another_home_never_retires_the_unmapped_debt(spawn):
    spawn, argv = spawn
    root = os.environ["HERMES_HOME"]  # the sandboxed home this update owns, with two profiles
    home_a, home_b = (os.path.join(root, "profiles", name) for name in ("a", "b"))
    os.makedirs(home_a)
    os.makedirs(home_b)
    old = spawn(home_a)
    entry = {"pid": old.pid, "argv": argv, "home": home_a, "ct": get_process_start_time(old.pid)}
    old.kill()
    old.wait(timeout=10)
    time.sleep(0.05)  # a later clock tick: the replacement is born after the process it replaces

    sibling = spawn(home_b)
    assert _unmapped_ready_filter(entry, set())([sibling.pid]) == [], \
        "home B's gateway with A's argv retired A's restart debt"
    replacement = spawn(home_a)
    assert _unmapped_ready_filter(entry, set())([sibling.pid, replacement.pid]) == [replacement.pid]


@pytest.mark.platforms("windows")  # the spec overlays the environment only on Windows
def test_an_unmapped_replay_runs_on_the_home_its_runtime_ran_on(tmp_path):
    from hermes_cli.gateway_windows import windowless_gateway_restart_spec

    paused_home = tmp_path / "profiles" / "a"
    paused_home.mkdir(parents=True)
    _argv, _cwd, env = windowless_gateway_restart_spec(
        [sys.executable, "-m", "hermes_cli.main", "gateway", "run"], home=str(paused_home))
    assert env["HERMES_HOME"] == str(paused_home.resolve()), "the replay ran on the updater's home"
