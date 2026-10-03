"""A killed or failed ``hermes update`` never strands — or wrongly starts — a paused gateway.

Failure class: Windows pause durability. ``hermes update`` stops every running gateway before it
touches the checkout. Cell (a): the updater is killed (``taskkill /F /T`` — console closed, Desktop
kill) right after the pause; the next plain ``hermes`` command must bring the gateway back.
Cell (c): a successful update brings the paused gateway back and it outlives the updater (the
record is discharged). Cell (b): the pull fails half-way (a held file makes git's fast-forward die
after writing part of the tree); the gateway must NOT be started onto that torn tree.
"""

from __future__ import annotations

import subprocess
import threading
import time

import psutil
import pytest

from tests.e2e.core.windows_update._machine import (
    REQUIRES_OPT_IN,
    fail_with,
    harness_git,
    new_machine,
)
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [pytest.mark.platforms("windows"), pytest.mark.integration,
              pytest.mark.live_system_guard_bypass, REQUIRES_OPT_IN]

_PAUSED = "Paused gateway profile"
_RECORD = ".hermes-update-paused-gateways.json"
_EARLY, _HELD = "AGENTS.md", "website/package.json"  # checkout order: the early file is written first


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).is_running() and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _running_gateway(machine, not_pid: int, timeout: float) -> dict | None:
    try:
        return machine.wait_gateway_running(not_pid=not_pid, timeout=timeout)
    except AssertionError:
        return None


def _update_killed_after_pause(machine) -> tuple[bool, str]:
    """Start ``hermes update`` and ``taskkill /F /T`` it the moment it reports the pause."""
    log = machine.logs / "update-killed.log"
    proc = subprocess.Popen([str(machine.hermes_exe), "update", "--yes"], cwd=machine.profile,
                            env=machine.env(), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    seen = threading.Event()
    lines: list[str] = []
    assert proc.stdout is not None

    def pump() -> None:
        with log.open("w", encoding="utf-8") as fh:
            for raw in iter(proc.stdout.readline, b""):
                text = raw.decode("utf-8", "replace").rstrip()
                lines.append(text)
                fh.write(text + "\n")
                fh.flush()
                if _PAUSED in text:
                    seen.set()

    threading.Thread(target=pump, daemon=True).start()
    deadline = time.monotonic() + 600
    while not seen.is_set() and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=60)
    proc.wait(timeout=60)
    return seen.is_set(), "\n".join(lines[-40:])


def _mint_two_file_change(machine) -> str:
    """A target commit on top of the installed HEAD that rewrites two existing tracked files."""
    base = machine.installed_head()
    index = machine.root / "torn.index"
    env = {"GIT_INDEX_FILE": str(index), "GIT_AUTHOR_NAME": "Hermes E2E", "GIT_AUTHOR_EMAIL": "e2e@hermes.invalid",
           "GIT_COMMITTER_NAME": "Hermes E2E", "GIT_COMMITTER_EMAIL": "e2e@hermes.invalid"}
    harness_git("-C", str(machine.serve), "read-tree", base, env=env)
    for path in (_EARLY, _HELD):
        src = machine.root / ("torn-" + path.replace("/", "_"))
        src.write_bytes((machine.install_dir / path).read_bytes() + b"\n")
        blob = harness_git("-C", str(machine.serve), "hash-object", "-w", "--no-filters", str(src))
        harness_git("-C", str(machine.serve), "update-index", "--cacheinfo", f"100644,{blob},{path}", env=env)
    tree = harness_git("-C", str(machine.serve), "write-tree", env=env)
    index.unlink(missing_ok=True)
    target = harness_git("-C", str(machine.serve), "commit-tree", tree, "-p", base, "-m", "e2e: torn target", env=env)
    harness_git("-C", str(machine.serve), "update-ref", "refs/heads/main", target)
    return base


@pytest.fixture(scope="module")
def journey(tmp_path_factory):
    out: dict = {}
    with FakeLLMServer() as srv:
        machine = new_machine(tmp_path_factory.mktemp("pd"), srv.base_url, label="pd", system_git=True)
        out["machine"] = machine
        try:
            install = machine.install()
            assert install.returncode == 0, fail_with(machine, f"install.ps1 exited {install.returncode}", install)
            with machine.gateway_phase():
                # (a) kill the updater right after it paused a real gateway
                machine.spawn_gateway()
                old = int(machine.wait_gateway_running().get("pid") or 0)
                machine.advance()
                out["a_paused"], out["a_tail"] = _update_killed_after_pause(machine)
                out["a_old_stopped"] = not _alive(old)
                out["a_record"] = (machine.hermes_home / _RECORD).is_file()
                out["a_launch"] = machine.hermes("gateway", "status", label="next-launch")
                out["a_after"] = _running_gateway(machine, old, 120)
                machine.kill_owned()

                # (c) a successful update: the gateway is back and still alive after the updater exits
                machine.spawn_gateway()
                old = int(machine.wait_gateway_running().get("pid") or 0)
                out["c_pre"] = machine.installed_head()
                out["c_update"] = machine.hermes("update", "--yes", label="update-ok", timeout=900)
                out["c_head"] = machine.installed_head()
                after = _running_gateway(machine, old, 120)
                time.sleep(30)
                out["c_after"] = after
                out["c_alive_30s"] = bool(after) and _alive(int(after.get("pid") or 0))
                out["c_record"] = (machine.hermes_home / _RECORD).is_file()
                machine.kill_owned()

                # (b) the pull fails with a torn tree
                machine.spawn_gateway()
                old = int(machine.wait_gateway_running().get("pid") or 0)
                out["b_pre"] = _mint_two_file_change(machine)
                with (machine.install_dir / _HELD).open("rb"):  # no FILE_SHARE_DELETE: git cannot replace it
                    out["b_update"] = machine.hermes("update", "--yes", label="update-torn", timeout=900)
                out["b_head"] = machine.installed_head()
                out["b_status"] = harness_git("-C", str(machine.install_dir), "status", "--porcelain",
                                              "--untracked-files=no")
                out["b_after"] = _running_gateway(machine, old, 60)
                out["b_record"] = (machine.hermes_home / _RECORD).is_file()
                machine.kill_owned()
            yield out
        finally:
            machine.teardown()


def test_killed_updater_gateway_resumed_by_next_launch(journey) -> None:
    m = journey["machine"]
    assert journey["a_paused"] and journey["a_old_stopped"], fail_with(
        m, f"premise: the update never paused the gateway before the kill\n{journey['a_tail']}")
    assert journey["a_after"] is not None, fail_with(
        m, "after `taskkill /F` of an updater that paused the gateway, the next `hermes` launch left it "
           f"stopped (pause record present after kill: {journey['a_record']})", journey["a_launch"])


def test_successful_update_gateway_outlives_the_updater(journey) -> None:
    m, run = journey["machine"], journey["c_update"]
    assert run.returncode == 0 and journey["c_head"] != journey["c_pre"], fail_with(
        m, f"premise: the update did not complete (rc={run.returncode}, HEAD {journey['c_pre']} -> {journey['c_head']})", run)
    assert journey["c_after"] is not None, fail_with(m, "the paused gateway was not restarted by a successful update", run)
    assert journey["c_alive_30s"], fail_with(
        m, f"the restarted gateway (pid {journey['c_after'].get('pid')}) died after `hermes update` exited", run)
    assert not journey["c_record"], fail_with(m, "a fully resumed pause left its record behind", run)


def test_failed_pull_does_not_start_gateway_on_torn_tree(journey) -> None:
    m, run = journey["machine"], journey["b_update"]
    torn = journey["b_head"] == journey["b_pre"] and _EARLY in journey["b_status"]
    assert run.returncode != 0 and torn, fail_with(
        m, f"premise: the held file did not tear the pull (rc={run.returncode}, HEAD={journey['b_head']}, "
           f"pre={journey['b_pre']}, status={journey['b_status']!r})", run)
    assert journey["b_after"] is None, fail_with(
        m, f"the gateway was restarted onto a torn checkout ({journey['b_status']!r}) after the failed update "
           f"(pid {journey['b_after'].get('pid') if journey['b_after'] else None})", run)
    assert journey["b_record"], fail_with(m, "the paused set was not kept for recovery after the torn pull", run)
