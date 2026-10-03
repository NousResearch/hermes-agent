"""``hermes update`` killed mid-flight on native Windows, then the user's next launch.

Failure class: an update that never finishes. A user closes the console, Task Manager
ends the tree, the machine loses power. On Windows the kill is ``taskkill /T /F``:
no signal handler, no ``finally``, no atexit — exactly what these cells do to the
real process tree that ``hermes.exe update`` (or the Desktop's hand-off script
``scripts/desktop-update/windows.ps1``) started.

Each cell kills at one point of the update and then asserts what the user is owed:

* the next ``hermes`` launch runs a turn (the install is runnable);
* the checkout is at the commit before the update or at its target — never a third,
  half-written state;
* nothing the dead update left blocks the next one: a plain ``hermes update`` then
  completes to the target and no ``.hermes-update-in-progress`` marker survives it
  (a marker naming a dead process that refuses every later update is the permanent
  marker the hand-off contract forbids).

Cells (one machine, in order; each publishes a fresh commit to update to):

* ``mid_git``: killed while the update's git child (fetch / merge / reset) is alive;
* ``tree_moved``: killed right after the checkout moved to the target, before the
  update finished (dependency sync, launcher refresh, completion stamp);
* ``desktop_handoff``: the Desktop hand-off script killed while its ``hermes update``
  child runs.

The crash-cell matrix (cell -> file -> fixing lane) is in
website/docs/developer-guide/source-update-completion.md.
"""

from __future__ import annotations

import json
import time

import pytest

from tests.e2e.core.windows_update._machine import (
    REQUIRES_OPT_IN,
    UPDATE_TIMEOUT,
    Journey,
    descendants,
    fail_with,
    failure_line,
    harness_git,
    new_machine,
    one_shot_turn,
    taskkill_tree,
)
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [pytest.mark.platforms("windows"), pytest.mark.integration,
              pytest.mark.live_system_guard_bypass, REQUIRES_OPT_IN]

MARKER = ".hermes-update-in-progress"
GIT_OPS = ("fetch", "merge", "reset", "checkout", "pull")


def _git_child(proc, machine, target) -> str | None:
    """The git operation the update tree is running right now, if any."""
    for child in descendants(proc):
        try:
            if "git" not in child.name().lower():
                continue
            argv = [a.lower() for a in child.cmdline()]
        except Exception:  # raced its exit
            continue
        op = next((a for a in argv[1:] if a in GIT_OPS), None)
        if op:
            return op
    return None


def _head(machine) -> str:
    try:
        return harness_git("-C", str(machine.install_dir), "rev-parse", "HEAD", timeout=30)
    except RuntimeError:
        return ""


def _kill_when(machine, proc, label: str, point, target: str) -> str:
    """Poll ``point(proc, machine, target)`` until it names the moment, then taskkill the whole tree.

    Returns what was observed. Raises when the process exits first: the cell never
    reached its kill point, which is a harness verdict, never a pass."""
    deadline = time.monotonic() + UPDATE_TIMEOUT
    while time.monotonic() < deadline:
        seen = point(proc, machine, target)
        if seen:
            taskkill_tree(proc.pid)
            proc.wait(timeout=60)
            machine.kill_owned()  # stragglers that left the tree (detached helpers)
            return seen
        if proc.poll() is not None:
            raise AssertionError(fail_with(
                machine, f"{label}: the update exited rc={proc.returncode} before the kill point "
                         f"(transcript {proc.transcript.name})"))
        time.sleep(0.02)
    taskkill_tree(proc.pid)
    raise AssertionError(fail_with(machine, f"{label}: kill point not reached within {UPDATE_TIMEOUT:.0f}s"))


def _crash(machine, srv, label: str, start, point) -> dict:
    """Publish a new commit, start the update, kill it at ``point``, then the next
    launch and the follow-up update. Returns everything the cell asserts on."""
    pre = _head(machine)
    target = machine.mint(pre, label)
    machine.publish(target)
    with machine.gateway_phase():
        proc = start()
        seen = _kill_when(machine, proc, label, point, target)
        after_kill = _head(machine)
        turn = one_shot_turn(machine, srv, f"{label}-next-launch")
        after_launch = _head(machine)
        marker_after_launch = (machine.hermes_home / MARKER).is_file()
        follow_up = machine.hermes("update", "--yes", label=f"{label}-follow-up-update", timeout=UPDATE_TIMEOUT)
    return {"pre": pre, "target": target, "seen": seen, "after_kill": after_kill,
            "turn": turn, "after_launch": after_launch, "marker_after_launch": marker_after_launch,
            "follow_up": follow_up, "final": _head(machine),
            "marker_final": (machine.hermes_home / MARKER).is_file()}


def _cli_update(machine, label: str):
    return lambda: machine.spawn_logged([str(machine.hermes_exe), "update", "--yes"], label)


def _handoff(machine, label: str):
    script = machine.install_dir / "scripts" / "desktop-update" / "windows.ps1"
    return lambda: machine.spawn_logged(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
         "-InstallRoot", str(machine.install_dir), "-DesktopPid", "0", "-NoUi", "-NoGateway"], label)


def _update_child(proc, machine, target) -> str | None:
    """The hand-off's ``hermes update`` child is running (marker claimed, tree not yet done)."""
    for child in descendants(proc):
        try:
            argv = [a.lower() for a in child.cmdline()]
        except Exception:
            continue
        if "update" in argv and "--yes" in argv and "--help" not in argv:
            return "update child " + str(child.pid)
    return None


def _tree_moved(proc, machine, target) -> str | None:
    """The checkout's HEAD is the target: git is done, the rest of the update is not."""
    return "checkout at target" if _head(machine) == target else None


@pytest.fixture(scope="module")
def journey(tmp_path_factory):
    with FakeLLMServer() as srv:
        machine = new_machine(tmp_path_factory.mktemp("crash"), srv.base_url, label="crash")
        j = Journey(machine)
        try:
            j.step("install", machine.install)
            j.step("installed", lambda: j.require(
                "install", j["install"].returncode == 0, "install.ps1 failed", j["install"]))
            if j.ok("installed"):
                j.step("mid_git", lambda: _crash(machine, srv, "mid-git",
                                                 _cli_update(machine, "mid-git-update"), _git_child))
                j.step("tree_moved", lambda: _crash(machine, srv, "tree-moved",
                                                    _cli_update(machine, "tree-moved-update"), _tree_moved))
                j.step("desktop_handoff", lambda: _crash(machine, srv, "handoff",
                                                         _handoff(machine, "handoff-script"), _update_child))
            yield j
        finally:
            machine.teardown()


def _assert_recovered(journey: Journey, cell: str) -> None:
    m, r = journey.machine, journey[cell]
    turn = r["turn"]
    assert turn.ok, fail_with(
        m, f"{cell}: the first launch after the killed update ran no turn "
           f"(killed at {r['seen']}; reply printed={turn.reply_id in turn.run.stdout}, "
           f"prompt reached provider={turn.reached_wire})", turn.run)
    assert r["after_launch"] in (r["pre"], r["target"]), fail_with(
        m, f"{cell}: after the killed update and the next launch the checkout is at {r['after_launch']}, "
           f"neither the pre-update {r['pre']} nor the target {r['target']}", turn.run)
    follow = r["follow_up"]
    assert follow.returncode == 0 and r["final"] == r["target"], fail_with(
        m, f"{cell}: the update after the killed one did not complete (rc={follow.returncode}, "
           f"checkout {r['final']}, target {r['target']}): {failure_line(follow)}", follow)
    assert not r["marker_final"], fail_with(
        m, f"{cell}: {MARKER} survived a completed follow-up update: "
           f"{_marker_text(m)}", follow)


def _marker_text(machine) -> str:
    try:
        return json.dumps(json.loads((machine.hermes_home / MARKER).read_text(encoding="utf-8-sig")))
    except (OSError, ValueError) as exc:
        return f"<unreadable: {exc}>"


def test_update_killed_mid_git_leaves_a_runnable_install(journey: Journey) -> None:
    _assert_recovered(journey, "mid_git")


def test_update_killed_after_the_tree_moved_leaves_a_runnable_install(journey: Journey) -> None:
    _assert_recovered(journey, "tree_moved")


def test_desktop_handoff_killed_mid_run_leaves_a_runnable_install(journey: Journey) -> None:
    _assert_recovered(journey, "desktop_handoff")
