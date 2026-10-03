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
* ``desktop_handoff``: the Desktop hand-off script's whole tree killed while its
  ``hermes update`` child runs;
* ``orphaned_update``: ONLY the hand-off script killed (its ``hermes update`` child keeps
  running, the shape of a closed progress window or an ended PowerShell). The update must
  finish, ``.hermes-update-in-progress`` must read LIVE for as long as it runs (contract
  C1: the owner or its line-4 delegate is alive) and be gone once it exits.

Kill points are observed states, never timings: a git child of the update in the
process tree, HEAD read straight from the ref files, the hand-off's ``hermes update``
child plus its claimed marker. Each waits with a bounded timeout and an update that
exits before its kill point is a harness verdict, never a pass.

The crash-cell matrix (cell -> file -> fixing lane) is in
website/docs/developer-guide/source-update-completion.md.
"""

from __future__ import annotations

import subprocess
import time

import psutil
import pytest

from tests.e2e.core._pending_fixes import known_failure

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


def _head_ref(machine) -> str:
    """HEAD read straight from ``.git`` (no subprocess), so a kill point polls it cheaply."""
    git_dir = machine.install_dir / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8-sig").strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:].strip()
        loose = git_dir / ref
        if loose.is_file():
            return loose.read_text(encoding="utf-8-sig").strip()
        for line in (git_dir / "packed-refs").read_text(encoding="utf-8-sig").splitlines():
            sha, _, name = line.partition(" ")
            if name.strip() == ref:
                return sha
    except OSError:  # mid-write by the update: the next poll reads it
        pass
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
        time.sleep(0.05)
    taskkill_tree(proc.pid)
    raise AssertionError(fail_with(machine, f"{label}: kill point not reached within {UPDATE_TIMEOUT:.0f}s"))


def _crash(machine, srv, label: str, start, point) -> dict:
    """Publish a new commit, start the update, kill it at ``point``, then the next
    launch and the follow-up update. Returns everything the cell asserts on."""
    # Cells share one machine. A git lock an earlier cell's kill left behind is that
    # cell's verdict, not this one's: remove it the way the refused update tells the
    # user to ("remove the file manually to continue"), and say so in the evidence.
    leftover = machine.install_dir / ".git" / "index.lock"
    if leftover.is_file():
        leftover.unlink()
        machine.timings.append((f"(harness removed a prior cell's {leftover.name} before {label})", 0.0))
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
    """The checkout's HEAD is the target: git is done, the rest of the update is not.

    Read from the ref files, not ``git rev-parse``: the window between HEAD moving and
    the update exiting is ~20 s on the runner (dependency sync, launcher refresh, skills,
    completion stamp; run 37147409475), and a file read cannot stall the way a spawned
    git can."""
    return "checkout at target" if _head_ref(machine) == target else None


# -- marker (contract C1, line format) -----------------------------------------------

CT_TOLERANCE = 1.0  # seconds; writers round ct to 3 decimals


def _parse_marker(text: str) -> dict:
    """``<pid>\\n<started_at>\\n[ct:<ct>\\n][delegate:<pid> ct:<ct>\\n]`` (positional)."""
    lines = text.lstrip("\ufeff").splitlines()

    def num(index: int, cast):
        try:
            return cast(lines[index].strip())
        except (IndexError, ValueError):
            return None

    def ct(field: str):
        field = field.strip()
        try:
            return float(field[3:]) if field.startswith("ct:") else None
        except ValueError:
            return None

    delegate = delegate_ct = None
    if len(lines) > 3 and lines[3].startswith("delegate:"):
        head, _, tail = lines[3][len("delegate:"):].partition(" ")
        delegate, delegate_ct = (int(head) if head.strip().isdigit() else None), ct(tail)
    return {"pid": num(0, int), "started_at": num(1, float),
            "ct": ct(lines[2]) if len(lines) > 2 else None,
            "delegate": delegate, "delegate_ct": delegate_ct}


def _identity_live(pid: int | None, ct: float | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        proc = psutil.Process(pid)
        if not proc.is_running():
            return False
        return ct is None or abs(proc.create_time() - ct) <= CT_TOLERANCE
    except psutil.Error:
        return False


def _marker_live(text: str) -> str | None:
    """Who keeps the marker LIVE (``"owner <pid>"`` / ``"delegate <pid>"``), or None."""
    m = _parse_marker(text)
    if _identity_live(m["pid"], m["ct"]):
        return f"owner {m['pid']}"
    if _identity_live(m["delegate"], m["delegate_ct"]):
        return f"delegate {m['delegate']}"
    return None


def _read_marker(machine) -> str | None:
    try:
        return (machine.hermes_home / MARKER).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    except OSError as exc:  # mid-replace by a writer
        return f"<unreadable: {exc}>"


def _marker_text(machine) -> str:
    text = _read_marker(machine)
    if text is None:
        return "<absent>"
    return f"{text!r} parsed={_parse_marker(text)} live={_marker_live(text)}"


# -- orphaned update: only the hand-off script dies --------------------------------


def _direct_update_child(proc) -> psutil.Process | None:
    try:
        children = psutil.Process(proc.pid).children()
    except psutil.Error:
        return None
    for child in children:
        try:
            argv = [a.lower() for a in child.cmdline()]
        except psutil.Error:
            continue
        if "update" in argv and "--yes" in argv:
            return child
    return None


def _orphan(machine, srv, label: str) -> dict:
    """Start the hand-off script, kill ONLY its powershell once its ``hermes update``
    child runs under the claimed marker, and watch that orphaned update to its end."""
    pre = _head(machine)
    target = machine.mint(pre, label)
    machine.publish(target)
    with machine.gateway_phase():
        proc = _handoff(machine, f"{label}-script")()
        deadline = time.monotonic() + UPDATE_TIMEOUT
        child = None
        while time.monotonic() < deadline:
            child = _direct_update_child(proc)
            if child is not None and _read_marker(machine) is not None:
                break
            child = None
            if proc.poll() is not None:
                raise AssertionError(fail_with(
                    machine, f"{label}: the hand-off exited rc={proc.returncode} before its hermes update "
                             f"child ran (transcript {proc.transcript.name})"))
            time.sleep(0.05)
        if child is None:
            taskkill_tree(proc.pid)
            raise AssertionError(fail_with(machine, f"{label}: no hermes update child within {UPDATE_TIMEOUT:.0f}s"))
        marker_at_kill = _read_marker(machine)
        # No /T: the script alone dies; its update child (in the script's job, which has
        # no KILL_ON_JOB_CLOSE) keeps running.
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/F"], capture_output=True, timeout=60)
        proc.wait(timeout=60)
        killed_at = time.monotonic()
        dead_while_running = None
        holders: set[str] = set()
        rc = None
        while time.monotonic() < killed_at + UPDATE_TIMEOUT:
            try:
                rc = child.wait(timeout=0.25)
                break
            except psutil.TimeoutExpired:
                pass
            except psutil.NoSuchProcess:
                break
            text = _read_marker(machine)
            if text is not None and text.startswith("<unreadable"):
                continue  # a writer is replacing it this instant; the next sample reads it
            who = _marker_live(text) if text is not None else None
            if not who:  # confirm on a second read: never call a mid-swap sample DEAD
                time.sleep(0.1)
                text = _read_marker(machine)
                if text is None:
                    who = None
                else:
                    who = "?" if text.startswith("<unreadable") else _marker_live(text)
            if who and who != "?":
                holders.add(who.split()[0])
            elif not who and dead_while_running is None and child.is_running():
                dead_while_running = (round(time.monotonic() - killed_at, 1),
                                      "<absent>" if text is None else repr(text))
        orphan_finished = not child.is_running()
        if not orphan_finished:
            machine.kill_owned()
        after_orphan = _head(machine)
        marker_after_orphan = _read_marker(machine)
        marker_after_orphan_text = _marker_text(machine)
        turn = one_shot_turn(machine, srv, f"{label}-next-launch")
        follow_up = machine.hermes("update", "--yes", label=f"{label}-follow-up-update", timeout=UPDATE_TIMEOUT)
    return {"pre": pre, "target": target, "seen": f"update child {child.pid}",
            "marker_at_kill": marker_at_kill, "orphan_finished": orphan_finished, "orphan_rc": rc,
            "dead_while_running": dead_while_running, "holders": sorted(holders),
            "after_orphan": after_orphan, "marker_after_orphan": marker_after_orphan,
            "marker_after_orphan_text": marker_after_orphan_text,
            "turn": turn, "after_launch": _head(machine), "follow_up": follow_up,
            "final": _head(machine), "marker_final": (machine.hermes_home / MARKER).is_file()}


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
                j.step("orphaned_update", lambda: _orphan(machine, srv, "orphan"))
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


# Red on main (wine2e run 37139409703): the killed git leaves .git/index.lock, which
# hermes_cli/gitlock.py only sweeps once it is 10 minutes old, and the launch-time
# interrupted-pull repair (hermes_cli/_early_recovery.py) dies with WinError 2 on a
# machine whose only Git is the installer's private copy — so the next update refuses.
# Fixed by #132361 (this branch is stacked on it).
def test_update_killed_mid_git_leaves_a_runnable_install(journey: Journey) -> None:
    _assert_recovered(journey, "mid_git")


def test_update_killed_after_the_tree_moved_leaves_a_runnable_install(journey: Journey) -> None:
    _assert_recovered(journey, "tree_moved")


def test_desktop_handoff_killed_mid_run_leaves_a_runnable_install(journey: Journey) -> None:
    _assert_recovered(journey, "desktop_handoff")


# Main's hand-off claims the marker with the script's pid and its update child runs
# under that claim without naming itself, so the marker reads DEAD the moment the
# script dies while the update still runs (a second update is admitted), and nothing
# removes it afterwards. The line-4 delegate (#132354 script side, #132365 Python side)
# keeps it LIVE. Merge-order safe: XFAILs only on exactly this gap.
ORPHAN_MARKER_GAP = (r"orphaned_update: \.hermes-update-in-progress (read DEAD|survived)",
                     "upd-txn: the line-4 delegate lands in #132354 + #132365")


def test_desktop_handoff_script_killed_alone_keeps_the_marker_live_until_its_update_ends(
        journey: Journey) -> None:
    m, r = journey.machine, journey["orphaned_update"]
    assert r["orphan_finished"], fail_with(
        m, f"orphaned_update: the orphaned hermes update was still running {UPDATE_TIMEOUT:.0f}s after "
           f"the script died")
    with known_failure(*ORPHAN_MARKER_GAP):
        assert r["dead_while_running"] is None, fail_with(
            m, f"orphaned_update: {MARKER} read DEAD {r['dead_while_running'][0]}s after the script died "
               f"while its hermes update still ran: {r['dead_while_running'][1]} "
               f"(at kill: {r['marker_at_kill']!r})")
    assert r["orphan_rc"] == 0 and r["after_orphan"] == r["target"], fail_with(
        m, f"orphaned_update: the hermes update orphaned by the dead script did not finish the update "
           f"(rc={r['orphan_rc']}, checkout {r['after_orphan']}, target {r['target']}; "
           f"marker holders seen: {r['holders']})")
    with known_failure(*ORPHAN_MARKER_GAP):
        assert r["marker_after_orphan"] is None, fail_with(
            m, f"orphaned_update: {MARKER} survived the orphaned update's exit: "
               f"{r['marker_after_orphan_text']}")
    turn = r["turn"]
    assert turn.ok, fail_with(m, "orphaned_update: the launch after the orphaned update ran no turn", turn.run)
    follow = r["follow_up"]
    assert follow.returncode == 0 and r["final"] == r["target"] and not r["marker_final"], fail_with(
        m, f"orphaned_update: the next update did not complete cleanly (rc={follow.returncode}, "
           f"checkout {r['final']}, marker left={r['marker_final']}): {failure_line(follow)}", follow)
