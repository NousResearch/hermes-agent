"""``hermes update`` killed mid-flight on native Windows, then the user's next launch.

Failure class: an update that never finishes. A user closes the console, Task Manager
ends the tree, the machine loses power. On Windows the kill is ``taskkill /T /F``:
no signal handler, no ``finally``, no atexit — exactly what these cells do to the
real process tree that ``hermes.exe update`` (or the Desktop's hand-off script
``scripts/desktop-update/windows.ps1``) started.

Each cell kills at one point of the update and then asserts what the user is owed:

* the next ``hermes`` launch runs a turn (the install is runnable);
* right after that launch the checkout IS the commit before the update or its target:
  HEAD is one of them and every tracked byte agrees with it (``git status`` clean,
  ``git diff --quiet HEAD``) — never a third, half-written tree. Each target edits
  modules every launch imports (``RUNTIME_FILES``), so a torn tree differs in code
  that runs, not only in a marker file;
* nothing the dead update left blocks the next one: a plain ``hermes update`` then
  completes to the target and no ``.hermes-update-in-progress`` marker survives it
  (a marker naming a dead process that refuses every later update is the permanent
  marker the hand-off contract forbids).

Cells (one machine; each publishes a fresh commit to update to; the ``mid_git`` cells run last):

* ``mid_fetch``: killed while the update's ``git fetch`` child runs (nothing local moved yet);
* ``mid_git``: killed while the update's local git write (the fast-forward ``merge``, or a
  ``reset`` / ``checkout``) is inside the checkout with ``.git/index.lock`` held. The kill is
  deterministic: a smudge filter the harness names in the install's own ``.git/config`` and
  ``.git/info/attributes`` (git's filter hook, nothing in the product) holds git while it
  writes ``hermes_constants.py``, the last changed path in index order. At that moment git
  has written the cell's new file and ``hermes_cli/main.py`` and unlinked
  ``hermes_constants.py``: the torn tree, stale ``index.lock`` and missing startup module
  a mid-merge kill leaves;
* ``mid_git_bootstrap``: the same hold on ``hermes_bootstrap.py``, the first changed path in
  index order and the module every launcher imports to reach the repair: the kill leaves it
  unlinked, so the next launch must repair from the recovery code the update published outside
  the tree before git wrote;
* ``tree_moved``: killed right after the checkout moved to the target, before the
  update finished (dependency sync, launcher refresh, completion stamp);
* ``desktop_handoff``: the Desktop hand-off script's whole tree killed while its
  ``hermes update`` child runs;
* ``orphaned_update``: ONLY the hand-off script killed, once its ``hermes update`` child
  holds its place under the script's claim (that child keeps running: the shape of an
  ended PowerShell). The update must
  finish, ``.hermes-update-in-progress`` must read LIVE for as long as it runs (contract
  C1: the owner or its line-4 delegate is alive) and be gone once it exits.

Kill points are observed states, never timings: a git child of the update in the
process tree (by its argv), the held filter plus ``index.lock``, HEAD read straight from
the ref files, the hand-off's ``hermes update``
child past its lock acquisition (its banner in update.log) plus the claimed marker. Each waits with a bounded timeout and an update that
exits before its kill point is a harness verdict, never a pass.

The crash-cell matrix (cell -> file -> fixing lane) is in
website/docs/developer-guide/source-update-completion.md.
"""

from __future__ import annotations

import contextlib
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from tests.e2e.core._pending_fixes import known_failure

from tests.e2e.core.windows._helpers import _decode
from tests.e2e.core.windows_update._machine import (
    REAL_GIT,
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
# Imported by every `hermes` launch: each crash target appends one statement to both.
RUNTIME_FILES = ("hermes_constants.py", "hermes_cli/main.py")


def _git_op(proc, ops: tuple[str, ...]) -> tuple[str, int] | None:
    """The first git child of the update tree whose argv names one of ``ops``: (op, pid)."""
    for child in descendants(proc):
        try:
            if "git" not in child.name().lower():
                continue
            argv = [a.lower() for a in child.cmdline()]
        except Exception:  # raced its exit
            continue
        op = next((a for a in argv[1:] if a in ops), None)
        if op:
            return op, child.pid
    return None


def _git_fetching(proc, machine, target) -> str | None:
    """The update's ``git fetch`` is running: the network half, before anything local moves."""
    seen = _git_op(proc, ("fetch",))
    return f"git fetch (pid {seen[1]})" if seen else None


# -- mid_git: the local write, held by git's own filter hook --------------------------------

# The update's git commands that write the checkout and its index.
LOCAL_GIT_OPS = ("merge", "reset", "checkout")
HOLD_FILTER = "hermes-e2e-hold"
# The last changed path in index order (".hermes-e2e-*" < "hermes_cli/main.py" <
# "hermes_constants.py"): when git reaches it, the other two are already written at the target.
HELD_PATH = "hermes_constants.py"
# The smudge filter. Git runs it while it checks out HELD_PATH, inside unpack_trees with
# .git/index.lock held and the old file already unlinked. One shot: the first run marks
# `held` and waits (the kill lands here); any later checkout passes the bytes through.
HOLD_SCRIPT = """\
import os, shutil, sys, time
from pathlib import Path

flags, budget = Path(sys.argv[1]), float(sys.argv[2])
try:
    fd = os.open(flags / "held", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
except FileExistsError:
    pass
else:
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline and not (flags / "release").exists():
        time.sleep(0.05)
shutil.copyfileobj(sys.stdin.buffer, sys.stdout.buffer)
"""


class _GitHold:
    """Arms the hold in the install's own git config (never the product), disarms after the kill.

    Git runs a filter command through the sh of the Git that runs it (PortableGit's, for
    the installer's Git); the command is the driver's own interpreter by absolute path.
    """

    def __init__(self, machine, path: str = HELD_PATH) -> None:
        self.machine = machine
        self.path = path
        self.flags = machine.root / "git-hold"
        self.attributes = machine.install_dir / ".git" / "info" / "attributes"
        self._saved_attributes: str | None = None
        self._held_since: float | None = None

    @property
    def held(self) -> bool:
        return (self.flags / "held").is_file()

    def arm(self) -> None:
        shutil.rmtree(self.flags, ignore_errors=True)
        self.flags.mkdir(parents=True)
        script = self.flags / "hold.py"
        script.write_text(HOLD_SCRIPT, encoding="utf-8")
        command = " ".join(shlex.quote(p) for p in (
            Path(sys.executable).as_posix(), script.as_posix(), self.flags.as_posix(), str(int(UPDATE_TIMEOUT))))
        harness_git("-C", str(self.machine.install_dir), "config", f"filter.{HOLD_FILTER}.smudge", command)
        if self.attributes.is_file():
            self._saved_attributes = self.attributes.read_text(encoding="utf-8-sig")
        self.attributes.parent.mkdir(parents=True, exist_ok=True)
        self.attributes.write_text((self._saved_attributes or "") + f"/{self.path} filter={HOLD_FILTER}\n",
                                   encoding="utf-8")

    def disarm(self) -> None:
        (self.flags / "release").touch()
        if self._saved_attributes is None:
            self.attributes.unlink(missing_ok=True)
        else:
            self.attributes.write_text(self._saved_attributes, encoding="utf-8")
        with contextlib.suppress(RuntimeError):  # never armed: nothing to remove
            harness_git("-C", str(self.machine.install_dir), "config", "--remove-section", f"filter.{HOLD_FILTER}")

    def point(self, proc, machine, target) -> str | None:
        """Git is inside the checkout: the filter holds it, a local write op is alive, index.lock is held."""
        if not self.held:
            return None
        seen = _git_op(proc, LOCAL_GIT_OPS)
        lock = machine.install_dir / ".git" / "index.lock"
        if seen is None or not lock.is_file():
            # Held by a git this cell does not model (or a lock-free write): name it, never
            # wait out the whole update budget on a kill point that cannot come.
            self._held_since = self._held_since or time.monotonic()
            if time.monotonic() - self._held_since > HOLD_SETTLE_SECONDS:
                gits = []
                for child in descendants(proc):
                    with contextlib.suppress(Exception):
                        if "git" in child.name().lower():
                            gits.append(" ".join(child.cmdline()[1:6]))
                raise AssertionError(fail_with(
                    machine, f"mid_git: the hold filter ran but no {'/'.join(LOCAL_GIT_OPS)} with "
                             f".git/index.lock held appeared in {HOLD_SETTLE_SECONDS:.0f}s "
                             f"(index.lock={lock.is_file()}, git children: {gits or 'none'})"))
            return None
        return f"git {seen[0]} (pid {seen[1]}) held writing {self.path}, .git/index.lock held"


HOLD_SETTLE_SECONDS = 30.0


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


def _tree(machine, label: str) -> dict:
    """The checkout against its own commit: HEAD, tracked files that differ from it, and
    whether the cell's target-only file is on disk. Read with the harness's git (never the
    product's) without taking the index lock."""
    def git(*args: str) -> tuple[int, str]:
        res = subprocess.run([REAL_GIT, "--no-optional-locks", "-c", "safe.directory=*",
                              "-C", str(machine.install_dir), *args],
                             env=machine.env(), capture_output=True, timeout=300)
        return res.returncode, (_decode(res.stdout) + _decode(res.stderr)).strip()

    rc_head, head = git("rev-parse", "HEAD")
    rc_status, status = git("status", "--porcelain=v1", "--untracked-files=no")
    rc_diff, diff = git("diff", "--quiet", "HEAD", "--")
    return {"head": head if rc_head == 0 else f"<rev-parse rc={rc_head}: {head[:200]}>",
            "dirty": status.splitlines()[:20] if rc_status == 0 else [f"<status rc={rc_status}: {status[:300]}>"],
            "diff_rc": rc_diff, "diff_err": diff[:300] if rc_diff not in (0, 1) else "",
            "target_file": (machine.install_dir / f".hermes-e2e-{label}").exists()}


def _tree_matches(tree: dict, commit: str, target: str) -> bool:
    """``tree`` is exactly ``commit``: HEAD, index and every tracked byte agree, and the
    target-only file exists iff ``commit`` is the target."""
    return (tree["head"] == commit and not tree["dirty"] and tree["diff_rc"] == 0
            and tree["target_file"] == (commit == target))


def _kill_when(machine, proc, label: str, point, target: str) -> str:
    """Poll ``point(proc, machine, target)`` until it names the moment, then taskkill the whole tree.

    Returns what was observed. Raises when the process exits first: the cell never
    reached its kill point, which is a harness verdict, never a pass."""
    deadline = time.monotonic() + UPDATE_TIMEOUT
    while time.monotonic() < deadline:
        try:
            seen = point(proc, machine, target)
        except BaseException:  # the point's own harness verdict: never leave the update running
            taskkill_tree(proc.pid)
            raise
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


def _crash(machine, srv, label: str, start, point, hold: _GitHold | None = None,
           runtime_files: tuple[str, ...] = RUNTIME_FILES) -> dict:
    """Publish a new commit, start the update, kill it at ``point``, then the next
    launch and the follow-up update. Returns everything the cell asserts on.

    ``hold`` is armed just before the update starts and disarmed right after the kill,
    before anything inspects the tree: the next launch and the follow-up run plain git."""
    # Cells share one machine. A git lock an earlier cell's kill left behind is that
    # cell's verdict, not this one's: remove it the way the refused update tells the
    # user to ("remove the file manually to continue"), and say so in the evidence.
    leftover = machine.install_dir / ".git" / "index.lock"
    if leftover.is_file():
        leftover.unlink()
        machine.timings.append((f"(harness removed a prior cell's {leftover.name} before {label})", 0.0))
    pre = _head(machine)
    target = machine.mint(pre, label, runtime_files)
    machine.publish(target)
    with machine.gateway_phase():
        if hold is not None:
            hold.arm()
        try:
            proc = start()
            seen = _kill_when(machine, proc, label, point, target)
        finally:
            if hold is not None:
                hold.disarm()
        index_lock_after_kill = leftover.is_file()
        tree_after_kill = _tree(machine, label)
        held_missing_after_kill = hold is not None and not (machine.install_dir / hold.path).exists()
        turn = one_shot_turn(machine, srv, f"{label}-next-launch")
        tree_after_launch = _tree(machine, label)
        marker_after_launch = (machine.hermes_home / MARKER).is_file()
        follow_up = machine.hermes("update", "--yes", label=f"{label}-follow-up-update", timeout=UPDATE_TIMEOUT)
    tree_final = _tree(machine, label)
    return {"label": label, "pre": pre, "target": target, "seen": seen, "tree_after_kill": tree_after_kill,
            "index_lock_after_kill": index_lock_after_kill, "held_missing_after_kill": held_missing_after_kill,
            "turn": turn, "tree_after_launch": tree_after_launch, "marker_after_launch": marker_after_launch,
            "follow_up": follow_up, "tree_final": tree_final,
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


# Printed by ``_cmd_update_impl``, which hermes_cli/main.py enters only after
# ``UpdateLock.acquire()`` returned: once update.log gains one, the update child has
# taken its place under the script's claim (and named itself in it, where it does).
UPDATE_BANNER = "Updating Hermes Agent..."


UPDATE_DONE = "Update complete!"
# CPython's exit status when the final flush of stdout fails at shutdown. The orphan's
# stdout is a pipe into the killed script, so its last flush has no reader; only that
# dead script ever waited on this exit code (run 37149266852: the update's own receipt
# says success, the checkout is at the target, the process exits 120).
PY_FINAL_FLUSH_FAILED = 120


def _update_banners(machine, banner: str = UPDATE_BANNER) -> int:
    try:
        return (machine.hermes_home / "logs" / "update.log").read_text(
            encoding="utf-8-sig", errors="replace").count(banner)
    except OSError:
        return 0


class _ExitCode:
    """An open handle on a process this test did not spawn, so its exit code stays readable
    after it exits. ``psutil.Process.wait`` returns None for a non-child that is already
    gone, which lost the orphaned update's rc whenever it exited between two samples."""

    def __init__(self, pid: int) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes, self._wintypes = ctypes, wintypes
        k = self._k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.OpenProcess.restype = wintypes.HANDLE
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.WaitForSingleObject.restype = wintypes.DWORD
        k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.GetExitCodeProcess.restype = wintypes.BOOL
        k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        # PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE
        self._h = k.OpenProcess(0x1000 | 0x00100000, False, pid)
        if not self._h:
            raise OSError(ctypes.get_last_error(), f"OpenProcess({pid}) failed")

    def wait(self, timeout: float) -> int | None:
        """The exit code once the process has exited, else None after ``timeout`` seconds."""
        if self._k.WaitForSingleObject(self._h, int(timeout * 1000)) != 0:  # WAIT_OBJECT_0
            return None
        code = self._wintypes.DWORD()
        if not self._k.GetExitCodeProcess(self._h, self._ctypes.byref(code)):
            raise OSError(self._ctypes.get_last_error(), "GetExitCodeProcess failed")
        return code.value

    def close(self) -> None:
        if self._h:
            self._k.CloseHandle(self._h)
            self._h = None


def _orphan(machine, srv, label: str) -> dict:
    """Start the hand-off script, kill ONLY its powershell once its ``hermes update``
    child runs under the claimed marker, and watch that orphaned update to its end."""
    pre = _head(machine)
    target = machine.mint(pre, label, RUNTIME_FILES)
    machine.publish(target)
    with machine.gateway_phase():
        # Both baselines BEFORE the script starts: a completion logged at any point after
        # this (even while taskkill / wait below run) is this run's, never the baseline's.
        banners = _update_banners(machine)
        done_before = _update_banners(machine, UPDATE_DONE)
        proc = _handoff(machine, f"{label}-script")()
        deadline = time.monotonic() + UPDATE_TIMEOUT
        child = None
        while time.monotonic() < deadline:
            # Kill point: the script's own `hermes update` child is past its lock
            # acquisition (a fresh banner in update.log) and the marker exists.
            child = _direct_update_child(proc)
            if child is not None and _read_marker(machine) is not None and _update_banners(machine) > banners:
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
        # Held from before the kill, while the update is minutes from done.
        exit_code = _ExitCode(child.pid)
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
            rc = exit_code.wait(0.25)
            if rc is not None:
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
            elif not who and dead_while_running is None and exit_code.wait(0) is None:
                dead_while_running = (round(time.monotonic() - killed_at, 1),
                                      "<absent>" if text is None else repr(text))
        if rc is None:
            rc = exit_code.wait(0)
        exit_code.close()
        orphan_finished = rc is not None
        if not orphan_finished:
            machine.kill_owned()
        tree_after_orphan = _tree(machine, label)
        orphan_reported_done = _update_banners(machine, UPDATE_DONE) > done_before
        marker_after_orphan = _read_marker(machine)
        marker_after_orphan_text = _marker_text(machine)
        turn = one_shot_turn(machine, srv, f"{label}-next-launch")
        follow_up = machine.hermes("update", "--yes", label=f"{label}-follow-up-update", timeout=UPDATE_TIMEOUT)
    return {"label": label, "pre": pre, "target": target, "seen": f"update child {child.pid}",
            "marker_at_kill": marker_at_kill, "orphan_finished": orphan_finished, "orphan_rc": rc,
            "orphan_reported_done": orphan_reported_done,
            "dead_while_running": dead_while_running, "holders": sorted(holders),
            "tree_after_orphan": tree_after_orphan, "marker_after_orphan": marker_after_orphan,
            "marker_after_orphan_text": marker_after_orphan_text,
            "turn": turn, "follow_up": follow_up,
            "tree_final": _tree(machine, label), "marker_final": (machine.hermes_home / MARKER).is_file()}


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
                j.step("mid_fetch", lambda: _crash(machine, srv, "mid-fetch",
                                                   _cli_update(machine, "mid-fetch-update"), _git_fetching))
                j.step("tree_moved", lambda: _crash(machine, srv, "tree-moved",
                                                    _cli_update(machine, "tree-moved-update"), _tree_moved))
                j.step("desktop_handoff", lambda: _crash(machine, srv, "handoff",
                                                         _handoff(machine, "handoff-script"), _update_child))
                j.step("orphaned_update", lambda: _orphan(machine, srv, "orphan"))
                # Last: a launch that cannot repair this tree leaves every later cell an
                # unrunnable install, which is this cell's verdict, not theirs.
                hold = _GitHold(machine)
                j.step("mid_git", lambda: _crash(machine, srv, "mid-git",
                                                 _cli_update(machine, "mid-git-update"), hold.point, hold))
                boot_hold = _GitHold(machine, "hermes_bootstrap.py")
                j.step("mid_git_bootstrap", lambda: _crash(
                    machine, srv, "mid-git-bootstrap", _cli_update(machine, "mid-git-bootstrap-update"),
                    boot_hold.point, boot_hold, runtime_files=("hermes_bootstrap.py", *RUNTIME_FILES)))
            yield j
        finally:
            machine.teardown()


def _tree_text(tree: dict) -> str:
    return (f"HEAD {tree['head']}, dirty tracked {tree['dirty'] or 'none'}, diff HEAD rc={tree['diff_rc']}"
            f"{' ' + tree['diff_err'] if tree['diff_err'] else ''}, target file present={tree['target_file']}")


def _assert_tree_is(m, cell: str, when: str, tree: dict, allowed: dict[str, str], target: str, run=None) -> None:
    """``tree`` is exactly one of ``allowed`` (name -> commit), byte for byte."""
    assert any(_tree_matches(tree, sha, target) for sha in allowed.values()), fail_with(
        m, f"{cell}: {when} the checkout is not the tree of "
           f"{' or '.join(f'the {name} {sha}' for name, sha in allowed.items())}: {_tree_text(tree)}", run)


def _assert_recovered(journey: Journey, cell: str) -> None:
    m, r = journey.machine, journey[cell]
    turn = r["turn"]
    assert turn.ok, fail_with(
        m, f"{cell}: the first launch after the killed update ran no turn "
           f"(killed at {r['seen']}; reply printed={turn.reply_id in turn.run.stdout}, "
           f"prompt reached provider={turn.reached_wire}; tree after the kill: {_tree_text(r['tree_after_kill'])}, "
           f".git/index.lock after the kill={r['index_lock_after_kill']})",
        turn.run)
    _assert_tree_is(m, cell, "after the killed update and the next launch",
                    r["tree_after_launch"], {"pre-update commit": r["pre"], "target": r["target"]},
                    r["target"], turn.run)
    follow = r["follow_up"]
    assert follow.returncode == 0, fail_with(
        m, f"{cell}: the update after the killed one did not complete (rc={follow.returncode}): "
           f"{failure_line(follow)}", follow)
    _assert_tree_is(m, cell, "after the follow-up update", r["tree_final"], {"target": r["target"]},
                    r["target"], follow)
    assert not r["marker_final"], fail_with(
        m, f"{cell}: {MARKER} survived a completed follow-up update: "
           f"{_marker_text(m)}", follow)


def test_update_killed_mid_fetch_leaves_a_runnable_install(journey: Journey) -> None:
    _assert_recovered(journey, "mid_fetch")


# Red on main (wine2e run 37139409703): the killed git leaves .git/index.lock, which
# hermes_cli/gitlock.py only sweeps once it is 10 minutes old, and the launch-time
# interrupted-pull repair (hermes_cli/_early_recovery.py) dies with WinError 2 on a
# machine whose only Git is the installer's private copy — so the next update refuses.
# Fixed by #132361. Until round 6 every green run killed at fetch (index.lock never
# existed); the hold now lands the kill inside the merge's checkout, lock held.
#
# What that exposed (native run 37190464303): the launcher imported hermes_constants from the
# checkout before `import hermes_bootstrap`, whose restore_interrupted_pull is the repair, so a
# merge killed after git unlinked hermes_constants.py bricked every launch. Fixed in #132361
# (launchers reach the repair first); this cell asserts recovery outright.


def test_update_killed_mid_git_leaves_a_runnable_install(journey: Journey) -> None:
    r = journey["mid_git"]
    assert r["index_lock_after_kill"], fail_with(
        journey.machine, f"mid_git: the kill did not land inside git's checkout (killed at {r['seen']}, "
                         f"no .git/index.lock after it): this cell proves nothing about a merge-time kill")
    _assert_recovered(journey, "mid_git")


# The repair's own entry: hermes_bootstrap.py unlinked by the killed merge. Red before #132361
# round 8 (D1): every launch died importing it, the marker kept, the tree never restored.
def test_update_killed_writing_the_launch_repairs_own_code_leaves_a_runnable_install(journey: Journey) -> None:
    r = journey["mid_git_bootstrap"]
    assert r["index_lock_after_kill"] and r["held_missing_after_kill"], fail_with(
        journey.machine, f"mid_git_bootstrap: the kill did not leave git's checkout holding with "
                         f"hermes_bootstrap.py unlinked (killed at {r['seen']}, index.lock="
                         f"{r['index_lock_after_kill']}, unlinked={r['held_missing_after_kill']})")
    _assert_recovered(journey, "mid_git_bootstrap")


def test_update_killed_after_the_tree_moved_leaves_a_runnable_install(journey: Journey) -> None:
    _assert_recovered(journey, "tree_moved")


def test_desktop_handoff_killed_mid_run_leaves_a_runnable_install(journey: Journey) -> None:
    _assert_recovered(journey, "desktop_handoff")


# Main's hand-off claims the marker with the script's pid and its update child runs
# under that claim without naming itself, so the marker reads DEAD the moment the
# script dies while the update still runs (a second update is admitted), and nothing
# removes it afterwards. The line-4 delegate (#132354 script side, #132365 Python side)
# keeps it LIVE. Merge-order safe: XFAILs only on exactly this gap, and only after every
# other assertion of the cell passed; an acceptance run of the integrated batch
# (HERMES_E2E_STRICT_ACCEPTANCE=upd-txn) fails on it instead.
ORPHAN_MARKER_GAP = (r"orphaned_update: \.hermes-update-in-progress (read DEAD|survived)",
                     "upd-txn: the line-4 delegate lands in #132354 + #132365")


def test_desktop_handoff_script_killed_alone_keeps_the_marker_live_until_its_update_ends(
        journey: Journey) -> None:
    m, r = journey.machine, journey["orphaned_update"]
    assert r["orphan_finished"], fail_with(
        m, f"orphaned_update: the orphaned hermes update was still running {UPDATE_TIMEOUT:.0f}s after "
           f"the script died")
    finished = r["orphan_reported_done"] and r["orphan_rc"] in (0, PY_FINAL_FLUSH_FAILED)
    assert finished, fail_with(
        m, f"orphaned_update: the hermes update orphaned by the dead script did not finish the update "
           f"(rc={r['orphan_rc']}, '{UPDATE_DONE}' logged={r['orphan_reported_done']}, "
           f"tree {_tree_text(r['tree_after_orphan'])}, target {r['target']}; marker holders seen: {r['holders']})")
    _assert_tree_is(m, "orphaned_update", "after the orphaned update finished", r["tree_after_orphan"],
                    {"target": r["target"]}, r["target"])
    turn = r["turn"]
    assert turn.ok, fail_with(m, "orphaned_update: the launch after the orphaned update ran no turn", turn.run)
    follow = r["follow_up"]
    assert follow.returncode == 0 and not r["marker_final"], fail_with(
        m, f"orphaned_update: the next update did not complete cleanly (rc={follow.returncode}, "
           f"marker left={r['marker_final']}): {failure_line(follow)}", follow)
    _assert_tree_is(m, "orphaned_update", "after the next update", r["tree_final"],
                    {"target": r["target"]}, r["target"], follow)
    # The marker contract last: the batch-owned gap, so its xfail can hide nothing above.
    gaps = []
    if r["dead_while_running"] is not None:
        gaps.append(f"{MARKER} read DEAD {r['dead_while_running'][0]}s after the script died while its "
                    f"hermes update still ran: {r['dead_while_running'][1]} (at kill: {r['marker_at_kill']!r})")
    if r["marker_after_orphan"] is not None:
        gaps.append(f"{MARKER} survived the orphaned update's exit: {r['marker_after_orphan_text']}")
    with known_failure(*ORPHAN_MARKER_GAP):
        assert not gaps, fail_with(m, "orphaned_update: " + "; ".join(gaps))
