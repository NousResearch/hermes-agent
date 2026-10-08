"""Cross-process update mutual exclusion (``hermes_cli.update_lock``).

Three surfaces can start an update of one install tree: a terminal ``hermes
update``, the dashboard's Update button (which spawns that same command
detached), and the desktop's Update button (Tauri updater → install-mode
bootstrap on its failure screen). Before the shared lock, two of them could run
concurrently and rewrite source under a live interpreter — observed in the wild
as an installer ``git checkout`` rewinding the checkout ~9k commits while a
dashboard-spawned ``hermes update`` was mid-``npm install``, which then failed
against the rewound tree's manifests.

These exercise the real marker file against a temp home — no mocks — because
the contract that matters is what the Rust updater and the Electron gate see on
disk.
"""

from __future__ import annotations

import errno
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli.update_lock import (
    HANDOFF_PID_ENV,
    UPDATE_MARKER_MAX_AGE_SECONDS,
    UpdateLock,
    checkout_lock_fds,
    checkout_lock_path,
    describe_holder,
    process_create_time,
    read_live_update,
    update_in_progress,
    update_marker_path,
)

# Repo root: the -I -S -B subprocesses insert it on sys.path to import the
# real hermes_cli without site-packages.
REPO_ROOT = Path(__file__).resolve().parents[2]

# A pid no live process owns. os.kill(pid, 0) must report it dead so a crashed
# updater can never wedge every future update. Deliberately larger than any
# platform's pid_t so it also covers the corrupt-marker path (OverflowError).
DEAD_PID = 4294967294


@pytest.fixture
def marker(tmp_path):
    return tmp_path / ".hermes-update-in-progress"


@pytest.fixture
def other_pid():
    """A live process that is not us: the stand-in for another updater (our own pid is ours)."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL)
    yield proc.pid
    proc.kill()
    proc.wait()


def _claim(marker, pid, started_at=None):
    """A v1 (legacy, pid-only) claim."""
    marker.write_text(f"{pid}\n{int(time.time() if started_at is None else started_at)}\n", encoding="utf-8")


def _claim_v2(marker, pid, started_at=None, ct_offset=0.0):
    started = int(time.time() if started_at is None else started_at)
    marker.write_text(f"{pid}\n{started}\nct:{process_create_time(pid) + ct_offset:.3f}\n", encoding="utf-8", newline="")


def test_marker_path_is_the_profile_tree_root(tmp_path, monkeypatch):
    """The lock must land where the Rust updater, Electron gate and hand-off scripts look.

    They all read the ROOT home; a sticky/`-p` profile re-homes HERMES_HOME to
    ``<root>/profiles/<p>``, and a marker there is invisible to every other owner (V9c).
    """
    root = tmp_path / "root"
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "work"))
    assert update_marker_path() == root / ".hermes-update-in-progress"
    monkeypatch.setenv("HERMES_HOME", str(root))
    assert update_marker_path() == root / ".hermes-update-in-progress"


def test_acquire_writes_pid_and_start_time(marker):
    lock = UpdateLock(path=marker)

    assert lock.acquire() is True
    assert lock.acquired is True

    lines = marker.read_text(encoding="utf-8-sig").splitlines()
    assert int(lines[0]) == os.getpid(), (
        "the Electron gate probes this pid for liveness"
    )
    assert int(lines[1]) == pytest.approx(time.time(), abs=5)
    assert float(lines[2].removeprefix("ct:")) == pytest.approx(process_create_time(), abs=0.01), (
        "line 3 is our creation time: readers tell a reused pid from us by it"
    )
    assert len(lines) == 3


def test_second_acquire_is_refused_while_the_first_is_live(marker, other_pid):
    """The bug: two updaters mutating one checkout at the same time."""
    _claim(marker, other_pid)

    second = UpdateLock(path=marker)
    assert second.acquire() is False
    assert second.holder is not None
    assert second.holder.pid == other_pid
    assert second.acquired is False


def test_refused_lock_does_not_delete_the_live_owners_marker(marker, other_pid):
    _claim(marker, other_pid)

    second = UpdateLock(path=marker)
    second.acquire()
    second.release()

    assert marker.exists(), "a refused claimant must never clear the live owner's lock"


@pytest.mark.platforms("posix")
def test_marker_owned_by_a_zombie_self_heals(marker):
    """A crashed updater lingering unreaped must not pin the lock for 20 minutes.

    ``os.kill(pid, 0)`` still succeeds for a zombie, so a marker owned by a
    dead-but-unreaped stage used to read as a live update until the age ceiling
    expired (#77259, #120635, #125932).
    """
    pid = os.fork()
    assert pid >= 0
    if pid == 0:
        os._exit(0)  # noqa: P111 — child exits without running pytest teardown

    # Keep the child unreaped (a zombie) and wait until the state probe
    # actually reports 'Z' so the assertion can't race the exit.
    from hermes_cli._early_recovery import _process_state

    became_zombie = False
    for _ in range(40):
        state = _process_state(pid)
        if state is not None and state.upper().startswith("Z"):
            became_zombie = True
            break
        time.sleep(0.05)
    assert became_zombie, "child never reached zombie state on this platform"

    try:
        _claim(marker, pid)
        assert read_live_update(path=marker) is None, "a zombie is not a live update"
        assert not marker.exists(), "the stale marker must self-heal (unlink)"
    finally:
        os.waitpid(pid, 0)  # reap so the test leaks no children


def test_marker_naming_our_own_pid_is_adopted(marker, monkeypatch):
    """A killed update's marker names the pid its retry gets (containers restart pid numbering).

    No other live process can hold our pid, so the claim is ours: take it instead of refusing
    "another update" for up to 20 minutes. It is a new attempt, so it is claimed fresh: a
    nearly-expired claim must still block a second updater for our whole run.
    """
    _claim(marker, os.getpid(), time.time() - UPDATE_MARKER_MAX_AGE_SECONDS + 5)

    lock = UpdateLock(path=marker)
    assert lock.acquire() is True
    assert lock.acquired is True

    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 60)
    holder = read_live_update(path=marker)
    assert holder is not None and holder.pid == os.getpid(), "a second updater would start mid-run"
    lock.release()
    assert not marker.exists()


def test_v2_marker_naming_our_pid_at_a_nearby_creation_time_is_a_killed_update(marker):
    """A fresh pid namespace (bwrap, a container) gives the next launch the killed updater's pid
    about a second after it started: inside the 2 s cross-writer skew, but not our creation time.
    Read as ours, the launch adopted a dead claim and silently skipped the tail the kill owed."""
    _claim_v2(marker, os.getpid(), ct_offset=-1.5)

    assert read_live_update(path=marker) is None, "a dead update's claim reads as a live one"
    _claim_v2(marker, os.getpid(), ct_offset=-1.5)
    lock = UpdateLock(path=marker)
    assert lock.acquire() is True and lock.acquired is True, "adopted a killed update's claim"
    assert marker.read_text(encoding="utf-8-sig").splitlines()[2] == f"ct:{process_create_time():.3f}"
    lock.release()


def test_one_incarnation_rule_when_our_own_creation_time_is_unreadable(marker, monkeypatch):
    """Degraded (our creation time unreadable): we write no-ct claims, so a no-ct claim naming our
    pid is ours and a ct one is a previous incarnation. The marker reader said so while the
    holder-record reader answered "unprovable" and named the dead incarnation as the holder."""
    from hermes_cli import update_lock

    monkeypatch.setattr(update_lock, "_OWN_CT", {"pid": os.getpid(), "ct": None})
    now = int(time.time())
    for text, ours in ((f"{os.getpid()}\n{now}\n", True), (f"{os.getpid()}\n{now}\nct:{now - 1}.000\n", False)):
        marker.write_text(text, encoding="utf-8")
        parsed = update_lock._parse_marker(marker.read_bytes())
        assert parsed.owner_live() is ours
        assert update_lock.incarnation_live(os.getpid(), parsed.create_time) is ours, text
        assert update_lock._lock_holder(marker).held is not ours, text


def test_release_leaves_a_marker_a_handoff_partner_now_owns(marker):
    """The desktop writes the marker, then the Tauri updater takes ownership.

    Releasing must not delete a marker whose pid is no longer ours — that would
    reopen the gate while the partner is still mid-update.
    """
    lock = UpdateLock(path=marker)
    lock.acquire()

    marker.write_text(f"{DEAD_PID}\n{int(time.time())}\n", encoding="utf-8")
    lock.release()

    assert marker.exists(), "the partner's marker is not ours to remove"


def test_dead_owner_is_reclaimed_not_honored(marker):
    marker.write_text(f"{DEAD_PID}\n{int(time.time())}\n", encoding="utf-8")

    lock = UpdateLock(path=marker)
    assert lock.acquire() is True
    assert int(marker.read_text(encoding="utf-8-sig").splitlines()[0]) == os.getpid()


def test_v1_owner_past_the_age_ceiling_is_reclaimed(marker, other_pid):
    """A legacy pid-only marker carries no creation time: only the ceiling exposes a reused pid."""
    _claim(marker, other_pid, int(time.time()) - UPDATE_MARKER_MAX_AGE_SECONDS - 60)

    lock = UpdateLock(path=marker)
    assert lock.acquire() is True and lock.acquired is True


def test_v2_live_owner_is_live_regardless_of_age(marker, other_pid):
    """An update running 25 minutes (Windows builds, long drains) still owns the lock (V3/V10/V11)."""
    _claim_v2(marker, other_pid, time.time() - 25 * 60)
    before = marker.read_bytes()

    lock = UpdateLock(path=marker)
    assert lock.acquire() is False
    assert lock.holder is not None and lock.holder.pid == other_pid
    assert read_live_update(path=marker) is not None
    assert marker.read_bytes() == before, "a refused claimant never rewrites the owner's marker"


def test_v2_marker_whose_pid_was_reused_is_reclaimed(marker, other_pid):
    """A live pid whose creation time does not match the record is a recycled pid (V22)."""
    _claim_v2(marker, other_pid, ct_offset=-100.0)

    lock = UpdateLock(path=marker)
    assert lock.acquire() is True and lock.acquired is True
    assert int(marker.read_text(encoding="utf-8-sig").splitlines()[0]) == os.getpid()


def test_delegate_keeps_the_claim_live_after_the_partner_dies(marker, monkeypatch, other_pid):
    """Rule 6: `hermes update` adopting a hand-off claim names itself on line 4, restores
    lines 1-3 byte-identical on exit, and is what keeps the claim visible if the partner dies."""
    _claim_v2(marker, other_pid)
    base = marker.read_bytes()
    monkeypatch.setenv(HANDOFF_PID_ENV, str(other_pid))

    lock = UpdateLock(path=marker)
    assert lock.acquire() is True and lock.acquired is False
    lines = marker.read_text(encoding="utf-8-sig").splitlines()
    assert marker.read_bytes().startswith(base) and lines[3].startswith(f"delegate:{os.getpid()} ct:")

    dead_partner = f"{DEAD_PID}\n{int(time.time())}\nct:1.000\n{lines[3]}\n"
    marker.write_text(dead_partner, encoding="utf-8", newline="")
    holder = read_live_update(path=marker)
    assert holder is not None and holder.pid == os.getpid(), "the running delegate keeps the update visible"
    marker.write_bytes(base + f"{lines[3]}\n".encode())
    lock.release()
    assert marker.read_bytes() == base, "the live partner gets its own claim back, byte-identical"


def test_a_coarse_creation_time_for_our_own_pid_is_still_us():
    """macOS hand-off: ``posix.sh``'s ``proc_ct`` reads ``ps -o lstart``, which has whole-second
    resolution there, while the runtime's psutil reports a sub-second creation time. The
    hand-off marker's ``delegate:`` line — the only link left after line 1 moved to a custodian
    the hand-off forked, which is a *sibling* of ``hermes update``, never an ancestor — then read
    as a previous incarnation of our own pid, so the update refused its own hand-off with exit 2
    ("Another Hermes update is already running (PID <the custodian>)") on every macOS launch.

    A whole-second value that floors our creation time is us. Anything else — a fractional
    value, another second, no creation time — keeps the strict rule.
    """
    from hermes_cli import update_lock

    # Halfway inside our own second, so the assertions are not decided by the 5 ms rule
    # itself (a real creation time within epsilon of the next second is legitimately "us").
    now = float(int(time.time())) + 0.5
    world = update_lock._World(pid=os.getpid(), ct=now, now=now,
                               alive=lambda pid: True, ct_of=lambda pid: now)
    floor = float(int(now))

    assert update_lock._incarnation(os.getpid(), floor, world) is True
    assert update_lock._incarnation(os.getpid(), floor - 1.0, world) is False, "the previous second is a previous incarnation"
    assert update_lock._incarnation(os.getpid(), floor + 1.0, world) is False, "the next second cannot be us"
    assert update_lock._incarnation(os.getpid(), now - 0.25, world) is False, "a sub-second claim stays strict (the pid-namespace rule)"
    assert update_lock._incarnation(os.getpid(), None, world) is False


def test_handoff_marker_with_a_coarse_delegate_ct_is_adopted(marker, other_pid, monkeypatch):
    """The whole hand-off shape, on a coarse-ct platform: line 1 names a live sibling (the
    custodian the hand-off forked) and line 4 names us with the shell's whole-second creation
    time. Adoption must succeed — refusing is the macOS "update always fails from the Desktop"
    loop, and the marker's live owner is not an ancestor, so the delegate line is the only route.
    """
    from hermes_cli import update_lock

    our_ct = float(int(time.time())) + 0.6  # what psutil reports; the shell can only write the floor
    monkeypatch.setattr(update_lock, "_own_create_time", lambda: our_ct)
    now = int(time.time())
    marker.write_text(
        f"{other_pid}\n{now}\nct:{process_create_time(other_pid):.3f}\n"
        f"delegate:{os.getpid()} ct:{float(int(our_ct)):.3f}\n",
        encoding="utf-8", newline="",
    )

    lock = UpdateLock(path=marker)
    assert lock.acquire() is True, "the hand-off's own update must adopt its claim"
    assert lock.acquired is False, "the hand-off's claim is not ours to own"
    holder = read_live_update(path=marker)
    assert holder is not None and holder.pid == other_pid, "the hand-off still owns the claim"

    lock.release()
    assert marker.exists(), "the live hand-off still needs its marker after our stage ends"


@pytest.mark.parametrize(
    "body",
    ["", "not-a-pid\n123\n", "\n\n", "12345", "{live}\n{now}.5\n", "1_0\n{now}\n", "{live}\nsoon\n"],
    ids=["empty", "garbage-pid", "blank-lines", "no-start-time", "fractional-start", "underscore-pid",
         "garbage-start"],
)
def test_malformed_markers_never_block_an_update(marker, body, other_pid):
    """Contract A2: line 1 and line 2 are integers or the marker is malformed, i.e. dead, in
    every reader (a fractional start time used to be live in Python and dead in the others)."""
    marker.write_text(body.format(live=other_pid, now=int(time.time())), encoding="utf-8")
    old = time.time() - 60
    os.utime(marker, (old, old))  # past the empty-file grace (A3)

    assert read_live_update(path=marker) is None
    assert UpdateLock(path=marker).acquire() is True


def test_marker_parse_follows_the_shared_contract(marker, other_pid):
    """Contract A2: BOM and CRLF are tolerated; a malformed ct line makes the marker v1 (so the
    age ceiling applies); only an exact line-4 delegate counts."""
    now, ct = int(time.time()), process_create_time(other_pid)
    live = {
        "bom": f"\ufeff{other_pid}\n{now}\nct:{ct:.3f}\n",
        "crlf": f"{other_pid}\r\n{now}\r\nct:{ct:.3f}\r\n",
        "bad-ct-fresh": f"{other_pid}\n{now}\nct:abc\n",
    }
    dead = {
        "bad-ct-aged": f"{other_pid}\n{now - UPDATE_MARKER_MAX_AGE_SECONDS - 300}\nct:abc\n",
        "spaced-delegate": f"{DEAD_PID}\n{now}\nct:1.000\ndelegate: {other_pid} ct:{ct:.3f}\n",
        "delegate-on-line-3": f"{DEAD_PID}\n{now}\ndelegate:{other_pid} ct:{ct:.3f}\n",
    }
    for name, body in {**live, **dead}.items():
        marker.write_text(body, encoding="utf-8", newline="")
        holder = read_live_update(path=marker)
        assert (holder is not None) == (name in live), name


def test_fresh_empty_marker_is_a_claim_in_flight(marker):
    """Contract A3: a writer without hard links publishes create-then-write; an empty marker that
    young is someone's claim in progress, so it is neither deleted nor taken over."""
    marker.write_bytes(b"")

    holder = read_live_update(path=marker)
    lock = UpdateLock(path=marker)
    assert holder is not None and holder.pid == 0
    assert lock.acquire() is False and marker.read_bytes() == b""


def test_claim_publishes_whole_and_reclaims_dead_writers_tmp_files(marker):
    """Contract A3 + litter: the claim lands complete (tmp + exclusive link) and leaves no tmp;
    a tmp left by a writer that died between write and publish is reclaimed, a live one is kept."""
    dead_tmp = marker.with_name(f"{marker.name}.{DEAD_PID}.tmp")
    mine = marker.with_name(f"{marker.name}.{os.getpid()}.ab12.tmp")
    dead_tmp.write_text("x")
    mine.write_text("x")

    lock = UpdateLock(path=marker)
    assert lock.acquire() is True
    assert marker.read_text(encoding="utf-8").startswith(f"{os.getpid()}\n")
    leftovers = sorted(p.name for p in marker.parent.iterdir() if p.name.endswith(".tmp"))
    assert leftovers == [mine.name]
    lock.release()


def test_stale_tmp_under_our_own_pid_number_is_reclaimed(marker):
    """Containers hand out the same pids every boot: a tmp a previous holder of our pid number
    left is litter once it is older than any write of ours, while a fresh one may be in flight."""
    stale = marker.with_name(f"{marker.name}.{os.getpid()}.deadbeef.tmp")
    fresh = marker.with_name(f"{marker.name}.{os.getpid()}.cafe.tmp")
    stale.write_bytes(b"x")
    fresh.write_bytes(b"x")
    hour_ago = time.time() - 3600
    os.utime(stale, (hour_ago, hour_ago))

    lock = UpdateLock(path=marker)
    assert lock.acquire() is True
    lock.release()
    assert not stale.exists()
    assert fresh.exists()


@pytest.mark.parametrize("owner", ["\u00b2", "\u2460"], ids=["superscript-two", "circled-one"])
def test_tmp_litter_with_a_digit_lookalike_pid_never_breaks_admission(marker, owner):
    """A sibling whose pid field is a Unicode digit lookalike is not a pid we can judge: it is
    left alone and the claim still lands (``str.isdigit`` accepts it, ``int`` refuses it)."""
    odd = marker.with_name(f"{marker.name}.{owner}.tmp")
    odd.write_bytes(b"x")

    lock = UpdateLock(path=marker)
    assert lock.acquire() is True
    assert marker.read_bytes().startswith(f"{os.getpid()}\n".encode())
    lock.release()


def _torn_write(after=None):
    """An ``os.write`` that lands part of the claim, then fails like a full disk."""
    real_write = os.write

    def torn(fd, data):
        real_write(fd, data[: len(data) // 2])
        if after is not None:
            after()
        raise OSError(errno.ENOSPC, "No space left on device")

    return torn


def _no_hard_links(*_args, **_kwargs):
    raise PermissionError(errno.EPERM, "Operation not permitted")


def test_failed_exclusive_create_leaves_no_torn_claim(marker, monkeypatch):
    """Contract A3 without hard links (FAT, some network mounts): a claim whose write fails
    part-way is withdrawn — a process that never acquired must not block every updater while
    it lives."""
    monkeypatch.setattr(os, "link", _no_hard_links)
    monkeypatch.setattr(os, "write", _torn_write())
    lock = UpdateLock(path=marker)
    assert lock.acquire() is False
    monkeypatch.undo()
    assert not marker.exists()
    assert read_live_update(path=marker) is None


@pytest.mark.platforms("posix")  # replacing a file someone holds open needs POSIX unlink
def test_withdrawing_a_torn_claim_never_deletes_a_replacement(marker, monkeypatch, other_pid):
    """The withdrawal deletes only the inode it created: a claimant that reclaimed the torn
    marker and published its own in between keeps its claim."""
    def replace():
        marker.unlink()
        _claim_v2(marker, other_pid)

    monkeypatch.setattr(os, "link", _no_hard_links)
    monkeypatch.setattr(os, "write", _torn_write(replace))
    assert UpdateLock(path=marker).acquire() is False
    monkeypatch.undo()
    assert marker.read_bytes().startswith(f"{other_pid}\n".encode())


def test_unreadable_creation_time_gets_the_v1_ceiling(marker, other_pid, monkeypatch):
    """Contract A1: a live pid whose creation time cannot be read (Windows denies it for
    elevated/other-user pids) is live only within the legacy ceiling: it may be a reused pid."""
    from hermes_cli import update_lock

    _claim_v2(marker, other_pid)
    fresh = marker.read_text(encoding="utf-8")
    monkeypatch.setattr(update_lock, "process_create_time", lambda pid=None: None)
    assert read_live_update(path=marker) is not None
    _claim_v2(marker, other_pid, started_at=time.time() - UPDATE_MARKER_MAX_AGE_SECONDS - 300)
    monkeypatch.undo()
    aged = marker.read_text(encoding="utf-8")
    assert read_live_update(path=marker) is not None, "a matching creation time is live at any age"
    monkeypatch.setattr(update_lock, "process_create_time", lambda pid=None: None)
    assert read_live_update(path=marker) is None and not marker.exists(), (fresh, aged)


def test_stale_marker_is_removed_on_read(marker):
    marker.write_text(f"{DEAD_PID}\n{int(time.time())}\n", encoding="utf-8")

    assert read_live_update(path=marker) is None
    assert not marker.exists(), "whoever notices a stale marker clears it"


def test_absent_marker_reports_no_live_update(marker):
    assert read_live_update(path=marker) is None


def test_context_manager_releases_even_on_exception(marker):
    with pytest.raises(RuntimeError):
        with UpdateLock(path=marker) as lock:
            assert lock.acquired is True
            raise RuntimeError("update blew up mid-flight")

    assert not marker.exists(), "a crashed update must not strand the lock"


def test_describe_holder_names_the_pid_and_elapsed_time(marker):
    lock = UpdateLock(path=marker)
    lock.acquire()

    holder = read_live_update(path=marker)
    assert holder is not None
    message = describe_holder(holder)

    assert str(os.getpid()) in message, (
        "the user needs the pid to find the other update"
    )
    assert "already running" in message


def test_unwritable_marker_location_refuses_instead_of_running_unlocked(tmp_path):
    """No "proceed unlocked": an update that cannot claim the lock cannot exclude another."""
    lock = UpdateLock(path=tmp_path / "nonexistent-file" / "marker")
    (tmp_path / "nonexistent-file").write_text(
        "i am a file, not a dir", encoding="utf-8"
    )

    assert lock.acquire() is False
    assert lock.holder is not None and lock.holder.reason
    assert "Cannot lock this install" in describe_holder(lock.holder)



def test_an_unwritable_marker_location_says_why_the_update_needs_it_and_how_to_fix_it(tmp_path):
    """Review P2 (5411135165): the refusal stays (the marker is what the Desktop gate, gateways and
    the other updaters read), so it must say what the path is for and how to make it writable."""
    blocker = tmp_path / "home"
    blocker.write_bytes(b"a file where the marker directory should be")
    lock = UpdateLock(path=blocker / "marker")

    assert lock.acquire() is False
    message = describe_holder(lock.holder)
    assert str(blocker) in message
    assert "Desktop app" in message and "other updaters" in message, message
    assert "read-only filesystem" in message, message

_HOLD_CHECKOUT = """
import sys, time
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import UpdateLock
lock = UpdateLock(path=__import__("pathlib").Path(sys.argv[3]), install_root=sys.argv[2])
assert lock.acquire()
print("held", flush=True)
time.sleep(120)
"""


@pytest.mark.platforms("posix")
def test_checkout_lock_excludes_an_update_from_another_home(tmp_path):
    """Two `hermes update` runs on one checkout from different homes exclude each other (V3)."""
    install = tmp_path / "checkout"
    install.mkdir()
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_CHECKOUT, str(REPO_ROOT), str(install), str(tmp_path / "homeA" / "m")],
        stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        lock = UpdateLock(path=tmp_path / "homeB-marker", install_root=install)
        assert lock.acquire() is False
        assert lock.holder is not None and lock.holder.pid == holder.pid
        assert not (tmp_path / "homeB-marker").exists()
        assert update_in_progress(install)
    finally:
        holder.kill()
        holder.wait()
    assert not update_in_progress(install), "the kernel frees the lock with its holder"
    lock = UpdateLock(path=tmp_path / "homeB-marker", install_root=install)
    assert lock.acquire() is True
    lock.release()


@pytest.mark.platforms("posix")
def test_an_unjudgeable_marker_never_admits_a_contender_past_a_held_checkout_lock(tmp_path, monkeypatch):
    """L5: read_live_update never raises, so a marker it cannot judge (any exception) reads as
    "no live update". That fails open on the marker only: the checkout kernel lock stays the
    guard, so update_in_progress still answers True and a contender's acquire is refused."""
    from hermes_cli import update_lock

    install = tmp_path / "checkout"
    install.mkdir()
    marker_path = tmp_path / "marker"
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_CHECKOUT, str(REPO_ROOT), str(install), str(tmp_path / "homeA" / "m")],
        stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        _claim_v2(marker_path, holder.pid)

        def unjudgeable(_path):
            raise RuntimeError("an error the parse was never expected to raise")

        monkeypatch.setattr(update_lock, "_read_marker", unjudgeable)
        assert read_live_update(path=marker_path, install_root=install) is None
        assert update_in_progress(install), "an unjudgeable marker hid a held checkout lock"
        lock = UpdateLock(path=marker_path, install_root=install)
        assert lock.acquire() is False, "a contender got past a held checkout lock"
    finally:
        holder.kill()
        holder.wait()


@pytest.mark.platforms("posix")
def test_checkout_lock_outlives_its_owner_while_an_inheriting_child_runs(tmp_path):
    """Invariant: lock free => no process of the update tree alive (pass_fds children)."""
    install = tmp_path / "checkout"
    install.mkdir()
    lock = UpdateLock(path=tmp_path / "marker", install_root=install)
    assert lock.acquire() is True
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                             pass_fds=checkout_lock_fds(install), stdin=subprocess.DEVNULL)
    try:
        lock.release()
        assert update_in_progress(install), "the owner's exit must not free a lock its child holds"
    finally:
        child.kill()
        child.wait()
    assert not update_in_progress(install)
    assert checkout_lock_path(install).name == ".hermes-update.lock"  # no .git: a ZIP install


def test_git_checkout_lock_lives_in_the_common_git_dir_and_never_dirties_the_tree(tmp_path):
    """A held lock must never show in `git status`: the updater's autostash would stash its own
    lock file in any tree whose .gitignore lacks it (fixtures, forks, a branch switch)."""
    install = tmp_path / "checkout"
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}

    def git(*args, cwd=install):
        return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True,
                              text=True, encoding="utf-8").stdout

    install.mkdir()
    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "one")
    linked = tmp_path / "linked"
    git("worktree", "add", "-q", "--detach", str(linked))
    common = Path(os.path.normpath(install / ".git"))
    assert checkout_lock_path(install) == common / "hermes-update.lock"
    assert checkout_lock_path(linked) == common / "hermes-update.lock"

    lock = UpdateLock(path=tmp_path / "marker", install_root=install)
    assert lock.acquire() is True
    try:
        assert git("status", "--porcelain", "--untracked-files=all") == ""
        assert update_in_progress(linked), "a linked worktree shares the repository's lock"
    finally:
        lock.release()


def test_unwritable_install_root_refuses(tmp_path):
    """A checkout we cannot write is refused through the exit-2 path, never updated unlocked."""
    install = tmp_path / "not-a-dir"
    install.write_text("file", encoding="utf-8")
    lock = UpdateLock(path=tmp_path / "marker", install_root=install)
    assert lock.acquire() is False
    assert lock.holder is not None and lock.holder.reason
    assert not (tmp_path / "marker").exists()


class TestHandoffFromOrchestratingUpdater:
    """The Tauri updater holds the marker, then spawns ``hermes update``.

    The regression: the child saw its own parent's live marker and exited 2,
    so every GUI update failed with "Hermes is still running" and retrying
    just re-ran the same self-deadlock. The parent names its pid in
    HANDOFF_PID_ENV; a live holder matching it is our own orchestrator.
    """

    def test_child_runs_under_the_parents_live_claim(self, marker, monkeypatch, other_pid):
        # other_pid stands in for the live parent updater.
        _claim(marker, other_pid)
        monkeypatch.setenv(HANDOFF_PID_ENV, str(other_pid))

        lock = UpdateLock(path=marker)
        assert lock.acquire() is True
        assert lock.acquired is False, "the parent's claim is not ours to own"

        lock.release()
        assert marker.exists(), "the parent still needs its marker after our stage ends"
        assert int(marker.read_text(encoding="utf-8-sig").splitlines()[0]) == other_pid

    def test_handoff_pid_that_is_not_the_live_holder_grants_nothing(
        self, marker, monkeypatch, other_pid
    ):
        """The env var alone must not bypass the lock."""
        _claim(marker, other_pid)
        monkeypatch.setenv(HANDOFF_PID_ENV, str(other_pid + 1))

        lock = UpdateLock(path=marker)
        assert lock.acquire() is False
        assert lock.holder is not None

    @pytest.mark.parametrize(
        "value",
        ["", "not-a-pid", "-1", "0"],
        ids=["empty", "garbage", "negative", "zero"],
    )
    def test_malformed_handoff_values_fall_back_to_refusal(
        self, marker, monkeypatch, value, other_pid
    ):
        _claim(marker, other_pid)
        monkeypatch.setenv(HANDOFF_PID_ENV, value)

        assert UpdateLock(path=marker).acquire() is False

    def test_handoff_env_with_no_marker_claims_normally(self, marker, monkeypatch):
        """A handoff pid must not stop us writing our own claim when unlocked."""
        monkeypatch.setenv(HANDOFF_PID_ENV, str(os.getpid()))

        lock = UpdateLock(path=marker)
        assert lock.acquire() is True
        assert lock.acquired is True
        assert int(marker.read_text(encoding="utf-8-sig").splitlines()[0]) == os.getpid()


class TestAncestryHandoff:
    """Staged updaters older than the HANDOFF_PID_ENV export never send it.

    ``hermes-setup`` under ``~/.hermes`` is only refreshed by a full installer
    run, so an updated checkout (new lock) driven by a pre-handoff staged
    updater (old parent) deadlocks on exit 2 forever unless the child also
    recognizes a live holder that is its own process ancestor.

    ``_pid_alive`` is pinned True here because the hermetic conftest guards
    ``os.kill`` probes of pids outside the test subtree (our ppid included);
    liveness has its own coverage above — ancestry is what's under test.
    """

    @pytest.fixture(autouse=True)
    def _liveness_pinned_true(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.update_lock._pid_alive", lambda pid: True)

    def test_marker_owned_by_our_parent_process_is_our_orchestrator(self, marker):
        marker.write_text(f"{os.getppid()}\n{int(time.time())}\n", encoding="utf-8")

        lock = UpdateLock(path=marker)
        assert lock.acquire() is True, "a live ancestor's claim is the one we run under"
        assert lock.acquired is False, "the parent's claim is not ours to own"

        lock.release()
        assert marker.exists(), "the parent still needs its marker after our stage ends"

    @pytest.mark.platforms("any")
    def test_grandchild_adopts_orchestrator_marker_without_psutil(self, marker, tmp_path):
        """Regression: the desktop hand-off's grandchild refused its own orchestrator.

        The posix shim (grandparent) holds the marker and spawns ``hermes update``
        (direct child — adopts via getppid). An old-updater update into a PM tree
        then hands off again: ``_old_updater._run_child`` spawns
        ``_update_takeover.py`` as ``python -I -S -B``, where psutil cannot import
        (-S skips site-packages). The psutil-only ancestry walk returned False for
        the two-hops-up shim, and the takeover child refused with exit 2 —
        "Another Hermes update is already running (PID <the shim itself>)" —
        observed live on a macOS rehearsal install, then again on Windows, where
        the stdlib walk had no /proc and no ps. Marked for every lane: the Windows
        lane only imports files carrying a platforms marker, which is how the
        Windows half went unseen. The stdlib fallback walk is what must adopt here.
        """
        import subprocess
        import sys
        from textwrap import dedent

        # Simulate the orchestrator: this test process holds the marker and
        # spawns the -I -B middle, which spawns the -I -S -B takeover-like leaf.
        marker.write_text(f"{os.getpid()}\n{int(time.time())}\n", encoding="utf-8")

        leaf = dedent(
            """
            import sys
            from pathlib import Path
            sys.path.insert(0, %(root)r)
            from hermes_cli.update_lock import UpdateLock
            lock = UpdateLock(path=Path(%(marker)r))
            if not lock.acquire():
                print("REFUSED", lock.holder.pid)
                raise SystemExit(2)
            assert lock.acquired is False, "the orchestrator's claim is not ours to own"
            print("ADOPTED")
            """
        ) % {"root": str(REPO_ROOT), "marker": str(marker)}
        middle = dedent(
            """
            import subprocess, sys
            code = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-c", %(leaf)r],
            ).returncode
            raise SystemExit(code)
            """
        ) % {"leaf": leaf}

        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", middle],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "ADOPTED" in result.stdout
        assert marker.exists(), "the orchestrator still needs its marker after the leaf ends"

    @pytest.mark.platforms("any")
    def test_unrelated_live_holder_is_still_refused_under_stdlib_walk(self, marker, tmp_path):
        """The stdlib fallback must not widen the lock: a foreign pid stays foreign.

        The marker owner is a live *sibling* of the leaf (a sleeper spawned by
        this test), never an ancestor of it — the shape of an unrelated
        concurrent updater. The leaf's ancestry walk dead-ends at pytest and
        the sibling must keep the lock.
        """
        import subprocess
        import sys
        import time as time_mod
        from textwrap import dedent

        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        try:
            marker.write_text(f"{sleeper.pid}\n{int(time_mod.time())}\n", encoding="utf-8")
            assert read_live_update(path=marker) is not None

            leaf = dedent(
                """
                import sys
                from pathlib import Path
                sys.path.insert(0, %(root)r)
                from hermes_cli.update_lock import UpdateLock
                lock = UpdateLock(path=Path(%(marker)r))
                print("ADOPTED" if lock.acquire() else "REFUSED")
                """
            ) % {"root": str(REPO_ROOT), "marker": str(marker)}
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-c", leaf],
                capture_output=True, text=True, timeout=120,
            )
            assert result.returncode == 0, result.stderr
            assert "REFUSED" in result.stdout
        finally:
            sleeper.kill()
            sleeper.wait()

    def test_live_non_ancestor_holder_is_still_refused(self, marker):
        """Ancestry must not open the lock to unrelated concurrent updaters."""
        marker.write_text(f"{DEAD_PID}\n{int(time.time())}\n", encoding="utf-8")

        lock = UpdateLock(path=marker)
        assert lock.acquire() is False
        assert lock.holder is not None
        assert lock.holder.pid == DEAD_PID


class _FakeProcess:
    """One link of a stubbed parent chain.

    ``error`` makes this link refuse inspection, standing in for a process the
    sandbox will not let us read (psutil raises ``AccessDenied`` there).
    """

    def __init__(self, pid, parent=None, error=None):
        self.pid = pid
        self._parent = parent
        self._error = error

    def parent(self):
        if self._error is not None:
            raise self._error
        return self._parent


def _pin_ancestry(monkeypatch, leaf):
    """Make ``psutil.Process()`` return the leaf of a stubbed chain."""
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "Process", lambda *a, **k: leaf)


def _chain(*pids, blocked_above=False):
    """Build us -> pids[0] -> pids[1] ... innermost first.

    ``blocked_above`` caps the chain with a link that raises instead of
    reporting its own parent — the sandboxed ``/proc/1`` case.
    """
    psutil = pytest.importorskip("psutil")
    top = _FakeProcess(1, error=psutil.AccessDenied(pid=1)) if blocked_above else None
    node = top
    for pid in reversed(pids):
        node = _FakeProcess(pid, parent=node)
    return _FakeProcess(os.getpid(), parent=node)


class TestAncestryUnderUnreadableProcesses:
    """Regression: #87514 — an unreadable process ABOVE the orchestrator.

    ``psutil.Process.parents()`` builds the whole chain before returning and
    only tolerates ``NoSuchProcess`` per link, so one ``AccessDenied`` high up
    threw away the ancestors already found. Under firejail with
    ``ptrace_scope=1`` (and in hardened containers) ``/proc/1`` is unreadable,
    so every desktop update refused its own orchestrator's fresh marker and
    exited 2 forever. Ancestry must be decided link by link.
    """

    def test_ancestor_below_an_unreadable_process_is_still_found(self, monkeypatch):
        from hermes_cli.update_lock import _is_ancestor_pid

        _pin_ancestry(monkeypatch, _chain(2000, blocked_above=True))

        assert _is_ancestor_pid(2000) is True, (
            "the orchestrator is one link up; a process we cannot read above "
            "it must not erase a match already proven"
        )

    def test_deeper_ancestor_below_an_unreadable_process_is_found(self, monkeypatch):
        from hermes_cli.update_lock import _is_ancestor_pid

        _pin_ancestry(monkeypatch, _chain(2000, 3000, blocked_above=True))

        assert _is_ancestor_pid(3000) is True

    def test_unreadable_process_below_the_match_still_refuses(self, monkeypatch):
        """Failing before a match keeps the conservative refusal."""
        from hermes_cli.update_lock import _is_ancestor_pid

        _pin_ancestry(monkeypatch, _chain(blocked_above=True))

        assert _is_ancestor_pid(2000) is False

    def test_unrelated_pid_is_refused_on_a_fully_readable_chain(self, monkeypatch):
        from hermes_cli.update_lock import _is_ancestor_pid

        _pin_ancestry(monkeypatch, _chain(2000, 3000))

        assert _is_ancestor_pid(DEAD_PID) is False

    def test_our_own_pid_is_never_an_ancestor(self, monkeypatch):
        from hermes_cli.update_lock import _is_ancestor_pid

        _pin_ancestry(monkeypatch, _chain(2000))

        assert _is_ancestor_pid(os.getpid()) is False

    def test_walk_is_bounded(self, monkeypatch):
        """A pathological chain terminates instead of spinning."""
        from hermes_cli.update_lock import _MAX_ANCESTRY_DEPTH, _is_ancestor_pid

        _pin_ancestry(monkeypatch, _chain(*range(2000, 2000 + _MAX_ANCESTRY_DEPTH * 2)))

        assert _is_ancestor_pid(2000 + _MAX_ANCESTRY_DEPTH * 2 - 1) is False

    def test_acquire_accepts_the_orchestrator_behind_an_unreadable_init(
        self, marker, monkeypatch
    ):
        """The reporter's end-to-end symptom: exit 2 on every GUI update."""
        monkeypatch.setattr("hermes_cli.update_lock._pid_alive", lambda pid: True)
        _pin_ancestry(monkeypatch, _chain(2000, blocked_above=True))
        marker.write_text(f"2000\n{int(time.time())}\n", encoding="utf-8")

        lock = UpdateLock(path=marker)
        assert lock.acquire() is True, "the hand-off child runs under 2000's claim"
        assert lock.acquired is False, "the orchestrator's claim is not ours to own"

        lock.release()
        assert marker.exists(), "the orchestrator still needs its marker"


@pytest.mark.platforms("posix")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes any file")
def test_unwritable_lock_file_still_locks_the_checkout(tmp_path):
    """Contract A5: a `sudo hermes update` leaves the lock file root-owned; later updates take the
    kernel lock on a read-only fd instead of refusing forever, and still exclude each other."""
    root = tmp_path / "install"
    root.mkdir()
    lock_file = checkout_lock_path(root)
    lock_file.write_text("0\n0\n")
    lock_file.chmod(0o444)

    first = UpdateLock(path=tmp_path / "a" / ".hermes-update-in-progress", install_root=root)
    assert first.acquire() is True, first.holder
    second = subprocess.run(
        [sys.executable, "-c",
         "import sys; from hermes_cli.update_lock import UpdateLock; "
         "sys.exit(0 if UpdateLock(path=__import__('pathlib').Path(sys.argv[1]), install_root=sys.argv[2]).acquire() else 3)",
         str(tmp_path / "b" / ".hermes-update-in-progress"), str(root)],
        cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}, stdin=subprocess.DEVNULL, timeout=60,
    )
    first.release()
    assert second.returncode == 3, "a second update ran while the read-only-locked checkout was held"


@pytest.mark.platforms("windows")
def test_read_only_lock_file_is_refused_on_windows(tmp_path):
    """Windows children cannot inherit the lock; they find their owner by the record in it. An
    owner that can only open the lock file read-only refuses up front instead of admitting
    itself and then having its own completion child refused."""
    root = tmp_path / "install"
    root.mkdir()
    lock_file = checkout_lock_path(root)
    lock_file.write_bytes(b"0\n0\n")
    lock_file.chmod(0o444)  # FILE_ATTRIBUTE_READONLY: O_RDWR is denied, the read-only open works
    marker = tmp_path / "home" / ".hermes-update-in-progress"
    try:
        lock = UpdateLock(path=marker, install_root=root)
        assert lock.acquire() is False
        assert lock.holder is not None and lock.holder.reason
        assert not marker.exists()
    finally:
        lock_file.chmod(0o666)


@pytest.mark.parametrize("link", [pytest.param("symlink", marks=pytest.mark.platforms("posix")), "link"])
def test_checkout_lock_never_writes_through_a_link_to_a_file_outside_the_install(tmp_path, link):
    """The holder record is written into the lock file itself: a symlink or hard link planted at
    its name is refused, never followed into truncating the file it points at."""
    root = tmp_path / "install"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"PRECIOUS USER DATA")
    getattr(os, link)(outside, checkout_lock_path(root))

    lock = UpdateLock(path=tmp_path / "home" / ".hermes-update-in-progress", install_root=root)
    assert lock.acquire() is False
    assert lock.holder is not None and lock.holder.reason
    assert outside.read_bytes() == b"PRECIOUS USER DATA"
