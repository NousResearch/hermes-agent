"""PID-namespace identity for gateway runtime records (#123081).

A gateway running under ``systemd PrivatePIDs=`` (or any container-per-gateway
layout with a shared state dir) records PID 1, because ``os.getpid()`` is
namespace-relative. Every reader outside that namespace resolves PID 1 to the
host's init, which is alive, is not a gateway, and — in the PID-file path —
caused the identity files to be unlinked. Deleting ``gateway.lock`` while the
live gateway still holds an flock on the unlinked inode let a second gateway
create a fresh lock and win: the singleton guard, the only atomic arbiter
against two gateways on one home, was bypassed and both ran.

These tests pin the three consumers that read another process' recorded PID, and
the tri-state namespace identity they share. Namespace identity is injected as
data (``local_pid_namespace``), never derived from ``sys.platform``, so the
suite is deterministic on every host; one test reads the real ``/proc`` on Linux.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_platform.host import pid_namespace as pns
from gateway import lifecycle_ledger as ll
from hermes_platform.host.pid_namespace import (
    LocalPidNamespace,
    _parse_pid_namespace_link,
    local_pid_namespace,
    pid_checkable_from,
    pid_namespace_id,
)
from gateway import scoped_lock_identity as sli
from gateway.scoped_lock_identity import scoped_lock_owned_by_self, scoped_lock_stale_locally

pytestmark = pytest.mark.platforms("linux")

_HOST_NS = "4026531834"
_OTHER_NS = "4026532999"

_LIVE = LocalPidNamespace(id=_HOST_NS, supported=True)
_UNKNOWN = LocalPidNamespace(id=None, supported=True)
_NONE = LocalPidNamespace(id=None, supported=False)


@pytest.fixture
def foreign_namespace(monkeypatch):
    """This process knows its own namespace, but the record claims a different one."""
    _set_local_namespace(monkeypatch, _LIVE)


def _set_local_namespace(monkeypatch, namespace):
    """Point BOTH the resolver and its in-tree consumers at a namespace identity.

    ``scoped_lock_identity``, ``status``, ``lifecycle_ledger`` and ``host_rendezvous``
    import ``local_pid_namespace`` by name, so patching only the defining module would
    leave them resolving the real namespace and the test would pass or fail for the wrong
    reason — a writer-side test would silently exercise the host's real namespace.

    ``monkeypatch`` undoes each of these at teardown, which is what keeps one case from
    deciding the next one's answer.
    """
    for module in (pns, sli, ll):
        monkeypatch.setattr(module, "local_pid_namespace", lambda: namespace)
    for name in ("gateway.status", "gateway.host_rendezvous"):
        try:
            module = importlib.import_module(name)
        except Exception:  # pragma: no cover - a lane without that module
            continue
        if hasattr(module, "local_pid_namespace"):
            monkeypatch.setattr(module, "local_pid_namespace", lambda: namespace)
    pns._clear_local_namespace_cache()


# ---------------------------------------------------------------------------
# The tri-state identity
# ---------------------------------------------------------------------------


def test_parse_pid_namespace_link_reads_the_kernel_inode():
    assert _parse_pid_namespace_link("pid:[4026531834]") == "4026531834"
    assert _parse_pid_namespace_link("pid:[4026531834]\n") == "4026531834"
    # Not a pid namespace link at all: a PID namespace inode is never empty.
    assert _parse_pid_namespace_link("pid:[]") is None
    assert _parse_pid_namespace_link("mnt:[4026531840]") is None
    assert _parse_pid_namespace_link("") is None


def test_a_successful_retry_is_promoted_into_the_cache(monkeypatch):
    """Failure → success → later failure must keep returning what we already learned.

    The cached resolver can hold ``_UNRESOLVED`` permanently: the direct retry in
    ``local_pid_namespace`` returned a real id, but nothing replaced the cached value, so a later
    transient failure dropped the process back to UNKNOWN for the rest of its life. That
    contradicts the stated contract ("cached once a definite answer was obtained") and reactivates
    every unknown-namespace refusal mid-process — including the branches that refuse to signal.
    """
    state = {"fail": True}

    def flaky_readlink(path):
        # Fail until the test flips the switch: deterministic, and independent of how many reads
        # the resolver happens to make.
        if state["fail"]:
            raise OSError()
        return f"pid:[{_HOST_NS}]"

    pns._clear_local_namespace_cache()
    monkeypatch.setattr(pns.os, "readlink", flaky_readlink)
    try:
        assert local_pid_namespace().known is False, "reads are failing"

        state["fail"] = False  # /proc recovers
        assert local_pid_namespace().id == _HOST_NS

        state["fail"] = True  # and fails again later
        assert local_pid_namespace().id == _HOST_NS, (
            "a namespace we already learned was forgotten when a later read failed"
        )
        assert local_pid_namespace().id == _HOST_NS
    finally:
        pns._clear_local_namespace_cache()


def test_a_learned_namespace_survives_a_failure_during_its_own_promotion(monkeypatch):
    """The retry's answer must be retained AS OBTAINED, not re-probed to be promoted (#123081).

    The old promotion cleared the memo and called the resolver again, so retaining the id
    depended on a SECOND read succeeding. Make that read fail and the "promotion" stored
    ``_UNRESOLVED`` — and returned it, so the very call that had just learned the namespace
    handed "unknown" back to its own caller. The observation already happened; keeping it
    must not require observing it again.
    """
    reads = {"n": 0}

    def fail_after_the_retry(path):
        # The first read (the cached attempt) fails, the second (the direct retry) succeeds,
        # and every read after that fails -- i.e. the promotion's own probe is the transient.
        reads["n"] += 1
        if reads["n"] == 2:
            return f"pid:[{_HOST_NS}]"
        raise OSError()

    pns._clear_local_namespace_cache()
    monkeypatch.setattr(pns.os, "readlink", fail_after_the_retry)
    try:
        learned = local_pid_namespace()
        assert learned.id == _HOST_NS, (
            "the retry learned a real id but its own promotion discarded it"
        )
        assert local_pid_namespace().id == _HOST_NS, (
            "a namespace already learned was lost on the next call"
        )
    finally:
        pns._clear_local_namespace_cache()


def _memo_read_and_probe_are_one_critical_section(monkeypatch):
    """Force a peer to retain the namespace while this call is deciding what to return.

    Returns ``(peer_retained, reader_value, foreign_still_refused)``.

    The synchronization point is the RETRY probe -- ``local_pid_namespace``'s second
    ``_resolve_local_pid_namespace()`` call -- because that one runs OUTSIDE the memo lock. The
    first probe is inside ``_local_pid_namespace_cached``'s critical section, so a peer
    publishing from there would deadlock against it and the race would resolve on a timeout
    instead of on the ordering we are trying to pin. Failing the retry therefore puts the peer
    and this call on either side of the narrow window the fix exists to close, and the events
    make that ordering deterministic rather than timed.
    """
    pns._clear_local_namespace_cache()
    retry_in_flight = threading.Event()
    peer_retained = threading.Event()
    calls = {"n": 0}

    def readlink(path):
        calls["n"] += 1
        if calls["n"] == 1:
            # First probe: the memo is empty and this call cannot proceed to the retry without
            # failing. This one runs under the memo lock, so do NOT block here.
            raise OSError()
        # Retry probe: outside the lock, so the peer can publish while this call is in it.
        retry_in_flight.set()
        # This assert IS the anti-timeout pin: it fails the case if the peer's retain was not
        # actually observed here, so a synchronization that only "passes" because a timeout
        # elapsed cannot go green.
        assert peer_retained.wait(timeout=10), (
            "the peer's retain was not observed by this retry probe -- the case would be resolving "
            "on the timeout rather than on the intended ordering"
        )
        raise OSError()

    monkeypatch.setattr(pns.os, "readlink", readlink)

    def peer():
        assert retry_in_flight.wait(timeout=10), "reader never reached its retry probe"
        pns._set_local_namespace(_LIVE)
        peer_retained.set()

    thread = threading.Thread(target=peer, daemon=True)
    thread.start()
    try:
        reader_value = pns.local_pid_namespace()
        # The raced value is what feeds the unlink policy, so assert the consequence while the
        # memo still holds the peer's namespace. Checked here rather than in a separate case: a
        # standalone predicate test passes with this fix reverted and proves nothing about it.
        foreign_refused = pns.record_unlinkable_from(_OTHER_NS)
    finally:
        thread.join(timeout=10)
        pns._clear_local_namespace_cache()
    return peer_retained.is_set(), reader_value, foreign_refused


def test_a_concurrent_retain_cannot_slip_past_the_memo_decision(monkeypatch):
    """A failed probe must not hand back an unresolved value a peer already superseded (#123081).

    The lock around "check the memo, then probe" does NOT close this on its own -- reverting it
    leaves the suite green. What closes it is the unresolved branch taking the memo POINTER before
    returning, so the caller gets the namespace that won rather than the read it happened to take
    first. The stale value is not inert: `record_unlinkable_from` reads `ours.known is False` as
    "cannot read a claim, so unlink", which makes a KNOWN foreign record eligible for cleanup.
    """
    retained, reader_value, foreign_refused = _memo_read_and_probe_are_one_critical_section(
        monkeypatch
    )

    assert retained is True, "the peer never managed to retain, so this case is not exercising the race"

    assert reader_value == _LIVE, (
        f"the raced call returned {reader_value} instead of the namespace the peer had retained"
    )
    assert foreign_refused is True, (
        "a canonical foreign record became unlinkable because our own namespace was read as "
        "unresolved"
    )


def test_local_namespace_is_stable_and_known_on_this_linux_host():
    """A definite answer is cached; the namespace cannot change under a live process."""
    first = local_pid_namespace()
    assert first.supported is True
    assert first.known is True
    assert first.id and first.id.isdigit()
    assert local_pid_namespace() is first


def test_real_readlink_agrees_with_the_public_resolver():
    """The cached identity is the kernel's own inode, not a guess."""
    import os as _os

    assert pid_namespace_id(_os.getpid()) == local_pid_namespace().id


def test_failed_lookup_is_not_cached_so_a_transient_proc_problem_recovers(monkeypatch):
    """A failed /proc read must not pin the process to "unknown" for its whole life.

    The resolver retries within a single call, so a transient failure is already
    invisible by the time it returns; what must not happen is caching that
    failure. With every read failing, each call re-reads rather than remembering.
    """
    reads = {"n": 0}

    def failing_readlink(path):
        reads["n"] += 1
        raise PermissionError("transient")

    pns._clear_local_namespace_cache()
    monkeypatch.setattr(pns.os, "readlink", failing_readlink)
    try:
        assert local_pid_namespace().known is False
        after_first = reads["n"]
        assert local_pid_namespace().known is False
        assert reads["n"] > after_first, "the failure was cached instead of retried"
    finally:
        pns._clear_local_namespace_cache()


def test_a_definite_answer_is_cached(monkeypatch):
    """The resolved namespace cannot change under a live process, so it is read once."""
    reads = {"n": 0}

    def counting_readlink(path):
        reads["n"] += 1
        return f"pid:[{_HOST_NS}]"

    pns._clear_local_namespace_cache()
    monkeypatch.setattr(pns.os, "readlink", counting_readlink)
    try:
        first = local_pid_namespace()
        assert first.id == _HOST_NS
        local_pid_namespace()
        assert reads["n"] == 1
    finally:
        pns._clear_local_namespace_cache()


# ---------------------------------------------------------------------------
# The predicate every consumer routes through
# ---------------------------------------------------------------------------


def test_pid_checkable_matches_only_inside_the_same_namespace(monkeypatch):
    _set_local_namespace(monkeypatch, _LIVE)
    assert pid_checkable_from(_HOST_NS) is True
    assert pid_checkable_from(_OTHER_NS) is False


def test_pid_checkable_keeps_hostname_only_semantics_where_no_namespace_exists(monkeypatch):
    """macOS/Windows: one namespace, so a bare PID keeps its meaning."""
    _set_local_namespace(monkeypatch, _NONE)
    assert pid_checkable_from(None) is True
    assert pid_checkable_from(_OTHER_NS) is True


def test_pid_checkable_fails_closed_when_our_own_lookup_failed(monkeypatch):
    """Absence of provenance cannot become provenance because the read failed.

    This is the SIGNAL-side predicate: refusing to act on a PID we cannot qualify is what
    keeps an unknown namespace from authorizing a kill. It is deliberately not the same
    predicate the unlink uses — see ``test_record_unlinkable_...`` below, where the same
    tri-state row has to keep main's behavior or the install cannot start.
    """
    _set_local_namespace(monkeypatch, _UNKNOWN)
    assert pid_checkable_from(_HOST_NS) is False
    assert pid_checkable_from(None) is False


@pytest.mark.parametrize(
    "recorded, unlinkable",
    [
        (_OTHER_NS, True),      # canonical and foreign — the only refusal
        (_HOST_NS, False),      # ours
        (None, False),          # unstamped legacy record
        ("4026532999.0", False),  # non-canonical: no claim to read
        ("", False),
        ("abc", False),
    ],
    ids=lambda v: repr(v),
)
def test_record_unlinkable_refuses_only_on_a_readable_foreign_stamp(
    monkeypatch, recorded, unlinkable
):
    """The unlink answers a different question than the signal, with opposite costs.

    Refusing to delete a dead owner's identity files is only justified by a claim we can
    actually read — a canonical namespace id that is not ours. Everything else (unstamped,
    corrupt, or our own namespace unknown right now) keeps main's behavior, because
    ``write_pid_file`` is O_EXCL and a refusal here becomes an install that cannot start.
    """
    from hermes_platform.host.pid_namespace import record_unlinkable_from

    _set_local_namespace(monkeypatch, _LIVE)
    assert record_unlinkable_from(recorded) is unlinkable


def test_record_unlinkable_never_refuses_when_our_namespace_is_unknown(monkeypatch):
    """A failed ``/proc`` lookup must not be able to block a cleanup, and therefore a start."""
    from hermes_platform.host.pid_namespace import record_unlinkable_from

    _set_local_namespace(monkeypatch, _UNKNOWN)
    assert record_unlinkable_from(_OTHER_NS) is False
    assert record_unlinkable_from(None) is False

    _set_local_namespace(monkeypatch, _NONE)
    assert record_unlinkable_from(_OTHER_NS) is False


@pytest.mark.parametrize(
    "corrupt",
    ["4026532999.0", " 4026532221 ", "", "abc", "pid:[4026531836]", "0x40265318", "4026532221\t"],
    ids=lambda v: repr(v),
)
def test_pid_checkable_reads_a_non_canonical_stamp_as_unstamped(monkeypatch, corrupt):
    """A corrupt stamp must not wedge the install.

    ``pid_checkable_from`` compares strings, so any non-canonical value compared unequal
    and read as FOREIGN — refusing to clean up a dead owner's files. Since
    ``write_pid_file`` is O_EXCL, that turns one bad byte into a gateway that cannot start
    at all, with no CLI route back. A value this module could never have written carries no
    identity claim, so it takes the unstamped path — main's behavior — while a real foreign
    stamp still refuses.
    """
    _set_local_namespace(monkeypatch, _LIVE)
    assert pid_checkable_from(corrupt) is True


@pytest.mark.parametrize("unicode_digits", ["²", "٣", "٤٠٢٦", "１２", "⁵"])
def test_pid_checkable_ignores_non_ascii_digit_forms(monkeypatch, unicode_digits):
    """``str.isdigit()`` accepts non-ASCII digit forms, which would read as FOREIGN.

    The canonical form is what ``/proc/<pid>/ns/pid`` yields — an ASCII integer string. A
    record holding any other digit spelling was not written by :func:`local_pid_namespace`,
    so it must take the unstamped path rather than be classified as a real identity and
    reintroduce the wedge. Hand-editing a record is the only route here, but the predicate
    documents that it only ever accepts what it writes, and that has to be true.
    """
    _set_local_namespace(monkeypatch, _LIVE)
    assert unicode_digits.isdigit() is True, "still a digit per str.isdigit — that is the trap"
    assert pid_checkable_from(unicode_digits) is True


def test_pid_checkable_still_refuses_a_real_foreign_stamp(monkeypatch):
    """The control for the corrupt-stamp tolerance: a canonical foreign id is still refused."""
    _set_local_namespace(monkeypatch, _LIVE)
    assert pid_checkable_from(_OTHER_NS) is False


def test_pid_checkable_keeps_legacy_unstamped_records_probeable(monkeypatch):
    """The rollout boundary: an unstamped record keeps main's behavior.

    Refusing to probe it would make every pre-upgrade record permanently
    unverifiable, silently disabling unclean-death detection for every install
    that had not yet restarted on a stamping build. Protection arrives when the
    gateway next restarts.
    """
    _set_local_namespace(monkeypatch, _LIVE)
    assert pid_checkable_from(None) is True


# ---------------------------------------------------------------------------
# Consumer 1: the runtime record carries the namespace that issued its PID
# ---------------------------------------------------------------------------


def test_a_corrupt_stamp_still_lets_the_gateway_start(tmp_path, monkeypatch):
    """The wedge this guards: O_EXCL + a refusing unlink = an install that cannot boot.

    ``write_pid_file`` refuses to clobber an existing record, so a non-canonical ``pidns``
    that read as foreign would keep a DEAD owner's files on disk and make every subsequent
    start fail with ``FileExistsError``. Drives the real ``get_running_pid()`` then the real
    ``write_pid_file()``, no mocked filesystem.
    """
    from gateway import status

    _set_local_namespace(monkeypatch, _LIVE)
    dead = 2 ** 22 + 4242
    _write_pair(tmp_path, monkeypatch, {
        "pid": dead, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": status._get_process_start_time(dead), "hermes_home": str(tmp_path),
        "pidns": "4026532221.0",
    })
    status.get_running_pid()
    assert not (tmp_path / "gateway.pid").exists()
    status.write_pid_file()  # RED without the tolerance: FileExistsError


def test_remove_pid_file_cleans_up_when_our_own_namespace_is_unreadable(tmp_path, monkeypatch):
    """``remove_pid_file`` deletes, so it must not use the fail-closed signal predicate.

    With an unresolvable local namespace the signal predicate refuses, this call site
    returned early, and the gateway left its OWN ``gateway.pid`` behind — a record whose
    stamp it could not even write, since the same lookup failed at write time. The docstring's
    early-return is meant for an absent file, where it protects nothing. This is the third site
    the unlink/signal split missed.
    """
    from gateway import status

    _set_local_namespace(monkeypatch, _UNKNOWN)
    pid_path = tmp_path / "gateway.pid"
    _write_pair(tmp_path, monkeypatch, {
        "pid": os.getpid(), "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "hermes_home": str(tmp_path),
    })

    status.remove_pid_file()

    assert not pid_path.exists()


def test_remove_pid_file_still_keeps_a_foreign_namespaces_record(tmp_path, monkeypatch, foreign_namespace):
    """The control in the other direction: a readable foreign stamp still blocks the unlink."""
    from gateway import status

    pid_path = tmp_path / "gateway.pid"
    _write_pair(tmp_path, monkeypatch, {
        "pid": os.getpid(), "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": status._get_process_start_time(os.getpid()), "hermes_home": str(tmp_path),
        "pidns": _OTHER_NS,
    })

    status.remove_pid_file()

    assert pid_path.exists()


def test_an_unresolvable_local_namespace_still_lets_the_gateway_start(tmp_path, monkeypatch):
    """A failed ``/proc`` lookup must not be able to wedge the install.

    ``pid_checkable_from`` fails closed when this process cannot name its own namespace,
    which is right before a signal. Applied to the unlink it was a trap: the owner's files
    stayed on disk, and since ``write_pid_file`` is O_EXCL every later start died on
    ``FileExistsError`` — reproduced here, three start attempts, all refused. ``procfs``
    unmounted or a permissions race is enough to get there, with no namespace involved.
    Drives the real ``get_running_pid()`` then the real ``write_pid_file()``.
    """
    from gateway import status

    _set_local_namespace(monkeypatch, _UNKNOWN)
    dead = 2 ** 22 + 7
    _write_pair(tmp_path, monkeypatch, {
        "pid": dead, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "hermes_home": str(tmp_path),  # unstamped — the documented legacy shape
    })

    status.get_running_pid()
    assert not (tmp_path / "gateway.pid").exists()
    status.write_pid_file()  # RED before the split: FileExistsError


def test_an_unresolvable_local_namespace_still_refuses_to_signal(tmp_path, monkeypatch):
    """The control for the split: failing closed stays in force before a signal.

    ``_validated_scoped_lock_gateway_owner`` is a different question from an unlink — a
    signal cannot be taken back, so an unknown namespace there still refuses.
    """
    from gateway import status

    _set_local_namespace(monkeypatch, _UNKNOWN)
    assert status._validated_scoped_lock_gateway_owner({
        "pid": 4242, "kind": "hermes-gateway", "start_time": 1,
        "hermes_home": str(tmp_path), "pidns": _HOST_NS,
    }) is None


def test_a_foreign_stamped_pid_is_never_reported_live(tmp_path, monkeypatch, foreign_namespace):
    """The identity predicate itself, not just the unlink paths.

    ``_live_pid_from_record`` is what ``get_running_pid`` (lock-held branch),
    ``get_running_pid_identity_strict``, ``get_runtime_status_running_pid`` and
    ``runtime_status_pid_is_live`` all return, and the first three feed callers that SIGNAL
    what they get. Under ``PrivatePIDs=`` the recorded 1 resolves to the host's init, which
    is very much alive — so with no ``start_time`` in the record and an unreadable cmdline,
    the old order answered "live, PID 1" and ``find_gateway_pids`` fed that to SIGTERM.
    """
    from gateway import status

    record = {
        "pid": os.getpid(), "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "hermes_home": str(tmp_path), "pidns": _OTHER_NS,
    }
    # No start_time and an unreadable cmdline: nothing but the namespace can answer.
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: None)

    assert status._live_pid_from_record(record) is None
    assert status.runtime_status_pid_is_live(record) is False


def test_our_own_stamped_pid_is_still_reported_live(tmp_path, monkeypatch):
    """The control: a same-namespace record without a start_time is unchanged."""
    from gateway import status

    _set_local_namespace(monkeypatch, _LIVE)
    record = {
        "pid": os.getpid(), "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "hermes_home": str(tmp_path), "pidns": _HOST_NS,
    }
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: None)

    assert status._live_pid_from_record(record) == os.getpid()


def _held_lock_carrying(tmp_path, monkeypatch, record):
    """Write ``record`` to gateway.pid + gateway.lock and hold a real flock on the lock.

    The record is written into the lock file too, because the held-lock branch reads both and
    each one alone can decide the cleanup. Returns ``(lock_path, open_handle)``.
    """
    from gateway import status

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "gateway.pid").write_text(json.dumps(record))
    lock = tmp_path / "gateway.lock"
    lock.write_text(json.dumps(record))
    handle = open(lock, "a+", encoding="utf-8")
    assert status._try_acquire_file_lock(handle), "test must start from a HELD lock"
    return lock, handle


def test_a_held_runtime_lock_is_never_unlinked_because_identity_is_unprovable(
    tmp_path, monkeypatch
):
    """A proven-HELD lock is the authority, not a PID we failed to name (#123081, #123109).

    ``_live_pid_from_record`` returns None whenever the namespace gate refuses — including for
    this build's own unresolved stamp — and the held-lock branch used to read that as "no live
    gateway", then cleaned up with ``unlink_lock=True``. The holder keeps its flock on the deleted
    inode, the next starter creates a fresh file and acquires it: the singleton bypass the whole
    change exists to close. Real flock, real files; the assertion is that the pathname survives.
    """
    from gateway import status

    _set_local_namespace(monkeypatch, _UNKNOWN)
    record = {
        "pid": os.getpid(), "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "hermes_home": str(tmp_path), "pidns": pns.PIDNS_UNRESOLVED,
    }
    lock, holder = _held_lock_carrying(tmp_path, monkeypatch, record)
    try:
        status.get_running_pid()
        assert lock.exists(), "the pathname of a proven-held runtime lock was unlinked"
        # The pathname surviving IS the guard: `flock` is per-inode, so the next starter's
        # open-for-write on this same path is what the held lock blocks. A separate file would
        # be a separate inode and could never be blocked — that asymmetry is the whole bypass.
    finally:
        holder.close()


def test_a_held_lock_survives_a_dead_local_pid_we_cannot_qualify(tmp_path, monkeypatch):
    """Held-lock protection must not depend on the local PID table (#123081, #123109).

    The test above pins the branch where the recorded PID happens to be locally live, so the
    local process table alone could satisfy it. This one removes that coincidence: the recorded
    PID does not exist here and the identity is unprovable, so the local table's silence is not
    evidence that our owner died. Only qualified identity may license that inference.
    """
    from gateway import status

    # The reader cannot name its own namespace, so `record_unlinkable_from` also answers False:
    # this is the only shape where the qualification guard is the sole thing standing between an
    # unprovable owner and an unlinked lock. A foreign stamp is already refused by the unlink
    # policy, so it cannot witness this.
    _set_local_namespace(monkeypatch, _UNKNOWN)
    absent_pid = 2 ** 22 + 12345  # beyond the default pid_max, so never allocatable here
    assert not status._pid_exists(absent_pid), "the control PID must not exist locally"
    record = {
        "pid": absent_pid, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "hermes_home": str(tmp_path), "pidns": pns.PIDNS_UNRESOLVED,
    }
    lock, holder = _held_lock_carrying(tmp_path, monkeypatch, record)
    try:
        status.get_running_pid()
        assert lock.exists(), (
            "a held lock was unlinked because the owner's PID was absent from THIS host's "
            "table -- while its identity is unprovable that absence is not death evidence"
        )
    finally:
        holder.close()


def test_a_current_writer_marks_its_own_unresolved_namespace(tmp_path, monkeypatch):
    """This build must not write a record indistinguishable from a pre-stamp one.

    The rollout boundary assumes ``pidns`` absent means "written by a build that predates
    the stamp". That is false: ``_build_pid_record`` only sets the key when
    ``local_pid_namespace().known``, so a transient or unmounted ``/proc`` makes THIS build emit
    an unstamped record from inside a namespace. A later reader, once ``/proc`` recovers, sees
    ``pidns is None`` and reclassifies it as legacy local authority — so the boundary did not
    protect the record, it opened it. An unresolved-but-supported writer therefore has to write
    an explicit state the reader can tell apart from absence.
    """
    from gateway import status

    _set_local_namespace(monkeypatch, _UNKNOWN)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record = status._build_pid_record()

    assert "pidns" in record, "an unresolved writer must say so, not omit it"
    assert record["pidns"] != _HOST_NS
    assert pns.is_unresolved_pid_namespace(record["pidns"]) is True


def test_an_unresolved_stamp_is_never_checkable_though_legacy_is(monkeypatch):
    """The reader half: a stamp meaning 'this build could not qualify it' must fail closed.

    Absence still means legacy and keeps main's behavior; the explicit unresolved state is
    different — this record knows it came from a namespace it could not name, and probing its PID
    here would read an unrelated process.
    """
    _set_local_namespace(monkeypatch, _LIVE)

    assert pns.pid_checkable_from(pns.PIDNS_UNRESOLVED) is False
    assert pns.pid_checkable_from(None) is True, "legacy absence keeps main's behavior"
    # …and it must not wedge the unlink either: that was the round-4 trap.
    assert pns.record_unlinkable_from(pns.PIDNS_UNRESOLVED) is False


def test_pid_record_stamps_the_pid_namespace(tmp_path, monkeypatch):
    from gateway import status

    # Compare against the writer's OWN resolver, not the defining module: the writer calls the
    # name it imported, and this suite deliberately points that at a fixture in other cases.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record = status._build_pid_record()
    assert record["pidns"] == status.local_pid_namespace().id

    # The stamp reaches the file the next process reads.
    status.write_pid_file()
    payload = json.loads((tmp_path / "gateway.pid").read_text())
    assert payload["pidns"] == status.local_pid_namespace().id


def test_pid_record_omits_pidns_where_no_namespace_exists(tmp_path, monkeypatch):
    from gateway import status

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Patch the CONSUMING module: status.py imports the resolver by name, so a
    # patch on the defining module would pass silently.
    monkeypatch.setattr(status, "local_pid_namespace", lambda: _NONE)
    assert "pidns" not in status._build_pid_record()


# ---------------------------------------------------------------------------
# Consumer 2: the unlink that bypassed the singleton guard
# ---------------------------------------------------------------------------


def _write_pair(tmp_path, monkeypatch, record):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "gateway.pid").write_text(json.dumps(record))
    (tmp_path / "gateway.lock").write_text(json.dumps(record))


def _held_lock(home):
    """Hold a real flock on ``home/gateway.lock`` for the duration of the test."""
    from gateway import status

    lock = home / "gateway.lock"
    handle = open(lock, "a+", encoding="utf-8")
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": 1, "kind": "hermes-gateway"}))
    handle.flush()
    assert status._try_acquire_file_lock(handle)
    return handle


def test_foreign_namespace_record_does_not_lose_its_live_runtime_lock(tmp_path, monkeypatch, foreign_namespace):
    """The #123081 incident, end to end.

    The live gateway holds gateway.lock from inside PrivatePIDs=. A checker
    outside the namespace reads its recorded PID 1 as the host's init and calls
    it dead. Before the fix that unlinked gateway.pid AND gateway.lock; the live
    flock stayed on the deleted inode, the next acquire created a fresh file and
    succeeded, and two gateways ran against one Telegram token.
    """
    from gateway import status

    foreign_record = {
        "pid": 1, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": status._get_process_start_time(1), "pidns": _OTHER_NS,
        "hermes_home": str(tmp_path),
    }
    _write_pair(tmp_path, monkeypatch, foreign_record)

    ready, release = threading.Event(), threading.Event()

    def hold():
        handle = _held_lock(tmp_path)
        ready.set()
        release.wait(30)
        status._release_file_lock(handle)
        handle.close()

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert ready.wait(5)
    time.sleep(0.2)
    try:
        assert status.is_gateway_runtime_lock_active(tmp_path / "gateway.lock") is True
        assert status.get_running_pid() is None  # unverifiable, not "dead"

        # The fix: the identity files survive, so the flock still arbitrates.
        assert (tmp_path / "gateway.pid").exists(), "refused to unlink a live gateway's pid file"
        assert (tmp_path / "gateway.lock").exists(), "refused to unlink a live gateway's lock"
        assert status.acquire_gateway_runtime_lock() is False, "second gateway was admitted"
    finally:
        status.release_gateway_runtime_lock()
        release.set()
        holder.join(5)


def test_same_namespace_record_still_cleans_up_a_genuine_poison_file(tmp_path, monkeypatch):
    """#89315 keeps working: a record we CAN verify is unlinked as before."""
    from gateway import status

    _set_local_namespace(monkeypatch, _LIVE)
    dead = 2 ** 22 + 12345
    _write_pair(tmp_path, monkeypatch, {
        "pid": dead, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": status._get_process_start_time(dead), "pidns": _HOST_NS,
        "hermes_home": str(tmp_path),
    })
    status.get_running_pid()
    assert not (tmp_path / "gateway.pid").exists()
    assert not (tmp_path / "gateway.lock").exists()


def test_unstamped_legacy_record_still_cleans_up(tmp_path, monkeypatch):
    """A pre-upgrade record (no pidns) on a host with no namespace concept is stale-checked as before."""
    from gateway import status

    _set_local_namespace(monkeypatch, _NONE)
    dead = 2 ** 22 + 12345
    _write_pair(tmp_path, monkeypatch, {
        "pid": dead, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": status._get_process_start_time(dead), "hermes_home": str(tmp_path),
    })
    status.get_running_pid()
    assert not (tmp_path / "gateway.pid").exists()


# ---------------------------------------------------------------------------
# Consumer 2b: the shutdown path, which unlinks on numeric PID alone
# ---------------------------------------------------------------------------


def test_exit_path_leaves_a_foreign_namespaces_pid_record_alone(tmp_path, monkeypatch, foreign_namespace):
    """``remove_pid_file`` must not erase another namespace's record when the PIDs collide.

    Under ``PrivatePIDs=`` this gateway IS pid 1 in its own namespace, so a
    record stamped elsewhere with ``pid=1`` passes the numeric-equality test that
    main uses to decide ownership. It runs at atexit (``run.py``'s
    ``_exit_after_graceful_shutdown``, ``run_shutdown``, ``shutdown_watchdog``,
    ``hermes gateway stop``), so an ordinary exit would delete the other
    namespace's ``gateway.pid`` while ``detect_unclean_exit()`` still reads that
    gateway as live. The flock is untouched here — this is not finding 1's bypass.
    """
    from gateway import status

    pid_path = tmp_path / "gateway.pid"
    _write_pair(tmp_path, monkeypatch, {
        "pid": os.getpid(), "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": status._get_process_start_time(os.getpid()), "hermes_home": str(tmp_path),
        "pidns": _OTHER_NS,
    })
    status.remove_pid_file()
    assert pid_path.exists()


def test_exit_path_still_removes_our_own_pid_record(tmp_path, monkeypatch):
    """The control: our own record is still cleaned up, stamped or not."""
    from gateway import status

    _set_local_namespace(monkeypatch, _LIVE)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "gateway.pid").write_text(json.dumps({
        "pid": os.getpid(), "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": status._get_process_start_time(os.getpid()), "hermes_home": str(tmp_path),
        "pidns": _HOST_NS,
    }))
    status.remove_pid_file()
    assert not (tmp_path / "gateway.pid").exists()


# ---------------------------------------------------------------------------
# Consumer 3: the lifecycle ledger verdict and the scoped bot-token lock
# ---------------------------------------------------------------------------


def test_mark_exited_does_not_clobber_another_namespaces_sentinel(tmp_path, monkeypatch, foreign_namespace):
    """``mark_exited`` is the mirror of finding 1: it rewrites by numeric PID alone.

    Two gateways in different namespaces can both be PID 1, so a gateway exiting normally
    would rewrite the other namespace's ``phase=running`` sentinel to ``phase=exited`` — and
    the next boot's ``record_startup`` would read its own sentinel as a clean life while the
    other gateway is still alive. Numeric equality is not ownership outside a shared namespace,
    the same rule ``_pid_is_sentinel_owner`` already applies on the read side.
    """
    from gateway import lifecycle_ledger as ll
    from gateway.lifecycle_ledger import get_lifecycle_sentinel_path

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = get_lifecycle_sentinel_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "phase": "running", "pid": os.getpid(), "pidns": _OTHER_NS,
        "start_time": time.time(), "create_time": 111.0,
    }), encoding="utf-8")

    ll.mark_exited(0, home=tmp_path)

    assert json.loads(path.read_text(encoding="utf-8"))["phase"] == "running"


def test_mark_exited_still_rewrites_our_own_sentinel(tmp_path, monkeypatch):
    """The control: our own sentinel is still marked exited, namespace and all."""
    from gateway import lifecycle_ledger as ll
    from gateway.lifecycle_ledger import get_lifecycle_sentinel_path

    _set_local_namespace(monkeypatch, _LIVE)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = get_lifecycle_sentinel_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "phase": "running", "pid": os.getpid(), "pidns": _HOST_NS,
        "start_time": time.time(), "create_time": 111.0,
    }), encoding="utf-8")

    ll.mark_exited(0, home=tmp_path)

    assert json.loads(path.read_text(encoding="utf-8"))["phase"] == "exited"


def test_lifecycle_ledger_does_not_call_a_foreign_namespace_gateway_unclean(monkeypatch):
    """`phase=running` from another namespace is "cannot verify", not "exited UNCLEANLY"."""
    from gateway import lifecycle_ledger as ll

    _set_local_namespace(monkeypatch, _LIVE)
    assert ll._pid_is_sentinel_owner(1, 12345.0, 12345.0, _OTHER_NS) is True
    # Same namespace: the real probe decides, unchanged.
    assert ll._pid_is_sentinel_owner(2 ** 22 + 12345, 12345.0, 12345.0, _HOST_NS) is False


def test_lifecycle_ledger_records_the_namespace_it_claimed_in(tmp_path, monkeypatch):
    from gateway import lifecycle_ledger as ll

    record = {"phase": "running", "pid": os.getpid(), "start_time": 1.0, "pidns": _HOST_NS}
    path = tmp_path / "state" / "gateway.lifecycle.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    # Our own live process in our own namespace is still a live owner.
    assert ll.detect_unclean_exit(tmp_path) is None


def test_scoped_lock_from_another_namespace_is_not_stolen(monkeypatch):
    """A second gateway must not take a live bot token because PID 1 is not a gateway cmdline."""
    from gateway import status

    _set_local_namespace(monkeypatch, _LIVE)
    live_pid = 1  # the namespaced gateway's own number, which is the host's init out here
    foreign = {
        "pid": live_pid, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": status._get_process_start_time(live_pid), "pidns": _OTHER_NS,
    }
    # Host init's cmdline is not a gateway, so every pre-fix staleness signal fires.
    assert status._read_process_cmdline(live_pid) is not None
    assert status._scoped_lock_record_is_stale(foreign, live_pid) is False


def test_scoped_lock_from_our_namespace_still_reclaims_a_dead_owner(monkeypatch):
    from gateway import status

    _set_local_namespace(monkeypatch, _LIVE)
    dead = 2 ** 22 + 12345
    same_ns = {
        "pid": dead, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": status._get_process_start_time(dead), "pidns": _HOST_NS,
    }
    assert status._scoped_lock_record_is_stale(same_ns, dead) is True


def test_the_owner_namespace_can_still_reclaim_its_own_dead_lock(monkeypatch):
    """The recovery path the refusal does NOT break.

    A crashed namespaced gateway's lock is not reclaimable from outside (that is the
    point), but the unit's own restart reclaims it from inside, where the record is
    checkable. Without this, a SIGKILLed gateway would leak its bot token to every
    outside starter with no way back short of deleting the lock by hand.
    """
    from gateway import status

    _set_local_namespace(monkeypatch, _LIVE)
    dead = 2 ** 22 + 12345
    own_namespace_record = {
        "pid": dead, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": status._get_process_start_time(dead), "pidns": _HOST_NS,
    }
    assert pid_checkable_from(own_namespace_record["pidns"]) is True
    assert status._scoped_lock_record_is_stale(own_namespace_record, dead) is True


# ---------------------------------------------------------------------------
# Consumer 4: the --replace takeover, where a wrong PID means an irreversible signal
# ---------------------------------------------------------------------------


def test_takeover_refuses_to_signal_a_foreign_namespace_holder(monkeypatch, tmp_path):
    """`--replace` must never SIGTERM a PID that was issued in another namespace.

    With the recorded PID being 1, the whole corroboration chain — start time,
    gateway cmdline, the target home's own PID record — is read against the host's
    init. A signal is irreversible, so the namespace gate is checked before any of
    it rather than relying on init's command line happening to look wrong.

    Init's real cmdline does refuse this today, so the fixture gives the checks
    behind it everything they would need to say "yes": the owner is alive, its
    start time agrees, its command line is a gateway's, and the target home's own
    PID record corroborates it. Only the namespace disagrees — which is the whole
    argument, because that corroboration is all being read about the wrong process.
    """
    from gateway import status

    home = Path(str(tmp_path)).resolve()
    _set_local_namespace(monkeypatch, _LIVE)
    record = {
        "pid": 1, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": 987654, "pidns": _OTHER_NS, "hermes_home": str(home),
    }
    (home / "gateway.pid").write_text(json.dumps(record))
    # Every check downstream of the namespace gate is made to pass.
    monkeypatch.setattr(status, "_scoped_lock_owner_state", lambda pid, st: "same")
    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: "/usr/bin/hermes gateway run")
    # Without the namespace guard this validates and the takeover SIGTERMs PID 1.
    assert status._validated_scoped_lock_gateway_owner(record) is None


def test_takeover_still_signals_a_holder_in_our_own_namespace(monkeypatch, tmp_path):
    """The control: a same-namespace owner is validated exactly as before.

    The owner is a stubbed live process, because the validator deliberately reads
    the real command line — this test process' argv is the test runner, not a
    gateway, and must not be made to look like one.
    """
    from gateway import status

    home = Path(str(tmp_path)).resolve()
    _set_local_namespace(monkeypatch, _LIVE)
    owner_pid = 4242
    start = 987654
    record = {
        "pid": owner_pid, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": start, "pidns": _HOST_NS, "hermes_home": str(home),
    }
    (home / "gateway.pid").write_text(json.dumps(record))
    monkeypatch.setattr(status, "_scoped_lock_owner_state", lambda pid, st: "same")
    monkeypatch.setattr(
        status, "_read_process_cmdline", lambda pid: "/usr/bin/hermes gateway run"
    )
    owner = status._validated_scoped_lock_gateway_owner(record)
    assert owner is not None and owner[0] == owner_pid


def test_takeover_still_refuses_a_same_namespace_claim_it_cannot_corroborate(monkeypatch, tmp_path):
    """The #123081 guard does not weaken the existing fail-closed owner checks:
    a checkable record with no corroborating PID record in the target home is refused."""
    from gateway import status

    home = Path(str(tmp_path)).resolve()
    _set_local_namespace(monkeypatch, _LIVE)
    record = {
        "pid": 4242, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": 987654, "pidns": _HOST_NS, "hermes_home": str(home),
    }
    monkeypatch.setattr(status, "_scoped_lock_owner_state", lambda pid, st: "same")
    monkeypatch.setattr(
        status, "_read_process_cmdline", lambda pid: "/usr/bin/hermes gateway run"
    )
    # No gateway.pid written in home: the corroboration step must still refuse.
    assert status._validated_scoped_lock_gateway_owner(record) is None


# ---------------------------------------------------------------------------
# Scoped-lock OWNERSHIP: equal PIDs in different namespaces are two gateways
# ---------------------------------------------------------------------------


def _lock_record(**overrides):
    record = {
        "pid": 12345, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": 987654, "pidns": _HOST_NS, "scope": "telegram-bot-token",
        "identity_hash": "abc", "hermes_home": "/tmp/whatever", "metadata": {},
        "updated_at": "2026-09-25T00:00:00+00:00",
    }
    record.update(overrides)
    return record


def test_equal_pid_from_a_foreign_namespace_is_not_self(monkeypatch):
    """Two gateways in different PID namespaces can both be PID 1.

    Numeric PID equality is not ownership outside a shared namespace, so a
    foreign record carrying our own number is another process — never 'self'.
    """
    _set_local_namespace(monkeypatch, _LIVE)
    monkeypatch.setattr(os, "getpid", lambda: 12345)
    assert scoped_lock_owned_by_self(_lock_record()) is True
    assert scoped_lock_owned_by_self(_lock_record(pidns=_OTHER_NS)) is False


def test_acquire_refuses_a_foreign_record_carrying_our_own_pid(monkeypatch, tmp_path):
    """P1: acquisition must not overwrite a foreign owner's record as 'self'."""
    from gateway import status

    _set_local_namespace(monkeypatch, _LIVE)
    monkeypatch.setattr(os, "getpid", lambda: 12345)
    lock = tmp_path / "telegram-bot-token.lock"
    foreign = _lock_record(pidns=_OTHER_NS, pid=12345)
    lock.write_text(json.dumps(foreign))
    monkeypatch.setattr(status, "_get_scope_lock_path", lambda scope, identity: lock)
    monkeypatch.setattr(status, "_pid_exists", lambda pid: True)

    acquired, existing = status.acquire_scoped_lock("telegram-bot-token", "secret")
    assert acquired is False
    assert existing == foreign
    assert json.loads(lock.read_text()) == foreign, "the foreign owner's record was rewritten"


def test_release_leaves_a_foreign_record_carrying_our_own_pid(monkeypatch, tmp_path):
    """P1: a release from an equal-PID process in another namespace must not unlink."""
    from gateway import status

    _set_local_namespace(monkeypatch, _LIVE)
    monkeypatch.setattr(os, "getpid", lambda: 12345)
    lock = tmp_path / "telegram-bot-token.lock"
    foreign = _lock_record(pidns=_OTHER_NS, pid=12345)
    lock.write_text(json.dumps(foreign))
    monkeypatch.setattr(status, "_get_scope_lock_path", lambda scope, identity: lock)

    status.release_scoped_lock("telegram-bot-token", "secret")
    assert lock.exists(), "released another namespace's lock"
    assert json.loads(lock.read_text()) == foreign


def test_bulk_release_skips_a_foreign_namespace_lock(monkeypatch, tmp_path):
    """The owner-filtered sweep carries the same qualification, not just the single-record API."""
    from gateway import status

    _set_local_namespace(monkeypatch, _LIVE)
    monkeypatch.setattr(status, "_get_lock_dir", lambda: tmp_path)
    (tmp_path / "mine.lock").write_text(json.dumps(_lock_record(pid=4242, pidns=_HOST_NS)))
    (tmp_path / "theirs.lock").write_text(json.dumps(_lock_record(pid=4242, pidns=_OTHER_NS)))

    removed = status.release_all_scoped_locks(owner_pid=4242)
    assert removed == 1
    assert not (tmp_path / "mine.lock").exists()
    assert (tmp_path / "theirs.lock").exists(), "swept another namespace's lock"


def test_local_pid_absence_does_not_authorize_reclaiming_a_foreign_owner(monkeypatch):
    """P1: an absent local number says nothing about an owner in another namespace.

    The PID-1 test above cannot reach this: 1 is present in its fixture
    environment, so the local liveness probe succeeds and the guard is reached.
    Here the number is absent locally, which is the branch that used to answer
    "reclaimable" before the namespace was ever consulted.
    """
    from gateway import status

    _set_local_namespace(monkeypatch, _LIVE)
    foreign = _lock_record(pid=2 ** 22 + 12345, pidns=_OTHER_NS)
    probed = []

    def is_pid_alive(pid):
        probed.append(pid)
        return False

    assert status._scoped_lock_record_is_stale(foreign, 2 ** 22 + 12345) is False
    # The local probe must not even be consulted about a foreign owner.
    monkeypatch.setattr(status, "_pid_exists", is_pid_alive)
    assert status._scoped_lock_record_is_stale(foreign, 2 ** 22 + 12345) is False
    assert probed == []


def test_acquire_refuses_a_foreign_record_whose_number_is_absent_locally(monkeypatch, tmp_path):
    """P1: the consumer must leave such a record in place, not claim the credential."""
    from gateway import status

    _set_local_namespace(monkeypatch, _LIVE)
    lock = tmp_path / "telegram-bot-token.lock"
    foreign = _lock_record(pid=2 ** 22 + 12345, pidns=_OTHER_NS)
    lock.write_text(json.dumps(foreign))
    monkeypatch.setattr(status, "_get_scope_lock_path", lambda scope, identity: lock)
    monkeypatch.setattr(status, "_pid_exists", lambda pid: False)

    acquired, existing = status.acquire_scoped_lock("telegram-bot-token", "secret")
    assert acquired is False
    assert json.loads(lock.read_text()) == foreign


def test_same_namespace_owner_still_reacquires_and_releases(monkeypatch, tmp_path):
    """The control: a real same-namespace owner reconnects, including a null start_time (#81468)."""
    from gateway import status

    _set_local_namespace(monkeypatch, _LIVE)
    monkeypatch.setattr(os, "getpid", lambda: 12345)
    # The freshly built record needs a resolvable start time for our stub PID.
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 987654)
    lock = tmp_path / "telegram-bot-token.lock"
    ours = _lock_record(start_time=None)
    lock.write_text(json.dumps(ours))
    monkeypatch.setattr(status, "_get_scope_lock_path", lambda scope, identity: lock)

    acquired, _ = status.acquire_scoped_lock("telegram-bot-token", "secret")
    assert acquired is True, "a null start_time must not break our own reconnect"
    assert json.loads(lock.read_text())["start_time"] == 987654  # refreshed

    lock.write_text(json.dumps(ours))
    status.release_scoped_lock("telegram-bot-token", "secret")
    assert not lock.exists()


def test_legacy_unstamped_record_keeps_main_behavior(monkeypatch, tmp_path):
    """A record written before the stamp is not retroactively foreign: our own PID is still self,
    and a dead one is still reclaimable. Refusing these would strand every pre-upgrade lock."""
    _set_local_namespace(monkeypatch, _LIVE)
    monkeypatch.setattr(os, "getpid", lambda: 12345)
    legacy = _lock_record(pid=12345)
    legacy.pop("pidns")
    assert scoped_lock_owned_by_self(legacy) is True

    # Liveness, not identity: a checkable record still follows the local probe exactly.
    dead = _lock_record(pid=2 ** 22 + 12345)
    dead.pop("pidns")
    assert scoped_lock_stale_locally(dead, lambda pid: False) is True
    assert scoped_lock_stale_locally(legacy, lambda pid: True) is False
    assert scoped_lock_owned_by_self(dead) is False  # a dead other's PID is never self


def test_unresolved_local_namespace_refuses_to_claim_ownership(monkeypatch):
    """When our own namespace cannot be resolved we have no authority over any number."""
    _set_local_namespace(monkeypatch, _UNKNOWN)
    monkeypatch.setattr(os, "getpid", lambda: 12345)
    assert scoped_lock_owned_by_self(_lock_record()) is False
    assert scoped_lock_stale_locally(_lock_record(), lambda pid: False) is False


def test_platform_without_namespaces_keeps_numeric_pid_behavior(monkeypatch):
    """macOS/Windows: one namespace, so a bare PID still identifies its owner."""
    _set_local_namespace(monkeypatch, _NONE)
    monkeypatch.setattr(os, "getpid", lambda: 12345)
    unstamped = _lock_record()
    unstamped.pop("pidns")
    assert scoped_lock_owned_by_self(unstamped) is True
    assert scoped_lock_stale_locally(unstamped, lambda pid: False) is True
    assert scoped_lock_stale_locally(unstamped, lambda pid: True) is False
