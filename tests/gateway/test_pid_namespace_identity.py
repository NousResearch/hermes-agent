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

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_platform.host import pid_namespace as pns
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

    ``scoped_lock_identity`` and ``status`` import ``local_pid_namespace`` by name,
    so patching only the defining module would leave them resolving the real
    namespace and the test would pass or fail for the wrong reason.
    """
    for module in (pns, sli):
        monkeypatch.setattr(module, "local_pid_namespace", lambda: namespace)


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

    pns._local_pid_namespace_cached.cache_clear()
    monkeypatch.setattr(pns.os, "readlink", failing_readlink)
    try:
        assert local_pid_namespace().known is False
        after_first = reads["n"]
        assert local_pid_namespace().known is False
        assert reads["n"] > after_first, "the failure was cached instead of retried"
    finally:
        pns._local_pid_namespace_cached.cache_clear()


def test_a_definite_answer_is_cached(monkeypatch):
    """The resolved namespace cannot change under a live process, so it is read once."""
    reads = {"n": 0}

    def counting_readlink(path):
        reads["n"] += 1
        return f"pid:[{_HOST_NS}]"

    pns._local_pid_namespace_cached.cache_clear()
    monkeypatch.setattr(pns.os, "readlink", counting_readlink)
    try:
        first = local_pid_namespace()
        assert first.id == _HOST_NS
        local_pid_namespace()
        assert reads["n"] == 1
    finally:
        pns._local_pid_namespace_cached.cache_clear()


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
    """Absence of provenance cannot become provenance because the read failed."""
    _set_local_namespace(monkeypatch, _UNKNOWN)
    assert pid_checkable_from(_HOST_NS) is False
    assert pid_checkable_from(None) is False


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


def test_pid_record_stamps_the_pid_namespace(tmp_path, monkeypatch):
    from gateway import status

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record = status._build_pid_record()
    assert record["pidns"] == local_pid_namespace().id

    # The stamp reaches the file the next process reads.
    status.write_pid_file()
    payload = json.loads((tmp_path / "gateway.pid").read_text())
    assert payload["pidns"] == local_pid_namespace().id


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
# Consumer 3: the lifecycle ledger verdict and the scoped bot-token lock
# ---------------------------------------------------------------------------


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
