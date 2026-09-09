"""Per-session exclusivity in the active-session registry.

The property under test is a correctness one, and it is separate from capacity:
AT MOST ONE LIVE OWNER MAY RUN A GIVEN STORED SESSION, whether or not an
operator configured ``max_concurrent_sessions``.

It matters because a second owner does not merely duplicate work. It loads its
own snapshot of the transcript, reasons from a history that does not include the
first owner's in-flight turn, and then appends to the same stored session -- so
the conversation ends up containing two replies that never saw each other. That
was observed in practice before this fence existed: two gateway processes
resumed one stored session, neither was refused, and both wrote to it.

The old behaviour is worth stating precisely, because it looked deliberate: with
no cap configured, ``try_acquire_active_session`` returned a disabled no-op lease
and never touched the registry at all. Capacity being unconfigured was silently
treated as "concurrent writers to one session are fine", which it never is.
"""

import itertools
import os

import pytest

from hermes_cli.active_sessions import (
    MAX_CONCURRENT_SESSIONS,
    PER_SESSION_EXCLUSIVE_SUBMIT,
    SESSION_NOT_OWNED,
    active_session_registry_snapshot,
    release_active_session,
    try_acquire_active_session,
)


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))


_owner_seq = itertools.count()


def acquire(session_id, config=None, surface="tui", live_id=None):
    """Acquire as a DISTINCT owner unless a live id is given explicitly.

    The live session id is half of writer identity, so a helper that reused one
    would model every acquisition as the same tab re-acquiring its own session --
    which is re-entrancy, not the concurrent-writer case these tests are about.
    Distinct by default; shared only where a test means it.
    """
    return try_acquire_active_session(
        session_id=session_id,
        surface=surface,
        config=config if config is not None else {},
        metadata={"live_session_id": live_id or f"live-{next(_owner_seq)}"},
    )


def test_no_cap_configured_still_fences_one_session():
    """The case the old code got wrong, and the reason this fence exists.

    ``max_concurrent_sessions`` unset must not mean "anyone may write to any
    session at any time".
    """
    lease_a, refused = acquire("S")
    assert lease_a is not None and refused is None

    lease_b, refusal = acquire("S")
    assert lease_b is None, "a second live owner of one session must be refused"
    assert refusal.reason == SESSION_NOT_OWNED
    # The message is for a person; the reason is the contract. Both are present.
    assert "S" in str(refusal)

    # And exactly one holder is recorded -- a refusal must not leave a slot behind.
    assert len(active_session_registry_snapshot()) == 1


def test_different_sessions_still_run_concurrently():
    """Exclusivity is PER SESSION. It is not a global mutex.

    Getting this wrong would be worse than the bug: it would serialise every
    conversation on the machine behind whichever one started first.
    """
    lease_1, refused_1 = acquire("S1")
    lease_2, refused_2 = acquire("S2")
    assert lease_1 is not None and refused_1 is None
    assert lease_2 is not None and refused_2 is None
    assert len(active_session_registry_snapshot()) == 2


def test_global_capacity_still_applies_independently():
    """The capacity policy is untouched, and refuses for its own reason."""
    config = {"max_concurrent_sessions": 2}
    assert acquire("S1", config)[0] is not None
    assert acquire("S2", config)[0] is not None

    lease, refusal = acquire("S3", config)
    assert lease is None
    assert refusal.reason == MAX_CONCURRENT_SESSIONS, (
        "a capacity refusal must not be reported as an ownership refusal: a client "
        "retries one and must not retry the other the same way"
    )
    assert "active session limit (2/2)" in str(refusal)


def test_capacity_and_exclusivity_are_not_the_same_switch():
    """Both refusals exist under a configured cap, and say different things."""
    config = {"max_concurrent_sessions": 4}
    assert acquire("S", config)[0] is not None
    lease, refusal = acquire("S", config)
    assert lease is None
    assert refusal.reason == SESSION_NOT_OWNED, (
        "with capacity to spare, the refusal can only be about ownership"
    )


def test_a_dead_owner_is_pruned_and_a_successor_may_acquire():
    """A crashed owner must not hold a session hostage forever.

    Written as a registry entry for a pid that cannot exist, because that is what
    a crashed process leaves behind -- the pruning path is what makes the fence
    survivable rather than a way to lock yourself out permanently.
    """
    lease, _ = acquire("S")
    assert lease is not None

    # Rewrite the holder as a process that is gone.
    from hermes_cli.active_sessions import _read_entries, _state_path, _write_entries

    entries = _read_entries(_state_path())
    assert len(entries) == 1
    entries[0]["pid"] = 0x7FFFFFFE
    entries[0]["process_start_time"] = 1.0
    _write_entries(_state_path(), entries)

    successor, refusal = acquire("S")
    assert successor is not None, f"a dead owner must not block a successor: {refusal}"
    assert len(active_session_registry_snapshot()) == 1


def test_a_recycled_pid_does_not_keep_a_lease_alive():
    """Identity is (pid, process start time), not a pid.

    A pid alone is not identity -- the number is reused, and on a busy machine it
    is reused quickly. An entry claiming OUR pid but a start time we never had is
    a dead owner whose number was handed to somebody else.
    """
    lease, _ = acquire("S")
    assert lease is not None

    from hermes_cli.active_sessions import _read_entries, _state_path, _write_entries

    entries = _read_entries(_state_path())
    entries[0]["pid"] = os.getpid()
    entries[0]["process_start_time"] = 1.0  # not when this process started
    _write_entries(_state_path(), entries)

    successor, refusal = acquire("S")
    assert successor is not None, f"a recycled pid must not hold a session: {refusal}"


def test_release_lets_the_next_owner_in():
    """The ordinary handoff: A finishes, B proceeds."""
    lease_a, _ = acquire("S")
    assert lease_a is not None
    assert acquire("S")[0] is None

    release_active_session(lease_a)
    assert active_session_registry_snapshot() == []

    lease_b, refusal = acquire("S")
    assert lease_b is not None, f"after release the session must be acquirable: {refusal}"


def test_release_is_idempotent_and_only_drops_its_own_lease():
    """Releasing twice must not free somebody else's session."""
    lease_a, _ = acquire("S1")
    lease_b, _ = acquire("S2")
    release_active_session(lease_a)
    release_active_session(lease_a)
    held = {entry.get("session_id") for entry in active_session_registry_snapshot()}
    assert held == {"S2"}
    assert lease_b is not None


def test_a_session_with_no_stored_id_is_exempt():
    """An unsaved draft has no identity, and must not exclude every other one.

    Treating "" as a session id would make the first unsaved composer on the
    machine refuse every other one -- a fence that fires on sessions that cannot
    collide by construction.
    """
    assert acquire("")[0] is not None
    assert acquire("")[0] is not None, "empty ids do not collide with each other"


def test_the_same_live_session_may_re_acquire_its_own_lease():
    """Re-entrancy, and the reason it is not a hole in the fence.

    A live session whose record was rebuilt in place loses its reference to the
    lease it already holds. Without this it would be fenced out of its own
    session by its own leak -- permanently, because pruning only removes entries
    whose PROCESS is dead and this one is alive.

    Identity is (pid, live session id): another process differs by pid, another
    tab in this process differs by live id. Only the same writer matches.
    """
    lease_a, _ = acquire("S", live_id="tab-1")
    assert lease_a is not None

    again, refusal = acquire("S", live_id="tab-1")
    assert again is not None, f"a writer must not be fenced out by its own leak: {refusal}"
    assert len(active_session_registry_snapshot()) == 1, "and it must not double-book"

    # A DIFFERENT live session in the same process is still a second writer: each
    # holds its own snapshot of the transcript, so the hazard is unchanged.
    other, refusal = acquire("S", live_id="tab-2")
    assert other is None
    assert refusal.reason == SESSION_NOT_OWNED


def test_the_capability_is_advertised_because_the_check_exists():
    """The flag lives beside the enforcement, so it cannot drift from it."""
    assert PER_SESSION_EXCLUSIVE_SUBMIT is True
