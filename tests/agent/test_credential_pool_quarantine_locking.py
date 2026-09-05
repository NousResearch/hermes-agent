"""Codex/nous quarantine paths must mutate self._entries under the lock.

Post-merge gate-sweep finding on the #71775 salvage (#77714). That PR moved
single-use-token refreshes OUTSIDE the pool lock to avoid stalling every
consumer during cross-process flock + OAuth network I/O — correct in intent,
but ``_refresh_entry_impl``'s three "terminal auth failure" quarantine paths
do a bare read-modify-write of ``self._entries``:

    removed_ids = [item.id for item in self._entries if ...]
    self._entries = [item for item in self._entries if ...]

Before #71775 those ran with the caller (``_available_entries``) holding the
lock. On the deferred path they now run unlocked, so a concurrent mutation
interleaved between the read and the write is silently lost.
"""

import threading

from agent.credential_pool import CredentialPool, PooledCredential


def _entry(entry_id: str, source: str) -> PooledCredential:
    return PooledCredential(
        id=entry_id,
        provider="anthropic",
        auth_type="oauth",
        access_token="tok",
        label=entry_id,
        source=source,
        priority=0,
    )


def _bare_pool(entries):
    pool = CredentialPool.__new__(CredentialPool)
    pool._lock = threading.RLock()
    pool._entries = list(entries)
    pool._active_leases = {}
    pool._current_id = None
    pool._max_concurrent = 2
    pool._unmatched_rotation_streak = 0
    pool.provider = "anthropic"
    return pool


def test_quarantine_read_modify_write_is_atomic():
    """A concurrent mutation must not be lost across the quarantine filter.

    The quarantine reads the surviving entries, then writes back a filtered
    list. If a concurrent writer lands between the read and the write and the
    section is unlocked, that write is clobbered. Under the lock the writer is
    serialized — it either lands fully before or fully after.
    """
    pool = _bare_pool([_entry("dc1", "device_code")])
    survivor = _entry("keep", "manual")
    started = threading.Event()

    def concurrent_add():
        started.set()
        with pool._lock:  # blocks while the quarantine holds the lock
            pool._entries = pool._entries + [survivor]

    t = threading.Thread(target=concurrent_add)

    with pool._lock:
        _removed = [i.id for i in pool._entries if i.source == "device_code"]
        t.start()
        started.wait(timeout=2)
        # Give the writer a chance to (incorrectly) interleave.
        t.join(timeout=0.2)
        pool._entries = [i for i in pool._entries if i.source != "device_code"]

    # Outside the lock the writer can now proceed; wait for it to finish.
    t.join(timeout=2)
    assert not t.is_alive(), "concurrent writer did not complete"

    ids = {e.id for e in pool._entries}
    assert "dc1" not in ids, "the device_code entry should be quarantined"
    assert "keep" in ids, (
        "the concurrent append was LOST — the quarantine read-modify-write "
        "of self._entries is not atomic"
    )


def test_quarantine_paths_hold_the_pool_lock():
    """Static guard: every bare ``self._entries = [`` inside
    _refresh_entry_impl must sit under a ``with self._lock`` block.

    The deferred-refresh call site runs outside the pool lock, so an
    unguarded rebind there is a lost-update window.
    """
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(CredentialPool._refresh_entry_impl))
    lines = src.splitlines()

    unguarded = []
    for idx, line in enumerate(lines):
        if "self._entries = [" not in line:
            continue
        indent = len(line) - len(line.lstrip())
        # Walk backwards for an enclosing `with self._lock` at lower indent.
        guarded = False
        for prev in range(idx - 1, -1, -1):
            p = lines[prev]
            if not p.strip():
                continue
            p_indent = len(p) - len(p.lstrip())
            if p_indent < indent:
                if "with self._lock" in p:
                    guarded = True
                    break
                if p.lstrip().startswith("def "):
                    break
        if not guarded:
            unguarded.append(line.strip())

    assert not unguarded, (
        "unguarded self._entries rebind(s) in _refresh_entry_impl — the "
        f"deferred refresh path runs outside the pool lock: {unguarded}"
    )


def test_rlock_allows_locked_callers_to_reenter():
    """The already-locked callers must still work after adding the lock.

    self._lock is an RLock, so a caller holding it can re-enter the new
    quarantine block without deadlocking.
    """
    pool = _bare_pool([_entry("dc1", "device_code")])

    with pool._lock:
        acquired = pool._lock.acquire(timeout=1)
        assert acquired, "RLock must allow same-thread re-entry"
        pool._lock.release()
