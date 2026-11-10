"""Storage and release for the per-message token-estimate memo (``agent.model_metadata``).

Each entry pins every string its fingerprint holds (the pin keeps the ``id()`` key sound), so the memo is
bounded two ways, both least-recently-used first so a conversation that is still running never pays:

* a retained-bytes budget: pinned string bytes plus the fingerprint structure around them;
* an entry count, the same 4,096 the memo always had.

Both bounds only govern IDLE entries (below), so they are set to hold no more than main's 4,096-entry memo
ever did: what a running conversation needs is protected by the active window, never by the budget.

Neither bound ever evicts an entry looked up in the last ``_ACTIVE_WINDOW_S`` seconds: a conversation larger
than the budget (image-heavy history) overshoots it while it runs instead of missing on every message, and the
overshoot is shed by the first insert after it goes idle. The protected entries pin strings the running
conversations hold anyway, so the overshoot costs only fingerprint structure.

``close()``, compaction and the compressor telemetry also drop entries they know are dead
(``forget_message_token_memo``). Eviction never changes an estimate: a later lookup recomputes the same value.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import OrderedDict
from typing import Any

# fingerprint -> (pins, text tokens, image count). Order is recency: a hit moves its entry to the end, and an
# OrderedDict evicts from the front in O(1) where a plain dict rescans its deleted slots on every eviction.
_MSG_TOKENS_CACHE: OrderedDict[Any, tuple[list, int, int]] = OrderedDict()
_MSG_TOKENS_CACHE_MAX = 4096
_MSG_TOKENS_CACHE_MAX_BYTES = 16 * 1024 * 1024
# Measured on real histories: the fingerprint tuple, its id int and the pin slot cost ~120 B per string leaf.
_LEAF_OVERHEAD_BYTES = 120
# An entry looked up this recently is never evicted (see module docstring).
_ACTIVE_WINDOW_S = 60.0
_clock = time.monotonic
_lock = threading.Lock()
_retained_bytes = 0


class MemoPins(list):
    """An entry's pinned strings, stamped with the clock time of the entry's last lookup."""

    __slots__ = ("touched",)


def lookup_message_tokens(key: Any) -> tuple[list, int, int] | None:
    """Return the entry for ``key`` (or None), marking it most recently used.

    The stamp and reorder share eviction's lock: a hit landing between the evictor's idle check of the
    oldest entry and its pop would otherwise move that entry to the end, so the pop took a different,
    active one."""
    with _lock:
        entry = _MSG_TOKENS_CACHE.get(key)
        if entry is not None:
            entry[0].touched = _clock()
            _MSG_TOKENS_CACHE.move_to_end(key)
    return entry


def _entry_bytes(entry: tuple[list, int, int]) -> int:
    pins = entry[0]
    return sum(map(sys.getsizeof, pins)) + _LEAF_OVERHEAD_BYTES * len(pins)


def remember_message_tokens(key: Any, entry: tuple[MemoPins, int, int]) -> None:
    """Insert ``entry`` as most recent, then evict least-recently-used entries past either bound, stopping at
    the first one looked up inside the active window (everything after it is more recent)."""
    global _retained_bytes
    size = _entry_bytes(entry)
    if size > _MSG_TOKENS_CACHE_MAX_BYTES:
        return  # one message larger than the whole budget is recomputed, never allowed to flush the memo
    cache = _MSG_TOKENS_CACHE
    now = _clock()
    entry[0].touched = now
    idle_before = now - _ACTIVE_WINDOW_S
    with _lock:
        if not cache:  # an emptied memo (tests clear it) restarts the count
            _retained_bytes = 0
        previous = cache.pop(key, None)
        if previous is not None:
            _retained_bytes -= _entry_bytes(previous)
        cache[key] = entry
        _retained_bytes += size
        while len(cache) > 1 and (len(cache) > _MSG_TOKENS_CACHE_MAX or _retained_bytes > _MSG_TOKENS_CACHE_MAX_BYTES):
            if next(iter(cache.values()))[0].touched >= idle_before:
                break
            _retained_bytes -= _entry_bytes(cache.popitem(last=False)[1])


def retained_bytes() -> int:
    """Current retained-bytes estimate of the memo (what the budget is compared against)."""
    return _retained_bytes


def _long_string_ids(messages: Any) -> set[int]:
    ids: set[int] = set()
    stack = [messages]
    while stack:
        value = stack.pop()
        if type(value) is dict:
            stack.extend(value.values())
        elif type(value) is list or type(value) is tuple:
            stack.extend(value)
        elif type(value) is str and len(value) > 64:  # short values ("assistant", tool names) are shared
            ids.add(id(value))
    return ids


def forget_message_token_memo(messages: Any, *, keep: Any = None) -> int:
    """Evict memo entries pinning any long string value of ``messages`` that ``keep`` does not also hold;
    return how many.

    Eviction never changes an estimate: a later lookup recomputes the same value.
    """
    global _retained_bytes
    owned = _long_string_ids(messages)
    if owned and keep is not None:
        owned -= _long_string_ids(keep)
    if not owned:
        return 0
    with _lock:
        stale = [key for key, (pins, _text, _images) in list(_MSG_TOKENS_CACHE.items())
                 if any(id(pin) in owned for pin in pins)]
        for key in stale:
            _retained_bytes -= _entry_bytes(_MSG_TOKENS_CACHE.pop(key))
    return len(stale)
