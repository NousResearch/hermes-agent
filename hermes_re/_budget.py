"""The compile cache's store: an LRU bounded by entry count AND by the size of what it holds.

An entry is any object with a ``_cost`` attribute: the estimated bytes of the RE2 programs built
for it so far (0 until first use). Entries leave oldest-first when there are more than
``MAX_ENTRIES`` of them or when their costs add up to more than ``MAX_BYTES``. Leaving the cache
frees a pattern only if nothing else holds it: a compiled pattern that a module keeps stays
built and usable, it is just no longer the cache's to count (as with stdlib's cache).

No lock: every step is one atomic container operation. Two threads may both compile a missing
key (the later insert wins) and the running total can drift when they interleave; it is
recomputed from the entries each time the limit is hit. So there is no lock for a forked child
to inherit.
"""
from __future__ import annotations

from collections import OrderedDict

MAX_ENTRIES = 4096    # distinct (pattern type, pattern, flags) keys; the repository has about 2,300
MAX_BYTES = 256 << 20  # estimated bytes of built programs held through the cache
entries = OrderedDict()  # key -> entry, least recently used first
_total = 0  # running sum of the entries' costs


def get(key):
    """The entry for ``key``, now the most recently used, or ``None``."""
    try:
        found = entries[key]
        entries.move_to_end(key)
    except KeyError:
        return None
    return found


def put(key, entry) -> None:
    global _total
    old = entries.get(key)
    entries[key] = entry
    _total += entry._cost - (old._cost if old is not None else 0)
    while len(entries) > MAX_ENTRIES:
        try:
            _total -= entries.popitem(last=False)[1]._cost
        except KeyError:  # another thread emptied the cache
            break
    if _total > MAX_BYTES:
        _shrink(entry)


def grew(key, entry, added: int) -> None:
    """``entry`` (cached under ``key`` when it was created) now costs ``added`` bytes more."""
    global _total
    if entries.get(key) is entry:
        _total += added
        if _total > MAX_BYTES:
            _shrink(entry)


def _shrink(keep) -> None:
    """Drop least recently used entries until the costs fit ``MAX_BYTES``. ``keep``, the entry
    being used right now, stays: dropping it would only make its next use compile it again."""
    global _total
    try:
        held = list(entries.items())
    except RuntimeError:  # another thread changed the cache meanwhile; the next growth retries
        return
    total = sum(entry._cost for _, entry in held)
    for key, entry in held:
        if total <= MAX_BYTES:
            break
        if entry is not keep and entries.pop(key, None) is not None:
            total -= entry._cost
    _total = total


def held() -> tuple:
    """``(number of entries, estimated bytes of their built programs)``."""
    return len(entries), sum(entry._cost for entry in list(entries.values()))


def recount() -> None:
    """Bring the running total back in line with the entries' costs."""
    global _total
    _total = held()[1]


def clear() -> None:
    global _total
    entries.clear()
    _total = 0
