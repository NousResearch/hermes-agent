"""Storage and release for the per-message token-estimate memo (``agent.model_metadata``).

Each entry pins every string its fingerprint holds (the pin keeps the ``id()`` key sound), so the memo is
bounded two ways, both least-recently-used first so a conversation that is still running never pays:

* a retained-bytes budget: pinned string bytes plus the fingerprint structure around them. A real-shaped
  1M-token history retains about 9 MB, so the default keeps several maxed conversations hot at once while
  the dead sessions of a long-lived process age out;
* an entry count, as a backstop for the bytes estimate.

``close()``, compaction and the compressor telemetry also drop entries they know are dead
(``forget_message_token_memo``). Eviction never changes an estimate: a later lookup recomputes the same value.
"""

from __future__ import annotations

import sys
import threading
from collections import OrderedDict
from typing import Any

# fingerprint -> (pins, text tokens, image count). Order is recency: a hit moves its entry to the end, and an
# OrderedDict evicts from the front in O(1) where a plain dict rescans its deleted slots on every eviction.
_MSG_TOKENS_CACHE: OrderedDict[Any, tuple[list, int, int]] = OrderedDict()
_MSG_TOKENS_CACHE_MAX = 65536
_MSG_TOKENS_CACHE_MAX_BYTES = 64 * 1024 * 1024
# Measured on real histories: the fingerprint tuple, its id int and the pin slot cost ~120 B per string leaf.
_LEAF_OVERHEAD_BYTES = 120
_lock = threading.Lock()
_retained_bytes = 0


def _entry_bytes(entry: tuple[list, int, int]) -> int:
    pins = entry[0]
    return sum(map(sys.getsizeof, pins)) + _LEAF_OVERHEAD_BYTES * len(pins)


def remember_message_tokens(key: Any, entry: tuple[list, int, int]) -> None:
    """Insert ``entry`` as most recent, then evict least-recently-used entries past either bound."""
    global _retained_bytes
    size = _entry_bytes(entry)
    cache = _MSG_TOKENS_CACHE
    with _lock:
        if not cache:  # an emptied memo (tests clear it) restarts the count
            _retained_bytes = 0
        previous = cache.pop(key, None)
        if previous is not None:
            _retained_bytes -= _entry_bytes(previous)
        cache[key] = entry
        _retained_bytes += size
        while len(cache) > 1 and (len(cache) > _MSG_TOKENS_CACHE_MAX or _retained_bytes > _MSG_TOKENS_CACHE_MAX_BYTES):
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
