"""Builder-declared stable prefixes for Anthropic prompt caching (#81867).

Skill, webhook, and cron builders concatenate a large static scaffold
(activation note + expanded skill body) with a small volatile invocation
tail (ticket payload, timestamps, run context) into one user-message
string. Only the builder knows the exact byte where the volatile tail
begins, so it registers the stable prefix here at construction time; the
cache planner consults the registry to place a cache breakpoint at that
boundary instead of caching the whole message as one atomic block.

This deliberately avoids re-parsing scaffold marker strings out of the
message at request time: markers can legitimately appear inside skill
bodies or inside event payloads (e.g. a helpdesk ticket quoting an agent
transcript), and any delimiter-search heuristic then either shrinks the
cached prefix or — worse — silently absorbs volatile bytes into it,
reintroducing the per-invocation cache miss this exists to fix.

The registry is process-local by design. A freshly fired webhook/cron
invocation is always built and sent by the same process, which is the
only window where the split pays off. Any miss (restart, eviction,
historic message) falls back to the pre-existing whole-message policy.

Split-shape lifetime: the split is applied only while the skill message is
one of the plan's marked endpoints (the last few cacheable messages). Once
later turns rotate it out of that window it ships as a single string block
again, which changes the block boundary once and re-ingests the prefix from
that message onward exactly one time in a long-lived session. Webhook/cron
invocations — the workload this exists for — send the skill turn as the
newest message every time, so they always hit the split shape; the one-time
re-ingest only affects long interactive sessions and nets out far below the
per-invocation full rewrite this removes.
"""

import threading
from collections import OrderedDict
from typing import Optional

# A couple dozen distinct active scaffolds (webhook routes x skills x cron
# jobs) is generous for one gateway process; beyond that, oldest entries
# fall back to whole-message caching rather than growing unboundedly.
_MAX_ENTRIES = 32

# Entries hold whole expanded skill bodies, so an entry count alone does not
# bound memory — a handful of large skills can retain tens of MB in a
# long-lived gateway process. Evict by total retained characters too (a
# conservative proxy for bytes: actual memory is 1–4x depending on the
# string's widest code point), always keeping the newest entry so a single
# oversized scaffold still gets a boundary instead of silently disabling
# the split.
_MAX_CHARS = 4 * 1024 * 1024

_lock = threading.Lock()
_prefixes: "OrderedDict[str, None]" = OrderedDict()


def register_stable_prefix(prefix: str) -> None:
    """Record ``prefix`` as the stable scaffold of a just-built message."""
    if not prefix:
        return
    with _lock:
        _prefixes[prefix] = None
        _prefixes.move_to_end(prefix)
        while len(_prefixes) > _MAX_ENTRIES:
            _prefixes.popitem(last=False)
        while len(_prefixes) > 1 and sum(map(len, _prefixes)) > _MAX_CHARS:
            _prefixes.popitem(last=False)


def find_stable_prefix(content: str) -> Optional[str]:
    """Longest registered prefix that is a *proper* prefix of ``content`` with non-whitespace tail.

    Proper with non-whitespace tail (``bool(content[len(prefix):].strip())``) so the
    split never produces an empty or whitespace-only volatile text block, which
    Anthropic rejects on the wire (HTTP 400).

    A hit refreshes the entry's LRU position: a scaffold fired every minute
    by cron must not be evicted by a burst of one-off skill invocations,
    which would silently drop it back to whole-message caching.
    """
    with _lock:
        best: Optional[str] = None
        for prefix in _prefixes:
            if content.startswith(prefix) and bool(content[len(prefix):].strip()):
                if best is None or len(prefix) > len(best):
                    best = prefix
        if best is not None:
            # After the scan so the OrderedDict is never mutated mid-iteration.
            _prefixes.move_to_end(best)
        return best


def clear_stable_prefixes() -> None:
    """Test isolation helper."""
    with _lock:
        _prefixes.clear()
