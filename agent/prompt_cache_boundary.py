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
"""

import threading
from collections import OrderedDict
from typing import Optional

# A couple dozen distinct active scaffolds (webhook routes x skills x cron
# jobs) is generous for one gateway process; beyond that, oldest entries
# fall back to whole-message caching rather than growing unboundedly.
_MAX_ENTRIES = 32

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


def find_stable_prefix(content: str) -> Optional[str]:
    """Longest registered prefix that is a *proper* prefix of ``content``.

    Proper (``len(content) > len(prefix)``) so the split never produces an
    empty volatile text block, which Anthropic rejects on the wire.
    """
    with _lock:
        candidates = list(_prefixes)
    best: Optional[str] = None
    for prefix in candidates:
        if len(content) > len(prefix) and content.startswith(prefix):
            if best is None or len(prefix) > len(best):
                best = prefix
    return best


def is_registered_stable_prefix(text: str) -> bool:
    """Exact-match check used when flattening a decorated split back."""
    with _lock:
        return text in _prefixes


def clear_stable_prefixes() -> None:
    """Test isolation helper."""
    with _lock:
        _prefixes.clear()
