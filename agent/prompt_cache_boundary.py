"""Builder-declared stable prefixes for Anthropic prompt caching.

Skill/webhook/cron builders concatenate a large static scaffold with a small volatile
tail into one user-message string. Only the builder knows where the tail begins, so it
registers the stable prefix here and the cache planner places a breakpoint at that
boundary. Re-parsing marker strings at request time is deliberately avoided: markers
can legitimately appear inside skill bodies, and any delimiter heuristic then shrinks
the cached prefix or absorbs volatile bytes. Process-local by design: any miss falls
back to the whole-message policy.
"""

import threading
from collections import OrderedDict
from typing import Optional

# A couple dozen active scaffolds is generous for one gateway process.
_MAX_ENTRIES = 32
# Entries hold whole expanded skill bodies, so also bound total retained chars.
# The newest entry is always kept so one oversized scaffold still gets a boundary.
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
    """Longest registered *proper* prefix of ``content`` with a non-whitespace tail.

    The tail must be non-whitespace so the split never yields an empty text
    block (Anthropic rejects it with HTTP 400). A hit refreshes the entry's LRU
    position so a scaffold fired every minute by cron is not evicted by a
    burst of one-off skill invocations.
    """
    with _lock:
        best: Optional[str] = None
        for prefix in _prefixes:
            if (
                content.startswith(prefix)
                and content[len(prefix):].strip()
                and (best is None or len(prefix) > len(best))
            ):
                best = prefix
        if best is not None:
            _prefixes.move_to_end(best)  # after the scan: never mutate mid-iteration
        return best


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def clear_stable_prefixes() -> None:
    """Test isolation helper."""
    with _lock:
        _prefixes.clear()
# ---- END PLUGIN-COMPAT ----
