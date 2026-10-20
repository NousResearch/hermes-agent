from __future__ import annotations

import re
from typing import Optional

RESERVED_SIGNAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"\bADR[-_ ]?\d{3,4}\b",
    r"\bHARD[-\s]?STOP\b",
    r"\breserved\b",
    r"\bruleset\s+\d+\b",
    r"\btelegram\b",
    r"\bmessage_id\b",
    r"\bdocs/\S+",
    r"\bBurak'?s?\b",
)]


def reserved_item_signal(reason: Optional[str]) -> Optional[str]:
    """The first reserved-item phrase found in a block reason, or ``None``.

    A reserved block (a merge-gate ruleset flip, a founder-only spend, an
    irreversible act — see ``~/AGENTS.md``) is *correctly* re-blocked on
    unchanged facts every time a worker re-verifies it: that repetition is
    what "waiting on the owner" means, not a spin loop. Both the backlog
    drain and the block-loop detector classify a reason through this same
    function so a reserved block is never mistaken for one.
    """
    if not reason:
        return None
    for pattern in RESERVED_SIGNAL_PATTERNS:
        match = pattern.search(reason)
        if match:
            return match.group(0)
    return None
