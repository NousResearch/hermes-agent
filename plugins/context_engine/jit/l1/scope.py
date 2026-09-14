"""Scope Resolver with Hysteresis for Hermes JIT Context Engine."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

EXPLICIT_SCOPE_PATTERNS = [
    r"(?:project|scope|switching to|switch to|switch project to|working on|open project):\s*([a-zA-Z0-9_\-]+)",
    r"(?:switch to project|work on project|open project|project:?)\s+([a-zA-Z0-9_\-]+)",
]


def resolve_scope(
    message: str,
    current_active_scope: str,
    pending_candidate_scope: Optional[str] = None,
    candidate_turns_count: int = 0,
) -> Tuple[str, List[str], Optional[str], int]:
    """Resolve active_scope and retrieval_scopes with hysteresis."""
    unquoted = re.sub(r"<ONA_CONTEXT[\s\S]*?</ONA_CONTEXT>", "", message)
    effective_text = unquoted if unquoted.strip() else message
    text_lower = effective_text.lower()

    for pat in EXPLICIT_SCOPE_PATTERNS:
        m = re.search(pat, text_lower)
        if m:
            cand = m.group(1).strip()
            if len(cand) >= 2:
                return cand, [cand], None, 0

    return current_active_scope, [current_active_scope], None, 0
