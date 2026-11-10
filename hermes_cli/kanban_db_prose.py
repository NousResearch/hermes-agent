"""Prose scan for ``t_<hex>`` card references in a completion's summary/result.

The scan feeds the advisory ``suspected_hallucinated_references`` flag: ids that
appear in prose but resolve to no task on this board. Only prose is read. A
completion summary quotes example commands (``hermes kanban complete
t_dead0001``) and pastes fenced transcripts as often as it cites cards, and a
branch or worktree name (``fix/t_66a9d5c0``, ``wt/t_ffd2508c``) is a git ref, not
a card — none of those tokens is a reference, and flagging them buries the
handful that are.
"""

from __future__ import annotations

import re

# ``kanban_create`` mints 12 hex chars and ``_new_task_id`` 8; 8+ for forward
# compat. The token must stand on its own: the tail of a path or ref
# (``/var/log/t_dead0003.log``, ``C:\tmp\t_dead0002\out.txt``) is a file name.
TASK_ID_PROSE_RE = re.compile(r"(?<![\w/\\-])t_[a-f0-9]{8,}(?![\w/\\-])")

_FENCED_CODE_RE = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=fence)[^\n]*$|\Z)",
    re.MULTILINE | re.DOTALL,
)
_INLINE_CODE_RE = re.compile(r"(`+)(?!`).*?(?<!`)\1", re.DOTALL)


def strip_code_examples(text: str) -> str:
    """``text`` with fenced code blocks and inline code spans blanked out.

    A fence left open is still code, so it swallows the rest of the text.
    """
    if not text:
        return ""
    return _INLINE_CODE_RE.sub(" ", _FENCED_CODE_RE.sub("\n", text))


def prose_task_ids(text: str) -> list[str]:
    """Ordered, de-duplicated ``t_<hex>`` ids that appear as prose references."""
    return list(dict.fromkeys(TASK_ID_PROSE_RE.findall(strip_code_examples(text))))
