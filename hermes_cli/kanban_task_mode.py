"""Rule 2 — task-mode (verification-rigor) classification.

Consulted by ``kanban_decompose.py`` at task-creation time to populate
``Task.task_mode``. Small standalone module (not folded into
``kanban_decompose.py``) because it is a pure classification concern with no
dependency on the decomposer's LLM plumbing or DB writes.

SCOPE: schema-population ONLY. Nothing in this diff makes any
dispatch/gate/enforcement decision read ``task_mode`` — it is set at
creation time and otherwise inert today. A misclassification here has no
downstream consequence until (if ever) something starts consulting the
field for gating; that is explicitly out of scope for this change.

Heuristic, not an LLM call — chosen over a second aux-LLM round-trip
because: (a) it composes cheap, stable structured signals (question
phrasing, keyword classes, file-mention counts) instead of paying
network/latency cost and LLM non-determinism on every task the decomposer
creates; (b) the decompose aux-LLM call already produces the title/body
text being classified, so a second LLM round-trip to re-read that same
text back would be redundant work for a field with no downstream gate.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

TaskMode = Literal["autocomplete", "chat", "agent"]

# Rule 2 modes: least to most verification rigor.
VALID_TASK_MODES: tuple[TaskMode, ...] = ("autocomplete", "chat", "agent")

# --- Chat signals: the task asks a question / wants an explanation or
# decision, not a code change. Checked first so a task that's purely a
# question is never misread as "agent" for mentioning an architecture noun
# or two filenames in passing.
_CHAT_PATTERNS = (
    re.compile(r"\bwhat\s+(does|is|are|should)\b", re.I),
    re.compile(r"\bwhy\s+(does|is|do|did)\b", re.I),
    re.compile(r"\bhow\s+does\b", re.I),
    re.compile(r"\bcan\s+(you|someone)\s+explain\b", re.I),
    re.compile(r"\bexplain\b", re.I),
    re.compile(r"\bclarify\b", re.I),
    re.compile(r"\?\s*$"),  # a title/body ending in a question mark
    re.compile(r"\bno\s+code\s+change\s+needed\b", re.I),
)

# --- Agent signals: multi-file / architectural / new-subsystem work that
# needs a full agentic loop (planning, multiple edits, its own tests).
_AGENT_KEYWORDS = (
    "architecture", "architectural", "migration", "migrate", "pipeline",
    "subsystem", "refactor", "redesign", "new module", "new abc",
    "schema migration", "design and build", "spanning", "multi-file",
    "multi-step", "several modules", "across the", "across several",
)
_AGENT_PATTERNS = tuple(re.compile(re.escape(kw), re.I) for kw in _AGENT_KEYWORDS)

# A body naming 2+ distinct file/module paths (word.py, word/word.py, ...)
# is a strong multi-file-scope signal independent of keywords.
_FILE_PATH_RE = re.compile(r"\b[\w/]+\.[a-zA-Z]{1,4}\b")

# --- Autocomplete signals: small, mechanical, single-location edits.
_AUTOCOMPLETE_KEYWORDS = (
    "typo", "rename", "one-line", "one line", "single-line", "single line",
    "single-file", "single file",
)
_AUTOCOMPLETE_PATTERNS = tuple(re.compile(re.escape(kw), re.I) for kw in _AUTOCOMPLETE_KEYWORDS)

# Short-body default threshold: bodies under this length with at most one
# file mention read as small, mechanical work.
_SHORT_BODY_CHARS = 80


def _distinct_file_mentions(text: str) -> int:
    return len(set(_FILE_PATH_RE.findall(text)))


def classify_task_mode(title: str, body: Optional[str]) -> TaskMode:
    """Best-effort heuristic classification of a task's verification-rigor
    mode. Never raises; composes structured signals rather than calling an
    LLM. Not consulted by dispatch/gate/enforcement logic — see module
    docstring.
    """
    title_text = title or ""
    body_text = body or ""
    combined = f"{title_text}\n{body_text}"

    if any(p.search(combined) for p in _CHAT_PATTERNS):
        return "chat"

    agent_hit = any(p.search(combined) for p in _AGENT_PATTERNS)
    multi_file = _distinct_file_mentions(combined) >= 2
    if agent_hit or multi_file:
        return "agent"

    if any(p.search(combined) for p in _AUTOCOMPLETE_PATTERNS):
        return "autocomplete"

    # Short-body-single-file default: short, single-sentence bodies with no
    # other signal read as small, mechanical work.
    if len(body_text.strip()) < _SHORT_BODY_CHARS and _distinct_file_mentions(combined) <= 1:
        return "autocomplete"

    return "chat"
