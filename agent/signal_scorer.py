"""
Signal-aware tool output scoring for context compression.

Adds a cheap, deterministic regex-based signal scorer that classifies tool
outputs by importance before pruning.  High-signal outputs (errors, crashes,
test failures) are preserved verbatim; low-signal outputs (success
confirmations, empty results) are pruned aggressively.

This is a standalone module extracted from the signal-aware compressor
prototype at ~/compression-harness/.  See ``PR_PROPOSAL.md`` for benchmarks
and design rationale.

Integration point: ``_prune_old_tool_results()`` in ``agent/context_compressor.py``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Signal patterns — classified by importance
# ---------------------------------------------------------------------------

# Patterns that indicate HIGH signal — outputs that must survive compression.
# Weight is the contribution per match; higher = more important to keep.
_HIGH_SIGNAL_PATTERNS: List[Tuple[re.Pattern, int]] = [
    (re.compile(r"(?i)\b(error|exception|traceback|fatal|crash|panic)\b"), 10),
    (re.compile(r"(?i)\b(fail(ed|ure)?|FAILED|FAIL)\b"), 8),
    (re.compile(r"(?i)\b(assert|assertion)\b.*\b(fail|error)\b"), 9),
    (re.compile(r"(?i)Traceback\s*\(most\s+recent\s+call\s+last\)"), 15),
    (re.compile(r"(?i)\b(segfault|segmentation\s+fault|core\s+dumped)\b"), 15),
    (re.compile(r"(?i)\b(cannot|could\s+not|unable\s+to|refused|denied|forbidden)\b"), 6),
    (re.compile(r"(?i)(\d+)\s+failed"), 7),
    (re.compile(r"(?i)\bFAILED\b.*\btest"), 8),
    (re.compile(r"(?i)\b(deprecated|breaking\s+change|must\s+migrate)\b"), 7),
    (re.compile(r"(?i)\b(security|vulnerability|CVE-|injection|exploit)\b"), 12),
    (re.compile(r"(?i)\b(performance|latency|throughput|bottleneck)\b"), 5),
    # Secrets — must NEVER survive compression.  Scored negative so they are
    # always suppressed even when surrounded by high-signal error context.
    (re.compile(r"(?i)\b(api[_\\s]?key|token|password|secret|credential)\b"), -5),
    # Distinctive values worth preserving
    (re.compile(r"[/\\\\][\\w.-]+[/\\\\][\\w.-]+"), 3),  # file paths
    (re.compile(r"https?://[^\\s]+"), 3),                 # URLs
    (re.compile(r"\b[0-9a-f]{7,40}\b"), 2),               # git SHAs
    (re.compile(r"\bv?\d+\.\d+\.\d+"), 2),                 # version numbers
]

# Patterns that indicate LOW signal — safe to prune aggressively.
_LOW_SIGNAL_PATTERNS: List[Tuple[re.Pattern, int]] = [
    (re.compile(r"(?i)\b(success(fully)?|completed|done|finished|ok)\b"), -3),
    (re.compile(r"(?i)\b(file\s+written|directory\s+created|saved|installed)\b"), -3),
    (re.compile(r"(?i)\b(unchanged|up.to.date|already\s+exists)\b"), -4),
    (re.compile(r"^\s*$"), -2),  # blank lines
]

# Tool-type baseline scores — some tools inherently carry more signal.
_TOOL_BASELINES: Dict[str, int] = {
    "terminal": 0,       # variable — depends on output content
    "read_file": 1,      # usually worth keeping some context
    "search_files": -2,  # search results are ephemeral
    "write_file": -2,    # confirmation mostly
    "patch": 0,          # diff output can be important
    "browser_vision": 1, # might show errors
    "browser_navigate": -2,
    "web_search": -2,
    "web_extract": -1,
    "delegate_task": 3,  # subagent results are high-signal
    "execute_code": 2,   # script output can have important results
}

# Thresholds for pruning actions.
_SIGNAL_KEEP_THRESHOLD = 3    # ≥3: keep verbatim (error/crash/decision)
_SIGNAL_PRUNE_THRESHOLD = -2  # ≤-2: prune aggressively (success confirmation)


def score_tool_output(tool_name: str, content: str) -> int:
    """Score a tool output by signal importance.

    Returns:
        int: Score where >0 = high signal (preserve), <0 = low signal (prune),
             0 = neutral.  Higher absolute value = stronger signal.
    """
    score = _TOOL_BASELINES.get(tool_name, 0)

    # High-signal patterns — critical patterns get higher match caps.
    for pattern, weight in _HIGH_SIGNAL_PATTERNS:
        matches = len(pattern.findall(content))
        if matches > 0:
            cap = 20 if weight >= 10 else 8 if weight >= 7 else 5
            score += weight * min(matches, cap)

    # Low-signal patterns — capped at 2 matches to prevent "OK\n" × 500
    # from drowning out a single "CRITICAL" error.
    for pattern, weight in _LOW_SIGNAL_PATTERNS:
        matches = len(pattern.findall(content))
        if matches > 0:
            score += weight * min(matches, 2)

    # Length heuristics
    content_len = len(content)
    if content_len < 50:
        score -= 2
    elif content_len > 5000:
        score += 1  # long outputs often contain diagnostics

    # Verbose but not signal-rich
    line_count = content.count("\n") + 1
    if line_count > 100 and score <= 0:
        score -= 1

    return score


def signal_aware_prune_action(tool_name: str, content: str) -> str:
    """Return the pruning action for a single tool output.

    Returns one of: ``"keep"``, ``"summarize"``, ``"prune"``.
    """
    score = score_tool_output(tool_name, content)
    if score >= _SIGNAL_KEEP_THRESHOLD:
        return "keep"
    if score <= _SIGNAL_PRUNE_THRESHOLD:
        return "prune"
    return "summarize"


# ---------------------------------------------------------------------------
# Smart content truncation for summarizer input
# ---------------------------------------------------------------------------

_MAX_CONTENT_CHARS = 6000  # total chars per message body in summarizer input

# Section headers that indicate important content blocks.
_SIGNAL_SECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)^[#=]{1,3}\s*(error|exception|failure|bug|issue|problem|warning|critical)", re.MULTILINE),
    re.compile(r"(?i)^(FAILED|ERROR|FAILURES|Traceback)", re.MULTILINE),
    re.compile(r"(?i)^={3,}\s*FAILURES\s*={3,}", re.MULTILINE),
    re.compile(r"(?i)^---\s*(FAIL|ERROR)", re.MULTILINE),
]


def _find_signal_ranges(content: str) -> List[Tuple[int, int, float]]:
    """Find high-signal regions (errors, failures, tracebacks) in content.

    Returns list of (start, end, importance) byte ranges.
    """
    ranges: List[Tuple[int, int, float]] = []

    for pattern in _SIGNAL_SECTION_PATTERNS:
        for match in pattern.finditer(content):
            start = max(0, match.start() - 200)
            end = min(len(content), match.end() + 2000)
            rest = content[match.end():match.end() + 3000]
            next_section = re.search(r"\n(?=[#=]{1,3}\s|\n\S)", rest)
            if next_section:
                end = min(len(content), match.end() + next_section.start())
            ranges.append((start, end, 1.0))

    # Traceback blocks (multi-line)
    tb_pattern = re.compile(
        r"(Traceback\s*\(most\s+recent\s+call\s+last\).*?)(?=\n\n\S|\n[^ \t]|\Z)",
        re.DOTALL,
    )
    for match in tb_pattern.finditer(content):
        ranges.append((match.start(), match.end(), 0.9))

    # Merge overlapping ranges
    if not ranges:
        return ranges
    ranges.sort()
    merged: List[Tuple[int, int, float]] = []
    current_start, current_end, current_imp = ranges[0]
    for start, end, imp in ranges[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            current_imp = max(current_imp, imp)
        else:
            merged.append((current_start, current_end, current_imp))
            current_start, current_end, current_imp = start, end, imp
    merged.append((current_start, current_end, current_imp))
    return merged


def smart_truncate_for_summarizer(content: str, max_chars: int = _MAX_CONTENT_CHARS) -> str:
    """Truncate content for the summarizer while preserving error sections.

    The stock compressor uses fixed head/tail truncation (4000 + 1500 chars).
    This version preserves signal-rich sections (errors, tracebacks, test
    failures) within the budget, then fills remaining space with head + tail.

    Args:
        content: Raw tool output or message content.
        max_chars: Maximum characters to keep (default 6000).

    Returns:
        Truncated string ≤ max_chars.
    """
    if len(content) <= max_chars:
        return content

    signal_ranges = _find_signal_ranges(content)
    head_chars = max_chars // 3
    tail_chars = max_chars // 4
    signal_budget = max_chars - head_chars - tail_chars

    pieces: List[str] = []
    used = 0

    if head_chars > 0:
        pieces.append(content[:head_chars])
        used += head_chars

    if signal_ranges and signal_budget > 100:
        for start, end, imp in signal_ranges:
            section = content[start:end]
            section_len = len(section)
            if used + section_len <= max_chars - tail_chars:
                if used > head_chars:
                    pieces.append("\n...\n")
                pieces.append(f"\n[IMPORTANT — score {imp:.1f}]:\n")
                pieces.append(section)
                used += section_len + 50
            elif signal_budget > 200:
                available = max_chars - tail_chars - used - 50
                if available > 200:
                    pieces.append("\n...\n")
                    pieces.append(f"\n[IMPORTANT (truncated) — score {imp:.1f}]:\n")
                    pieces.append(section[:available])
                    used += available + 50
                break

    if tail_chars > 0 and len(content) > head_chars + 50:
        remaining = max_chars - used
        if remaining > 200:
            pieces.append(f"\n...[truncated {len(content) - used - remaining} chars]...\n")
            pieces.append(content[-remaining:])

    result = "".join(pieces)
    return result[:max_chars]
