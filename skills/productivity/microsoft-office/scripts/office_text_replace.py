#!/usr/bin/env python3
"""Run-preserving, non-recursive text replacement shared by Office tools."""
from __future__ import annotations


def replace_in_runs(runs, old: str, new: str) -> int:
    """Replace matches present in the original run text exactly once.

    Matches are applied from right to left, so inserted text is never searched
    again and original offsets remain valid. A cross-run replacement keeps the
    formatting of the run where the match begins.
    """
    if not old:
        return 0
    original = [run.text for run in runs]
    full = "".join(original)
    starts = []
    cursor = 0
    while True:
        start = full.find(old, cursor)
        if start < 0:
            break
        starts.append(start)
        cursor = start + len(old)

    boundaries = []
    position = 0
    for text in original:
        boundaries.append((position, position + len(text)))
        position += len(text)

    for start in reversed(starts):
        end = start + len(old)
        spans = []
        for index, (run_start, run_end) in enumerate(boundaries):
            if run_end > start and run_start < end:
                spans.append((
                    index,
                    max(start, run_start) - run_start,
                    min(end, run_end) - run_start,
                ))
        for offset, (index, cut_start, cut_end) in enumerate(spans):
            text = runs[index].text
            replacement = new if offset == 0 else ""
            runs[index].text = text[:cut_start] + replacement + text[cut_end:]
    return len(starts)


def contiguous_run_groups(runs):
    """Split XML-backed runs at soft breaks, fields, and other boundaries."""
    groups = []
    for run in runs:
        if groups and groups[-1][-1]._r.getnext() is run._r:
            groups[-1].append(run)
        else:
            groups.append([run])
    return groups
