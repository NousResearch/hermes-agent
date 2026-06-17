#!/usr/bin/env python3
"""Sanitize fork overlay text before 3-way merge replay."""

from __future__ import annotations


def _find_anchor_line(lines: list[str], anchor: str, *, start: int = 0) -> int:
    for index in range(start, len(lines)):
        if anchor in lines[index]:
            return index
    return -1


def _drop_line_substrings(text: str, substrings: list[str]) -> str:
    if not substrings:
        return text
    kept: list[str] = []
    for line in text.splitlines(keepends=True):
        if any(substring in line for substring in substrings):
            continue
        kept.append(line)
    return "".join(kept)


def _replace_fork_region_with_upstream(
    fork_text: str,
    upstream_text: str,
    *,
    start_anchor: str,
    end_anchor: str,
) -> str:
    fork_lines = fork_text.splitlines(keepends=True)
    upstream_lines = upstream_text.splitlines(keepends=True)

    fork_start = _find_anchor_line(fork_lines, start_anchor)
    fork_end = _find_anchor_line(fork_lines, end_anchor, start=fork_start + 1 if fork_start >= 0 else 0)
    upstream_start = _find_anchor_line(upstream_lines, start_anchor)
    upstream_end = _find_anchor_line(
        upstream_lines,
        end_anchor,
        start=upstream_start + 1 if upstream_start >= 0 else 0,
    )
    if min(fork_start, fork_end, upstream_start, upstream_end) < 0:
        return fork_text

    merged_lines = (
        fork_lines[: fork_start + 1]
        + upstream_lines[upstream_start + 1 : upstream_end]
        + fork_lines[fork_end:]
    )
    return "".join(merged_lines)


def sanitize_fork_overlay_text(
    path: str,
    fork_text: str,
    upstream_text: str,
    sanitizers: dict[str, dict[str, object]],
) -> str:
    """Apply per-path overlay sanitizers before git merge-file replay."""
    spec = sanitizers.get(path)
    if not spec:
        return fork_text

    sanitized = fork_text
    region = spec.get("replace_fork_region_with_upstream")
    if isinstance(region, dict):
        start_anchor = str(region.get("start_anchor", ""))
        end_anchor = str(region.get("end_anchor", ""))
        if start_anchor and end_anchor:
            sanitized = _replace_fork_region_with_upstream(
                sanitized,
                upstream_text,
                start_anchor=start_anchor,
                end_anchor=end_anchor,
            )

    drop_substrings = spec.get("drop_fork_line_substrings")
    if isinstance(drop_substrings, list):
        sanitized = _drop_line_substrings(sanitized, [str(item) for item in drop_substrings])

    return sanitized


def load_overlay_sanitizers(strategy_payload: dict[str, object]) -> dict[str, dict[str, object]]:
    raw = strategy_payload.get("overlay_sanitizers")
    if not isinstance(raw, dict):
        return {}
    sanitizers: dict[str, dict[str, object]] = {}
    for path, spec in raw.items():
        if isinstance(spec, dict):
            sanitizers[str(path).replace("\\", "/")] = spec
    return sanitizers
