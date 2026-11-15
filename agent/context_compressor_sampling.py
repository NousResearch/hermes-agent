"""Bounded, record-preserving summarizer input for context compression.

Pure class-level helpers; ``_SUMMARY_INPUT_MAX_CHARS`` / ``_SAMPLED_INPUT_SLICES`` come from the
compressor class so subclasses and tests can override them.
"""

from __future__ import annotations

from typing import Optional, Sequence

from agent.compression_marker import _elision_marker, elide_middle


class SummaryInputSamplingMixin:
    @classmethod
    def _bound_summary_input(cls, content: str) -> str:
        """Cap total summarizer input, keeping head and tail and marking the omitted middle."""
        if len(content) <= cls._SUMMARY_INPUT_MAX_CHARS:
            return content

        marker_template = (
            "\n\n...[summary input truncated: omitted "
            "{omitted:,} chars from the middle to keep compression prompt bounded]...\n\n"
        )
        # Marker width can change with the omitted count; estimate, then rebuild once.
        omitted = len(content)
        for _ in range(2):
            marker = marker_template.format(omitted=omitted)
            remaining = max(cls._SUMMARY_INPUT_MAX_CHARS - len(marker), 0)
            head_chars = int(remaining * 0.45)
            tail_chars = remaining - head_chars
            omitted = max(len(content) - head_chars - tail_chars, 0)
        tail = content[-tail_chars:].lstrip() if tail_chars else ""
        return content[:head_chars].rstrip() + marker + tail

    # Lean-mode sampling slice count: 8 keeps slices ~20K chars at the 160K cap.
    _SAMPLED_INPUT_SLICES = 8

    @classmethod
    def _bound_oversized_record(cls, record: str, limit: int) -> str:
        """Bound an oversized record with an explicit intra-record truncation marker."""
        if len(record) <= limit:
            return record
        marker_reserve = len(_elision_marker(omitted=len(record), total=len(record)))
        if limit <= marker_reserve:
            return record[:limit]
        head_len = (limit - marker_reserve) // 2
        return elide_middle(record, head_len, limit - marker_reserve - head_len)

    @classmethod
    def _sample_summary_records(
        cls, records: Sequence[str], slices_out: Optional[list] = None,
    ) -> tuple[str, dict[str, int]]:
        """Sample complete serialized records while retaining the character bound.
        ``slices_out`` receives the kept ``(start, display_records)`` runs for typed input.

        Returns the bounded transcript and record-level coverage counters for compression
        telemetry. `input_chars` counts raw serialized record content; `sampled_chars` counts the
        *display* chars of retained records (after intra-record truncation by
        `_bound_oversized_record`); neither includes separators or elision markers, so
        `omitted_chars = input_chars - sampled_chars` also covers truncated-away bytes.
        """
        input_chars = sum(len(r) for r in records)

        def _coverage(sampled_chars: int, sampled_record_count: int) -> dict[str, int]:
            return {
                "input_chars": input_chars, "sampled_chars": sampled_chars,
                "omitted_chars": input_chars - sampled_chars, "record_count": len(records),
                "sampled_record_count": sampled_record_count,
                "elided_record_count": len(records) - sampled_record_count,
            }

        if not records:
            return "", _coverage(0, 0)

        separator = "\n\n"
        total_len = input_chars + len(separator) * (len(records) - 1)
        if total_len <= cls._SUMMARY_INPUT_MAX_CHARS:
            if slices_out is not None:
                slices_out.append((0, list(records)))
            return separator.join(records), _coverage(input_chars, len(records))

        n = max(1, min(cls._SAMPLED_INPUT_SLICES, len(records)))
        marker_template = (
            "\n\n...[records {first:,}-{last:,}: {elided:,} chars elided — recover via session_search]...\n\n"
        )
        marker_len = len(marker_template.format(first=len(records), last=len(records), elided=total_len))
        budget = max(cls._SUMMARY_INPUT_MAX_CHARS - marker_len * (n - 1), 1)
        target = max(1, budget // n)

        # Oversized records are bounded to slice target with explicit intra-record truncation markers
        # so they cannot consume other regions' budget or evict the newest record.
        display_records = [cls._bound_oversized_record(r, target) for r in records]

        def _merged(slices: list[tuple[int, int]]) -> list[tuple[int, int]]:
            out: list[tuple[int, int]] = []
            for s, e in slices:
                if out and s <= out[-1][1]:
                    out[-1] = (out[-1][0], max(out[-1][1], e))
                else:
                    out.append((s, e))
            return out

        starts = [round(i * len(records) / n) for i in range(n)]
        selected: list[tuple[int, int]] = []
        for index, start in enumerate(starts):
            if index == len(starts) - 1:
                # Anchor the last slice to the newest record at the end of the history.
                end = len(records)
                start = end - 1
                size = len(display_records[start])
                while start > 0 and size + len(separator) + len(display_records[start - 1]) <= target:
                    start -= 1
                    size += len(separator) + len(display_records[start])
            else:
                end = start
                size = 0
                while end < len(records) and (size == 0 or size + len(display_records[end]) + len(separator) <= target):
                    size += len(display_records[end]) + (len(separator) if end > start else 0)
                    end += 1
            if end > start:
                selected.append((start, end))
        selected = _merged(selected)

        def _render(slices: list[tuple[int, int]]) -> str:
            parts: list[str] = []
            cursor = 0
            for s, e in slices:
                if s > cursor:
                    sep_count = (s - cursor) if cursor == 0 else (s - cursor + 1)
                    elided = sum(len(records[i]) for i in range(cursor, s)) + len(separator) * sep_count
                    parts.append(marker_template.format(first=cursor + 1, last=s, elided=elided))
                parts.append(separator.join(display_records[s:e]))
                cursor = e
            return "".join(parts)

        # Budget extension: the greedy fill leaves each slice short of `target` by up to one record
        # (5-43% of the cap unused for 8-20K records). Spend the headroom on whole neighbouring
        # records, round-robin one record per slice per round so every region keeps an even share
        # (the newest slice grows backward, older slices grow forward) — never past cap.
        cap = cls._SUMMARY_INPUT_MAX_CHARS
        rendered_len = len(_render(selected))
        grew = True
        while grew:
            grew = False
            for idx in range(len(selected) - 1, -1, -1):
                s, e = selected[idx]
                if idx == len(selected) - 1:
                    nxt, grown = s - 1, (s - 1, e)
                    if nxt < (selected[idx - 1][1] if idx else 0):
                        continue
                else:
                    nxt, grown = e, (s, e + 1)
                    if nxt >= selected[idx + 1][0]:
                        continue
                if rendered_len + len(separator) + len(display_records[nxt]) > cap:
                    continue
                selected[idx] = grown
                new_len = len(_render(_merged(selected)))
                # The pre-check above bounds the added record; the exact re-render catches the
                # one thing it cannot see — a gap's first index gaining a digit or comma in the
                # marker (e.g. 999 -> 1,000) when the render is already at cap.
                if new_len > cap:
                    selected[idx] = (s, e)
                    continue
                rendered_len = new_len
                grew = True
        selected = _merged(selected)

        # No overflow trim is needed: every slice holds <= `target` display chars (records are
        # pre-bounded to `target`), there are <= n-1 markers each <= `marker_len` (widths computed
        # at their maxima), and n*target + (n-1)*marker_len <= _SUMMARY_INPUT_MAX_CHARS by
        # construction; the extension pass above only adds a record when the result stays <= cap.
        shown = [i for s, e in selected for i in range(s, e)]
        if slices_out is not None:
            slices_out.extend((s, display_records[s:e]) for s, e in selected)
        return _render(selected), _coverage(sum(len(display_records[i]) for i in shown), len(shown))
