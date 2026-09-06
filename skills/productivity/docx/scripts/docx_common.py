#!/usr/bin/env python3
# MIT License. Shared helpers for the docx skill scripts.
"""Shared helpers: paragraph iteration and run-preserving text replacement."""
from __future__ import annotations


def iter_all_paragraphs(doc, include_headers_footers: bool = True):
    """Yield every paragraph in body, tables (recursively), headers, footers."""
    yield from _iter_container(doc)
    if include_headers_footers:
        for section in doc.sections:
            for part in (
                section.header, section.footer,
                section.first_page_header, section.first_page_footer,
                section.even_page_header, section.even_page_footer,
            ):
                if part is not None:
                    yield from _iter_container(part)


def _iter_container(container):
    for para in container.paragraphs:
        yield para
    for table in container.tables:
        yield from _iter_table(table)


def _iter_table(table):
    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                yield para
            for nested in cell.tables:
                yield from _iter_table(nested)


def iter_part_roots(doc):
    """Yield the XML root of the body plus every header/footer part."""
    yield doc.element.body
    seen = set()
    for section in doc.sections:
        for part in (
            section.header, section.footer,
            section.first_page_header, section.first_page_footer,
            section.even_page_header, section.even_page_footer,
        ):
            if part is not None and id(part._element) not in seen:
                seen.add(id(part._element))
                yield part._element


def replace_in_paragraph(para, old: str, new: str) -> int:
    """Replace `old` with `new` in a paragraph, preserving run formatting.

    Two regimes, chosen by whether the replacement can feed itself:

    * `old not in new` (ordinary case): the historical two-stage behavior
      is preserved exactly -- first every occurrence fully contained in a
      single run is replaced (formatting fully preserved), then remaining
      cross-run occurrences are collapsed and each replacement inherits
      the formatting of the run where its match starts.

    * `old in new` (self-containing, e.g. X -> XX): the same two-stage
      priority is computed once over the paragraph's ORIGINAL run texts,
      and the edits are applied right-to-left, so inserted replacement
      text is never rescanned.  The operation always terminates and each
      original occurrence is replaced exactly once.

    Returns number of replacements made.
    """
    if not old or old not in para.text:
        return 0
    if old not in new:
        count = 0
        # Pass 1: within-run replacements.
        for run in para.runs:
            if old in run.text:
                count += run.text.count(old)
                run.text = run.text.replace(old, new)
        # Pass 2: cross-run occurrences.
        while old in para.text:
            runs = para.runs
            # Map paragraph text offsets to (run_index, offset_in_run).
            full = "".join(r.text for r in runs)
            start = full.find(old)
            if start < 0:
                break
            end = start + len(old)
            pos = 0
            spans = []  # (run_idx, cut_start, cut_end) portions inside the match
            for i, r in enumerate(runs):
                r_start, r_end = pos, pos + len(r.text)
                if r_end > start and r_start < end:
                    spans.append((i, max(start, r_start) - r_start,
                                  min(end, r_end) - r_start))
                pos = r_end
            first = True
            for i, cs, ce in spans:
                t = runs[i].text
                if first:
                    runs[i].text = t[:cs] + new + t[ce:]
                    first = False
                else:
                    runs[i].text = t[:cs] + t[ce:]
            count += 1
        return count

    # Self-containing replacement: bounded snapshot semantics over the
    # ORIGINAL run texts (never rescan mutated text).
    runs = list(para.runs)
    originals = [r.text for r in runs]
    offsets = []
    acc = 0
    for t in originals:
        offsets.append(acc)
        acc += len(t)
    full = "".join(originals)

    # Pass 1 (priority): occurrences fully contained in one run, greedy
    # non-overlapping left-to-right inside each run (str.replace order).
    # Ranges are only RECORDED here; every edit happens in the single
    # right-to-left application pass below (never apply ranges twice).
    texts = list(originals)
    within = 0
    taken = []  # full-coordinate ranges consumed by pass 1
    for i, t in enumerate(originals):
        if old in t:
            within += t.count(old)
            base = offsets[i]
            pos = t.find(old)
            while pos >= 0:
                taken.append((base + pos, base + pos + len(old)))
                pos = t.find(old, pos + len(old))

    # Pass 2: remaining cross-run occurrences over the ORIGINAL text.  A
    # candidate fully inside one run was already consumed by pass 1; a
    # candidate overlapping any selected range is consumed by that
    # replacement (within-run-before-cross-run precedence).
    ranges = list(taken)
    pos = 0
    while True:
        idx = full.find(old, pos)
        if idx < 0:
            break
        end = idx + len(old)
        inside_run = False
        for i in range(len(originals)):
            r_start = offsets[i]
            r_end = r_start + len(originals[i])
            if r_start <= idx and end <= r_end:
                inside_run = True
                break
        if inside_run:
            pos = idx + 1
            continue
        if any(idx < e and s < end for s, e in ranges):
            pos = idx + 1
            continue
        ranges.append((idx, end))
        pos = idx + 1

    # Apply every selected range right-to-left: a mutation can only shift
    # offsets at positions >= the current range's start, which no later
    # (more-leftward) range ever touches.
    for start, end in sorted(ranges, key=lambda r: r[0], reverse=True):
        for i, t in enumerate(texts):
            r_start = offsets[i]
            r_end = r_start + len(originals[i])
            if r_end <= start or r_start >= end:
                continue  # run lies entirely outside this match
            cs = max(start, r_start) - r_start
            ce = min(end, r_end) - r_start
            if r_start <= start:
                # Run hosting the match start: replacement goes here and
                # inherits this run's formatting.
                texts[i] = t[:cs] + new + t[ce:]
            else:
                # Continuation run: strip its matched portion.
                texts[i] = t[:cs] + t[ce:]
    for run, t, orig in zip(runs, texts, originals):
        if t != orig:
            run.text = t
    return within + len(ranges) - len(taken)
