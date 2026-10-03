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

    Strategy: first replace occurrences fully contained in a single run
    (formatting fully preserved). If the needle spans multiple runs, the
    matched runs are collapsed: the replacement inherits the formatting of
    the run where the match starts. Returns number of replacements made.
    """
    if not old or old == new or old not in para.text:
        return 0
    count = 0
    # Pass 1: within-run replacements.
    for run in para.runs:
        if old in run.text:
            count += run.text.count(old)
            run.text = run.text.replace(old, new)
    # Pass 2: occurrences that span a run boundary. Collect those spans from the text
    # as it stands -- never re-scan text this pass writes, or a replacement that
    # contains the needle (`a` -> `aa`) matches itself and the loop never ends -- then
    # apply them right-to-left so the earlier offsets stay valid.
    runs = para.runs
    run_spans = []
    pos = 0
    for r in runs:
        run_spans.append((pos, pos + len(r.text)))
        pos += len(r.text)
    full = "".join(r.text for r in runs)
    crossings = []
    idx = full.find(old)
    while idx != -1:
        end = idx + len(old)
        if sum(1 for s, e in run_spans if e > idx and s < end) > 1:
            crossings.append((idx, end))
        idx = full.find(old, end)
    for start, end in reversed(crossings):
        runs = para.runs
        pos = 0
        cuts = []  # (run_idx, slice_start, slice_end) portions inside the match
        for i, r in enumerate(runs):
            r_start, r_end = pos, pos + len(r.text)
            if r_end > start and r_start < end:
                cuts.append((i, max(start, r_start) - r_start, min(end, r_end) - r_start))
            pos = r_end
        first = True
        for i, cs, ce in cuts:
            t = runs[i].text
            if first:
                runs[i].text = t[:cs] + new + t[ce:]
                first = False
            else:
                runs[i].text = t[:cs] + t[ce:]
        count += 1
    return count
