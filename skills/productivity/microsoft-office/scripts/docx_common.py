#!/usr/bin/env python3
# MIT License. Shared helpers for the docx skill scripts.
"""Shared helpers: paragraph iteration and run-preserving text replacement."""
from __future__ import annotations

from office_text_replace import replace_in_runs

W_RUN = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r"


def paragraph_runs(para):
    """Return all visible runs in document order, including hyperlinks."""
    from docx.text.run import Run

    return [Run(element, para) for element in para._p.iter(W_RUN)]


def iter_all_paragraphs(doc, include_headers_footers: bool = True):
    """Yield every paragraph in body, tables (recursively), headers, footers."""
    yield from _iter_container(doc)
    if include_headers_footers:
        seen = set()
        for section in doc.sections:
            for part in (
                section.header, section.footer,
                section.first_page_header, section.first_page_footer,
                section.even_page_header, section.even_page_footer,
            ):
                if part is not None and id(part._element) not in seen:
                    seen.add(id(part._element))
                    yield from _iter_container(part)


def _iter_container(container):
    yield from container.paragraphs
    for table in container.tables:
        yield from _iter_table(table)


def _iter_table(table, seen_cells=None):
    seen_cells = set() if seen_cells is None else seen_cells
    for row in table.rows:
        for cell in row.cells:
            identity = id(cell._tc)
            if identity in seen_cells:
                continue
            seen_cells.add(identity)
            yield from cell.paragraphs
            for nested in cell.tables:
                yield from _iter_table(nested, seen_cells)


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
    if not old:
        return 0
    runs = paragraph_runs(para)
    if old not in "".join(run.text for run in runs):
        return 0
    return replace_in_runs(runs, old, new)
