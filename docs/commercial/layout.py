"""Layout helpers shared by the document scaffold and the content modules.

Separated out because build.py imports the content and the content needs these — keeping
them in build.py would be a circular import, and putting them in theme.py would mix the
visual system with document structure.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.platypus import Image, NextPageTemplate, PageBreak, Paragraph
from reportlab.platypus.tableofcontents import TableOfContents

from theme import CONTENT_W, Gap, S, SectionOpener

ASSETS = Path(__file__).resolve().parent / "assets"


def img(name, width=CONTENT_W):
    """An asset scaled to a column width, preserving aspect. Never wider than the frame."""
    from reportlab.lib.utils import ImageReader

    reader = ImageReader(str(ASSETS / name))
    iw, ih = reader.getSize()
    return Image(str(ASSETS / name), width=width, height=width * ih / iw)



def contents_flowable():
    """The table of contents.

    Built by ReportLab's own two-pass machinery rather than a hand-kept list: a manual
    one drifts the first time a section moves, and a contents page that lies is worse
    than none.
    """
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"], S["toc2"]]
    toc.dotsMinLevel = 0
    return toc


def section(number, title, subtitle=""):
    return [NextPageTemplate("body"), PageBreak(), SectionOpener(number, title, subtitle),
            Gap(7)]


def h2(text):
    return Paragraph(text, S["h2"])


def h3(text):
    return Paragraph(text, S["h3"])


def bullets(items, style="bullet"):
    return [Paragraph(f"<bullet>&#8226;</bullet>{t}", S[style]) for t in items]


def caption(text):
    return Paragraph(text, S["caption"])
