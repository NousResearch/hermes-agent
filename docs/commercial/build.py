"""Builds NOVA_Commercial_and_AWS_Deployment_Playbook.pdf.

Content rules this file is written under, because they are the whole point of the
document:

* **Every capability claim traces to the repository.** Where a claim has a file behind it,
  the file is named. Where a claim would need a real AWS account or a real model provider
  to be true, it is marked as unproven rather than softened.
* **Every external number carries a source**, listed in §25. Where a number is an
  assumption for the sake of an example, the words "illustrative assumption" appear beside
  it, not in a footnote.
* **AWS prices are not quoted.** The official pricing pages were unreachable from the build
  environment (the egress proxy blocks aws.amazon.com), so the document gives cost
  *structure* and the official URLs, and says plainly that it is not quoting figures.
"""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer)
from reportlab.platypus.tableofcontents import TableOfContents

from theme import (ACCENT, ACCENT_SOFT, AQUA, BODY_FONT, BOLD_FONT, CONTENT_W, CRITICAL,
                   GOOD, HAIRLINE, HAIRLINE_HI, INK, INK_2, LADDER, MARGIN_B, MARGIN_L,
                   MARGIN_R, MARGIN_T, MONO_FONT, PAGE_H, PAGE_W, PANEL, PANEL_HI, S,
                   SERIOUS, TEXT, TEXT_FAINT, TEXT_MUTED, WARM, WARNING, Checklist, Flow,
                   Gap, P, Panel, Rule, SectionOpener, Timeline, callout, status_chip,
                   table)

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
OUT = HERE / "NOVA_Commercial_and_AWS_Deployment_Playbook.pdf"

VERSION = "1.0"
BUILT = _dt.date(2026, 9, 15)
COMMIT = "b9319677"

_SECTION = {"title": ""}




# -- page furniture ------------------------------------------------------------


def _ground(canvas, doc):
    """The page ground: flat navy, one very soft bloom, and a faint dot texture.

    The texture is drawn at about 4% opacity on a 9mm grid. It has to be felt rather than
    seen — anything stronger competes with 9pt body text, which is the failure this kind of
    document falls into most often.
    """
    canvas.saveState()
    canvas.setFillColor(INK)
    canvas.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)

    canvas.setFillColor(colors.HexColor("#101B30"))
    canvas.setFillAlpha(0.55)
    canvas.circle(PAGE_W * 0.86, PAGE_H * 0.93, 62 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#0E1A2C"))
    canvas.circle(PAGE_W * 0.08, PAGE_H * 0.12, 54 * mm, stroke=0, fill=1)
    canvas.setFillAlpha(1)

    canvas.setFillColor(colors.HexColor("#1B2942"))
    canvas.setFillAlpha(0.42)
    step = 9 * mm
    y = 6 * mm
    while y < PAGE_H:
        x = 6 * mm
        while x < PAGE_W:
            canvas.circle(x, y, 0.28, stroke=0, fill=1)
            x += step
        y += step
    canvas.setFillAlpha(1)
    canvas.restoreState()


def _furniture(canvas, doc):
    """Header rule, running title, and the footer with the page number."""
    canvas.saveState()
    canvas.setFont(BODY_FONT, 7)
    canvas.setFillColor(TEXT_FAINT)
    canvas.drawString(MARGIN_L, PAGE_H - 13 * mm, "NOVA  ·  AI Workforce Commercial & Deployment Playbook")
    running = _SECTION["title"]
    if running:
        canvas.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 13 * mm, running)
    canvas.setStrokeColor(HAIRLINE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN_L, PAGE_H - 15.6 * mm, PAGE_W - MARGIN_R, PAGE_H - 15.6 * mm)

    canvas.line(MARGIN_L, 12.4 * mm, PAGE_W - MARGIN_R, 12.4 * mm)
    canvas.setFillColor(TEXT_FAINT)
    canvas.drawString(MARGIN_L, 9 * mm, f"v{VERSION}  ·  {BUILT:%d %B %Y}  ·  repository @ {COMMIT}")
    canvas.setFillColor(TEXT_MUTED)
    canvas.setFont(BOLD_FONT, 8)
    canvas.drawRightString(PAGE_W - MARGIN_R, 9 * mm, str(canvas.getPageNumber()))
    canvas.restoreState()


def on_page(canvas, doc):
    _ground(canvas, doc)
    _furniture(canvas, doc)


def on_cover(canvas, doc):
    # multiBuild lays the story out more than once, and the running title is a module
    # global — without this reset, pass two starts with pass one's last section name and
    # the contents page is headed "25 · Sources and references".
    _SECTION["title"] = ""
    canvas.saveState()
    canvas.setFillColor(INK)
    canvas.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#122038"))
    canvas.setFillAlpha(0.85)
    canvas.circle(PAGE_W * 0.78, PAGE_H * 0.80, 88 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#0F1B2E"))
    canvas.circle(PAGE_W * 0.12, PAGE_H * 0.26, 70 * mm, stroke=0, fill=1)
    canvas.setFillAlpha(1)

    canvas.setFillColor(colors.HexColor("#1B2942"))
    canvas.setFillAlpha(0.5)
    step = 7 * mm
    y = 5 * mm
    while y < PAGE_H:
        x = 5 * mm
        while x < PAGE_W:
            canvas.circle(x, y, 0.3, stroke=0, fill=1)
            x += step
        y += step
    canvas.setFillAlpha(1)

    # Wordmark
    canvas.setFillColor(ACCENT)
    canvas.rect(MARGIN_L, PAGE_H - 62 * mm, 2.2 * mm, 17 * mm, stroke=0, fill=1)
    canvas.setFillColor(TEXT)
    canvas.setFont(BOLD_FONT, 46)
    canvas.drawString(MARGIN_L + 7 * mm, PAGE_H - 60 * mm, "NOVA")
    canvas.setFillColor(TEXT_MUTED)
    canvas.setFont(BODY_FONT, 9)
    canvas.drawString(MARGIN_L + 7.6 * mm, PAGE_H - 68 * mm,
                      "ENTERPRISE  AI  WORKFORCE  PLATFORM")

    canvas.setFillColor(TEXT)
    canvas.setFont(BOLD_FONT, 27)
    canvas.drawString(MARGIN_L, PAGE_H - 108 * mm, "AI Workforce Commercial")
    canvas.drawString(MARGIN_L, PAGE_H - 120 * mm, "& Deployment Playbook")

    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(2)
    canvas.line(MARGIN_L, PAGE_H - 129 * mm, MARGIN_L + 34 * mm, PAGE_H - 129 * mm)

    canvas.setFillColor(TEXT_MUTED)
    canvas.setFont(BODY_FONT, 12)
    canvas.drawString(MARGIN_L, PAGE_H - 141 * mm,
                      "Product  ·  Market  ·  Sales  ·  Pricing  ·  Marketing  ·  AWS Deployment")

    # Evidence strip — the cover states the document's own posture, which is the
    # single most important thing a reader needs before page one.
    box_y = 58 * mm
    canvas.setFillColor(PANEL)
    canvas.setStrokeColor(HAIRLINE)
    canvas.setLineWidth(0.7)
    canvas.roundRect(MARGIN_L, box_y, CONTENT_W, 40 * mm, 3 * mm, stroke=1, fill=1)
    canvas.setFillColor(WARNING)
    canvas.setFont(BOLD_FONT, 8.4)
    canvas.drawString(MARGIN_L + 7 * mm, box_y + 31 * mm, "READ THIS FIRST")
    canvas.setFillColor(TEXT)
    canvas.setFont(BODY_FONT, 8.6)
    for i, line in enumerate([
        "Every capability claim in this document is graded against the repository at the commit below.",
        "930 platform tests and 6 browser suites pass. 49 container checks pass on this machine.",
        "Nothing in NOVA has yet run on a real AWS account, and no agent has yet called a real model",
        "provider. Those two facts are stated wherever they matter, and they are the top of §24.",
    ]):
        canvas.drawString(MARGIN_L + 7 * mm, box_y + 24 * mm - i * 5.2 * mm, line)

    canvas.setFillColor(TEXT_FAINT)
    canvas.setFont(BODY_FONT, 8)
    canvas.drawString(MARGIN_L, 34 * mm, f"Version {VERSION}")
    canvas.drawString(MARGIN_L, 29 * mm, f"{BUILT:%d %B %Y}")
    canvas.drawString(MARGIN_L, 24 * mm, f"Repository state: commit {COMMIT} (branch main)")
    canvas.drawString(MARGIN_L, 19 * mm, "Internal master document — not for external distribution as-is")
    canvas.restoreState()


class Doc(BaseDocTemplate):
    """Two templates: a cover with no furniture, and the body.

    Section titles are collected on the way through so the running header and the table of
    contents both come from the document itself rather than from a hand-kept list that
    would drift the first time a section moved.
    """

    def __init__(self, path):
        super().__init__(path, pagesize=(PAGE_W, PAGE_H),
                         leftMargin=MARGIN_L, rightMargin=MARGIN_R,
                         topMargin=MARGIN_T, bottomMargin=MARGIN_B,
                         title="NOVA — AI Workforce Commercial & Deployment Playbook",
                         author="NOVA", subject="Commercial and AWS deployment playbook")
        frame = Frame(MARGIN_L, MARGIN_B, CONTENT_W,
                      PAGE_H - MARGIN_T - MARGIN_B, id="body",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[frame], onPage=on_cover),
            PageTemplate(id="body", frames=[frame], onPage=on_page),
        ])

    def afterFlowable(self, flowable):
        if isinstance(flowable, SectionOpener):
            _SECTION["title"] = f"{flowable.number} · {flowable.title}"
            self.notify("TOCEntry", (0, f"{flowable.number}   {flowable.title}", self.page))
        elif isinstance(flowable, Paragraph) and flowable.style.name == "h2":
            self.notify("TOCEntry", (1, flowable.getPlainText(), self.page))




from content import build_story  # noqa: E402


def main():
    story = build_story()
    doc = Doc(str(OUT))
    doc.multiBuild(story)
    print(f"wrote {OUT}")
    return OUT


if __name__ == "__main__":
    main()
