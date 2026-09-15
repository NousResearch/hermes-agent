"""The visual system for the NOVA playbook: palette, typography, and the flowables.

Kept apart from the content so a wording change never risks the layout and a design
change never risks a claim. Nothing here imports NOVA — this is a document toolchain that
happens to live in the repository, and the repository's implementation stays the source of
truth for everything the document says.

**Colour.** A deep navy ground with one cool accent and one warm one. The two chart hues
are slots 1 and 2 of the data-viz reference palette, stepped for a dark surface, and were
run through that skill's validator against this document's own panel colour (#141c2e)
rather than against the default dark surface: all five checks pass, worst adjacent CVD
ΔE 26.8. Status colours are the fixed status palette and are never reused as series hues.

**Contrast.** Body text is #E8ECF4 on #0B1220 — about 15:1. Secondary text is #9FB0CC at
about 7.5:1. Nothing load-bearing is carried by colour alone: every status chip pairs its
colour with a word.
"""

from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Flowable, Paragraph, Table, TableStyle

# -- palette -------------------------------------------------------------------

INK          = colors.HexColor("#0B1220")   # page ground
INK_2        = colors.HexColor("#0E1626")   # band behind section openers
PANEL        = colors.HexColor("#141C2E")   # glass panel fill
PANEL_HI     = colors.HexColor("#1B2438")   # raised panel / table header
HAIRLINE     = colors.HexColor("#243047")   # 1px rules and panel borders
HAIRLINE_HI  = colors.HexColor("#33415E")

TEXT         = colors.HexColor("#E8ECF4")
TEXT_MUTED   = colors.HexColor("#9FB0CC")
TEXT_FAINT   = colors.HexColor("#6C7F9E")

ACCENT       = colors.HexColor("#3987E5")   # cool accent — series slot 1
ACCENT_SOFT  = colors.HexColor("#1D3352")
WARM         = colors.HexColor("#D95926")   # warm accent — series slot 2
AQUA         = colors.HexColor("#199E70")

GOOD         = colors.HexColor("#0CA30C")
WARNING      = colors.HexColor("#FAB219")
SERIOUS      = colors.HexColor("#EC835A")
CRITICAL     = colors.HexColor("#E8563F")

#: The capability ladder, in the repository's own order, weakest first. Each carries the
#: colour it is drawn in and the single word a reader needs. A rung is never rendered as a
#: bare tick: the word IS the claim.
LADDER = {
    "NOT IMPLEMENTED":   (colors.HexColor("#5A6A86"), "absent"),
    "DECLARED":          (colors.HexColor("#8093B4"), "config exists, no runtime consumer proven"),
    "WIRED":             (colors.HexColor("#6FA8DC"), "a runtime call site reads it"),
    "ENFORCED":          (ACCENT,                      "the runtime refuses or acts on it"),
    "TESTED":            (colors.HexColor("#33B08A"), "covered by a test in this repository"),
    "PARTIAL":           (WARNING,                     "some of it holds; the rest does not"),
    "LIVE LOCAL PROVEN": (AQUA,                        "observed running, on this machine"),
    "LIVE FIELD PROVEN": (GOOD,                        "observed running, on real AWS with a real provider"),
}

PAGE_W, PAGE_H = 210 * mm, 297 * mm
MARGIN_L, MARGIN_R = 18 * mm, 18 * mm
MARGIN_T, MARGIN_B = 22 * mm, 18 * mm
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R

BODY_FONT = "Helvetica"
BOLD_FONT = "Helvetica-Bold"
MONO_FONT = "Courier"


def styles() -> dict[str, ParagraphStyle]:
    """Every paragraph style the document uses, built once."""
    base = getSampleStyleSheet()["Normal"]
    s: dict[str, ParagraphStyle] = {}

    def add(name, **kw):
        s[name] = ParagraphStyle(name, parent=base, **kw)

    add("body", fontName=BODY_FONT, fontSize=9.4, leading=14.2, textColor=TEXT,
        spaceAfter=7, alignment=TA_LEFT)
    add("body_muted", fontName=BODY_FONT, fontSize=9, leading=13.6,
        textColor=TEXT_MUTED, spaceAfter=6)
    add("lede", fontName=BODY_FONT, fontSize=11, leading=16.5, textColor=TEXT,
        spaceAfter=10)
    add("h1", fontName=BOLD_FONT, fontSize=21, leading=25, textColor=TEXT,
        spaceBefore=0, spaceAfter=3)
    add("h2", fontName=BOLD_FONT, fontSize=13.5, leading=17, textColor=TEXT,
        spaceBefore=13, spaceAfter=5)
    add("h3", fontName=BOLD_FONT, fontSize=10.6, leading=14, textColor=ACCENT,
        spaceBefore=10, spaceAfter=3)
    add("kicker", fontName=BOLD_FONT, fontSize=7.6, leading=10, textColor=ACCENT,
        spaceAfter=3)
    add("bullet", fontName=BODY_FONT, fontSize=9.4, leading=14, textColor=TEXT,
        leftIndent=11, bulletIndent=2, spaceAfter=3.5)
    add("cell", fontName=BODY_FONT, fontSize=8, leading=10.6, textColor=TEXT)
    add("cell_muted", fontName=BODY_FONT, fontSize=7.7, leading=10.4, textColor=TEXT_MUTED)
    add("cell_head", fontName=BOLD_FONT, fontSize=7.6, leading=10, textColor=TEXT)
    add("mono", fontName=MONO_FONT, fontSize=8.2, leading=12, textColor=colors.HexColor("#BFD4F2"))
    # Leading sized for the largest run inside it. A mixed-size line in a body style
    # overlaps whatever follows, which is exactly what happened on the pricing cards.
    add("price", fontName=BODY_FONT, fontSize=9, leading=23, textColor=TEXT,
        spaceBefore=2, spaceAfter=5)
    add("caption", fontName=BODY_FONT, fontSize=7.6, leading=10.4, textColor=TEXT_FAINT,
        spaceBefore=3, spaceAfter=8)
    add("toc1", fontName=BOLD_FONT, fontSize=9.6, leading=17, textColor=TEXT)
    add("toc2", fontName=BODY_FONT, fontSize=8.8, leading=14, textColor=TEXT_MUTED,
        leftIndent=12)
    add("cover_title", fontName=BOLD_FONT, fontSize=34, leading=38, textColor=TEXT)
    add("cover_sub", fontName=BODY_FONT, fontSize=13, leading=19, textColor=TEXT_MUTED)
    add("right_muted", fontName=BODY_FONT, fontSize=7.6, leading=10,
        textColor=TEXT_FAINT, alignment=TA_RIGHT)
    return s


S = styles()


def P(text, style="body"):
    return Paragraph(text, S[style])


# -- flowables -----------------------------------------------------------------


class Rule(Flowable):
    """A hairline. Thin on purpose: a rule is punctuation, not a design element."""

    def __init__(self, width=CONTENT_W, colour=HAIRLINE, thickness=0.5, space=4):
        super().__init__()
        self.width, self.colour, self.thickness, self.space = width, colour, thickness, space
        self.height = space

    def wrap(self, *_):
        return self.width, self.height

    def draw(self):
        self.canv.setStrokeColor(self.colour)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.height / 2, self.width, self.height / 2)


class Gap(Flowable):
    def __init__(self, height=6):
        super().__init__()
        self.height = height
        self.width = 0

    def wrap(self, *_):
        return 0, self.height

    def draw(self):
        pass


class SectionOpener(Flowable):
    """The large numeral and title that starts each part.

    A big number is doing real work here rather than decoration: this document is read by
    jumping into it, and a reader flicking pages needs to know which part they are in
    before they read a word.
    """

    HEIGHT = 34 * mm

    def __init__(self, number, title, subtitle=""):
        super().__init__()
        self.number, self.title, self.subtitle = number, title, subtitle
        self.width, self.height = CONTENT_W, self.HEIGHT

    def wrap(self, *_):
        return self.width, self.height

    def draw(self):
        c = self.canv
        h = self.height
        c.saveState()
        c.setFillColor(PANEL)
        c.roundRect(0, 0, self.width, h, 3 * mm, stroke=0, fill=1)
        c.setStrokeColor(HAIRLINE)
        c.setLineWidth(0.6)
        c.roundRect(0, 0, self.width, h, 3 * mm, stroke=1, fill=0)
        # Accent spine on the left edge.
        c.setFillColor(ACCENT)
        c.rect(0, 3 * mm, 1.4 * mm, h - 6 * mm, stroke=0, fill=1)

        c.setFillColor(colors.HexColor("#22314C"))
        c.setFont(BOLD_FONT, 44)
        c.drawString(9 * mm, h - 24 * mm, self.number)

        left = 9 * mm + c.stringWidth(self.number, BOLD_FONT, 44) + 7 * mm
        c.setFillColor(TEXT)
        c.setFont(BOLD_FONT, 17)
        c.drawString(left, h - 17 * mm, self.title)
        if self.subtitle:
            c.setFillColor(TEXT_MUTED)
            c.setFont(BODY_FONT, 9)
            c.drawString(left, h - 24 * mm, self.subtitle)
        c.restoreState()


class Panel(Flowable):
    """A glass panel wrapping other flowables.

    Implemented by measuring the inner content first and painting the panel behind it, so a
    panel is never the wrong height for what is inside it — the single most common way this
    kind of document ends up with clipped text.
    """

    def __init__(self, content, width=CONTENT_W, pad=6 * mm, fill=PANEL,
                 border=HAIRLINE, accent=None, radius=2.6 * mm):
        super().__init__()
        self.content = content
        self.width = width
        self.pad = pad
        self.fill = fill
        self.border = border
        self.accent = accent
        self.radius = radius
        self._heights: list[float] = []
        self.height = 0.0

    def wrap(self, availWidth, availHeight):
        inner = self.width - 2 * self.pad - (2.6 * mm if self.accent else 0)
        self._heights = []
        total = 0.0
        for f in self.content:
            _, h = f.wrap(inner, availHeight)
            self._heights.append(h)
            total += h
        self.height = total + 2 * self.pad
        return self.width, self.height

    def split(self, availWidth, availHeight):
        """Break the panel across a page boundary rather than jumping it whole.

        Without this, a panel taller than the space left on a page moves entirely to the
        next one — which is how a document like this ends up with half-empty pages. The
        split is greedy: take whatever inner flowables fit, and carry the rest into a
        second panel that paints its own background.

        Returning ``[]`` means "do not split, move me whole". That is the answer whenever
        the first piece would still not fit, because a first part that overflows makes no
        progress and ReportLab correctly refuses it as a splitting error.
        """
        if availHeight >= self.height or len(self.content) < 2:
            return []
        inner = self.width - 2 * self.pad - (2.6 * mm if self.accent else 0)
        budget = availHeight - 2 * self.pad
        if budget <= 0:
            return []
        taken, used = [], 0.0
        for f in self.content:
            _, h = f.wrap(inner, availHeight)
            if used + h > budget:
                break
            taken.append(f)
            used += h
        rest = self.content[len(taken):]
        # Everything fit, or too little did. Fewer than two flowables in the first part
        # means a lone heading would end the page with its content overleaf, which reads
        # worse than moving the whole panel.
        if len(taken) < 2 or not rest:
            return []
        first = Panel(taken, width=self.width, pad=self.pad, fill=self.fill,
                      border=self.border, accent=self.accent, radius=self.radius)
        second = Panel(rest, width=self.width, pad=self.pad, fill=self.fill,
                       border=self.border, accent=self.accent, radius=self.radius)
        return [first, second]

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(self.fill)
        c.roundRect(0, 0, self.width, self.height, self.radius, stroke=0, fill=1)
        if self.border is not None:
            c.setStrokeColor(self.border)
            c.setLineWidth(0.6)
            c.roundRect(0, 0, self.width, self.height, self.radius, stroke=1, fill=0)
        x = self.pad
        if self.accent:
            c.setFillColor(self.accent)
            c.rect(0, 2 * mm, 1.3 * mm, self.height - 4 * mm, stroke=0, fill=1)
            x += 2.6 * mm
        y = self.height - self.pad
        inner = self.width - 2 * self.pad - (2.6 * mm if self.accent else 0)
        for f, h in zip(self.content, self._heights):
            y -= h
            f.drawOn(c, x, y)
        c.restoreState()


def callout(title, body, kind="note"):
    """A titled card. ``kind`` picks the accent and is always paired with the title word."""
    accents = {"note": ACCENT, "good": GOOD, "warn": WARNING, "risk": CRITICAL,
               "aqua": AQUA, "warm": WARM}
    accent = accents.get(kind, ACCENT)
    head = ParagraphStyle("calloutHead", parent=S["body"], fontName=BOLD_FONT,
                          fontSize=9.2, leading=12.5, textColor=accent, spaceAfter=3)
    inner = [Paragraph(title, head)]
    for para in ([body] if isinstance(body, str) else body):
        inner.append(Paragraph(para, S["body"]))
    inner[-1] = Paragraph(
        (body if isinstance(body, str) else body[-1]),
        ParagraphStyle("calloutLast", parent=S["body"], spaceAfter=0),
    )
    return Panel(inner, fill=PANEL_HI, accent=accent, pad=5 * mm)


def status_chip(label):
    """A ladder rung drawn as a chip. Colour plus the word, never colour alone."""
    colour, _ = LADDER.get(label, (TEXT_MUTED, ""))
    return Paragraph(
        f'<font color="#{colour.hexval()[2:]}"><b>&#9632;</b></font> '
        f'<font color="#{TEXT.hexval()[2:]}" size="7.2">{label}</font>',
        ParagraphStyle("chip", parent=S["cell"], fontSize=7.2, leading=9.6),
    )


def table(rows, widths, header=True, align=None, zebra=True, font_size=8,
          pad=3.2, head_fill=PANEL_HI, body_fill=PANEL):
    """A table styled once, everywhere.

    Text cells are Paragraphs rather than strings so every column wraps instead of
    overflowing — the failure this document format is most prone to.
    """
    body_style = ParagraphStyle("tcell", parent=S["cell"], fontSize=font_size,
                                leading=font_size * 1.34)
    head_style = ParagraphStyle("thead", parent=S["cell_head"], fontSize=font_size - 0.4,
                                leading=font_size * 1.3)
    data = []
    for r, row in enumerate(rows):
        out = []
        for cell in row:
            if isinstance(cell, Flowable):
                out.append(cell)
            else:
                st = head_style if (header and r == 0) else body_style
                out.append(Paragraph(str(cell), st))
        data.append(out)

    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ("BACKGROUND", (0, 0), (-1, -1), body_fill),
        ("LINEBELOW", (0, 0), (-1, -2), 0.35, HAIRLINE),
        ("BOX", (0, 0), (-1, -1), 0.6, HAIRLINE),
    ]
    if header:
        cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), head_fill),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, HAIRLINE_HI),
        ]
    if zebra:
        start = 1 if header else 0
        for i in range(start, len(data)):
            if (i - start) % 2 == 1:
                cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#172034")))
    if align:
        for col, how in align.items():
            cmds.append(("ALIGN", (col, 0), (col, -1), how))
    t.setStyle(TableStyle(cmds))
    return t


class Checklist(Flowable):
    """An unticked checklist. Boxes are drawn, not typed, so they cannot fail to render.

    Items wrap inside their column and the row height grows to fit, because a checklist
    label is the kind of text that is always one word longer than the column — and an
    item that runs off the page is worse than a slightly taller list.
    """

    def __init__(self, items, width=CONTENT_W, columns=2, line_h=4.0 * mm,
                 pad=2.4 * mm, font_size=8.2):
        super().__init__()
        self.items = items
        self.width = width
        self.columns = columns
        self.line_h = line_h
        self.pad = pad
        self.font_size = font_size
        self._lines: list[list[str]] = []
        self.height = 0.0

    def _layout(self, width):
        """Split every item into the lines it needs at this width, and size the rows."""
        from reportlab.pdfbase.pdfmetrics import stringWidth

        self.width = width
        col_w = width / self.columns
        text_w = col_w - 7.4 * mm          # box + gutter
        self._lines = []
        for item in self.items:
            words, lines, cur = item.split(), [], ""
            for w in words:
                trial = f"{cur} {w}".strip()
                if stringWidth(trial, BODY_FONT, self.font_size) <= text_w:
                    cur = trial
                else:
                    if cur:
                        lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
            self._lines.append(lines or [""])

        # Rows are as tall as the tallest item in them, so the two columns stay aligned.
        self._row_heights = []
        for r in range(0, len(self.items), self.columns):
            tallest = max(len(self._lines[i])
                          for i in range(r, min(r + self.columns, len(self.items))))
            self._row_heights.append(tallest * self.line_h + self.pad)
        self.height = sum(self._row_heights)

    def wrap(self, availWidth, availHeight):
        # Honour the width we are actually given: inside a Panel that is narrower than the
        # page column, and ignoring it is how a checklist ends up past the margin.
        self._layout(availWidth or self.width)
        return self.width, self.height

    def split(self, availWidth, availHeight):
        """Break between rows so a long checklist flows instead of jumping a page.

        A checklist is the one flowable here that is routinely taller than the space left,
        and moving it whole is what strands its heading at the bottom of a page.
        """
        self._layout(availWidth or self.width)
        if availHeight >= self.height:
            return []
        rows = len(self._row_heights)
        used, keep = 0.0, 0
        for r in range(rows):
            if used + self._row_heights[r] > availHeight:
                break
            used += self._row_heights[r]
            keep = r + 1
        if keep == 0 or keep == rows:
            return []
        cut = keep * self.columns
        first = Checklist(self.items[:cut], width=self.width, columns=self.columns,
                          line_h=self.line_h, pad=self.pad, font_size=self.font_size)
        second = Checklist(self.items[cut:], width=self.width, columns=self.columns,
                           line_h=self.line_h, pad=self.pad, font_size=self.font_size)
        return [first, second]

    def draw(self):
        c = self.canv
        col_w = self.width / self.columns
        y_top = self.height
        for i, lines in enumerate(self._lines):
            col, row = i % self.columns, i // self.columns
            x = col * col_w
            row_top = y_top - sum(self._row_heights[:row])
            box_y = row_top - self.line_h + 0.6 * mm
            c.setStrokeColor(HAIRLINE_HI)
            c.setFillColor(colors.HexColor("#101829"))
            c.setLineWidth(0.7)
            c.roundRect(x, box_y, 3.4 * mm, 3.4 * mm, 0.6 * mm, stroke=1, fill=1)
            c.setFillColor(TEXT)
            c.setFont(BODY_FONT, self.font_size)
            for n, line in enumerate(lines):
                c.drawString(x + 5.6 * mm, box_y + 0.9 * mm - n * self.line_h, line)


class Flow(Flowable):
    """A horizontal process flow: numbered nodes joined by arrows.

    Used where a sequence is the point. Wraps to a second band when the caller asks for
    one, because five nodes across A4 leaves each label too narrow to read.
    """

    def __init__(self, steps, width=CONTENT_W, node_h=19 * mm, per_row=None):
        super().__init__()
        self.steps = steps
        self.width = width
        self.node_h = node_h
        self.per_row = per_row or len(steps)
        self.rows = (len(steps) + self.per_row - 1) // self.per_row
        self.height = self.rows * (node_h + 5 * mm)

    def wrap(self, *_):
        return self.width, self.height

    def draw(self):
        c = self.canv
        gap = 4 * mm
        node_w = (self.width - gap * (self.per_row - 1)) / self.per_row
        for i, (title, sub) in enumerate(self.steps):
            col, row = i % self.per_row, i // self.per_row
            x = col * (node_w + gap)
            y = self.height - (row + 1) * (self.node_h + 5 * mm) + 5 * mm
            c.setFillColor(PANEL_HI)
            c.setStrokeColor(HAIRLINE)
            c.setLineWidth(0.6)
            c.roundRect(x, y, node_w, self.node_h, 2 * mm, stroke=1, fill=1)
            c.setFillColor(ACCENT)
            c.setFont(BOLD_FONT, 7)
            c.drawString(x + 3 * mm, y + self.node_h - 5.4 * mm, f"{i + 1:02d}")
            c.setFillColor(TEXT)
            c.setFont(BOLD_FONT, 8)
            last = _wrapped(c, title, x + 3 * mm, y + self.node_h - 9.4 * mm,
                            node_w - 6 * mm, BOLD_FONT, 8, 3.4 * mm, 2)
            if sub:
                c.setFillColor(TEXT_MUTED)
                c.setFont(BODY_FONT, 6.6)
                # Starts below wherever the title actually ended, so a two-line title
                # pushes the subtitle down instead of being written over.
                _wrapped(c, sub, x + 3 * mm, last - 4 * mm, node_w - 6 * mm,
                         BODY_FONT, 6.6, 2.9 * mm, 3)
            if col < self.per_row - 1 and i < len(self.steps) - 1:
                cx = x + node_w + gap / 2
                cy = y + self.node_h / 2
                c.setStrokeColor(HAIRLINE_HI)
                c.setLineWidth(0.9)
                c.line(cx - 1.4 * mm, cy, cx + 1.1 * mm, cy)
                c.setFillColor(HAIRLINE_HI)
                p = c.beginPath()
                p.moveTo(cx + 1.7 * mm, cy)
                p.lineTo(cx + 0.5 * mm, cy + 0.8 * mm)
                p.lineTo(cx + 0.5 * mm, cy - 0.8 * mm)
                p.close()
                c.drawPath(p, stroke=0, fill=1)


def _wrap_lines(c, text, max_w, font, size, max_lines):
    """Break text into at most *max_lines* lines that fit *max_w*."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if c.stringWidth(trial, font, size) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
            if len(lines) == max_lines:
                return lines
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return lines[:max_lines]


def _wrapped(c, text, x, top_y, max_w, font, size, leading, max_lines):
    """Draw wrapped text downward from *top_y*, returning the y it finished at.

    Top-anchored deliberately. The previous version grew upward from its baseline, so a
    two-line subtitle climbed into the title above it — which is exactly what happened in
    the process-flow nodes.
    """
    lines = _wrap_lines(c, text, max_w, font, size, max_lines)
    for i, line in enumerate(lines):
        c.drawString(x, top_y - i * leading, line)
    return top_y - max(len(lines) - 1, 0) * leading


class Timeline(Flowable):
    """A vertical month-by-month roadmap band."""

    def __init__(self, entries, width=CONTENT_W, row_h=11.5 * mm):
        super().__init__()
        self.entries, self.width, self.row_h = entries, width, row_h
        self.height = len(entries) * row_h

    def wrap(self, *_):
        return self.width, self.height

    def draw(self):
        c = self.canv
        spine = 21 * mm
        c.setStrokeColor(HAIRLINE)
        c.setLineWidth(0.8)
        c.line(spine, 1 * mm, spine, self.height - 1 * mm)
        for i, (label, title, detail, colour) in enumerate(self.entries):
            y = self.height - (i + 1) * self.row_h
            c.setFillColor(TEXT_MUTED)
            c.setFont(BOLD_FONT, 7.6)
            c.drawRightString(spine - 4 * mm, y + self.row_h - 5.2 * mm, label)
            c.setFillColor(colour)
            c.circle(spine, y + self.row_h - 4.6 * mm, 1.5 * mm, stroke=0, fill=1)
            c.setFillColor(TEXT)
            c.setFont(BOLD_FONT, 8.6)
            c.drawString(spine + 5 * mm, y + self.row_h - 5.2 * mm, title)
            c.setFillColor(TEXT_MUTED)
            c.setFont(BODY_FONT, 7.4)
            _wrapped(c, detail, spine + 5 * mm, y + self.row_h - 9 * mm,
                     self.width - spine - 6 * mm, BODY_FONT, 7.4, 3.1 * mm, 2)
