#!/usr/bin/env python3
"""Build a Persian RTL .pptx from a JSON spec.

Slide types: title, agenda, bullets, split (image/map + bullets + overlay
labels), table, timeline, cards, sources, closing. Every paragraph is RTL
(pPr rtl=1 + right align); every run carries complex-script font props so
Persian renders correctly in both PowerPoint and Impress. Latin digits are
auto-converted to Persian unless "persian_digits": false.

Spec schema: see references/spec.md in the skill directory.
"""
import argparse
import json
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml
from pptx.util import Inches, Pt

A_NS = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'

DEFAULT_THEME = {
    "deep": "0E3B2E", "gold": "C9A227", "gold_lt": "D9C998",
    "cream": "F7F3E9", "ink": "20261C", "muted": "6E705F",
    "card": "FFFFFF", "card_alt": "F1ECDC", "note": "EFEAD6",
    "title_size": 36, "body_size": 22, "footer": "دفاع مقدس",
}

# ---------------------------------------------------------------- helpers
FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
AUTO_PERSIAN_DIGITS = True


def fa(s):
    return str(s).translate(FA_DIGITS) if AUTO_PERSIAN_DIGITS else str(s)


def set_rtl(p, align="r"):
    pPr = p._p.get_or_add_pPr()
    pPr.set("rtl", "1")
    pPr.set("algn", align)


def style_run(r, font_name, size, bold=False, color="20261C", italic=False):
    f = r.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.name = font_name
    f.color.rgb = RGBColor.from_string(color)
    rPr = r._r.get_or_add_rPr()
    rPr.set("lang", "fa-IR")
    rPr.set("altLang", "en-US")
    rPr.set("dirty", "0")
    rPr.set("sz", str(int(size * 100)))
    rPr.set("szCs", str(int(size * 100)))
    if bold:
        rPr.set("b", "1")
        rPr.set("bCs", "1")
    if italic:
        rPr.set("i", "1")
        rPr.set("iCs", "1")
    rPr.set("rtl", "1")
    latin = rPr.find(qn("a:latin"))
    cs = parse_xml(f'<a:cs {A_NS} typeface="{font_name}"/>')
    if latin is not None:
        latin.addnext(cs)
    else:
        rPr.append(cs)


def para(tf, text, font_name, size=22, bold=False, color="20261C", align="r",
         space_after=8, space_before=0, line=1.18, first=False,
         marker=False, marker_color="C9A227"):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    set_rtl(p, align)
    p.space_after = Pt(space_after)
    p.space_before = Pt(space_before)
    p.line_spacing = line
    if marker:
        m = p.add_run()
        style_run(m, font_name, size, bold=True, color=marker_color)
        m.text = "•  "
    r = p.add_run()
    style_run(r, font_name, size, bold=bold, color=color)
    r.text = fa(text)
    return p


def tbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    return tb, tf


def rect(slide, x, y, w, h, fill=None, line=None, lw=1.0,
         shape=MSO_SHAPE.RECTANGLE, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = RGBColor.from_string(fill)
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = RGBColor.from_string(line)
        s.line.width = Pt(lw)
    s.shadow.inherit = False
    s.text_frame.word_wrap = True
    if name:
        s.name = name
    return s


def set_bg(slide, color):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string(color)


def set_gradient(slide, c1, c2, angle=125.0):
    try:
        fill = slide.background.fill
        fill.gradient()
        stops = fill.gradient_stops
        stops[0].color.rgb = RGBColor.from_string(c1)
        stops[1].color.rgb = RGBColor.from_string(c2)
        fill.gradient_angle = angle
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- builder
class Deck:
    def __init__(self, spec):
        global AUTO_PERSIAN_DIGITS
        AUTO_PERSIAN_DIGITS = spec.get("persian_digits", True)
        self.spec = spec
        self.t = dict(DEFAULT_THEME)
        self.t.update(spec.get("theme", {}))
        self.font = spec.get("font", "Vazirmatn")
        self.W, self.H = 13.333, 7.5
        self.M = 0.65
        self.body_w = self.W - 2 * self.M
        self.prs = Presentation()
        self.prs.slide_width = Inches(self.W)
        self.prs.slide_height = Inches(self.H)
        self.footer_txt = spec.get("footer", self.t["footer"])
        self.n = 0

    def blank(self):
        return self.prs.slides.add_slide(self.prs.slide_layouts[6])

    def footer(self, slide):
        rect(slide, self.M, 6.98, self.body_w, 0.014, fill="DCD2A6", name="footline")
        _, tf = tbox(slide, self.M, 7.08, 3.2, 0.3)
        para(tf, str(self.n), self.font, size=12, color=self.t["muted"],
             first=True, space_after=0)
        _, tf2 = tbox(slide, self.W - self.M - 5.0, 7.08, 5.0, 0.3)
        para(tf2, self.footer_txt, self.font, size=12, color=self.t["muted"],
             first=True, space_after=0)

    def header(self, slide, title, subtitle=None):
        _, tf = tbox(slide, self.M, 0.42, self.body_w, 0.82)
        para(tf, title, self.font, size=self.t["title_size"], bold=True,
             color=self.t["deep"], first=True, space_after=0, line=1.0)
        rect(slide, self.M, 1.34, self.body_w, 0.02, fill="DCD2A6")
        rect(slide, self.M + self.body_w - 4.4, 1.28, 4.4, 0.055, fill=self.t["gold"])
        if subtitle:
            _, tf2 = tbox(slide, self.M, 1.55, self.body_w, 0.36)
            para(tf2, subtitle, self.font, size=15, color=self.t["muted"],
                 first=True, space_after=0)

    def notes(self, slide, text):
        tf = slide.notes_slide.notes_text_frame
        p = tf.paragraphs[0]
        set_rtl(p, "r")
        r = p.add_run()
        r.text = text
        r.font.name = self.font
        r.font.size = Pt(13)
        r.font.color.rgb = RGBColor.from_string("333333")

    def bullets(self, slide, items, x=None, y=1.95, w=None, size=None):
        x = self.M if x is None else x
        w = self.body_w if w is None else w
        size = self.t["body_size"] if size is None else size
        _, tf = tbox(slide, x, y, w, 4.6)
        for i, it in enumerate(items):
            if isinstance(it, dict) and it.get("sub"):
                para(tf, it["text"], self.font, size=size, first=(i == 0),
                     space_after=10, line=1.22, marker=True, marker_color=self.t["gold"])
                para(tf, it["sub"], self.font, size=size - 3,
                     color=self.t["muted"], space_after=10, line=1.22)
            else:
                text = it["text"] if isinstance(it, dict) else it
                size_i = it.get("size", size) if isinstance(it, dict) else size
                para(tf, text, self.font, size=size_i, first=(i == 0),
                     space_after=10, line=1.22, marker=True, marker_color=self.t["gold"])
        return tf

    # ---- slide types ----
    def s_title(self, s):
        slide = s["_shapes_slide"]
        ok = set_gradient(slide, "0B2E22", "1B4633")
        if not ok:
            set_bg(slide, self.t["deep"])
        pattern = s.get("pattern")
        if pattern:
            slide.shapes.add_picture(pattern, Inches(0), Inches(0),
                                     width=Inches(self.W), height=Inches(self.H)).name = "pattern"
        rect(slide, 0.30, 0.30, self.W - 0.60, self.H - 0.60,
             fill=None, line=self.t["gold"], lw=1.25, name="frame")
        rect(slide, 0.38, 0.38, self.W - 0.76, self.H - 0.76,
             fill=None, line="8A6F2A", lw=0.5, name="frame2")
        _, tf = tbox(slide, 1.1, 1.28, 8.0, 0.5)
        para(tf, s.get("kicker", ""), self.font, size=16, bold=True,
             color=self.t["gold"], first=True, space_after=0)
        _, tf = tbox(slide, 1.1, 1.95, 11.0, 1.3)
        para(tf, s.get("title", ""), self.font, size=54, bold=True,
             color="F3EBD4", first=True, space_after=0, line=1.0)
        rect(slide, 1.15, 3.28, 4.6, 0.05, fill=self.t["gold"])
        _, tf = tbox(slide, 1.1, 3.55, 9.6, 0.9)
        if s.get("subtitle"):
            para(tf, s["subtitle"], self.font, size=26, color="E4DCC0",
                 first=True, space_after=6)
        if s.get("subtitle2"):
            para(tf, s["subtitle2"], self.font, size=19, color="BFAF82", space_after=0)
        _, tf = tbox(slide, 1.1, 5.85, 7.0, 0.9)
        if s.get("presenter"):
            para(tf, s["presenter"], self.font, size=18, color="D8CFB4",
                 first=True, space_after=6)
        if s.get("date"):
            para(tf, s["date"], self.font, size=18, color="D8CFB4", space_after=0)

    def s_agenda(self, s):
        slide = s["_shapes_slide"]
        set_bg(slide, self.t["cream"])
        self.header(slide, s.get("title", "فهرست مطالب"))
        y0, row_h = 1.85, 0.60
        for i, item in enumerate(s["items"]):
            yy = y0 + i * row_h
            c = rect(slide, self.W - self.M - 0.44, yy, 0.44, 0.44,
                     fill=self.t["gold"], shape=MSO_SHAPE.OVAL)
            ctf = c.text_frame
            ctf.margin_left = ctf.margin_right = ctf.margin_top = ctf.margin_bottom = 0
            para(ctf, str(i + 1), self.font, size=17, bold=True, color="FFFFFF",
                 align="c", first=True, space_after=0, line=1.0)
            _, tf = tbox(slide, self.M, yy + 0.03, self.body_w - 0.70, 0.5)
            para(tf, item, self.font, size=self.t["body_size"], color=self.t["ink"],
                 first=True, space_after=0, line=1.0)
        self.footer(slide)

    def s_bullets(self, s):
        slide = s["_shapes_slide"]
        set_bg(slide, self.t["cream"])
        self.header(slide, s["title"], s.get("subtitle"))
        self.bullets(slide, s["items"], y=1.95)
        self.footer(slide)

    def s_split(self, s):
        """image on the left, bullets on the right (RTL reading order)."""
        slide = s["_shapes_slide"]
        set_bg(slide, self.t["cream"])
        self.header(slide, s["title"], s.get("subtitle"))
        img = s["image"]
        ix, iy, iw, ih = img["x"], img["y"], img["w"], img["h"]
        if img.get("card", False):
            rect(slide, ix - 0.12, iy - 0.12, iw + 0.24, ih + 0.24,
                 fill=self.t["card"], line="DCD2A6", lw=0.75)
        slide.shapes.add_picture(img["path"], Inches(ix), Inches(iy),
                                 width=Inches(iw), height=Inches(ih)).name = "picture"
        bx = ix + iw + 0.25
        bw = self.W - self.M - bx
        self.bullets(slide, s["bullets"], x=bx, y=img.get("bullets_y", 1.95), w=bw,
                     size=img.get("bullets_size", 20))
        for key, lb in s.get("labels", {}).items():
            text = lb["text"]
            size = lb.get("size", 13)
            w_box = lb.get("w") or max(1.15 if len(text) <= 3 else 1.55,
                                       0.12 * len(text) + 0.35)
            px = ix + lb["fx"] * iw
            py = iy + lb["fy"] * ih
            if lb.get("align", "l") == "r":
                x = px - w_box - 0.04
            else:
                x = px + 0.04
            x = min(max(x, ix + 0.03), ix + iw - w_box - 0.03)
            y = min(max(py - 0.14, iy + 0.03), iy + ih - 0.30)
            _, tf = tbox(slide, x, y, w_box, 0.28)
            para(tf, text, self.font, size=size, bold=lb.get("bold", False),
                 color=lb.get("color", self.t["ink"]), first=True, space_after=0, line=1.0)
        self.footer(slide)

    def s_table(self, s):
        slide = s["_shapes_slide"]
        set_bg(slide, self.t["cream"])
        self.header(slide, s["title"], s.get("subtitle"))
        headers = s["headers"]
        rows = s["rows"]
        ncol = len(headers)
        gw = self.body_w / ncol
        nrow = len(rows) + 1
        row_h = s.get("row_height", 0.86)
        tbl = slide.shapes.add_table(nrow, ncol, Inches(self.M), Inches(1.80),
                                     Inches(self.body_w), Inches(row_h * nrow)).table
        for c in range(ncol):
            tbl.columns[c].width = Inches(gw)
        tbl.rows[0].height = Inches(0.62)
        for c, htext in enumerate(headers):
            cell = tbl.cell(0, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string(self.t["deep"])
            cell.margin_left = cell.margin_right = Inches(0.22)
            cell.margin_top = cell.margin_bottom = Inches(0.08)
            para(cell.text_frame, htext, self.font, size=20, bold=True,
                 color="EFE7CD", first=True, space_after=0, line=1.0)
        for r, row in enumerate(rows):
            tbl.rows[r + 1].height = Inches(row_h)
            for c, val in enumerate(row):
                cell = tbl.cell(r + 1, c)
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor.from_string(
                    self.t["card"] if r % 2 == 0 else self.t["card_alt"])
                cell.margin_left = cell.margin_right = Inches(0.22)
                cell.margin_top = cell.margin_bottom = Inches(0.10)
                para(cell.text_frame, val, self.font, size=18, color=self.t["ink"],
                     first=True, space_after=0, line=1.12)
        if s.get("note"):
            ny = 1.80 + row_h * nrow + 0.25
            nb = rect(slide, self.M, ny, self.body_w, s.get("note_height", 0.95),
                      fill=self.t["note"], line="D9C998", lw=0.75)
            ntf = nb.text_frame
            ntf.margin_left = ntf.margin_right = Inches(0.25)
            ntf.margin_top = Inches(0.12)
            para(ntf, s["note"], self.font, size=15, color=self.t["ink"],
                 first=True, space_after=0, line=1.15)
        self.footer(slide)

    def s_timeline(self, s):
        slide = s["_shapes_slide"]
        set_bg(slide, self.t["cream"])
        self.header(slide, s["title"], s.get("subtitle"))
        events = s["events"]
        y0, row_h = 1.88, 0.74
        line_x = self.W - self.M - 1.05
        rect(slide, line_x, y0 + 0.18, 0.02, row_h * (len(events) - 1) + 0.1,
             fill=self.t["gold"], name="spine")
        for i, (date, txt) in enumerate(events):
            yy = y0 + i * row_h
            rect(slide, line_x - 0.08, yy + 0.13, 0.18, 0.18,
                 fill=self.t["gold"], shape=MSO_SHAPE.OVAL)
            _, tf = tbox(slide, line_x - 4.35, yy + 0.04, 4.05, 0.34)
            para(tf, date, self.font, size=18, bold=True, color=self.t["deep"],
                 first=True, space_after=0, line=1.0)
            _, tf = tbox(slide, self.M, yy + 0.04, line_x - self.M - 4.55, 0.34)
            para(tf, txt, self.font, size=18, color=self.t["ink"],
                 first=True, space_after=0, line=1.0)
        self.footer(slide)

    def s_cards(self, s):
        slide = s["_shapes_slide"]
        set_bg(slide, self.t["cream"])
        self.header(slide, s["title"], s.get("subtitle"))
        cards = s["cards"][:2]
        gap = 0.33
        cw = (self.body_w - gap) / 2
        for i, card in enumerate(cards):
            cx = self.M + i * (cw + gap)
            rect(slide, cx, 1.95, cw, card.get("h", 2.5), fill=self.t["card"],
                 line="D9C998", lw=0.75)
            rect(slide, cx, 1.95, 0.09, card.get("h", 2.5), fill=card["color"])
            _, tf = tbox(slide, cx + 0.45, 2.25, cw - 0.70, card.get("h", 2.5) - 0.4)
            para(tf, card["title"], self.font, size=21, bold=True,
                 color=card["color"], first=True, space_after=10)
            for line in card.get("lines", []):
                para(tf, line, self.font, size=19, color=self.t["ink"],
                     space_after=8, line=1.2)
        if s.get("note"):
            _, tf = tbox(slide, self.M, 4.85, self.body_w, 1.3)
            para(tf, s["note"], self.font, size=17, color=self.t["muted"],
                 first=True, space_after=0, line=1.3)
        self.footer(slide)

    def s_sources(self, s):
        slide = s["_shapes_slide"]
        set_bg(slide, self.t["cream"])
        self.header(slide, s.get("title", "منابع"))
        y0 = 1.85
        for i, src in enumerate(s["items"]):
            yy = y0 + i * s.get("row_height", 0.58)
            _, tf = tbox(slide, self.M + 0.55, yy, self.body_w - 0.55, 0.44)
            para(tf, f"{i+1}.  {src}", self.font, size=18,
                 color=self.t["ink"], first=True, space_after=0, line=1.05)
        self.footer(slide)

    def s_closing(self, s):
        slide = s["_shapes_slide"]
        ok = set_gradient(slide, "0B2E22", "1B4633")
        if not ok:
            set_bg(slide, self.t["deep"])
        if s.get("pattern"):
            slide.shapes.add_picture(s["pattern"], Inches(0), Inches(0),
                                     width=Inches(self.W), height=Inches(self.H)).name = "pattern"
        rect(slide, 0.30, 0.30, self.W - 0.60, self.H - 0.60,
             fill=None, line=self.t["gold"], lw=1.25, name="frame")
        _, tf = tbox(slide, 1.1, 2.55, 11.0, 1.2)
        para(tf, s.get("title", "با تشکر از توجه شما"), self.font, size=44,
             bold=True, color="F3EBD4", first=True, space_after=0, line=1.0)
        rect(slide, 1.15, 3.85, 4.0, 0.05, fill=self.t["gold"])
        _, tf = tbox(slide, 1.1, 4.15, 9.5, 0.8)
        if s.get("subtitle"):
            para(tf, s["subtitle"], self.font, size=22, color="C9BC93",
                 first=True, space_after=0)

    def build(self):
        for s in self.spec["slides"]:
            kind = s["type"]
            slide = self.blank()
            s["_shapes_slide"] = slide
            s["_shapes"] = slide.shapes
            if kind not in ("title", "closing"):
                self.n += 1
            getattr(self, "s_" + kind)(s)
            if s.get("notes"):
                self.notes(slide, s["notes"])
        return self.prs


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="path to JSON deck spec")
    ap.add_argument("output", help="output .pptx path")
    args = ap.parse_args(argv)
    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)
    prs = Deck(spec).build()
    prs.save(args.output)
    print(json.dumps({"ok": True, "output": args.output,
                      "slides": len(prs.slides._sldIdLst)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
