"""Renders the PDF to images and reports anything that looks wrong.

The checks below are the ones a generator can make honestly: text running past the frame,
a page that is nearly empty, an image wider than the text column, and pages whose content
box overflows the margins. Everything else needs eyes, which is what the PNGs are for.
"""

import sys
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
PDF = HERE / "NOVA_Commercial_and_AWS_Deployment_Playbook.pdf"
OUT = HERE / "_pages"

# The frame the document lays out into, in points (1mm = 2.8346pt).
MM = 2.834645669
LEFT, RIGHT = 18 * MM, (210 - 18) * MM
TOP, BOTTOM = 22 * MM, (297 - 18) * MM
PAGE_W, PAGE_H = 210 * MM, 297 * MM


def opens_a_section(page):
    """True when this page starts with a section opener.

    Detected by the opener's own 44pt numeral, which nothing else in the document uses.
    The page before one of these is allowed to be short: every section starts on a fresh
    page by design, so trailing whitespace there is layout, not a fault.
    """
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", ()):
            for span in line.get("spans", ()):
                if span["size"] > 40 and span["bbox"][1] < 120:
                    return True
    return False


def main(render=True):
    doc = fitz.open(PDF)
    OUT.mkdir(exist_ok=True)
    problems = []
    print(f"pages: {doc.page_count}")

    for i, page in enumerate(doc, start=1):
        rect = page.rect
        blocks = page.get_text("blocks")
        # Header and footer live outside the frame by design; ignore them.
        body = [b for b in blocks if b[1] > 16 * MM and b[3] < (297 - 13) * MM]

        for b in body:
            x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
            if x1 > RIGHT + 2:
                problems.append((i, "overflows right margin",
                                 f"x1={x1:.1f} > {RIGHT:.1f}: {text[:60]!r}"))
            if x0 < LEFT - 2:
                problems.append((i, "overflows left margin",
                                 f"x0={x0:.1f} < {LEFT:.1f}: {text[:60]!r}"))
            if y1 > BOTTOM + 2:
                problems.append((i, "overflows bottom margin",
                                 f"y1={y1:.1f}: {text[:60]!r}"))

        for im in page.get_images(full=True):
            for r in page.get_image_rects(im[0]):
                if r.x1 > RIGHT + 2 or r.x0 < LEFT - 2:
                    problems.append((i, "image outside the column",
                                     f"{r.x0:.1f}..{r.x1:.1f}"))

        text = page.get_text().strip()
        # A near-empty page usually means a big flowable pushed itself to the next page.
        if i > 2 and len(text) < 260 and not page.get_images():
            problems.append((i, "page is nearly empty", f"{len(text)} chars"))

        # Large trailing whitespace. Legitimate at the end of a section (the next section
        # always opens a new page), so this is reported for review rather than as a fault.
        spans = [b[3] for b in body] + [r.y1 for im in page.get_images(full=True)
                                        for r in page.get_image_rects(im[0])]
        next_opens = i < doc.page_count and opens_a_section(doc[i])  # doc[i] is page i+1
        if spans and i < doc.page_count and not next_opens:
            lowest = max(spans)
            used = (lowest - TOP) / (BOTTOM - TOP)
            if used < 0.60:
                problems.append((i, "large trailing whitespace",
                                 f"content fills {used:.0%} of the frame"))

        if render:
            page.get_pixmap(dpi=105).save(OUT / f"p{i:03d}.png")

    print(f"\nproblems: {len(problems)}")
    for page_no, kind, detail in problems:
        print(f"  p{page_no:>3}  {kind:28s} {detail}")
    return doc.page_count, problems


if __name__ == "__main__":
    main(render="--no-render" not in sys.argv)
