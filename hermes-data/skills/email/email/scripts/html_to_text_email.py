#!/usr/bin/env python3
"""Convert a hand-authored HTML email into a readable plain-text alternative.

Usage:
    python3 html_to_text_email.py email.html [email.txt]
    python3 html_to_text_email.py email.html --stdout

Why this exists: a naive `<[^>]+>` strip leaves table rows as `Structure |` fragments
scattered over separate lines, and hand-formatted HTML routinely puts a row's two cells
on different source lines. This converter turns each two-cell row into a single
`Label — value` line, keeps paragraph breaks, and drops the trailing separator that the
row's last empty cell leaves behind.

Used for the text/plain part of a multipart/alternative draft; see
references/html-email-mime-and-render-verification.md for the full packaging recipe.
"""
import html as htmllib
import re
import sys

CELL_SEP = " \u2014\x03"       # em dash + marker written by </td>
MARKER = "\x03"


def html_to_text(html_body: str) -> str:
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html_body, flags=re.S)
    t = re.sub(r"<t[dh][^>]*>", "", t)
    t = re.sub(r"</t[dh]>", CELL_SEP, t)          # cell boundary -> "label — value"
    t = re.sub(r"</tr>", "\n", t)
    t = re.sub(r"<br\s*/?>", "\n", t)
    t = re.sub(r"</(p|li|h1|h2|h3|div|header|footer|blockquote)>", "\n\n", t)
    t = htmllib.unescape(re.sub(r"<[^>]+>", "", t))

    raw = [re.sub(r"[ \t]+", " ", line).strip() for line in t.split("\n")]

    out, i = [], 0
    while i < len(raw):
        line = raw[i]
        sep = MARKER in line
        line = line.replace(MARKER, "").strip()
        if sep:
            # the row's value sits on the next non-empty line
            j = i + 1
            while j < len(raw) and not raw[j].strip():
                j += 1
            if j < len(raw):
                val = raw[j].replace(MARKER, "").strip()
                row = "%s %s" % (line, val)
                row = re.sub(r"\s*\u2014\s*$", "", row)   # trailing empty cell
                out.append(row)
                i = j + 1
                continue
        out.append(line)
        i += 1

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    src = open(argv[1], encoding="utf-8").read()
    text = html_to_text(src)
    if "--stdout" in argv:
        print(text)
        return 0
    dst = argv[2] if len(argv) > 2 and not argv[2].startswith("-") else argv[1].rsplit(".", 1)[0] + ".txt"
    open(dst, "w", encoding="utf-8").write(text)
    print("wrote %s (%d chars)" % (dst, len(text)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
