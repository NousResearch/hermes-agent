#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Exact word counts for a document, as JSON.

Emits the document's total word count and a per-heading-section breakdown so
an editorial review can ground word-impact estimates and reduction
percentages in real numbers instead of guessing. Sections are delimited by
markdown headings (# through ######); heading markers inside fenced code
blocks are ignored (fences pair CommonMark-style: a fence closes only on a
run of the same character at least as long, so ```` fences may embed ```
examples). A word is any whitespace-separated token, plus one word per CJK
character since those scripts do not space-delimit words. For non-markdown
input the result is a single section holding the full text.
"""

import argparse
import json
import re
import sys
from pathlib import Path

HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯ｦ-ﾟ]")


def word_count(text: str) -> int:
    cjk = len(CJK.findall(text))
    return cjk + len(CJK.sub(" ", text).split())


def section_metrics(text: str) -> list[dict]:
    sections = []
    current = {"heading": "(preamble)", "level": 0, "body": []}
    open_fence = None  # (char, length) while inside a fenced block
    for line in text.splitlines():
        fence = FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if open_fence is None:
                open_fence = (marker[0], len(marker))
            elif marker[0] == open_fence[0] and len(marker) >= open_fence[1] and line.strip() == marker:
                open_fence = None
            current["body"].append(line)
            continue
        match = None if open_fence else HEADING.match(line)
        if match:
            sections.append(current)
            current = {
                "heading": match.group(2).strip(),
                "level": len(match.group(1)),
                "body": [],
            }
        else:
            current["body"].append(line)
    sections.append(current)

    out = []
    for section in sections:
        words = word_count("\n".join(section["body"]))
        if section["heading"] == "(preamble)" and words == 0:
            continue
        out.append(
            {"heading": section["heading"], "level": section["level"], "words": words}
        )
    return out


def metrics(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "file": str(path),
        "total_words": word_count(text),
        "sections": section_metrics(text),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # JSON is UTF-8 regardless of locale code page
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="document to measure")
    parser.add_argument("-o", "--output", help="write JSON here (default: stdout)")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_file():
        print(f"error: not a readable file: {path}", file=sys.stderr)
        return 2

    result = json.dumps(metrics(path), indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(result + "\n", encoding="utf-8")
    else:
        print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
