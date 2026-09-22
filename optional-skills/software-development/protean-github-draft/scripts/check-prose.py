#!/usr/bin/env python3
"""Blocking prose gate for anything posted where a human reads it.

Run this BEFORE posting any external text (PR/issue body or comment, review, docs,
client deliverable). Non-zero exit means DO NOT POST. It is mechanical on purpose:
these tells are invisible while you write them, and "I will be careful" has already
failed at least once.

    check-prose.py <file> [<file> ...]
    check-prose.py --text "the string I am about to post"

Exit codes: 0 clean, 1 tells found, 3 usage/IO error.

The rules are intentionally mechanical. A hit is a candidate to fix, not a verdict: a quote from someone else, a code block,
and an em dash inside a quoted command are all legitimate. Read each hit.
"""
import re
import sys

EM_DASH = "\u2014"
EM_DASH_ALLOWED_IN = ("```",)  # fenced code keeps the source text verbatim

BANNED_PHRASES = [
    # frame openers that announce the message instead of being it
    r"two things worth stating",
    r"findings first",
    r"the honest gap",
    r"i want to be upfront",
    r"one caveat worth flagging",
    r"two notes for",
    r"note on mechanism",
    r"a few thoughts",
    r"worth settling",
    r"worth pinning down",
    r"something to consider",
    # cardinality prefaces: the list itself shows the count ("Two integration notes.")
    r"\b(two|three|four|five) ([a-z]+ ){0,3}(notes|points|things|thoughts|observations|findings)[.:]",
    r"^one ([a-z]+ ){1,4}notes?\s*[:.]",
    r"^a quick note\b",
    # self-referential meta-framing
    r"i'?d frame this differently",
    r"i'?d argue",
    r"i would say",
    r"my earlier note",
    r"this deserves",
    r"stand on their own",
    r"deserves precision",
    # narration of the search
    r"after tracing",
    r"it turns out that",
    r"i checked .{0,20} and found",
    # verdicts the reader can form
    r"the right shape",
    r"right design",
    r"exactly what is needed",
    r"the natural single base",
    r"the obvious choice",
    r"a step in the right direction",
    # signposts and closers
    r"^notably",
    r"^importantly",
    r"^it is worth noting",
    r"^at its core",
    r"that is okay",
    r"no shame",
    r"none subsumes another",
    r"the rest follows",
    r"nothing else to add",
]

IDENTIFIER_PATTERNS = [
    # private names and paths must never be published
    r"/Users/", r"/home/", r"/var/folders", r"/private/tmp", r"\b[A-Za-z]:\\",
    r"(?i)\b(api[_-]?key|token|password|secret|credential)\b\s*[:=]\s*\S+",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"(?i)\b(localhost|127\.0\.0\.1|0\.0\.0\.0)\b",
]


def strip_code(text):
    """Drop fenced blocks and inline code so quoted material is not flagged.

    Fence lines are replaced with an empty line rather than removed, so reported line
    numbers still map onto the file the reader will open.
    """
    out, in_fence = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append("")
            continue
        out.append("" if in_fence else re.sub(r"`[^`]*`", "", line))
    return "\n".join(out)


def check(name, raw):
    prose = strip_code(raw)
    findings = []

    for n, line in enumerate(prose.splitlines(), 1):
        if EM_DASH in line:
            findings.append(f"{name}:{n}: em dash (T2) -> use a period or comma: {line.strip()[:110]}")
        if ";" in line and len(line) > 60:
            findings.append(f"{name}:{n}: semicolon stitching clauses (T2) -> prefer a period: {line.strip()[:110]}")

    for pat in BANNED_PHRASES:
        for m in re.finditer(pat, prose, re.IGNORECASE | re.MULTILINE):
            line_no = prose[: m.start()].count("\n") + 1
            findings.append(f"{name}:{line_no}: banned phrase '{m.group(0)}' (AI tell)")

    for pat in IDENTIFIER_PATTERNS:
        for m in re.finditer(pat, raw, re.IGNORECASE):
            line_no = raw[: m.start()].count("\n") + 1
            findings.append(f"{name}:{line_no}: private identifier '{m.group(0)}' -> redact or replace with a generic placeholder")

    # line length: a posted paragraph should scan, not wall
    longs = [n for n, l in enumerate(prose.splitlines(), 1) if len(l) > 420 and not l.strip().startswith("|")]
    if longs:
        findings.append(f"{name}: paragraph over 420 chars on line(s) {longs} -> split it")

    return findings


def main(argv):
    if not argv:
        print(__doc__)
        return 3
    sources = []
    if argv[0] == "--text":
        sources = [("<text>", " ".join(argv[1:]))]
    else:
        for p in argv:
            try:
                with open(p, encoding="utf-8") as fh:
                    sources.append((p, fh.read()))
            except OSError as exc:
                print(f"check-prose: cannot read {p}: {exc}", file=sys.stderr)
                return 3

    findings = []
    for name, text in sources:
        findings += check(name, text)

    if findings:
        print(f"check-prose: {len(findings)} finding(s). DO NOT POST until each is fixed or justified:\n")
        for f in findings:
            print(f"  {f}")
        return 1

    print(f"check-prose: clean ({len(sources)} file(s), {sum(len(t.splitlines()) for _, t in sources)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
