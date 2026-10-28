#!/usr/bin/env python3
"""Normalize Docusaurus-era authored syntax in docs content to Fumadocs forms.

Docusaurus and Fumadocs disagree on two authored forms, and the old forms
render wrong (literal text, wrong fragment IDs) under the Fumadocs MDX
pipeline:

- titled admonitions: ``:::tip Title`` (Docusaurus) vs ``:::tip[Title]``
- heading anchors:    ``## Heading {#id}`` vs ``## Heading [#id]``

Untitled admonitions (``:::info`` … ``:::``) are already valid Fumadocs
syntax and are left alone. Everything inside fenced code blocks is left
alone — code examples never carry authored admonition syntax.

Comments differ per file kind, so the direction is explicit:

- ``target="md"``   (default): whole-line MDX comments ``{/* ... */}``
  render as *visible text* in plain Markdown pages and become ``<!-- ... -->``.
- ``target="mdx"``: HTML comments ``<!-- ... -->`` are not valid in MDX
  (and the root ``no-tmp`` lint marker rides inside one) and become
  ``{/* ... */}``.

Both directions re-anchor stale ``website/…`` script paths that moved to
``apps/docs/…`` in the relocation, so the generator's own comment stays true.

The converter is idempotent (normalized text passes through unchanged) and
never invents identifiers: a malformed anchor (``{#}``, ``{#with space}``,
``{#中文}``) is returned in ``flagged`` for a human decision, never rewritten.

Usage:
    python3 apps/docs/scripts/normalize_content_syntax.py [--check] [path ...]

``--check`` reports what would change (and any flagged token) without
writing; a non-zero exit means the tree is not yet normalized.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_APP = SCRIPT_DIR.parent
REPO = DOCS_APP.parent.parent
CONTENT = DOCS_APP / "content"

# Paths that moved with the relocation and are quoted inside content prose
# (the generated-page comment names its generator). Keep them pointed at the
# file that actually produces the text.
_MOVED_PATHS = {
    "website/scripts/generate-skill-docs.py": "apps/docs/scripts/generate-skill-docs.py",
    "website/scripts/check_doc_links.py": "apps/docs/scripts/check_doc_links.py",
}

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_ADMON_RE = re.compile(r"^(?P<indent>\s*):::(?P<kind>[A-Za-z][A-Za-z0-9_-]*)(?P<rest>\s+.+?)?\s*$")
_HEADING_RE = re.compile(r"^(?P<marks>#{1,6}\s+)(?P<text>.+)$")
# A Docusaurus heading anchor, only at the very end of the line.
_ANCHOR_TAIL_RE = re.compile(r"\{#(?P<id>[^}]*)\}\s*$")
_ANCHOR_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_MDX_COMMENT_RE = re.compile(r"^\s*\{/\*?(?P<body>.*?)\*/\}\s*$")
_HTML_COMMENT_RE = re.compile(r"<!--\s*(?P<body>.*?)\s*-->")


def _remap_moved_paths(text: str) -> str:
    for old, new in _MOVED_PATHS.items():
        text = text.replace(old, new)
    return text


def _convert_admonition_line(line: str) -> str | None:
    """``:::kind Title`` → ``:::kind[Title]``; None when the line is not one."""
    m = _ADMON_RE.match(line)
    if m is None or not m.group("rest"):
        return None  # untitled admonition (or not an admonition): valid as-is
    title = m.group("rest").strip()
    if not title or title.startswith("["):
        return None  # empty title, or already in Fumadocs bracket form
    return f"{m.group('indent')}:::{m.group('kind')}[{title}]"


def _convert_anchor_line(line: str, flagged: list[str], ln: int) -> str:
    """``## Heading {#id}`` → ``## Heading [#id]``, the id verbatim."""
    if not _HEADING_RE.match(line):
        return line
    m = _ANCHOR_TAIL_RE.search(line)
    if m is None:
        return line
    anchor_id = m.group("id")
    # An anchor inside inline code (`{#…}`) is a literal example, not an anchor:
    # an odd number of backticks before it means we are inside a code span.
    if line[: m.start()].count("`") % 2 == 1:
        return line
    head = line[: m.start()].rstrip()
    if not _ANCHOR_ID_RE.match(anchor_id):
        # Never invent a fix: report the exact token for a human decision.
        flagged.append(f"line {ln}: malformed heading anchor {{{anchor_id!r}}} left as-is")
        return line
    return f"{head} [#{anchor_id}]"


def _convert_comments(line: str, target: str) -> str:
    if target == "md":
        m = _MDX_COMMENT_RE.match(line)
        if m:
            return f"<!-- {_remap_moved_paths(m.group('body')).strip()} -->"
        return line
    # target == "mdx": HTML comments are not valid MDX; only the comment span
    # is rewritten, so prose sharing the line survives verbatim.
    return _HTML_COMMENT_RE.sub(
        lambda m: f"{{/* {_remap_moved_paths(m.group('body')).strip()} */}}",
        line,
    )


def iter_fence_aware_lines(text: str) -> Iterator[tuple[bool, str]]:
    """Yield ``(in_fence, line)`` for every line; fenced code stays fenced.

    The opening fence line and everything up to (and including) the closing
    fence line of matching char and length are reported as ``in_fence=True``.
    """
    in_fence = False
    fence_char = ""
    fence_len = 0
    for line in text.split("\n"):
        m = _FENCE_RE.match(line)
        if in_fence:
            yield True, line
            if m and m.group(1)[0] == fence_char and len(m.group(1)) >= fence_len:
                in_fence = False
            continue
        if m:
            fence_char, fence_len = m.group(1)[0], len(m.group(1))
            in_fence = True
            yield True, line
            continue
        yield False, line


def iter_markdown_link_targets(line: str) -> Iterator[str]:
    """Inline ``[text](target)`` targets on one line, outside inline code."""
    probe = re.sub(r"`+[^`]*`+", "", line)
    yield from re.findall(r"\]\(([^)\s]+)\)", probe)


def normalize_fumadocs_syntax(text: str, *, target: str = "md") -> tuple[str, list[str]]:
    """Convert authored Docusaurus syntax in ``text`` to Fumadocs forms.

    Returns ``(normalized_text, flagged)``. ``flagged`` lists malformed anchor
    tokens (and nothing else): those lines are never modified here — they
    need a human decision about the intended identifier.

    ``target`` picks the comment direction; see the module docstring.
    """
    out: list[str] = []
    flagged: list[str] = []
    for ln, (in_fence, line) in enumerate(iter_fence_aware_lines(text), 1):
        if in_fence:
            out.append(line)
            continue
        converted = _convert_admonition_line(line)
        if converted is None:
            converted = _convert_anchor_line(line, flagged, ln)
        out.append(_convert_comments(converted, target))
    return "\n".join(out), flagged


def risky_admonition_titles(text: str) -> list[str]:
    """Admonition titles that nest brackets inside ``:::kind[...]``.

    Fumadocs' deprecated ``remarkAdmonition`` handles nested brackets by
    slicing between the first ``[`` and the last ``]``; the recommended
    ``remarkDirectiveAdmonition`` parses the label with remark-directive.
    These titles must be render-verified against whichever plugin ships —
    the conversion itself is mechanical and already applied, so this scan is
    informational, unlike ``flagged`` (tokens left as-is).
    """
    findings: list[str] = []
    for ln, (in_fence, line) in enumerate(iter_fence_aware_lines(text), 1):
        if in_fence:
            continue
        m = re.match(
            r"^(?P<indent>\s*):::(?P<kind>[A-Za-z][A-Za-z0-9_-]*)\[(?P<title>.*)\]\s*$", line
        )
        if m is None:
            continue
        title = m.group("title")
        if "[" in title or "]" in title or "{" in title or "}" in title:
            findings.append(f"line {ln}: admonition title nests brackets: {line.strip()!r}")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="*", default=None,
                        help="files or directories to normalize (default: apps/docs/content)")
    parser.add_argument("--check", action="store_true",
                        help="report without writing; exit 1 if anything would change")
    args = parser.parse_args(argv)

    roots = [Path(p) for p in (args.paths or [])] or [CONTENT]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(p for p in sorted(root.rglob("*"))
                         if p.suffix in (".md", ".mdx") and p.is_file())

    changed_files: list[str] = []
    flagged_all: list[str] = []
    risky_all: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        target = "mdx" if path.suffix == ".mdx" else "md"
        new_text, flagged = normalize_fumadocs_syntax(text, target=target)
        shown = path.relative_to(REPO) if path.is_relative_to(REPO) else path
        if new_text != text:
            changed_files.append(shown.as_posix())
            if not args.check:
                path.write_text(new_text, encoding="utf-8")
        if flagged:
            flagged_all.extend(f"{shown}: {msg}" for msg in flagged)
        risky_all.extend(f"{shown}: {msg}" for msg in risky_admonition_titles(new_text))

    for rel in changed_files:
        print(f"{'would normalize' if args.check else 'normalized'}: {rel}")
    for msg in flagged_all:
        print(f"FLAGGED (left as-is, needs a human decision): {msg}", file=sys.stderr)
    for msg in risky_all:
        print(f"REVIEW (converted, render-verify the bracketed title): {msg}", file=sys.stderr)

    if args.check:
        return 1 if changed_files else 0
    print(f"{len(files)} file(s) scanned, {len(changed_files)} normalized, "
          f"{len(flagged_all)} token(s) flagged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())