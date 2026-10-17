"""Classic-CLI rendering of an edit's inline diff: a ``┊ ✎ edited path  +N −M`` header per file,
then rows with a line-number gutter and full-width tinted backgrounds for added/removed lines.

Input is the ``inline_diff`` text ``agent.display.render_edit_diff_with_delta`` produced (the
same payload the TUI and desktop receive, so their wire format is untouched): an optional review
header, ``a/x → b/x`` file lines, ``@@`` hunks and ``+``/``-``/`` `` rows, plus an optional
"omitted" summary. Re-rendered here, client side only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from agent.i18n import t
from hermes_cli.cli_tool_render import DIM, RST, clean_text, clip, error_color, ok_color

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_FILE_RE = re.compile(r"^(?P<a>\S.*?) → (?P<b>\S.*)$")
_INDENT = "    "
_TOKEN_RE = re.compile(r"\w+|\s+|[^\w\s]")
# Paired lines this different are a rewrite, not an edit: highlighting most of the row says nothing.
_MIN_PAIR_SIMILARITY = 0.4
# SequenceMatcher is quadratic in the worst case (one value changed in a long single-line JSON
# array): pairs longer than this get plain row colours instead of a word highlight.
_MAX_PAIR_CHARS = 400
_MAX_PAIR_TOKENS = 200


@dataclass
class DiffFile:
    path: str
    rows: list[tuple[str, int | None, str]] = field(default_factory=list)
    added: int = 0
    removed: int = 0


def _file_path(match: re.Match) -> str:
    a, b = match.group("a").strip(), match.group("b").strip()
    path = a if b == "/dev/null" else b
    return path[2:] if path[:2] in ("a/", "b/") else path


class _Parser:
    def __init__(self):
        self.files: list[DiffFile] = []
        self.trailer: list[str] = []
        self.old = self.new = 0

    def current(self) -> DiffFile:
        if not self.files:
            self.files.append(DiffFile("?"))
        return self.files[-1]

    def hunk(self, match: re.Match) -> None:
        current = self.current()
        if current.rows:
            current.rows.append(("gap", None, ""))
        self.old, self.new = int(match.group(1)), int(match.group(2))

    def row(self, kind: str, body: str) -> None:
        current = self.current()
        if kind == "+":
            current.rows.append(("+", self.new, body))
            current.added += 1
            self.new += 1
        elif kind == "-":
            current.rows.append(("-", self.old, body))
            current.removed += 1
            self.old += 1
        else:
            current.rows.append((" ", self.new, body))
            self.old += 1
            self.new += 1

    def feed(self, line: str) -> None:
        if line.startswith("\\"):
            return
        hunk = _HUNK_RE.match(line)
        if hunk:
            return self.hunk(hunk)
        if line[:1] in ("+", "-", " ") and self.files:
            return self.row(line[0], line[1:])
        found = _FILE_RE.match(line)
        if found:
            self.files.append(DiffFile(_file_path(found)))
        elif line.strip():
            self.trailer.append(line.strip())


def _is_review_header(line: str) -> bool:
    from agent.display import get_skin_tool_prefix
    stripped = line.strip()
    return stripped.startswith(("┊", get_skin_tool_prefix())) or stripped == clean_text(t("display.diff.review_header")).strip()


def parse_inline_diff(text: str) -> tuple[list[DiffFile], list[str]]:
    """``(files, trailer)`` from rendered inline-diff text (escape sequences ignored)."""
    parser = _Parser()
    for index, raw in enumerate((text or "").splitlines()):
        line = clean_text(raw)
        if index == 0 and _is_review_header(line):
            continue
        parser.feed(line)
    return parser.files, parser.trailer


def _pad(text: str, width: int) -> str:
    from prompt_toolkit.utils import get_cwidth
    text = clip(text, width)
    return text + " " * max(0, width - get_cwidth(text))


def changed_spans(old: str, new: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Character spans that differ between a removed line and the added line paired with it,
    compared token by token (words, runs of spaces, punctuation). ``None`` when the lines share
    too little for a word-level highlight to help, or when either line is too long to compare
    cheaply."""
    if len(old) > _MAX_PAIR_CHARS or len(new) > _MAX_PAIR_CHARS:
        return None
    a, b = _TOKEN_RE.findall(old), _TOKEN_RE.findall(new)
    if len(a) > _MAX_PAIR_TOKENS or len(b) > _MAX_PAIR_TOKENS:
        return None
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    if matcher.ratio() < _MIN_PAIR_SIMILARITY:
        return None
    offsets_a, offsets_b = [0], [0]
    for token in a:
        offsets_a.append(offsets_a[-1] + len(token))
    for token in b:
        offsets_b.append(offsets_b[-1] + len(token))
    spans_a, spans_b = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:
            spans_a.append((offsets_a[i1], offsets_a[i2]))
        if j2 > j1:
            spans_b.append((offsets_b[j1], offsets_b[j2]))
    return spans_a, spans_b


def _pair_rows(rows: list[tuple[str, int | None, str]]) -> dict[int, list[tuple[int, int]]]:
    """Row index -> changed spans, for each ``-`` run followed by a ``+`` run (paired in order)."""
    spans: dict[int, list[tuple[int, int]]] = {}
    index = 0
    while index < len(rows):
        if rows[index][0] != "-":
            index += 1
            continue
        minus_start = index
        while index < len(rows) and rows[index][0] == "-":
            index += 1
        plus_start = index
        while index < len(rows) and rows[index][0] == "+":
            index += 1
        for m, p in zip(range(minus_start, plus_start), range(plus_start, index)):
            found = changed_spans(rows[m][2], rows[p][2])
            if found is not None:
                spans[m], spans[p] = found
    return spans


def _highlight(body: str, spans: list[tuple[int, int]], row: str, emphasis: str) -> str:
    """``body`` with each span on the brighter ``emphasis`` background, returning to ``row``."""
    out, pos = [], 0
    for start, end in spans:
        out.append(body[pos:start])
        out.append(f"{emphasis}{body[start:end]}{row}")
        pos = end
    out.append(body[pos:])
    return "".join(out)


def _render_row(kind: str, number: int | None, body: str, digits: int, width: int, colors: dict,
                spans: list[tuple[int, int]] | None = None) -> str:
    if kind == "gap":
        return f"{DIM}{_INDENT}{' ' * digits} ⋮{RST}"
    text = f"{_INDENT}{str(number).rjust(digits)} {kind} {body}"
    if kind == " ":
        return f"{colors['dim']}{clip(text, width)}{RST}"
    key = "plus" if kind == "+" else "minus"
    row = colors[key]
    padded = _pad(text, width)
    emphasis = colors.get(f"{key}_em")
    if not spans or not emphasis:
        return f"{row}{padded}{RST}"
    from prompt_toolkit.utils import get_cwidth
    prefix = f"{_INDENT}{str(number).rjust(digits)} {kind} "
    shown = padded[len(prefix):]
    visible = shown.rstrip(" ")
    if visible.endswith("…") and get_cwidth(text) > width:  # clipped: never highlight past the cut
        visible = visible[:-1]
    clipped = [(start, min(end, len(visible))) for start, end in spans if start < len(visible)]
    return f"{row}{prefix}{_highlight(shown, clipped, row, emphasis)}{RST}"


def _header(diff: DiffFile) -> str:
    from agent.display import get_skin_tool_prefix
    counts = f"{ok_color()}+{diff.added}{RST} {error_color()}−{diff.removed}{RST}"
    return f"  {get_skin_tool_prefix()} ✎ {t('cli.tool_render.edited', path=diff.path)}  {counts}"


def render_inline_diff(text: str, width: int) -> list[str]:
    """Header + gutter rows per file; added/removed rows are padded to ``width`` cells so the
    background reads as a bar. Unparseable input is returned line by line, unchanged."""
    files, trailer = parse_inline_diff(text)
    if not any(f.rows for f in files):
        return (text or "").rstrip("\n").splitlines()
    from agent.display import _diff_ansi
    colors = _diff_ansi()
    width = max(20, width - 1)  # never write the last column: some terminals wrap there
    out: list[str] = []
    for diff in files:
        out.append(_header(diff))
        numbers = [n for _, n, _ in diff.rows if n is not None]
        digits = len(str(max(numbers))) if numbers else 1
        spans = _pair_rows(diff.rows)
        out.extend(_render_row(kind, number, body, digits, width, colors, spans.get(index))
                   for index, (kind, number, body) in enumerate(diff.rows))
    out.extend(f"{DIM}{_INDENT}{line}{RST}" for line in trailer)
    return out


def plain_diff(text: str) -> str:
    """The diff without escape sequences or the review header, for the transcript viewer."""
    lines = clean_text(text or "").strip("\n").splitlines()
    if lines and _is_review_header(lines[0]):
        lines = lines[1:]
    return "\n".join(lines)
