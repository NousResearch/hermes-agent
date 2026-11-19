#!/usr/bin/env python3
"""Desktop redesign design-token guard (drift guard G1) and DESIGN.md token-table generator.

The redesign ("portal shell", ``:root[data-shell='portal']``) keeps every design value in one
folder, ``apps/desktop/src/app/redesign/tokens/``. Outside it, redesign CSS reads tokens; it does
not restate values. This guard finds the declarations that still do:

  raw-color      a hex, rgb()/hsl()/… or named colour (transparent/currentColor/inherit are fine)
  inline-mix     an inline ``color-mix(…var(--ui-base) N%…)`` or brand-fill mix (use an ink/role token)
  radius         a px/rem/em literal in ``border-*-radius``
  font-size      a px/rem literal in ``font-size`` or the ``font`` shorthand
  letter-spacing a non-zero length literal in ``letter-spacing``
  line-height    a px/rem literal in ``line-height`` (unitless ratios are fine)
  duration       an ms/s literal in a ``transition*``/``animation*`` declaration
  z-index        an integer literal above 1
  token-literal  a ``--portal-*`` property defined outside tokens/ with a literal value

Scope: every rule gated on ``[data-shell='portal']`` in any ``apps/desktop/src`` CSS file, plus
every rule of the redesign-only files (``_REDESIGN_ONLY``), never ``tokens/`` itself. Classic CSS
is never judged.

Ratchet: ``scripts/desktop_design_tokens_baseline.json`` holds each file's finding count from the
day the guard landed; a file may only go down. A higher count fails; a lower one is an advisory
to refresh the baseline (``--write-baseline``). Four findings always fail, baseline or not:

  undefined-token  ``var(--portal-…)`` naming a property defined nowhere (a CSS declaration, a TS
                   style key or ``setProperty``; a comment neither defines nor reads): a typo or a
                   token deleted while still in use
  page-column      ``var(--portal-column-page)`` read outside tokens/ and the page stage contract
                   (``app/redesign/portal-pages.css``): a page sizing its own column instead of
                   marking its stage ``page-stage``, which is how pages drift apart
  page-top         a negated ``--portal-page-top``/``--portal-settings-top`` on a block-start margin,
                   inset or translate: a row pulled up into the page's top inset, onto the stage's
                   edge (the routine detail's breadcrumb did this)
  stale-docs       the generated token tables in ``apps/desktop/DESIGN.md`` differ from tokens/

Tokens defined in tokens/ that nothing reads are listed as an advisory.

Opt out of one declaration with ``/* token-exempt: <why> */`` directly before it or after it on
the same line, or of a whole rule with the comment directly before its selector. Use it for
values the language deliberately does not tokenise (a Figma optical offset, art-direction
percentages), never to dodge a token that exists.

Run: python scripts/check_desktop_design_tokens.py [--list] [--write-baseline] [--write-docs]
  --list            print every finding (burn-down view)
  --write-baseline  rewrite the baseline from the current tree
  --write-docs      regenerate the token tables in apps/desktop/DESIGN.md
Exit 1 on a violation, 0 when clean.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DESKTOP = REPO / "apps" / "desktop"
SRC = DESKTOP / "src"
TOKENS = "app/redesign/tokens/"
DESIGN_MD = DESKTOP / "DESIGN.md"
BASELINE = REPO / "scripts" / "desktop_design_tokens_baseline.json"
# The rail pages' one column and the only file outside tokens/ that may lay it out.
PAGE_COLUMN = "--portal-column-page"
PAGE_STAGE_CONTRACT = "app/redesign/portal-pages.css"

# Stylesheets whose rules only redesign markup can match: their classes and hooks
# (``[data-page-surface]``, ``.onboarding-v2``, …) are emitted only on the redesign's branch of the
# TSX. Which branch emits a class is not visible from the CSS, so the set is named, not derived.
_REDESIGN_ONLY = (
    re.compile(r"^app/redesign/[^/]+\.css$"),
    re.compile(r"^app/redesign/design-system/[^/]+\.css$"),
    re.compile(r"^app/chat/sidebar/portal/[^/]+\.css$"),
    re.compile(r"^components/assistant-ui/thread/redesign\.css$"),
    re.compile(r"^app/agent-settings/[^/]+\.css$"),
    re.compile(r"^app/settings/(global-overviews|appearance-overview|portal-safety-fields|agent-config-fields)\.css$"),
    re.compile(r"^app/routines/[^/]+\.css$"),
    re.compile(r"^app/overlays/panel\.css$"),
    re.compile(r"^app/(library/library|messaging/messaging|session-import/session-import|page-search-shell)\.css$"),
    re.compile(r"^app/capabilities/ui/capability-search\.css$"),
    re.compile(r"^app/chat/(stage|items/chat-items)\.css$"),
    re.compile(r"^components/(onboarding/surface|ui/pixel-icon)\.css$"),
)
_GATE = re.compile(r"data-shell=['\"]?portal")
# Token files in the order the cascade imports them (tokens/index.css); the docs follow it.
TOKEN_FILES = ("raw", "color", "type", "space", "shape", "motion", "layers", "bridge", "sidebar")
DOCS_BEGIN = "<!-- design-tokens:begin (generated by scripts/check_desktop_design_tokens.py --write-docs) -->"
DOCS_END = "<!-- design-tokens:end -->"


# ── A small CSS reader (no dependencies: the guard also runs in hook snapshots) ─────────────────

@dataclass
class Decl:
    prop: str
    value: str
    line: int
    exempt: bool = False
    note: str = ""


@dataclass
class Block:
    prelude: str  # selector, or "@media …" for an at-rule
    line: int
    parent: Block | None = None
    exempt: bool = False
    decls: list[Decl] = field(default_factory=list)
    children: list[Block] = field(default_factory=list)

    @property
    def is_rule(self) -> bool:
        return bool(self.prelude) and not self.prelude.startswith("@")

    def gated(self) -> bool:
        node: Block | None = self
        while node is not None:
            if node.is_rule and _GATE.search(node.prelude):
                return True
            node = node.parent
        return False

    def walk(self) -> Iterator[Block]:
        yield self
        for child in self.children:
            yield from child.walk()


def parse_css(text: str) -> Block:
    """Blocks with their declarations. Comments carrying ``token-exempt`` mark the next declaration
    or rule (or, trailing on the same line, the declaration before them); other comments directly
    before a declaration become its ``note`` (the docs' description column)."""
    root = Block("", 1)
    stack = [root]
    buf: list[str] = []
    buf_line = 0
    line = 1
    i = 0
    n = len(text)
    pending_exempt = False
    pending_note = ""
    last_decl: tuple[Decl, int] | None = None  # (decl, line its ';' was on)
    depth = 0

    def flush_decl(end_line: int) -> None:
        nonlocal pending_exempt, pending_note, last_decl
        stmt = "".join(buf).strip()
        buf.clear()
        if not stmt or ":" not in stmt or stmt.startswith("@"):
            pending_exempt = False
            pending_note = ""
            return
        prop, _, value = stmt.partition(":")
        decl = Decl(prop.strip(), " ".join(value.split()), buf_line or end_line, pending_exempt, pending_note)
        stack[-1].decls.append(decl)
        last_decl = (decl, end_line)
        pending_exempt = False
        pending_note = ""

    while i < n:
        ch = text[i]
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            end = n if end < 0 else end
            body = text[i + 2:end].strip()
            if "token-exempt" in body:
                if not "".join(buf).strip() and last_decl and last_decl[1] == line:
                    last_decl[0].exempt = True
                else:
                    pending_exempt = True
            elif not "".join(buf).strip():
                pending_note = " ".join(body.split())
            line += text.count("\n", i, end)
            i = end + 2
            continue
        if ch in "'\"":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            if not "".join(buf).strip():
                buf_line = line
            buf.append(text[i:j + 1])
            line += text.count("\n", i, j + 1)
            i = j + 1
            continue
        if ch == "\n":
            line += 1
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if depth == 0 and ch == "{":
            prelude = " ".join("".join(buf).split())
            buf.clear()
            block = Block(prelude, buf_line or line, stack[-1], pending_exempt)
            stack[-1].children.append(block)
            stack.append(block)
            pending_exempt = False
            pending_note = ""
            last_decl = None
        elif depth == 0 and ch == ";":
            flush_decl(line)
        elif depth == 0 and ch == "}":
            flush_decl(line)
            if len(stack) > 1:
                stack.pop()
            last_decl = None
        else:
            if not ch.isspace() and not "".join(buf).strip():
                buf_line = line
            buf.append(ch)
        i += 1
    return root


# ── Findings ─────────────────────────────────────────────────────────────────────────────────

_NAMED_COLORS = frozenset("""
aliceblue antiquewhite aqua aquamarine azure beige bisque black blanchedalmond blue blueviolet brown
burlywood cadetblue chartreuse chocolate coral cornflowerblue cornsilk crimson cyan darkblue darkcyan
darkgoldenrod darkgray darkgreen darkgrey darkkhaki darkmagenta darkolivegreen darkorange darkorchid
darkred darksalmon darkseagreen darkslateblue darkslategray darkslategrey darkturquoise darkviolet
deeppink deepskyblue dimgray dimgrey dodgerblue firebrick floralwhite forestgreen fuchsia gainsboro
ghostwhite gold goldenrod gray green greenyellow grey honeydew hotpink indianred indigo ivory khaki
lavender lavenderblush lawngreen lemonchiffon lightblue lightcoral lightcyan lightgoldenrodyellow
lightgray lightgreen lightgrey lightpink lightsalmon lightseagreen lightskyblue lightslategray
lightslategrey lightsteelblue lightyellow lime limegreen linen magenta maroon mediumaquamarine
mediumblue mediumorchid mediumpurple mediumseagreen mediumslateblue mediumspringgreen
mediumturquoise mediumvioletred midnightblue mintcream mistyrose moccasin navajowhite navy oldlace
olive olivedrab orange orangered orchid palegoldenrod palegreen paleturquoise palevioletred
papayawhip peachpuff peru pink plum powderblue purple rebeccapurple red rosybrown royalblue
saddlebrown salmon sandybrown seagreen seashell sienna silver skyblue slateblue slategray slategrey
snow springgreen steelblue tan teal thistle tomato turquoise violet wheat white whitesmoke yellow
yellowgreen
""".split())
_NO_COLOR_PROPS = frozenset({
    "font-family", "content", "grid-template-areas", "grid-area", "transition-property", "will-change",
    "animation-name", "container", "container-name", "view-transition-name", "anchor-name",
    "position-anchor", "counter-reset", "counter-increment", "list-style-type", "font-feature-settings",
})
_HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")
_COLOR_FN = re.compile(r"\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch)\(", re.IGNORECASE)
_WORD = re.compile(r"(?<![\w-])([a-zA-Z]+)(?![\w-])")
_INK_MIX = re.compile(r"color-mix\([^;]*?var\(\s*--(?:ui-base|portal-brand-fill|portal-raw-blue)\s*\)", re.IGNORECASE)
_LENGTH = re.compile(r"(?<![\w.#-])(-?\d*\.?\d+)(px|rem|em)\b")
_TIME = re.compile(r"(?<![\w.#-])(\d*\.?\d+)(ms|s)\b")
_VAR_NAME = re.compile(r"--[\w-]+")
# The page's top inset negated (``calc(-1 * var(--portal-page-top))``, ``calc(0px - …)``): a row pulled
# up into the space above a page's first row, which leaves it on the stage's edge.
_PAGE_TOP_TOKENS = r"--portal-(?:page|settings)-top\b"
_PAGE_TOP_PULL = re.compile(
    rf"-\s*(?:1\s*\*\s*)?var\(\s*{_PAGE_TOP_TOKENS}|var\(\s*{_PAGE_TOP_TOKENS}\s*\)\s*\*\s*-1"
)
_BLOCK_START_PROPS = frozenset({
    "margin", "margin-top", "margin-block", "margin-block-start", "top", "inset", "inset-block", "inset-block-start",
    "translate",
})
_STRINGS = re.compile(r"'[^']*'|\"[^\"]*\"|url\([^)]*\)")


def _bare(value: str) -> str:
    return _VAR_NAME.sub("", _STRINGS.sub("", value))


def _nonzero_lengths(value: str, units: tuple[str, ...]) -> bool:
    return any(float(num) != 0 and unit in units for num, unit in _LENGTH.findall(_bare(value)))


def declaration_findings(decl: Decl) -> list[str]:
    prop = decl.prop.lower()
    bare = _bare(decl.value)
    kinds: list[str] = []
    if prop not in _NO_COLOR_PROPS and (
        _HEX.search(bare) or _COLOR_FN.search(bare) or any(w.lower() in _NAMED_COLORS for w in _WORD.findall(bare))
    ):
        kinds.append("raw-color")
    if _INK_MIX.search(decl.value):
        kinds.append("inline-mix")
    if re.fullmatch(r"border(-[a-z]+)*-radius", prop) and _nonzero_lengths(decl.value, ("px", "rem", "em")):
        kinds.append("radius")
    if prop in ("font-size", "font") and _nonzero_lengths(decl.value, ("px", "rem")):
        kinds.append("font-size")
    if prop == "letter-spacing" and _nonzero_lengths(decl.value, ("px", "rem", "em")):
        kinds.append("letter-spacing")
    if prop == "line-height" and _nonzero_lengths(decl.value, ("px", "rem")):
        kinds.append("line-height")
    if prop.startswith(("transition", "animation")) and any(float(x) != 0 for x, _ in _TIME.findall(bare)):
        kinds.append("duration")
    if prop == "z-index" and re.fullmatch(r"\d+", decl.value.strip()) and int(decl.value) > 1:
        kinds.append("z-index")
    if prop in _BLOCK_START_PROPS and _PAGE_TOP_PULL.search(decl.value):
        kinds.append("page-top")
    if prop.startswith("--portal-") and "var(" not in decl.value and (
        _LENGTH.search(bare) or _TIME.search(bare) or "raw-color" in kinds or re.fullmatch(r"-?\d*\.?\d+%?", decl.value.strip())
    ):
        kinds.append("token-literal")
    return kinds


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    kind: str
    text: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: {self.kind}: {self.text}"


def _rel(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def css_files(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*.css") if "node_modules" not in p.parts)


def file_findings(rel: str, text: str) -> list[Finding]:
    if rel.startswith(TOKENS):
        return []
    whole = any(rx.match(rel) for rx in _REDESIGN_ONLY)
    if not whole and not _GATE.search(text):
        return []
    found: list[Finding] = []
    for block in parse_css(text).walk():
        if block.exempt or not block.decls or not (whole or block.gated()):
            continue
        for decl in block.decls:
            if decl.exempt:
                continue
            for kind in declaration_findings(decl):
                found.append(Finding(rel, decl.line, kind, f"{decl.prop}: {decl.value}"))
    return found


# ── Definitions and references (undefined / unused tokens) ───────────────────────────────────

_CSS_DEF = re.compile(r"(?<![\w-])(--portal-[\w-]+)\s*:")
_REF = re.compile(r"var\(\s*(--portal-[\w-]+)|\((?:[a-z-]+:)?(--portal-[\w-]+)\)")
# What publishes a token at runtime from TS/TSX: a style-object key (``'--portal-x': …``), CSS text in a
# template (``--portal-x: …;``), a Tailwind arbitrary property (``[--portal-x:…]``) or setProperty.
_TS_DEF = re.compile(
    r"(?<![\w-])(--portal-[\w-]+)['\"`]?\s*:|setProperty\(\s*['\"`](--portal-[\w-]+)['\"`]"
)


def _script_files(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*") if p.suffix in (".ts", ".tsx", ".mts") and "node_modules" not in p.parts)


def strip_comments(text: str, script: bool) -> str:
    """``text`` with every comment blanked to spaces (newlines kept, so line numbers hold): CSS block
    comments, plus line comments in a script. Strings and template literals (with their ``${…}``)
    are skipped whole, so a ``//`` in a URL is not a comment."""
    out = list(text)
    n = len(text)

    def blank(start: int, end: int) -> None:
        for k in range(start, end):
            if out[k] != "\n":
                out[k] = " "

    def string_end(i: int) -> int:
        quote = text[i]
        i += 1
        while i < n and text[i] not in (quote, "\n"):
            i += 2 if text[i] == "\\" else 1
        return i + 1

    def template_end(i: int) -> int:
        i += 1
        while i < n and text[i] != "`":
            if text[i] == "\\":
                i += 2
            elif text.startswith("${", i):
                i = code_end(i + 2, closing=True)
            else:
                i += 1
        return i + 1

    def code_end(i: int, closing: bool) -> int:
        depth = 0
        while i < n:
            ch = text[i]
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                end = n if end < 0 else end + 2
                blank(i, end)
                i = end
            elif script and text.startswith("//", i):
                end = text.find("\n", i)
                end = n if end < 0 else end
                blank(i, end)
                i = end
            elif ch in "'\"":
                i = string_end(i)
            elif script and ch == "`":
                i = template_end(i)
            else:
                if closing and ch == "{":
                    depth += 1
                elif closing and ch == "}":
                    if depth == 0:
                        return i + 1
                    depth -= 1
                i += 1
        return n

    code_end(0, closing=False)
    return "".join(out)


def references(src: Path) -> tuple[set[str], dict[str, list[tuple[str, int]]]]:
    """(names defined anywhere, name → reference sites), comments excluded. A TS/TSX definition is a
    style-object key, CSS text, a Tailwind arbitrary property or a setProperty call, never a mere
    mention."""
    defined: set[str] = set()
    refs: dict[str, list[tuple[str, int]]] = {}
    for path in [*css_files(src), *_script_files(src)]:
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
        if "--portal-" not in raw:
            continue
        script = path.suffix != ".css"
        text = strip_comments(raw, script)
        rel = path.relative_to(src).as_posix()
        for m in _REF.finditer(text):
            refs.setdefault(m.group(1) or m.group(2), []).append((rel, text.count("\n", 0, m.start()) + 1))
        if script:
            defined.update(m.group(1) or m.group(2) for m in _TS_DEF.finditer(text))
        else:
            defined.update(_CSS_DEF.findall(text))
    return defined, refs


# ── Generated docs ───────────────────────────────────────────────────────────────────────────

# A token file's root-level blocks: the base (both themes), the dark override and the light one.
_MODE_SELECTORS = {
    ":root[data-shell='portal']": "base",
    ":root.dark[data-shell='portal']": "dark",
    ":root[data-shell='portal']:not(.dark)": "light",
}
# The root element under extra conditions (a host, a pointer) inside an at-rule.
_ROOT_VARIANT = re.compile(r"^(?:html|:root)(?:\[[^\]]+\]|\.[\w-]+|:not\([^)]*\))+$")
_FILE_TITLES = {
    "raw": "Primitives (`raw.css`): the only colour literals",
    "color": "Colour roles (`color.css`)",
    "type": "Type (`type.css`)",
    "space": "Space and layout (`space.css`)",
    "shape": "Shape: radii, strokes, shadows (`shape.css`)",
    "motion": "Motion (`motion.css`)",
    "layers": "Layers (`layers.css`)",
    "bridge": "Bridge (`bridge.css`): portal roles under the names shared components read",
    "sidebar": "Chat sidebar (`sidebar.css`): the Variation 1 variant's own roles",
}


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


@dataclass
class TokenRow:
    values: dict[str, str] = field(default_factory=dict)  # mode → value
    note: str = ""
    overrides: list[str] = field(default_factory=list)  # "`(condition)`: `value`", in source order

    def cells(self) -> tuple[str, str, str]:
        dark = self.values.get("dark", self.values.get("base", ""))
        light = self.values.get("light", self.values.get("base", ""))
        light_cell = "″" if not light or light == dark else f"`{_cell(light)}`"
        note = " ".join(filter(None, [self.note, "; ".join(self.overrides) + "." if self.overrides else ""]))
        return f"`{_cell(dark)}`", light_cell, _cell(note)


def _token_rows(css: str) -> dict[str, TokenRow]:
    """A token file's custom properties: the theme values from its root-level mode blocks, then each
    override an at-rule makes (a width class, a coarse pointer) noted with its condition. Only
    root-level blocks set the base: a nested block's value applies under its condition alone."""
    rows: dict[str, TokenRow] = {}
    for block in parse_css(css).children:
        mode = _MODE_SELECTORS.get(block.prelude)
        if mode is not None:
            for decl in block.decls:
                if decl.prop.startswith("--"):
                    row = rows.setdefault(decl.prop, TokenRow())
                    row.values[mode] = decl.value
                    row.note = row.note or decl.note
        elif block.prelude.startswith("@"):
            condition = block.prelude.split(None, 1)[-1]
            for rule in block.children:
                rule_mode = _MODE_SELECTORS.get(rule.prelude)
                if rule_mode == "base":
                    scope = ""
                elif rule_mode is not None:
                    scope = f" ({rule_mode})"
                elif _ROOT_VARIANT.match(rule.prelude):
                    scope = f" on `{rule.prelude}`"
                else:
                    continue
                for decl in rule.decls:
                    if decl.prop.startswith("--"):
                        rows.setdefault(decl.prop, TokenRow()).overrides.append(
                            f"`{condition}`{scope}: `{decl.value}`"
                        )
    return rows


def render_docs(tokens_dir: Path) -> str:
    out = [DOCS_BEGIN, ""]
    for stem in TOKEN_FILES:
        path = tokens_dir / f"{stem}.css"
        if not path.exists():
            continue
        rows = _token_rows(path.read_text(encoding="utf-8-sig"))
        if not rows:
            continue
        out += [f"#### {_FILE_TITLES[stem]}", "", "| Token | Dark | Light | Note |", "|---|---|---|---|"]
        for name, row in rows.items():
            dark, light, note = row.cells()
            out.append(f"| `{name}` | {dark} | {light} | {note} |")
        out.append("")
    out.append(DOCS_END)
    return "\n".join(out)


def docs_block(markdown: str) -> str | None:
    start = markdown.find(DOCS_BEGIN)
    end = markdown.find(DOCS_END)
    if start < 0 or end < 0:
        return None
    return markdown[start:end + len(DOCS_END)]


# ── Main ─────────────────────────────────────────────────────────────────────────────────────

def collect(src: Path) -> list[Finding]:
    found: list[Finding] = []
    for path in css_files(src):
        found += file_findings(_rel(path), path.read_text(encoding="utf-8-sig", errors="replace"))
    return found


def page_column_findings(refs: dict[str, list[tuple[str, int]]]) -> list[Finding]:
    """Reads of the page column outside tokens/ and the page stage contract."""
    return sorted(
        Finding(rel, line, "page-column", PAGE_COLUMN)
        for rel, line in refs.get(PAGE_COLUMN, [])
        if not rel.startswith(TOKENS) and rel != PAGE_STAGE_CONTRACT
    )


def ratchet(counts: dict[str, int], baseline: dict[str, int]) -> tuple[list[str], list[str]]:
    """(files over their baseline, files under it)."""
    over = sorted(f for f, n in counts.items() if n > baseline.get(f, 0))
    under = sorted(f for f, n in baseline.items() if counts.get(f, 0) < n)
    return over, under


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="print every finding")
    parser.add_argument("--write-baseline", action="store_true", help="rewrite the per-file baseline")
    parser.add_argument("--write-docs", action="store_true", help="regenerate DESIGN.md token tables")
    args = parser.parse_args(argv)

    # A tree without the redesign's tokens folder (a partial checkout, a hook's snapshot of an older
    # commit) has nothing to judge.
    if not (SRC / TOKENS).is_dir():
        print("desktop design tokens: no tokens folder in this tree, nothing to check")
        return 0

    found = collect(SRC)
    counts: dict[str, int] = {}
    for f in found:
        counts[f.file] = counts.get(f.file, 0) + 1

    if args.write_baseline:
        BASELINE.write_text(json.dumps({"files": dict(sorted(counts.items()))}, indent=2) + "\n", encoding="utf-8")
        print(f"desktop design tokens: baseline written ({len(found)} findings in {len(counts)} files)")
    docs = render_docs(SRC / TOKENS)
    markdown = DESIGN_MD.read_text(encoding="utf-8-sig")
    if args.write_docs:
        current = docs_block(markdown)
        if current is None:
            print(f"desktop design tokens: {DESIGN_MD} has no {DOCS_BEGIN!r} marker", file=sys.stderr)
            return 1
        DESIGN_MD.write_text(markdown.replace(current, docs), encoding="utf-8")
        markdown = DESIGN_MD.read_text(encoding="utf-8-sig")
        print("desktop design tokens: DESIGN.md token tables regenerated")

    baseline = json.loads(BASELINE.read_text(encoding="utf-8-sig"))["files"] if BASELINE.exists() else {}
    over, under = ratchet(counts, baseline)
    defined, refs = references(SRC)
    undefined = sorted(
        Finding(rel, line, "undefined-token", name) for name, sites in refs.items() if name not in defined for rel, line in sites
    )
    token_names = set()
    for stem in TOKEN_FILES:
        path = SRC / TOKENS / f"{stem}.css"
        if path.exists():
            token_names.update(_CSS_DEF.findall(path.read_text(encoding="utf-8-sig")))
    unused = sorted(name for name in token_names if name not in refs)
    stale = docs_block(markdown) != docs
    own_column = page_column_findings(refs)
    pulled = [f for f in found if f.kind == "page-top"]

    if args.list:
        for f in found:
            print(f)
    failed = bool(over or undefined or stale or own_column or pulled)
    for rel in over:
        print(f"{rel}: {counts[rel]} design-token findings, baseline {baseline.get(rel, 0)} (it may only go down):")
        for f in found:
            if f.file == rel:
                print(f"  {f}")
    for f in undefined:
        print(f"{f}  (defined nowhere: a typo or a deleted token)")
    for f in own_column:
        print(f"{f}  (mark the page's stage `page-stage`; only {PAGE_STAGE_CONTRACT} lays the column)")
    for f in pulled:
        print(f"{f}  (a page's first row starts at its top inset; put the row in the flow under it)")
    if stale:
        print("apps/desktop/DESIGN.md token tables are stale: run python scripts/check_desktop_design_tokens.py --write-docs")
    if under and not args.write_baseline:
        print("advisory: below baseline, lock it in with --write-baseline: "
              + ", ".join(f"{r} {counts.get(r, 0)}<{baseline[r]}" for r in under))
    if unused:
        print(f"advisory: {len(unused)} token(s) in tokens/ that nothing reads: {', '.join(unused)}")
    print(f"desktop design tokens: {len(found)} findings in {len(counts)} files (ratcheted), "
          f"{len(undefined)} undefined, docs {'stale' if stale else 'current'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
