"""Markdown -> Telegram MarkdownV2 conversion (the legacy, non-rich send path)."""

import bisect
import itertools
import re

from gateway.platforms.helpers import convert_table_to_bullets as _wrap_markdown_tables
from plugins.platforms.telegram.rich_code_regions import map_outside_code
from plugins.platforms.telegram.telegram_latex import normalize_latex_macros, replace_macros_outside

# Stands in for the ** of a line-start **> expandable-quote opener while emphasis is resolved.
_EXPANDABLE_QUOTE = "\x00EQ\x00"

# Every char MarkdownV2 requires backslash-escaped outside code spans/fences.
_MDV2_ESCAPE_RE = re.compile(r'([_*\[\]()~`>#\+\-=|{}.!\\])')
_EMPHASIS_MARK = {"bold": "*", "italic": "_"}


# A line opening with one of these prefixes starts a Telegram blockquote.
_QUOTE_LINE_PREFIXES = (_EXPANDABLE_QUOTE + ">", ">")


def _line_begins_quote(text: str, start: int) -> bool:
    """True when the line beginning at *start* is a blockquote line."""
    return text.startswith(_QUOTE_LINE_PREFIXES, start)


def _bordering_quote_newline(text: str, pos: int, line_start: int) -> bool:
    """True when the newline at *pos* closes or opens a quote line (*line_start* starts the line it closes)."""
    return _line_begins_quote(text, line_start) or _line_begins_quote(text, pos + 1)


class _SpanEdges:
    """Sorted line edges of one text, so delimiter pairing answers its span questions by bisect.

    Re-slicing ``text[o_end:start]`` and re-finding its newlines for every candidate pair cost
    closers × span: one distant unmatched ``**`` opener made a 32 KB reply take ~1.4 s (200 KB
    ~106 s) on the gateway event loop (#151723).
    """

    __slots__ = ("blank_starts", "newlines", "quote_edges")

    def __init__(self, text: str) -> None:
        self.newlines = [m.start() for m in re.finditer("\n", text)]
        self.blank_starts = [p for p in self.newlines if text.startswith("\n", p + 1)]
        edges: list[int] = []
        line_start = 0
        for pos in self.newlines:
            if _bordering_quote_newline(text, pos, line_start):
                edges.append(pos)
            line_start = pos + 1
        self.quote_edges = edges

    def has_newline(self, start: int, end: int) -> bool:
        """A newline inside ``text[start:end]``."""
        i = bisect.bisect_left(self.newlines, start)
        return i < len(self.newlines) and self.newlines[i] < end

    def has_blank_line(self, start: int, end: int) -> bool:
        """A ``\\n\\n`` fully inside ``text[start:end]``."""
        i = bisect.bisect_left(self.blank_starts, start)
        return i < len(self.blank_starts) and self.blank_starts[i] + 2 <= end

    def crosses_quote_edge(self, start: int, end: int) -> bool:
        """A newline inside ``text[start:end]`` bordering a quote line."""
        i = bisect.bisect_left(self.quote_edges, start)
        return i < len(self.quote_edges) and self.quote_edges[i] < end


def _resolve_emphasis(text: str, stash, escape) -> str:
    """Resolve **bold** and *italic* delimiter runs together so nested/combined emphasis keeps its
    inner markers (***x*** → _*x*_, **a *i* b** → *a _i_ b*) instead of the bold pass swallowing
    them. Bold may span a single newline within a paragraph; italic stays single-line; unmatched
    markers survive as literal text."""
    return _render_emphasis(_pair_emphasis(text), stash, escape)


def _pair_emphasis(text: str) -> list:
    """The tokens of *text*, each ``*`` delimiter run carrying its pairings and unpaired remainder."""
    tokens = []  # ("t", s) literal | ["d", start, end, can_open, can_close, remaining, opens, closes]
    pos = 0
    for m in re.finditer(r"\*+", text):
        run_start = m.start()
        slash = run_start
        while slash > 0 and text[slash - 1] == "\\":
            slash -= 1
        if (run_start - slash) % 2:
            run_start += 1  # a backslash-escaped * is literal text (**Use \*args**, #151722)
            if run_start == m.end():
                continue
        if run_start > pos:
            tokens.append(("t", text[pos:run_start]))
        before = text[run_start - 1] if run_start > 0 else ""
        after = text[m.end()] if m.end() < len(text) else ""
        can_open = bool(after) and not after.isspace()
        can_close = bool(before) and not before.isspace()
        tokens.append([
            "d",
            run_start,
            m.end(),
            can_open,
            can_close,
            m.end() - run_start,
            [],
            [],
        ])
        pos = m.end()
    if pos < len(text):
        tokens.append(("t", text[pos:]))

    edges = _SpanEdges(text)
    stack = []  # indexes of delimiter tokens that may still close
    near2 = []  # per stack slot: the slot of the nearest opener at or below it with 2+ stars left
    bottoms: dict = {}  # (closer length % 3, closer can open) -> token index a failed search reached
    for idx, tok in enumerate(tokens):
        if tok[0] != "d":
            continue
        _, _start, _end, can_open, can_close, remaining, _opens, closes = tok
        while can_close and remaining > 0 and stack:
            depth = _pick_opener(tokens, stack, near2, tok, bottoms, edges)
            if depth is None:
                break
            del stack[depth + 1:], near2[depth + 1:]  # openers between the pair stay literal (CommonMark)
            opener = tokens[stack[-1]]
            use = 2 if opener[5] >= 2 and tok[5] >= 2 else 1
            kind = "bold" if use == 2 else "italic"
            opener[6].append((kind, idx))
            closes.append((kind, stack[-1]))
            opener[5] -= use
            tok[5] = remaining = remaining - use
            if opener[5] == 0:
                stack.pop()
                near2.pop()
            elif opener[5] < 2:
                near2[-1] = near2[-2] if len(near2) > 1 else -1
        if remaining > 0 and can_open:
            stack.append(idx)
            near2.append(len(stack) - 1 if remaining >= 2 else (near2[-1] if near2 else -1))
    return tokens


def _rule_of_three_refuses(opener: list, closer: list) -> bool:
    """CommonMark's rule of 3: when either run can both open and close, the two may not pair if
    their lengths sum to a multiple of 3 unless both are multiples of 3. Without it the ``*`` of
    ``2*3`` closed one star of a ``**`` opener, so ``**2*3 = 6**`` turned italic and lost its
    ``*`` (#151722)."""
    o_len, c_len = opener[2] - opener[1], closer[2] - closer[1]
    return bool(opener[4] or closer[3]) and (o_len + c_len) % 3 == 0 and bool(o_len % 3 or c_len % 3)


def _span_allows(edges: "_SpanEdges", opener: list, closer: list) -> bool:
    """Italic stays on one line; bold stays within one paragraph and on one side of a quote edge."""
    if opener[5] >= 2 and closer[5] >= 2:
        return not (edges.has_blank_line(opener[2], closer[1]) or edges.crosses_quote_edge(opener[2], closer[1]))
    return not edges.has_newline(opener[2], closer[1])


def _pick_opener(tokens: list, stack: list, near2: list, closer: list, bottoms: dict, edges):
    """Stack slot of the opener *closer* pairs with, or ``None``.

    The nearest opener the rule of 3 allows wins, as in CommonMark, except that a bare ``**``
    closer reaches past single-``*`` openers to a ``**`` opener: chat text means
    ``**rm -rf *.tmp**`` as bold around a literal ``*`` (CommonMark nests two italics there, which
    Telegram cannot show, and drops the star). A ``***`` closer still pairs its ``*`` first
    (``**a *b***``). A failed
    search records how far down it looked for closers of the same class, so a long run of refused
    openers is walked once, not once per closer.
    """
    key = ((closer[2] - closer[1]) % 3, closer[3])
    floor = bottoms.get(key, -1)
    nearest = None
    for depth in range(len(stack) - 1, -1, -1):
        if stack[depth] <= floor:
            break
        if not _rule_of_three_refuses(tokens[stack[depth]], closer):
            nearest = depth
            break
    if nearest is None:
        bottoms[key] = stack[-1]
        return None
    candidates = [nearest]
    if closer[2] - closer[1] == closer[5] == 2 and tokens[stack[nearest]][5] < 2 and near2[nearest] >= 0:
        double = tokens[stack[near2[nearest]]]
        if not _rule_of_three_refuses(double, closer):
            candidates.insert(0, near2[nearest])
    return next((d for d in candidates if _span_allows(edges, tokens[stack[d]], closer)), None)


def _crosses_quote_edge(text: str, start: int, end: int) -> bool:
    """True when a newline in text[start:end] borders a quote line: Telegram ends a blockquote at a
    newline into non-quote text, an entity still open there is rejected, and an entity spanning
    quote lines would swallow the next line's > marker. A span with no newline cannot cross one, so
    the prefix scan that locates the enclosing line only runs once a newline is present: a long
    single line of short links stays linear (#151728 review)."""
    pos = text.find("\n", start, end)
    if pos < 0:
        return False
    line_start = text.rfind("\n", 0, start) + 1
    while pos >= 0:
        if _bordering_quote_newline(text, pos, line_start):
            return True
        line_start = pos + 1
        pos = text.find("\n", line_start, end)
    return False


def _emphasis_marks(tok, depth: dict) -> tuple:
    """The style kinds a delimiter run closes (innermost first) and opens (outermost first).

    A style that is the last to close and the first to reopen at one run continues unbroken;
    emitting both marks would read as "__" (underline) or an empty bold to Telegram."""
    closes = [kind for kind, _ in tok[7]]
    opens = [kind for kind, _ in reversed(tok[6])]
    for kind in ("bold", "italic"):
        if closes[-1:] == opens[:1] == [kind] and closes.count(kind) == depth[kind]:
            closes.remove(kind)
            opens.remove(kind)
    return closes, opens


def _render_emphasis(tokens: list, stash, escape) -> str:
    """Emit each delimiter's own boundary marks so combined runs that end at different places
    (***a* b**) keep every closing mark. MarkdownV2 cannot nest an entity inside itself, so a kind
    already open adds no further marks (****x**** -> *x*). Each outermost emphasized span is escaped
    and stashed whole, so later passes cannot cross it."""
    depth = {"bold": 0, "italic": 0}
    parts, group = [], None

    def emit_literal(value: str) -> None:
        if group is None:
            parts.append(value)
        else:
            group.append(escape(value))

    for tok in tokens:
        if tok[0] == "t":
            emit_literal(tok[1])
            continue
        closes, opens = _emphasis_marks(tok, depth)
        for kind in closes:
            depth[kind] -= 1
            if depth[kind] == 0:
                group.append(_EMPHASIS_MARK[kind])
        if group is not None and not any(depth.values()):
            parts.append(stash("".join(group)))
            group = None
        emit_literal("*" * tok[5])
        for kind in opens:
            if group is None:
                group = []
            if depth[kind] == 0:
                group.append(_EMPHASIS_MARK[kind])
            depth[kind] += 1
    return "".join(parts)


# A link target Telegram can open: a scheme URL (https:, tg:, mailto:) or a bare domain. A schemeless
# title ("[Title](Title)") or Hermes' desktop-only "@session:" reference is shown as raw syntax.
_RENDERABLE_LINK_TARGET_RE = re.compile(
    r'(?i)^(?:[a-z][a-z0-9+.-]*:[^\s@]\S*|[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:[/?#:]\S*)?)$')
_MD_LINK_RE = re.compile(r'\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)')


def link_target_renders(target: str) -> bool:
    return bool(_RENDERABLE_LINK_TARGET_RE.match(target.strip()))


def degrade_unrenderable_links(markdown: str) -> str:
    """``[label](target)`` → ``label`` when Telegram cannot open ``target`` (rich-message path; code
    spans and fences stay verbatim)."""
    if '](' not in markdown:
        return markdown
    return map_outside_code(markdown, lambda prose: _MD_LINK_RE.sub(_link_or_label, prose))


def _link_or_label(m) -> str:
    return m.group(0) if link_target_renders(m.group(2)) else m.group(1)


def _escape_mdv2(text: str) -> str:
    """Escape Telegram MarkdownV2 special characters with a preceding backslash."""
    return _MDV2_ESCAPE_RE.sub(r'\\\1', text)


def to_markdown_v2(content: str) -> str:
    """Convert standard markdown to Telegram MarkdownV2: code is stashed behind placeholders first (never
    modified), markdown constructs become MarkdownV2 syntax, everything else is escaped."""
    if not content:
        return content
    placeholders: dict = {}
    counter = [0]

    def _ph(value: str) -> str:
        """Stash *value* behind a placeholder token that survives escaping."""
        key = f"\x00PH{counter[0]}\x00"
        counter[0] += 1
        placeholders[key] = value
        return key

    def _ph_wrap(open_: str, close: str):
        return lambda m: _ph(f"{open_}{_escape_mdv2(m.group(1))}{close}")

    # 0) LaTeX macros in prose (\rightarrow, $x \leq y$) become Unicode: MarkdownV2 has no math, so
    #    they would reach the chat as raw backslash text. Code keeps them verbatim. Then rewrite
    #    GFM-style pipe tables into Telegram-friendly row groups.
    if "\\" in content:
        content = map_outside_code(content, normalize_latex_macros)
    text = _wrap_markdown_tables(content)

    # 1) Protect fenced code blocks (``` ... ```)
    #    Per MarkdownV2 spec, \ and ` inside pre/code must be escaped.
    #    A fence still opens on its own line — the opening ``` must be the
    #    first triple-backtick run on that line and must end it — but the
    #    line may carry arbitrary leading whitespace (list/blockquote-
    #    nested code indents fences by 4+ spaces) or lead-in prose
    #    ("Here is the code: ```"), both of which the line-start-only
    #    anchor silently downgraded from <pre> to escaped literal text.
    #    Requiring the rest of the opening line to be backtick-free is
    #    what keeps *inline* triple backticks (e.g. "the syntax is
    #    ```x``` inline") out of the match: a closing run can never sit
    #    on the same line as the opener, and the tempered prefix cannot
    #    skip past an earlier run to a later one.  The closing fence must
    #    sit on its own line (any indent), with optional trailing
    #    whitespace and an optional ``\r`` so CRLF-terminated fences
    #    (Windows-authored content) match too.
    def _protect_fenced(m):
        prefix = m.group(1)   # lead-in text / indent before the opening fence
        opening = m.group(2)  # opening ``` (with optional language) and newline
        body = m.group(3)     # code body (may be empty)
        closing = m.group(4)   # closing fence (with its indent)
        body = body.replace('\\', '\\\\').replace('`', '\\`')
        return prefix + _ph(opening + body + closing)

    text = re.sub(
        r'(?m)^((?:(?!```)[^\n])*)(```[^`\n]*\n)([\s\S]*?)(^[ \t]*```)[ \t]*\r?$',
        _protect_fenced,
        text,
    )

    # 2) Protect inline code (`...` and GFM's backtick-free ``...``) in one pass.
    text = _stash_inline_code(text, _ph)

    # 3) Convert markdown links – escape the display text; inside the URL
    #    only ')' and '\' need escaping per the MarkdownV2 spec.
    def _convert_link(m):
        if _crosses_quote_edge(text, m.start(), m.end()):
            return m.group(0)  # Telegram ends the quote at the newline, cutting the link: keep it literal
        if not link_target_renders(m.group(2)):
            return _ph(_escape_mdv2(m.group(1)))  # Telegram would show the raw [label](target) syntax
        url = m.group(2).replace('\\', '\\\\').replace(')', '\\)')
        return _ph(f'[{_escape_mdv2(m.group(1))}]({url})')

    text = _MD_LINK_RE.sub(_convert_link, text)
    text = _stash_bare_urls(text, _ph)
    # 4) Headers (## Title) → bold *Title*, stripping redundant ** inside the header
    def _convert_header(m):
        inner = re.sub(r'\*\*(.+?)\*\*', r'\1', m.group(1).strip())
        return _ph(f'*{_escape_mdv2(inner)}*')

    text = re.sub(r'^#{1,6}\s+(.+)$', _convert_header, text, flags=re.MULTILINE)
    # 4b) An expandable quote opener (**> at line start) is quote syntax, not a bold delimiter: hide its
    #     ** behind a sentinel so step 5 cannot pair it with a later ** in the quoted text.
    text = re.sub(r'(?m)^\*\*(?=>{1,3} )', _EXPANDABLE_QUOTE, text)
    # 5) Bold **text** and *italic* resolve together so nested emphasis survives; unmatched
    # markers stay literal (bullet lists rely on that); 6) Strikethrough; 7) Spoiler kept as-is.
    text = _resolve_emphasis(text, _ph, _escape_mdv2)
    text = re.sub(r'~~(.+?)~~', _ph_wrap('~', '~'), text)
    text = re.sub(r'\|\|(.+?)\|\|', _ph_wrap('||', '||'), text)
    # 9) Blockquotes: protect leading > from escaping; expandable quotes (**> starts, || ends the block's LAST line).
    def _convert_blockquote(m):  # one quote block; a raw || anywhere but its last line is an unclosed spoiler
        lines = [line.split(' ', 1) for line in m.group(0).split('\n')]  # [prefix, content]; prefix: >, >>, **> …
        expandable = lines[0][0].startswith(_EXPANDABLE_QUOTE)
        lines[0][0] = lines[0][0].replace(_EXPANDABLE_QUOTE, '**')
        last = len(lines) - 1 if expandable and lines[-1][1].endswith('||') else -1
        return '\n'.join(_ph(f'{p} {_escape_mdv2(c[:-2])}||' if i == last else f'{p} {_escape_mdv2(c)}')
                         for i, (p, c) in enumerate(lines))

    text = re.sub(rf'^(?:{_EXPANDABLE_QUOTE})?>{{1,3}} .+$(?:\n>{{1,3}} .+$)*', _convert_blockquote, text,
                  flags=re.MULTILINE)
    # A sentinel left over means the opener had no quote text after all: restore its literal **.
    text = text.replace(_EXPANDABLE_QUOTE, '**')
    # 10) Escape remaining special characters in plain text
    text = _escape_mdv2(text)
    # 11) Restore placeholders in one pass: a value only nests keys stashed before it, so expanding
    #     values in insertion order resolves every nesting without one whole-text scan per key.
    expanded: dict = {}
    for key, value in placeholders.items():
        expanded[key] = _expand_placeholders(value, expanded)
    text = _expand_placeholders(text, expanded)
    # 12) Safety net: escape bare ( ) { } that slipped through, but never inside ``` or ` spans.
    _safe_parts = []
    for _idx, _seg in enumerate(re.split(r'(```[\s\S]*?```|`[^`\r\n]+`)', text)):
        if _idx % 2 == 1:
            _safe_parts.append(_seg)  # inside code — untouched
        else:
            _safe_parts.append(re.sub(r'[(){}]', lambda m, _seg=_seg: _escape_bare_bracket(m, _seg), _seg))
    return ''.join(_safe_parts)


_PLACEHOLDER_RE = re.compile('\x00PH\\d+\x00')


def _expand_placeholders(text: str, expanded: dict) -> str:
    if '\x00' not in text:
        return text
    return _PLACEHOLDER_RE.sub(lambda m: expanded.get(m.group(0), m.group(0)), text)


# A bare scheme URL; trailing punctuation is prose, not URL.
_BARE_URL_RE = re.compile(r'(?i)\b[a-z][a-z0-9+.-]*://[^\s<>\x00]+')
_URL_TAIL_CHARS = '.,;:!?\'"'
# A run of emphasis/strike/spoiler marks with prose after it (the end, punctuation, non-ASCII) ends a
# URL: ``**https://x.com**—next``. Followed by more URL text (``a*b*c``, ``x||y``) it is URL text.
_URL_MARK_RUN_RE = re.compile(r'[*~|]+')
_URL_PROSE_AFTER = frozenset('.,;:!?\'")]}')
_SPAN_MARKS = {'~~': re.compile(r'~~(.+?)~~'), '||': re.compile(r'\|\|(.+?)\|\|')}  # steps 6/7
# Characters re-paired while deciding ``/``-ended mark runs (two passes over the paragraph each); past
# it the rest read as delimiters, as on a URL-free text, so a huge reply stays linear.
_SLASH_RUN_BUDGET = 200_000


def _url_end(url: str) -> int:
    """Length of *url* without trailing prose punctuation; a ``)`` stays only while it closes a ``(``.
    One pass: the paren balance is counted once and updated while trimming."""
    end, unclosed = len(url), url.count(')') - url.count('(')
    while end:
        ch = url[end - 1]
        if ch == ')' and unclosed > 0:
            unclosed -= 1
        elif ch == '(':
            unclosed += 1
            break
        elif ch not in _URL_TAIL_CHARS:
            break
        end -= 1
    return end


class _BareUrl:
    """One bare URL: ``text[start:core]`` is URL text; ``text[core:run_end]`` is the mark run prose
    follows, which closes emphasis (main's reading) unless it is a ``/web/*`` wildcard."""

    __slots__ = ("core", "keep_run", "run_end", "slash_run", "start")

    def __init__(self, text: str, m):
        self.start = m.start()
        url = m.group(0)
        run = _closing_mark_run(url)
        if run is None:
            self.core = self.run_end = self.start + _url_end(url)
        else:
            self.core = self.start + _url_end(url[:run.start()])
            self.run_end = self.start + run.end()
        self.slash_run = run is not None and self.core == self.start + run.start() and text[self.core - 1] == '/'
        self.keep_run = False

    @property
    def stop(self) -> int:
        return self.run_end if self.keep_run else self.core


def _closing_mark_run(url: str):
    """The first mark run of *url* that prose follows and no later run of the same marks reopens as
    URL text: in ``a*;b*c`` the ``*;`` pairs with ``*c``, so ``;`` is internal punctuation and the URL
    runs on; in ``a**,next`` nothing does, so ``**`` closes emphasis. One right-to-left pass."""
    runs = list(_URL_MARK_RUN_RE.finditer(url))
    reopened, closing = set(), None
    for r in reversed(runs):
        if not _prose_follows(url, r.end()):
            reopened.add(r.group())
        elif r.group() not in reopened:
            closing = r
    return closing


def _prose_follows(url: str, end: int) -> bool:
    nxt = url[end:end + 1]
    return not nxt or nxt in _URL_PROSE_AFTER or not nxt.isascii()


def _unpaired_marks(text: str) -> int:
    """Delimiter marks of *text* that pair with nothing and so would show as literal characters."""
    count = sum(tok[5] for tok in _pair_emphasis(text) if tok[0] == 'd')
    for mark, pat in _SPAN_MARKS.items():
        count += pat.sub('', text).count(mark)
    return count


def _dry_paragraph(text: str, start: int, end: int, urls: list) -> str:
    """``text[start:end]`` with each URL as one opaque word, its mark run kept only as a delimiter."""
    parts, pos = [], start
    for url in urls:
        parts += [text[pos:url.start], 'u']
        pos = url.stop
    return ''.join(parts) + text[pos:end]


def _decide_slash_runs(text: str, urls: list) -> None:
    """A mark run right after a URL's ``/`` is ambiguous: ``**see https://x.com/**`` closes bold,
    ``https://web.archive.org/web/*`` is a wildcard. Left to right, keep it as URL text only when that
    leaves fewer marks unpaired in its paragraph than reading it as a delimiter; the real pairing
    decides, so nested and multi-line emphasis pair exactly as they do around any other word."""
    if not any(u.slash_run for u in urls):
        return
    bounds = [0] + [m.end() for m in re.finditer(r'\n\n', text)] + [len(text) + 1]
    budget = _SLASH_RUN_BUDGET
    for a, b in itertools.pairwise(bounds):
        lo, hi = bisect.bisect_left(urls, a, key=lambda u: u.start), bisect.bisect_left(urls, b, key=lambda u: u.start)
        para = urls[lo:hi]
        runs = [u for u in para if u.slash_run]
        budget -= 2 * len(runs) * (b - a)
        if not runs or budget < 0:
            continue
        for url in runs:
            closing = _unpaired_marks(_dry_paragraph(text, a, b, para))
            url.keep_run = True
            if _unpaired_marks(_dry_paragraph(text, a, b, para)) >= closing:
                url.keep_run = False


def _stash_bare_urls(text: str, stash) -> str:
    """Stash bare URLs escaped whole, so a ``*``, ``~~`` or ``||`` inside one stays literal URL text
    instead of pairing into italic/strike/spoiler marks that drop characters or hide the URL."""
    urls, pos = [], 0
    while (m := _BARE_URL_RE.search(text, pos)) is not None:
        urls.append(_BareUrl(text, m))
        pos = max(urls[-1].run_end, m.start() + 1)
    _decide_slash_runs(text, urls)
    out, pos = [], 0
    for url in urls:
        out += [text[pos:url.start], stash(_escape_mdv2(text[url.start:url.stop]))]
        pos = url.stop
    return ''.join(out) + text[pos:]


def _stash_inline_code(text: str, stash) -> str:
    """Stash inline code spans: a backtick run opens a span that the next run of the same width on
    the same line closes (CommonMark), so runs inside an open span are content, not new spans.

    Telegram code spans are single-backtick, so a span keeps its content verbatim with ` and \\
    escaped. A ``...`` span whose content holds backticks, or a ```...``` run, keeps its outer
    ticks as literal text and its content is scanned for spans of its own (`` `code` ``)."""
    runs = list(re.finditer(r'`+', text))
    breaks = [m.start() for m in re.finditer(r'[\r\n]', text)] + [len(text)]
    next_same, last = [None] * len(runs), {}
    for k in range(len(runs) - 1, -1, -1):  # index of the next run of the same width
        width = len(runs[k].group())
        next_same[k], last[width] = last.get(width), k
    literal, out, pos, i = set(), [], 0, 0
    while i < len(runs):
        opener, width, close = runs[i], len(runs[i].group()), next_same[i]
        if close is not None and runs[close].start() > breaks[bisect.bisect_left(breaks, opener.end())]:
            close = None  # a code span never crosses a line
        if i in literal or close is None:
            i += 1
            continue
        content = text[opener.end():runs[close].start()]
        if width > 2 or (width == 2 and '`' in content):
            literal.add(close)
            i += 1
            continue
        out.append(text[pos:opener.start()])
        out.append(stash('`' + content.replace('\\', '\\\\').replace('`', '\\`') + '`'))
        pos, i = runs[close].end(), close + 1
    out.append(text[pos:])
    return ''.join(out)


def _escape_bare_bracket(m, seg: str) -> str:
    """Escape a bare ( ) { } unless it is already escaped or delimits a ``[text](url)`` link."""
    s = m.start()
    ch = m.group(0)
    if s > 0 and seg[s - 1] == '\\':  # already escaped
        return ch
    if ch == '(' and s > 0 and seg[s - 1] == ']':  # opens a link [text](url)
        return ch
    if ch == ')':  # closes a link URL? walk back matching depth
        before = seg[:s]
        if '](http' in before or '](' in before:
            depth = 0
            for j in range(s - 1, max(s - 2000, -1), -1):
                if seg[j] == '(':
                    depth -= 1
                    if depth < 0:
                        if j > 0 and seg[j - 1] == ']':
                            return ch
                        break
                elif seg[j] == ')':
                    depth += 1
    return '\\' + ch


# The formula bodies are possessive: none can consume $, so a match ends a body at the first
# character it can't take, and a failed span must not retry every split of a TeX command (the
# escape runs on the gateway event loop for every streamed draft frame).
# Regions the rich-message dollar escape leaves verbatim (code is skipped before this runs, see
# ``rich_code_regions``; $$ display math, $x$ inline math, digit-led formulas such as $2+2$
# that carry an operator or close tight like $4$ (an amount is never closed by a $ that no digit or
# word follows), spaced formulas such as $2x + 1$, $2\,x$ or $5! = 120$ (any symbol but . , and `,
# or a TeX command, whose {...} arguments may hold words ($1\text{ kg}$); letters only as one-letter variables, so "$5 or 10$" and "$5 - $3" stay prose:
# the closing $ is never followed by a digit, so an amount's $ cannot close it), already-escaped \$),
# and the currency $ it escapes: a lone $ before an
# amount, or after a currency prefix (R$ 1.000, US$ 1,200). Telegram's rich Markdown pairs any two
# bare $ as inline LaTeX, so "$395k vs $483k" otherwise renders as one garbled math span.
# A $$ pair stays inside one stretch of prose: its body never holds a backtick (a code span
# split the prose there, so a prose "Use $$ for the PID" would pair across it with the next
# formula), a blank line, or a bare $ before a digit (an amount, never display math).
_RICH_DOLLAR_RE = re.compile(
    r'(?P<keep>\\\$|\$\$(?:[^$\n`]|(?<=\\)\$|(?<!\\)\$(?![\d$])|\n(?![ \t>]*(?:\n|$)))*?\$\$'
    r'|(?<![\w$])\$(?![\d\s$])(?:\\.|[^\\\n$])+?(?<!\s)\$(?![\d$])'
    r'|(?<![\w$\\])\$\d[\w.,]*(?:[-+*/^=<>]|\\[a-zA-Z]++)(?:[\w.,(){}^_+*/=<>-]|\\[a-zA-Z]+)*+(?<![-+*/^=<>])\$(?![\w$])'
    r'|(?<![\w$\\])\$\d[\w.,]*(?<![.,])\$(?![\w$])'
    r'|(?<![\w$\\])\$\d(?=[^$\n]*?(?:[^\w\s$.,`]|\\))'
    r'(?:\\[a-zA-Z]+(?:\{[^{}$\n\\]*\})*+|\\[^a-zA-Z\s$]|[^\W\d_](?![^\W\d_])|[^$\n\\\w`]|[\d_])*+(?<![-+*/^=<>\s])\$(?![\w$]|\s?\d))'
    r'|(?P<currency>(?<![\\$])\$(?!\$)(?=\d)|(?<=[A-Z])\$(?=\s?\d))'
)


_RICH_BRACKET_MATH_RE = re.compile(r'\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)')


def normalize_rich_latex(text: str) -> str:
    """Bare LaTeX macros in rich-message prose become Unicode. Code, ``\\(...\\)``/``\\[...\\]`` and every
    dollar span the currency escape keeps as math (``_RICH_DOLLAR_RE``) stay verbatim for the client."""
    return map_outside_code(text, _rich_latex_prose) if '\\' in text else text


def _rich_latex_prose(prose: str) -> str:
    if '\\' not in prose:
        return prose
    spans = [m.span() for m in _RICH_DOLLAR_RE.finditer(prose) if m.group('keep')]
    spans += [m.span() for m in _RICH_BRACKET_MATH_RE.finditer(prose)]
    return replace_macros_outside(prose, sorted(spans))


def escape_rich_currency_dollars(text: str) -> str:
    """Escape currency ``$`` for ``sendRichMessage`` markdown; real math, code and ``\\$`` are kept."""
    if '$' not in text:
        return text
    return map_outside_code(text, _escape_prose_dollars)


def _escape_prose_dollars(prose: str) -> str:
    if '$' not in prose:
        return prose
    return _RICH_DOLLAR_RE.sub(lambda m: m.group(0) if m.group('keep') else '\\$', prose)
