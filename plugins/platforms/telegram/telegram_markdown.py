"""Markdown -> Telegram MarkdownV2 conversion (the legacy, non-rich send path)."""

import bisect
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
        tok = _star_token(text, m.start(), m.end())
        if tok is None:
            continue
        if tok[1] > pos:
            tokens.append(("t", text[pos:tok[1]]))
        tokens.append(tok)
        pos = m.end()
    if pos < len(text):
        tokens.append(("t", text[pos:]))
    pairing = _StarPairing(tokens, text)
    for idx, tok in enumerate(tokens):
        if tok[0] == "d":
            pairing.feed(idx)
    return tokens


def _star_token(text: str, start: int, end: int):
    """The delimiter token for the ``*`` run text[start:end], or ``None`` when it is one escaped star."""
    slash = start
    while slash > 0 and text[slash - 1] == "\\":
        slash -= 1
    if (start - slash) % 2:
        start += 1  # a backslash-escaped * is literal text (**Use \*args**, #151722)
        if start == end:
            return None
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return ["d", start, end, bool(after) and not after.isspace(), bool(before) and not before.isspace(),
            end - start, [], []]


class _StarPairing:
    """The ``*`` run pairing rules as incremental state, shared by :func:`_pair_emphasis` and the
    bare-URL scan so both split and pair a run the same way (``***a* b**`` closes one star, then two).

    :meth:`feed` pairs a run for real; :meth:`probe` asks what a run would close and records every
    change in a journal that :meth:`rollback` undoes, so a URL's mark run can be tried as a closer."""

    def __init__(self, tokens: list, text: str) -> None:
        self.tokens, self.edges = tokens, _SpanEdges(text)
        self.stack = []  # indexes of delimiter tokens that may still close
        self.near2 = []  # per stack slot: the slot of the nearest opener at or below it with 2+ stars left
        self.bottoms: dict = {}  # (closer length % 3, closer can open) -> token index a failed search reached

    def feed(self, idx: int) -> list:
        """Pair tokens[idx] with open runs, then keep its unpaired remainder open; the (kind, opener) pairs."""
        tok = self.tokens[idx]
        pairs = self._close(tok, idx, 0, None)
        if tok[5] > 0 and tok[3]:
            self.stack.append(idx)
            self.near2.append(len(self.stack) - 1 if tok[5] >= 2 else (self.near2[-1] if self.near2 else -1))
        return pairs

    def probe(self, tok: list, journal: list) -> list:
        """The pairs *tok* would close among the top openers; *tok* keeps what it could not pair."""
        journal.append(lambda b=dict(self.bottoms): (self.bottoms.clear(), self.bottoms.update(b)))
        return self._close(tok, None, max(0, len(self.stack) - _CLOSER_LOOKBACK), journal)

    def rollback(self, journal: list) -> None:
        for undo in reversed(journal):
            undo()
        journal.clear()

    def _close(self, tok: list, idx, lowest: int, journal) -> list:
        stack, near2, pairs = self.stack, self.near2, []
        while tok[4] and tok[5] > 0 and len(stack) > lowest:
            depth = _pick_opener(self.tokens, stack, near2, tok, self.bottoms, self.edges, lowest)
            if depth is None:
                break
            self._cut(depth + 1, journal)  # openers between the pair stay literal (CommonMark)
            opener = self.tokens[stack[-1]]
            use = 2 if opener[5] >= 2 and tok[5] >= 2 else 1
            kind = "bold" if use == 2 else "italic"
            pairs.append((kind, stack[-1]))
            if journal is None:
                opener[6].append((kind, idx))
                tok[7].append((kind, stack[-1]))
            else:
                journal.append(lambda o=opener, r=opener[5]: o.__setitem__(5, r))
            opener[5] -= use
            tok[5] -= use
            if opener[5] == 0:
                self._cut(len(stack) - 1, journal)
            elif opener[5] < 2:
                if journal is not None:
                    journal.append(lambda j=len(near2) - 1, v=near2[-1]: near2.__setitem__(j, v))
                near2[-1] = near2[-2] if len(near2) > 1 else -1
        return pairs

    def _cut(self, keep: int, journal) -> None:
        stack, near2 = self.stack, self.near2
        if journal is not None and keep < len(stack):
            journal.append(lambda s=stack[keep:], n=near2[keep:]: (stack.extend(s), near2.extend(n)))
        del stack[keep:], near2[keep:]


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


def _pick_opener(tokens: list, stack: list, near2: list, closer: list, bottoms: dict, edges, lowest: int = 0):
    """Stack slot of the opener *closer* pairs with, or ``None``.

    The nearest opener the rule of 3 allows wins, as in CommonMark, except that a bare ``**``
    closer reaches past single-``*`` openers to a ``**`` opener: chat text means
    ``**rm -rf *.tmp**`` as bold around a literal ``*`` (CommonMark nests two italics there, which
    Telegram cannot show, and drops the star). A ``***`` closer still pairs its ``*`` first
    (``**a *b***``). A failed
    search records how far down it looked for closers of the same class, so a long run of refused
    openers is walked once, not once per closer. A probe passes *lowest*, the deepest slot it may
    reach; a pairing that would reach below it reports ``None``.
    """
    key = ((closer[2] - closer[1]) % 3, closer[3])
    floor = bottoms.get(key, -1)
    nearest = None
    complete = lowest == 0
    for depth in range(len(stack) - 1, lowest - 1, -1):
        if stack[depth] <= floor:
            complete = True
            break
        if not _rule_of_three_refuses(tokens[stack[depth]], closer):
            nearest = depth
            break
    if nearest is None:
        if complete:
            bottoms[key] = stack[-1]
        return None
    candidates = [nearest]
    if closer[2] - closer[1] == closer[5] == 2 and tokens[stack[nearest]][5] < 2 and near2[nearest] >= 0:
        if near2[nearest] < lowest:
            return None
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
_URL_MARK_RUN_RE = re.compile(r'[*~|]+')
_URL_PROSE_AFTER = frozenset('.,;:!?\'")]}')
# Prose delimiters the URL scan tracks: paragraph breaks, line breaks, ``*`` runs, strike, spoiler.
_PROSE_MARK_RE = re.compile(r'\n\n+|\n|\*+|~~|\|\|')
_SINGLE_LINE_MARKS = ('*', '***', '~~', '||')  # bold (``**``) alone may span one newline
_CLOSER_LOOKBACK = 8  # live openers a URL mark run may close at once (``~~**x/**~~``)


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


def _escaped(text: str, pos: int) -> bool:
    """True when text[pos] follows an odd run of backslashes."""
    slash = pos
    while slash > 0 and text[slash - 1] == "\\":
        slash -= 1
    return (pos - slash) % 2 == 1


def _prose_follows(url: str, end: int) -> bool:
    nxt = url[end:end + 1]
    return not nxt or nxt in _URL_PROSE_AFTER or not nxt.isascii()


class _Opener:
    """An emphasis/strike/spoiler opener seen in prose, and the URL mark run that may close it.
    *mark* is ``~~``/``||``, or the token index of a ``*`` run (its stars left live in the token)."""

    __slots__ = ("alive", "close", "mark", "spent")

    def __init__(self, mark) -> None:
        self.mark, self.alive, self.close = mark, True, None
        self.spent = False  # its URL closer would pair every star it has left


class _UrlEmphasisScan:
    """One rule decides where each bare URL ends: a mark run inside a URL is URL text unless it
    closes emphasis that opened in prose before the URL. Left to right, prose delimiters keep a stack
    of open marks; ``*`` runs pair through :class:`_StarPairing`, the rules the renderer uses, so a
    run may close part of an opener (``***bold* https://x/**``). In a URL, the first mark run that
    closes the innermost open marks exactly and has prose after it (``**https://x/a**,next``) is that
    emphasis's fallback closer. A later prose closer of the same emphasis wins
    (``*Open https://x/a*;b*c now*``, ``**b https://x/p* d*`` keep the URL whole) unless the URL run
    would have spent that opener's stars (``*o **https://x/** e*``: ``e*`` closes the outer ``*``); an opener left unclosed when its
    line/paragraph ends, or when an enclosing emphasis closes over it, takes its fallback, and the
    URL ends there. Every other mark is URL text. One pass, linear."""

    def __init__(self, text: str) -> None:
        self.text, self.stack, self.by_mark = text, [], {}
        self.tokens: list = []  # every prose ``*`` run, as :func:`_pair_emphasis` tokens
        self.stars = _StarPairing(self.tokens, text)
        self.star_openers: dict = {}  # token index -> its _Opener
        self.ends: dict = {}  # URL match start -> where its protected text ends, for released runs

    def run(self, urls: list) -> dict:
        pos = 0
        for m in urls:
            self._prose(pos, m.start())
            self._url(m)
            pos = m.end()
        self._prose(pos, len(self.text))
        self._release(self.stack)
        return self.ends

    def _prose(self, start: int, end: int) -> None:
        text = self.text
        for m in _PROSE_MARK_RE.finditer(text, start, end):
            tok = m.group()
            if tok[0] == '\n':
                self._release(self.stack if len(tok) > 1 else self._single_line_openers())
            elif tok[0] == '*':
                self._star_run(m.start(), m.end())
            elif not (m.start() and text[m.start() - 1] == '\\'):  # an escaped mark is literal
                self._delimiter(tok)

    def _single_line_openers(self) -> list:
        """Openers a newline ends: strike, spoiler, and ``*`` runs with other than two stars left."""
        out = self.by_mark.pop('~~', []) + self.by_mark.pop('||', [])
        # A run filed under ``*`` that is down to two stars is filed under ``**`` too: it survives.
        return out + [o for o in self.by_mark.pop('*', []) if self.tokens[o.mark][5] != 2]

    def _star_run(self, start: int, end: int) -> None:
        tok = _star_token(self.text, start, end)
        if tok is None:
            return
        idx = len(self.tokens)
        self.tokens.append(tok)
        for _kind, at in self.stars.feed(idx):
            opener = self.star_openers[at]
            if not opener.alive:
                continue
            self._unwind(opener)
            if opener.close is not None and (self.tokens[at][5] == 0 or not opener.spent):
                opener.close[0] = False  # prose closed what the URL run would: it stays URL text
            if self.tokens[at][5] == 0:
                opener.alive = False
                self.stack.pop()
            else:
                self._file_star(opener)
        if tok[5] and tok[3]:
            opener = _Opener(idx)
            self.star_openers[idx] = opener
            self.stack.append(opener)
            self._file_star(opener)

    def _file_star(self, opener) -> None:
        """File a ``*`` opener by what a newline does to it; a run is refiled as it loses stars."""
        bucket = '**' if self.tokens[opener.mark][5] == 2 else '*'
        self.by_mark.setdefault(bucket, []).append(opener)

    def _unwind(self, target) -> None:
        while (top := self.stack[-1]) is not target:
            self.stack.pop()
            self._release([top])  # an opener inside the closed span stays unclosed

    def _delimiter(self, mark: str) -> None:
        same = self.by_mark.get(mark, [])
        while same and not same[-1].alive:
            same.pop()
        if same:
            target = same.pop()
            self._unwind(target)
            self.stack.pop()
            target.alive = False
            if target.close is not None:
                target.close[0] = False  # prose closed it: the URL run stays URL text
        else:
            opener = _Opener(mark)
            self.stack.append(opener)
            self.by_mark.setdefault(mark, []).append(opener)

    def _url(self, m) -> None:
        url = m.group()
        for run in _URL_MARK_RUN_RE.finditer(url):
            if not _prose_follows(url, run.end()):
                continue
            # A backslash-escaped mark is URL text (``https://x/a\**``): only the marks after it may close.
            start = run.start() + _escaped(url, run.start())
            if start == run.end():
                continue
            openers = self._innermost(m.start() + start, url[start:run.end()])
            if openers:
                fallback = [True, m.start(), m.start() + _url_end(url[:start])]
                for opener in openers:
                    opener.close = fallback
                return

    def _innermost(self, at: int, run: str) -> list:
        """The live openers, innermost first, that *run* (at text offset *at*) closes, or ``[]``. Every
        mark must pair except stars left over at its end; pairing may reach past openers it leaves
        literal (``**rm *x https://x/y**``)."""
        while self.stack and not self.stack[-1].alive:
            self.stack.pop()
        live = [o for o in self.stack[-1:-_CLOSER_LOOKBACK - 1:-1] if o.alive]
        taken, journal, pos, reach = [], [], 0, 0
        try:
            while pos < len(run):
                if run[pos] == '*':
                    stars = len(run) - pos - len(run[pos:].lstrip('*'))
                    closed, spare = self._star_closes(at + pos, at + pos + stars, journal)
                    pos += stars
                    if spare and pos < len(run):
                        return []  # stars the pairing leaves over end the closing run
                else:
                    nxt = live[reach] if reach < len(live) else None
                    closed = [nxt] if nxt and isinstance(nxt.mark, str) and run.startswith(nxt.mark, pos) else None
                    pos += 2
                if not closed or any(o.close is not None or o not in live[reach:] for o in closed):
                    return []
                if pos < len(run) and any(isinstance(o.mark, int) and self.tokens[o.mark][5] for o in closed):
                    return []  # a partly closed run must be the last thing this URL run closes
                reach = max(live.index(o) for o in closed) + 1
                taken += closed
            for opener in taken:
                opener.spent = isinstance(opener.mark, str) or self.tokens[opener.mark][5] == 0
            return taken
        finally:
            self.stars.rollback(journal)

    def _star_closes(self, start: int, end: int, journal: list) -> tuple:
        """The ``*`` openers, innermost first, a URL star run text[start:end] pairs with (``None`` for
        none), and whether stars are left over: ``**bold* https://x/p**`` closes one, one stays literal."""
        tok = _star_token(self.text, start, end)
        if tok is None:
            return None, False
        pairs = self.stars.probe(tok, journal)
        return list(dict.fromkeys(self.star_openers[at] for _kind, at in pairs)) or None, tok[5] > 0

    def _release(self, openers: list) -> None:
        """*openers* end unclosed by prose: each takes its URL fallback closer, if it still has one."""
        for opener in openers:
            if opener.alive and opener.close is not None and opener.close[0]:
                self.ends.setdefault(opener.close[1], opener.close[2])
            opener.alive = False
        if openers is self.stack:
            self.stack, self.by_mark = [], {}


def _stash_bare_urls(text: str, stash) -> str:
    """Stash bare URLs escaped whole, so a ``*``, ``~~`` or ``||`` inside one stays literal URL text
    instead of pairing into italic/strike/spoiler marks that drop characters or hide the URL; only
    a run closing emphasis opened before the URL is released (:class:`_UrlEmphasisScan`)."""
    urls = list(_BARE_URL_RE.finditer(text))
    if not urls:
        return text
    ends = _UrlEmphasisScan(text).run(urls)
    out, pos = [], 0
    for m in urls:
        stop = ends.get(m.start(), m.start() + _url_end(m.group()))
        out += [text[pos:m.start()], stash(_escape_mdv2(text[m.start():stop]))]
        pos = stop
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
