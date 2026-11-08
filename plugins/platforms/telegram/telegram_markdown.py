"""Markdown -> Telegram MarkdownV2 conversion (the legacy, non-rich send path)."""

import bisect
import re

from gateway.platforms.helpers import convert_table_to_bullets as _wrap_markdown_tables

# Stands in for the ** of a line-start **> expandable-quote opener while emphasis is resolved.
_EXPANDABLE_QUOTE = "\x00EQ\x00"

# Every char MarkdownV2 requires backslash-escaped outside code spans/fences.
_MDV2_ESCAPE_RE = re.compile(r'([_*\[\]()~`>#\+\-=|{}.!\\])')
_EMPHASIS_MARK = {"bold": "*", "italic": "_"}


def _resolve_emphasis(text: str, stash, escape) -> str:
    """Resolve **bold** and *italic* delimiter runs together so nested/combined emphasis keeps its
    inner markers (***x*** → _*x*_, **a *i* b** → *a _i_ b*) instead of the bold pass swallowing
    them. Bold may span a single newline within a paragraph; italic stays single-line; unmatched
    markers survive as literal text."""

    tokens = []  # ("t", s) literal | ["d", start, end, can_open, can_close, remaining, opens, closes]
    pos = 0
    for m in re.finditer(r"\*+", text):
        if m.start() > pos:
            tokens.append(("t", text[pos : m.start()]))
        before = text[m.start() - 1] if m.start() > 0 else ""
        after = text[m.end()] if m.end() < len(text) else ""
        can_open = bool(after) and not after.isspace()
        can_close = bool(before) and not before.isspace()
        tokens.append([
            "d",
            m.start(),
            m.end(),
            can_open,
            can_close,
            m.end() - m.start(),
            [],
            [],
        ])
        pos = m.end()
    if pos < len(text):
        tokens.append(("t", text[pos:]))

    stack = []  # indexes of delimiter tokens that may still close
    for idx, tok in enumerate(tokens):
        if tok[0] != "d":
            continue
        _, start, _end, can_open, can_close, remaining, _opens, closes = tok
        if can_close:
            while remaining > 0 and stack:
                opener = tokens[stack[-1]]
                o_end, o_remaining = opener[2], opener[5]
                use = 2 if o_remaining >= 2 and remaining >= 2 else 1
                span = text[o_end:start]
                if use == 1 and "\n" in span:
                    break  # italic must not cross a line
                if use == 2 and ("\n\n" in span or _crosses_quote_edge(text, o_end, start)):
                    break  # bold stays within one paragraph and on one side of a quote's edge
                kind = "bold" if use == 2 else "italic"
                opener[6].append((kind, idx))
                closes.append((kind, stack[-1]))
                opener[5] = o_remaining - use
                remaining -= use
                if opener[5] == 0:
                    stack.pop()
        tok[5] = remaining
        if remaining > 0 and can_open:
            stack.append(idx)

    return _render_emphasis(tokens, stash, escape)


def _crosses_quote_edge(text: str, start: int, end: int) -> bool:
    """True when a newline in text[start:end] borders a quote line: Telegram ends a blockquote at a
    newline into non-quote text, an entity still open there is rejected, and an entity spanning
    quote lines would swallow the next line's > marker."""
    quote = re.compile(rf'(?:{_EXPANDABLE_QUOTE})?>')
    is_quote = lambda pos: quote.match(text, text.rfind('\n', 0, pos) + 1) is not None
    return any(is_quote(start + m.start()) or is_quote(start + m.end()) for m in re.finditer('\n', text[start:end]))


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

    # 0) Rewrite GFM-style pipe tables into Telegram-friendly row groups
    #    before the normal MarkdownV2 conversions run.
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
        url = m.group(2).replace('\\', '\\\\').replace(')', '\\)')
        return _ph(f'[{_escape_mdv2(m.group(1))}]({url})')

    text = re.sub(r'\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)', _convert_link, text)
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
    # 11) Restore placeholders in reverse insertion order so nested placeholders resolve.
    for key in reversed(list(placeholders.keys())):
        text = text.replace(key, placeholders[key])
    # 12) Safety net: escape bare ( ) { } that slipped through, but never inside ``` or ` spans.
    _safe_parts = []
    for _idx, _seg in enumerate(re.split(r'(```[\s\S]*?```|`[^`\r\n]+`)', text)):
        if _idx % 2 == 1:
            _safe_parts.append(_seg)  # inside code — untouched
        else:
            _safe_parts.append(re.sub(r'[(){}]', lambda m, _seg=_seg: _escape_bare_bracket(m, _seg), _seg))
    return ''.join(_safe_parts)


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
# Regions the rich-message dollar escape leaves verbatim (fenced code, inline code delimited by
# matching backtick-run widths, $$ display math, $x$ inline math, digit-led formulas such as $2+2$
# that carry an operator or close tight like $4$ (an amount is never closed by a $ that no digit or
# word follows), spaced formulas such as $2x + 1$ or $2\,x$ (an operator or TeX command, letters
# only as one-letter variables, so "$5 or 10$" and "$5 - $3" stay prose), already-escaped \$),
# and the currency $ it escapes: a lone $ before an
# amount, or after a currency prefix (R$ 1.000, US$ 1,200). Telegram's rich Markdown pairs any two
# bare $ as inline LaTeX, so "$395k vs $483k" otherwise renders as one garbled math span.
_RICH_DOLLAR_RE = re.compile(
    r'(?P<keep>```[\s\S]*?```|(?P<ticks>`+)[^\n]*?(?<!`)(?P=ticks)(?!`)|\\\$|\$\$[\s\S]*?\$\$'
    r'|(?<![\w$])\$(?![\d\s$])(?:\\.|[^\\\n$])+?(?<!\s)\$(?![\d$])'
    r'|(?<![\w$\\])\$\d[\w.,]*(?:[-+*/^=<>]|\\[a-zA-Z]++)(?:[\w.,(){}^_+*/=<>-]|\\[a-zA-Z]+)*+(?<![-+*/^=<>])\$(?![\w$])'
    r'|(?<![\w$\\])\$\d[\w.,]*(?<![.,])\$(?![\w$])'
    r'|(?<![\w$\\])\$\d(?=[^$\n]*?(?:[-+*/^=<>]|\\))'
    r'(?:[\d.,(){}^_+*/=<> -]|\\[a-zA-Z]+|\\[^a-zA-Z\s$]|[a-zA-Z](?![a-zA-Z]))*+(?<![-+*/^=<>\s])\$(?![\w$]|\s?\d))'
    r'|(?P<currency>(?<![\\$])\$(?!\$)(?=\d)|(?<=[A-Z])\$(?=\s?\d))'
)


def escape_rich_currency_dollars(text: str) -> str:
    """Escape currency ``$`` for ``sendRichMessage`` markdown; real math, code and ``\\$`` are kept."""
    if '$' not in text:
        return text
    return _RICH_DOLLAR_RE.sub(lambda m: m.group(0) if m.group('keep') else '\\$', text)
