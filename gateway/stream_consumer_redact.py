"""Streamed-egress secret redaction for GatewayStreamConsumer (#56039).

A streamed reply is redacted while it is still arriving, so a pattern can only be trusted on text
the redactor has seen in full context. Everything the consumer emits mid-stream (preview edits,
draft/native frames, sealed overflow heads, split points) therefore stops at a *settled* boundary:
text after it could still be, or still complete, a secret — the token being typed, an
``Authorization:``/``Bearer``/``password:`` label whose value has not arrived, an open quote after
such a label, a private-key ``-----BEGIN`` block without its END, a PuTTY ``Private-Lines:`` body, a key
fragment the next line may continue. Withholding it keeps frames prefix-stable (a partial key is
never shown, then re-masked) and keeps a cut from separating a label from its value. The boundary
never moves back within a message: settled text has been emitted, and a frame that drops it is
not a prefix of the one before (stream-is-the-message drafts re-append the whole snapshot).

The settled text is redacted in place, incrementally: only the newly settled text plus a bounded
lookback is re-redacted per tick, so a long reply costs linear redaction work, not quadratic.
"""

from __future__ import annotations

import re

from agent.redact import _PREFIX_SUBSTRINGS
from agent.redact_phone_policy import redact_for_chat_egress as redact_for_egress

_TAIL_SCAN = 512        # how far back from the end (or from an open value's start) a label can start
_LABEL_MAX = 256        # longest label (plus separators) an open value can follow
_BLOCK_SCAN = 16384     # how far back an open key block can start (an 8192-bit PEM is ~6.5 KB)
_LOOKBACK = 1024        # most settled text re-redacted with each newly settled chunk
# Continuations a still-arriving token is probed with: an opaque body (masked only when what came
# before makes it a key), and the separators that join one into a credential (``id.secret``,
# ``bot:token``, ``user@host``).
_PROBE = "Q7mZ2xW9rT4kL8vB3nH6pD1sF5gJ0cYa8Kd2LmP4"
_CONTINUATIONS = (_PROBE, "." + _PROBE, ":" + _PROBE, "-" + _PROBE, "_" + _PROBE, "@example.com/")
# Literal starts of secrets the prefix list does not carry (JWT, base64 PEM, age).
_LITERALS = tuple(lit for lit in (*_PREFIX_SUBSTRINGS, "eyJ", "LS0tLS1CRUdJTi", "AGE-SECRET-KEY-1") if lit)
_LITERAL_MAX = max(map(len, _LITERALS))
_NON_KEY_CHAR_RE = re.compile(r"[^A-Za-z0-9_\-]")
# A Zhipu key's 32-hex id still arriving: no probe completes it until its ``.secret`` part does.
_HEX_ID_TAIL_RE = re.compile(r"(?<![A-Za-z0-9_.-])[0-9a-f]{8,32}\Z")

_KW = r"(?:key|token|secret|passw(?:or)?d|pass(?:phrase)?|pwd|pw|credentials?|auth\w*|cookie|sig\w*|bearer)"
_OPEN_VALUE = r"(?:\\?\"[^\"\n]*|\\?'[^'\n]*|\S*)"  # an open quoted value (to its own quote) or a token
# A CLI flag's quoted value may span lines (agent.redact._QUOTED_FLAG_VALUE): hold to its closing quote.
_OPEN_FLAG_VALUE = r"(?:\\?\"(?:\\[\s\S]|[^\"\\]){0,512}\\?|\\?'(?:'\\''|[^']){0,512}|\S*)"
_OPEN_TAIL_RE = re.compile(
    rf"\b(?:proxy-)?authorization\s*(?::\s*(?:\S+[ \t]+)?\S*)?\Z"            # header, scheme, credential
    rf"|\bbearer\s*\S*\Z"
    rf"|(?<![\w-])-{{1,2}}[\w-]*{_KW}[\w-]*(?:\s*=\s*|\s+){_OPEN_FLAG_VALUE}\Z"  # CLI flag + value
    rf"|(?<![\w-])-p\s+{_OPEN_FLAG_VALUE}\Z"                                     # docker login -p
    rf"|(?<![\w.-])\\?[\"']?[\w.-]*{_KW}[\w.-]*\\?[\"']?\s*(?:[:=]\s*{_OPEN_VALUE})?\Z"  # NAME[:=] value
    rf"|\b(?:password|pwd)[ \t]*=[^\n]*\Z"     # connection string: a later host=/user= masks it
    rf"|\b(?:set-)?cookie:[ \t]*(?:[^\s;]+;[ \t]*)*\S*\Z",
    re.IGNORECASE,
)
# The labels alone (every alternative without its ``\Z``): where an open label can start.
_OPEN_HEAD_RE = re.compile(_OPEN_TAIL_RE.pattern.replace(r"\Z", ""), re.IGNORECASE)
_PEM_END_RE = re.compile(r"-----END[A-Z0-9 ]*-----")
_LINE_OPENER_RE = re.compile(r"\b(?:(?:set-)?cookie:|(?:password|pwd)[ \t]*=)", re.IGNORECASE)
_PEM_BEGIN = "-----BEGIN"
_PEM_HEADER_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")  # an END line's own dashes
# A BEGIN header still being typed, or a closed one naming a block the redactor masks (PEM and
# PGP private keys). A certificate or public key header releases the hold.
_PEM_HOLD_RE = re.compile(r"-----BEGIN(?:[A-Z0-9 ]*-{0,4}\Z|[A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----)")
_PPK_OPEN_RE = re.compile(
    r"Private-Lines:[ \t]*\d*[ \t]*"
    r"(?:(?:\r?\n|\\r\\n|\\n)[ \t]*(?:\d+[|:\t][ \t]*)?[A-Za-z0-9+/=]*[ \t]*)*")
# A token run broken only by control/zero-width characters: the redactor joins it (#77484).
_CONTROL_SPLIT_RUN_RE = re.compile(
    r"[A-Za-z0-9_.\-\x00-\x1f\u200b-\u200f\u2028-\u202f\u2060\ufeff]*"
    r"[\x00-\x1f\u200b-\u200f\u2028-\u202f\u2060\ufeff]\Z")


def _token_may_be_secret(context: str, token: str) -> bool:
    """True when the trailing, still-arriving ``token`` is, or could grow into, a secret: it
    holds a secret's literal start or ends in the beginning of one, is masked already, or is
    masked once a continuation is appended."""
    if any(lit in token for lit in _LITERALS) or _HEX_ID_TAIL_RE.search(token):
        return True
    for i in range(max(0, len(token) - _LITERAL_MAX), len(token)):
        if (not i or not token[i - 1].isalnum()) and any(lit.startswith(token[i:]) for lit in _LITERALS):
            return True
    base = redact_for_egress(context + token)
    return (base != context + token and not base.endswith(token)) or any(
        redact_for_egress(context + token + c) != base + c for c in _CONTINUATIONS)


def _open_key_block(text: str, lo: int, end: int) -> int:
    """Start of the earliest key block past ``lo`` whose END is not settled by ``end`` (or of a
    trailing partial ``-----BEGIN``), else -1: a cut inside the block leaves both halves raw."""
    for m in _PEM_HOLD_RE.finditer(text, lo):
        if m.start() >= end:
            break
        close = _PEM_END_RE.search(text, m.end())
        if close is None or close.end() > end:
            return m.start()
    tail = text[-len(_PEM_BEGIN):]
    return next((len(text) - k for k in range(len(tail), 0, -1) if _PEM_BEGIN.startswith(tail[-k:])
                 and (k == len(text) or text[-k - 1] not in _PEM_HEADER_CHARS)), -1)


def _probe_start(text: str, token_start: int) -> int:
    """Where to probe the trailing token from. A token longer than any secret (``....`` rules,
    long paths) is probed from a character no key body holds, within _TAIL_SCAN of the end, so
    a key still arriving is seen whole; -1 when there is none (an opaque blob: hold all of it)."""
    n = len(text)
    if n - token_start <= _TAIL_SCAN:
        return token_start
    m = _NON_KEY_CHAR_RE.search(text, n - _TAIL_SCAN, n)
    return -1 if m is None else m.start()


def _long_token_head_may_be_secret(text: str, token_start: int) -> bool:
    """A token past _TAIL_SCAN is probed near its end, which no longer sees how it began: a
    ``eyJ…`` JWT or ``sk-…`` key with a long body would settle raw. Its head, with the line
    before it, is checked as a token still arriving (the cost stays bounded by _TAIL_SCAN)."""
    line_start = max(text.rfind("\n", 0, token_start) + 1, token_start - _TAIL_SCAN)
    return _token_may_be_secret(text[line_start:token_start], text[token_start:token_start + _TAIL_SCAN])


def _open_anchors(text: str, end: int) -> set[int]:
    """Where the open region ending at ``end`` can start: the trailing token, an unclosed quote
    on the line, a ``Cookie:`` list / ``password=`` on the line. An open value has no fixed
    length (an STS SessionToken is 1-2 KB), so its label is looked for just before one of these,
    not in a fixed window before ``end``, which loses the label once the value outgrows it."""
    line_start = text.rfind("\n", 0, end) + 1
    token_start = max(text.rfind(c, 0, end) for c in " \t\r\n") + 1
    anchors = {token_start}
    # A label inside the token (``?token=…``, ``--key=…``): its value starts after the last
    # separator, not counting a value's own trailing base64 ``=`` padding.
    body_end = len(text[token_start:end].rstrip("="))
    sep = max(text.rfind("=", token_start, token_start + body_end), text.rfind(":", token_start, end))
    if sep >= 0:
        anchors.add(sep)
    for quote in (text.rfind('"', line_start, end), text.rfind("'", line_start, end)):
        if quote >= 0:  # each kind: a later apostrophe must not hide the value's opening quote
            anchors.add(quote)
    if (opener := _last_line_opener(text, line_start, end)) >= 0:
        anchors.add(opener)
    return anchors


def _last_line_opener(text: str, lo: int, end: int) -> int:
    """Start of the last ``Cookie:`` list or connection-string ``password=`` in ``text[lo:end]``:
    their values span spaces, so the trailing token does not anchor them."""
    last = -1
    for m in _LINE_OPENER_RE.finditer(text, lo, end):
        last = m.start()
    return last


def _open_tail_start(text: str, end: int) -> int:
    """Start of the earliest open label/value that runs to ``end``, else -1. Labels are looked
    for within _TAIL_SCAN of ``end`` and within _TAIL_SCAN before each open-region anchor;
    candidates there are found with the label-only pattern and confirmed with an anchored
    match, so a tick's cost does not grow with the length of the value."""
    m = _OPEN_TAIL_RE.search(text, max(0, end - _TAIL_SCAN), end)
    best = m.start() if m else -1
    tail_lo = max(0, end - _TAIL_SCAN)
    for anchor in _open_anchors(text, end):
        lo, hi = max(0, anchor - _TAIL_SCAN), min(anchor + 1, tail_lo)
        while lo < hi and (head := _OPEN_HEAD_RE.search(text, lo, min(end, hi + _LABEL_MAX))) and (
                head.start() < hi):
            if _OPEN_TAIL_RE.match(text, head.start(), end):
                best = head.start() if best < 0 else min(best, head.start())
                break
            lo = head.start() + 1
    return best


def settled_end(text: str) -> int:
    """Index where the text that can no longer become part of a secret ends."""
    n = len(text)
    if not n:
        return 0
    end = n
    token_start = max(text.rfind(c) for c in " \t\r\n") + 1
    if token_start < n:
        probe_start = _probe_start(text, token_start)
        line_start = max(text.rfind("\n", 0, probe_start) + 1, probe_start - _TAIL_SCAN)
        if probe_start < 0 or _token_may_be_secret(text[line_start:probe_start], text[probe_start:]) or (
                probe_start > token_start and _long_token_head_may_be_secret(text, token_start)):
            end = token_start
    if (start := _open_tail_start(text, n)) >= 0:
        end = min(end, start)
    block_lo = max(0, n - _BLOCK_SCAN)
    ppk = text.rfind("Private-Lines:", block_lo)
    if ppk >= 0 and _PPK_OPEN_RE.fullmatch(text, ppk):
        end = min(end, ppk)
    if (m := _CONTROL_SPLIT_RUN_RE.search(text, max(0, n - _TAIL_SCAN))) and m.start() < end:
        run = m.group(0)
        if redact_for_egress(run + _PROBE) != redact_for_egress(run) + _PROBE:
            end = m.start()
    # Fixpoint: the settled text must not itself end in an open label (``Authorization: Bearer ``
    # with its value held by a later rule), and no boundary may cut a key block. Settling a label
    # apart from its value lets its redaction mask the label alone (the scheme word becomes
    # ``***``), after which the value no longer redacts at all.
    while True:
        prev = end
        if 0 <= (start := _open_tail_start(text, end)) < end:
            end = start
        if (begin := _open_key_block(text, block_lo, end)) >= 0:
            end = min(end, begin)
        if end == prev:
            return end


def _lookback_start(text: str, upto: int) -> int:
    """Where re-redaction restarts: the start of the line holding ``upto`` (line-anchored
    patterns such as YAML keys and connection strings need it), at most _LOOKBACK back.
    Multi-line candidates never reach back here: settled_end holds them until complete."""
    lo = max(0, upto - _LOOKBACK)
    newline = text.rfind("\n", lo, upto)
    if newline >= 0 or lo == 0:
        return newline + 1
    return max(text.rfind(" ", 0, lo), text.rfind("\t", 0, lo)) + 1


def redact_to_fixpoint(text: str) -> str:
    """Redact until stable. The redactor re-masks its own output (``Bearer abcd...wxyz`` ->
    ``Bearer ***`` -> ``***``), and the settled text is redacted again with later text and at
    segment end: emitting only a fixpoint keeps those passes from rewriting what was shown. Each
    pass only adds masks, so the loop ends."""
    for _ in range(4):
        again = redact_for_egress(text)
        if again == text:
            break
        text = again
    return text


def redact_settled(text: str, redacted_prefix: str, *, done: bool) -> tuple[str, int]:
    """Redact ``text`` up to its settled boundary (all of it when ``done``); returns the new text
    and where its settled part ends. ``redacted_prefix`` is the settled part returned last time:
    while ``text`` still starts with it, only the new text plus a bounded lookback is redacted."""
    upto = len(redacted_prefix) if text.startswith(redacted_prefix) else 0
    # Monotonic: once settled, text stays settled. A hold rule that fires on a later character
    # (``https://a`` may start ``am_…``) only holds what follows, so frames stay prefix-stable.
    settled = len(text) if done else max(settled_end(text), upto)
    if upto == settled and redacted_prefix:
        return text, settled
    start = _lookback_start(text, upto)
    if done:
        # A value longer than _LOOKBACK may sit across ``upto``: redact back to its line's start.
        start = min(start, text.rfind("\n", 0, upto) + 1)
    if settled <= start:
        return text, settled
    redacted = redact_to_fixpoint(text[start:settled])
    if not done and not redacted.startswith(text[start:upto]):
        # The new text changes how emitted text redacts. Emitted text cannot be rewritten and
        # nothing may be appended after it, since the redaction of the new text is only correct
        # in that changed context: hold until the segment ends and is redacted whole.
        return text, upto
    return text[:start] + redacted + text[settled:], start + len(redacted)
