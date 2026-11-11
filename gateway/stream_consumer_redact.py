"""Streamed-egress secret redaction for GatewayStreamConsumer (#56039).

A streamed reply is redacted while it is still arriving, so a pattern can only be trusted on text
the redactor has seen in full context. Everything the consumer emits mid-stream (preview edits,
draft/native frames, sealed overflow heads, split points) therefore stops at a *settled* boundary:
text after it could still be, or still complete, a secret — the token being typed, an
``Authorization:``/``Bearer``/``password:`` label whose value has not arrived, an open quote after
such a label, a ``-----BEGIN`` block without its END, a PuTTY ``Private-Lines:`` body, a key
fragment the next line may continue. Withholding it keeps frames prefix-stable (a partial key is
never shown, then re-masked) and keeps a cut from separating a label from its value.

The settled text is redacted in place, incrementally: only the newly settled text plus a bounded
lookback is re-redacted per tick, so a long reply costs linear redaction work, not quadratic.
"""

from __future__ import annotations

import re

from agent.redact import _PREFIX_SUBSTRINGS, redact_for_egress

_TAIL_SCAN = 512        # how far back an open label/value can start
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

_KW = r"(?:key|token|secret|passw(?:or)?d|pass(?:phrase)?|pwd|pw|credentials?|auth\w*|cookie|sig\w*|bearer)"
_OPEN_VALUE = r"(?:\\?[\"'][^\"'\n]*|\S*)"  # an open quoted value or a still-arriving token
_OPEN_TAIL_RE = re.compile(
    rf"\b(?:proxy-)?authorization\s*(?::\s*(?:\S+[ \t]+)?\S*)?\Z"            # header, scheme, credential
    rf"|\bbearer\s*\S*\Z"
    rf"|(?<![\w-])-{{1,2}}[\w-]*{_KW}[\w-]*(?:\s*=\s*|\s+){_OPEN_VALUE}\Z"     # CLI flag + value
    rf"|(?<![\w-])-p\s+\S*\Z"                                                # docker login -p
    rf"|(?<![\w.-])\\?[\"']?[\w.-]*{_KW}[\w.-]*\\?[\"']?\s*(?:[:=]\s*{_OPEN_VALUE})?\Z"  # NAME[:=] value
    rf"|\b(?:password|pwd)[ \t]*=[^\n]*\Z"     # connection string: a later host=/user= masks it
    rf"|\b(?:set-)?cookie:[ \t]*(?:[^\s;]+;[ \t]*)*\S*\Z",
    re.IGNORECASE,
)
_PEM_END_RE = re.compile(r"-----END[A-Z0-9 ]*-----")
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
    if any(lit in token for lit in _LITERALS):
        return True
    for i in range(max(0, len(token) - _LITERAL_MAX), len(token)):
        if (not i or not token[i - 1].isalnum()) and any(lit.startswith(token[i:]) for lit in _LITERALS):
            return True
    base = redact_for_egress(context + token)
    return (base != context + token and not base.endswith(token)) or any(
        redact_for_egress(context + token + c) != base + c for c in _CONTINUATIONS)


def _probe_start(text: str, token_start: int) -> int:
    """Where to probe the trailing token from. A token longer than any secret (``....`` rules,
    long paths) is probed from a character no key body holds, within _TAIL_SCAN of the end, so
    a key still arriving is seen whole; -1 when there is none (an opaque blob: hold all of it)."""
    n = len(text)
    if n - token_start <= _TAIL_SCAN:
        return token_start
    m = _NON_KEY_CHAR_RE.search(text, n - _TAIL_SCAN, n)
    return -1 if m is None else m.start()


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
        if probe_start < 0 or _token_may_be_secret(text[line_start:probe_start], text[probe_start:]):
            end = token_start
    if m := _OPEN_TAIL_RE.search(text, max(0, n - _TAIL_SCAN)):
        end = min(end, m.start())
    block_lo = max(0, n - _BLOCK_SCAN)
    begin = text.rfind("-----BEGIN", block_lo)
    if begin >= 0 and not _PEM_END_RE.search(text, begin):
        end = min(end, begin)
    ppk = text.rfind("Private-Lines:", block_lo)
    if ppk >= 0 and _PPK_OPEN_RE.fullmatch(text, ppk):
        end = min(end, ppk)
    if (m := _CONTROL_SPLIT_RUN_RE.search(text, max(0, n - _TAIL_SCAN))) and m.start() < end:
        run = m.group(0)
        if redact_for_egress(run + _PROBE) != redact_for_egress(run) + _PROBE:
            end = m.start()
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


def redact_settled(text: str, redacted_prefix: str, *, done: bool) -> tuple[str, int]:
    """Redact ``text`` up to its settled boundary (all of it when ``done``); returns the new text
    and where its settled part ends. ``redacted_prefix`` is the settled part returned last time:
    while ``text`` still starts with it, only the new text plus a bounded lookback is redacted."""
    settled = len(text) if done else settled_end(text)
    upto = min(settled, len(redacted_prefix)) if text.startswith(redacted_prefix) else 0
    if upto == settled and redacted_prefix:
        return text, settled
    start = _lookback_start(text, upto)
    if settled <= start:
        return text, settled
    redacted = redact_for_egress(text[start:settled])
    return text[:start] + redacted + text[settled:], start + len(redacted)
