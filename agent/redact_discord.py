"""Discord bot token redaction for agent.redact (#117848)."""

import base64
import re

# Discord bot tokens: ``<base64 user id>.<base64 timestamp>.<hmac>``, plus the ``mfa.<secret>``
# form. Shape alone is not enough — a long dotted CamelCase identifier
# (``SomeVeryLongClassName.Config.AnotherLongAttributeName``) fits it — so a candidate only
# counts when its first segment base64-decodes to a Discord snowflake (17-20 ASCII digits),
# which is what the real first segment always is. JWTs are handled by _JWT_RE. Without this,
# ``DISCORD_BOT_TOKEN=<tok>`` was masked by the assignment pass but the same token as a JSON
# value or bare in tool output went through verbatim (#117848) — in a project whose Discord
# adapter keeps one in every install's ``.env``. The first character is the top six bits of
# the id's leading ASCII digit, so ids starting 0-3 encode to ``M``, 4-7 to ``N`` and 8-9 to
# ``O`` at every snowflake length; ``[MNO]`` covers them all, not just the older ``M`` bots.
# Neither form has a left-boundary lookbehind: a token glued to an escape spelling
# (``\\n``/``%0A``/``\\u0020`` in JSON-dumped or URL-encoded tool output, an ANSI colour
# code) has a letter/digit as its left neighbor and would slip a ``(?<![A-Za-z0-9])`` guard.
# Candidates are found overlapping (lookahead); the snowflake decode rejects every id-form
# start that is not a real user id, and _mfa_left_ok does the same job for ``mfa.``.
# The right boundary admits a dot that ends the run (``<tok>.`` / ``<tok>...`` at a sentence
# end) but not one that continues a dotted identifier (``<tok>.foo``).
_TAIL = r"(?![A-Za-z0-9_-]|\.[A-Za-z0-9_-])"
_DISCORD_TOKEN_RE = re.compile(
    rf"(?=(mfa\.[A-Za-z0-9_-]{{20,}}){_TAIL})"
    rf"|(?=([MNO][A-Za-z0-9_-]{{22,27}}\.[A-Za-z0-9_-]{{6,7}}\.[A-Za-z0-9_-]{{27,}}){_TAIL})"
)
# Escape spellings whose last character is a letter or digit glued to what follows.
_ESCAPE_TAIL_RE = re.compile(
    r"(?:\\[nrt]|%[0-9A-Fa-f]{2}|\\u[0-9A-Fa-f]{4}|(?:\x1b|\\x1b|\\033|\\e)\[[0-9;]*m)\Z"
)
_TOKEN_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")


def _mfa_left_ok(text: str, start: int) -> bool:
    """``mfa.`` has no id to decode, so it needs a word boundary or an escape spelling on its left."""
    if start == 0 or text[start - 1] not in _TOKEN_CHARS:
        return True
    return bool(_ESCAPE_TAIL_RE.search(text, max(0, start - 12), start))


def _is_discord_token(candidate: str) -> bool:
    """True when *candidate* carries a real Discord user id, not just the token shape."""
    head = candidate.split(".", 1)[0]
    try:
        decoded = base64.urlsafe_b64decode(head + "=" * (-len(head) % 4))
    except ValueError:  # binascii.Error subclasses ValueError
        return False
    return decoded.isdigit() and 17 <= len(decoded) <= 20


def _redact_discord_tokens(text: str, mask_fn) -> str:
    parts, pos = [], 0
    for m in _DISCORD_TOKEN_RE.finditer(text):
        mfa = m.group(1) is not None
        token = m.group(1) if mfa else m.group(2)
        start = m.start(1) if mfa else m.start(2)
        if start < pos or not (_mfa_left_ok(text, start) if mfa else _is_discord_token(token)):
            continue
        parts += (text[pos:start], mask_fn(token))
        pos = start + len(token)
    return "".join(parts) + text[pos:] if parts else text
