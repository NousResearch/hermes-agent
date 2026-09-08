"""Normalization for Telegram's permissive Rich Markdown block parser."""

import re


# Consume code/math before looking for block prefixes so their literal contents
# survive unchanged, including an unfinished code fence in a streaming draft.
_LITERAL_HASH_RE = re.compile(
    r"(?P<fence>`{3,}|~{3,})[^\n]*\n.*?(?:(?P=fence)(?![`~])|\Z)"
    r"|(?P<ticks>`+)(?!`).*?(?<!`)(?P=ticks)(?!`)"
    r"|\$\$.*?(?:\$\$|\Z)"
    r"|(?P<prefix>^[ \t]*(?:(?:>[ \t]*|[-+*][ \t]+|\d+[.)][ \t]+))*)"
    r"(?P<hashes>#+)(?=[^\s#])",
    re.MULTILINE | re.DOTALL,
)


def escape_literal_hash_prefixes(text: str) -> str:
    """Keep ``#89``/``#tag`` as prose, including in lists and blockquotes.

    Telegram accepts these as headings even without the whitespace required by
    standard Markdown. Escape only such block-start hashes, not real headings,
    inline references, links, code, or already escaped prefixes.
    """
    def replace(match: re.Match[str]) -> str:
        hashes = match.group("hashes")
        if hashes is None:
            return match.group(0)
        return match.group("prefix") + r"\#" * len(hashes)

    return _LITERAL_HASH_RE.sub(replace, text)
