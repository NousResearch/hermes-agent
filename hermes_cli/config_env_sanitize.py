"""Semantics-preserving physical-line normalization for dotenv write paths."""

def sanitize_env_lines(lines: list) -> list:
    """Normalize safe .env formatting without changing value semantics.

    A quoted dotenv value may span physical lines. Every byte inside that
    binding — including whitespace and ``#`` markers — is value content, so a
    per-line ``strip()`` would corrupt it (#76571). Use python-dotenv's pinned
    parser only to discover binding boundaries: preserve multiline bindings,
    keep the existing normalization for ordinary physical lines, and fail
    closed by leaving the whole file untouched if any binding is malformed.

    The caller's ``lines`` are the physical lines: the result is aligned
    one-to-one with them and never splits a line again, so U+2028 and the
    other extra ``str.splitlines()`` boundaries stay value content. Blank
    lines python-dotenv folds into the front of a binding are not part of
    its value and do not make it multiline.

    Content after the first ``=`` remains opaque value data. A known variable
    name embedded in that value must never be reinterpreted as another
    assignment; concatenated assignments remain on one line.
    """
    from bisect import bisect_right
    from io import StringIO

    from dotenv.parser import parse_stream

    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    bindings = list(parse_stream(StringIO("".join(lines))))
    if any(binding.error for binding in bindings):
        return list(lines)

    preserved: set[int] = set()
    offset = 0
    for binding in bindings:
        original = binding.original.string
        start, offset = offset, offset + len(original)
        if binding.key is None:
            continue
        key_at = start + len(original) - len(original.lstrip())
        first = bisect_right(line_starts, key_at) - 1
        last = bisect_right(line_starts, offset - 1) - 1
        if last > first:
            preserved.update(range(first, last + 1))
    if offset != sum(len(line) for line in lines):
        return list(lines)  # bindings must tile the input exactly; otherwise fail closed

    sanitized: list[str] = []
    for index, line in enumerate(lines):
        raw = line.rstrip("\r\n")
        stripped = raw.strip()

        # A multiline binding is an indivisible value-bearing unit. Only
        # normalize its physical line endings; preserve all other bytes.
        # Blank lines and comments are preserved as well.
        if index in preserved or not stripped or stripped.startswith("#"):
            sanitized.append(raw + "\n")
            continue

        sanitized.append(stripped + "\n")

    return sanitized
