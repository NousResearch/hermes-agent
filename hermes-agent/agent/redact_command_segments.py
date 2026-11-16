"""Quote-aware pipeline/sequence splitting for agent.redact's command classifiers (moved from agent.redact)."""


def _command_segments(command: str) -> list[str]:
    """Pipeline/sequence segments, split only on unquoted ``| ; &`` so an
    ``awk '{print $1; print $2}'`` program or ``grep 'foo|bar'`` pattern stays one
    segment. Backslash is not an escape (Windows ``C:\\Users\\...``)."""
    segments: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    for ch in command:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            buf.append(ch)
            continue
        if ch in "|;&":
            seg = "".join(buf).strip()
            if seg:
                segments.append(seg)
            buf = []
            continue
        buf.append(ch)
    seg = "".join(buf).strip()
    if seg:
        segments.append(seg)
    return segments
