"""Native Gemini URL composition keeps endpoint query options out of the path."""

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def _query_pairs(parsed, default_query: dict) -> list[tuple[str, str]]:
    """Endpoint query options in their original order, minus what the client defaults override.

    The endpoint's own repeats and blank values are data (a relay can key on ``?tag=a&tag=b`` or
    ``?route=``), so they survive: only the overridden keys are replaced.
    """
    overrides = dict(default_query or {})
    pairs = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
             if key not in overrides]
    for key, value in overrides.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        pairs.extend((str(key), str(item)) for item in values)
    return pairs


def gemini_method_url(base_url: str, model: str, method: str, default_query: dict, *, stream: bool = False) -> str:
    """Client defaults override endpoint query values; repeats and blanks survive, SSE owns ``alt``."""
    parsed = urlsplit(base_url)
    query = _query_pairs(parsed, default_query)
    if stream:
        query = [pair for pair in query if pair[0] != "alt"] + [("alt", "sse")]
    return urlunsplit(parsed._replace(
        path=f"{parsed.path.rstrip('/')}/models/{model}:{method}",
        query=urlencode(query), fragment="",
    ))
