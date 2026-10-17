"""Native Gemini URL composition keeps endpoint query options out of the path."""

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def gemini_method_url(base_url: str, model: str, method: str, default_query: dict, *, stream: bool = False) -> str:
    """Client defaults override endpoint query values; SSE always owns the alt parameter."""
    parsed = urlsplit(base_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(default_query)
    if stream:
        query["alt"] = "sse"
    return urlunsplit(parsed._replace(
        path=f"{parsed.path.rstrip('/')}/models/{model}:{method}",
        query=urlencode(query, doseq=True), fragment="",
    ))
