from __future__ import annotations

from contextlib import ExitStack, contextmanager
from urllib.parse import urlsplit

_MAX_REDIRECTS = 20
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_USER_AGENT = "hermes-agent-petdex"


class PetdexNetworkError(RuntimeError):
    pass


def _is_petdex_url(url: str) -> bool:
    import httpx

    try:
        parsed = httpx.URL(url)
        raw = urlsplit(url)
        port = raw.port
    except (httpx.InvalidURL, TypeError, ValueError):
        return False
    host = parsed.host.lower()
    return (
        parsed.scheme == "https"
        and (host == "petdex.dev" or host.endswith(".petdex.dev"))
        and raw.username is None
        and raw.password is None
        and port in (None, 443)
    )


@contextmanager
def stream_petdex(url: str, *, timeout: float):
    import httpx

    if not _is_petdex_url(url):
        raise PetdexNetworkError("refusing non-petdex URL")

    current_url = url
    redirects = 0
    with httpx.Client(
        follow_redirects=False,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        while True:
            with ExitStack() as response_stack:
                try:
                    response = response_stack.enter_context(
                        client.stream("GET", current_url, timeout=timeout)
                    )
                except httpx.RemoteProtocolError as exc:
                    raise PetdexNetworkError(
                        "refusing invalid petdex redirect"
                    ) from exc
                if response.status_code not in _REDIRECT_STATUSES:
                    yield response
                    return
                location = response.headers.get("Location", "").strip()
                if not location:
                    raise PetdexNetworkError("petdex redirect missing Location")
                if redirects >= _MAX_REDIRECTS:
                    raise PetdexNetworkError("too many petdex redirects")
                try:
                    raw_location = urlsplit(location)
                    if (
                        raw_location.username is not None
                        or raw_location.password is not None
                    ):
                        raise PetdexNetworkError("refusing non-petdex redirect")
                    next_url = str(response.url.join(location))
                except (httpx.InvalidURL, TypeError, ValueError) as exc:
                    raise PetdexNetworkError(
                        "refusing invalid petdex redirect"
                    ) from exc
                if not _is_petdex_url(next_url):
                    raise PetdexNetworkError("refusing non-petdex redirect")
                current_url = next_url
                redirects += 1
