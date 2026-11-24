#!/usr/bin/env python3
"""Fetch a docs catalog URL the way Hermes Desktop does, and fail when a browser would block it.

The Desktop renderer fetches ``/docs/api/plugins.json`` cross-origin (a packaged build sends
``Origin: null``). A browser checks CORS on every response of a redirect chain, so the docs host's
redirect to GitHub Pages needs ``Access-Control-Allow-Origin`` as much as the final response does
(#136483). ``curl -L`` never checks it, which is why the skills-index watchdog stayed green while
every ``hermes://plugin/install?catalog=`` link failed.

Exit 0: every hop allows the origin and the final body is a non-empty JSON array.
Exit 1: a browser would reject the fetch (missing header, error status, unusable body).
Exit 2: the host could not be reached after retries.
Prints one summary line for the workflow.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from email.message import Message
from urllib.parse import urljoin

DEFAULT_URL = "https://hermes-agent.nousresearch.com/docs/api/plugins.json"
ORIGIN = "null"
MAX_HOPS = 5
TIMEOUT_SECONDS = 20
ATTEMPTS = 3
RETRY_DELAY_SECONDS = 10.0
_REDIRECTS = (301, 302, 303, 307, 308)


class Rejected(Exception):
    """A browser would refuse to hand this response to the renderer."""


class _HopByHop(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # surface each hop so its own headers are checked


def _fetch(opener: urllib.request.OpenerDirector, url: str) -> tuple[int, Message, bytes]:
    request = urllib.request.Request(url, headers={"Origin": ORIGIN, "Accept": "application/json"})
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:  # unfollowed redirects and error statuses land here
        if exc.code >= 500:
            raise OSError(f"{url} answered {exc.code}") from exc
        return exc.code, exc.headers, exc.read()


def check(url: str) -> str:
    opener = urllib.request.build_opener(_HopByHop)
    for hops in range(MAX_HOPS + 1):
        status, headers, body = _fetch(opener, url)
        allowed = headers.get("Access-Control-Allow-Origin")
        if allowed not in ("*", ORIGIN):
            raise Rejected(f"{status} from {url} sends Access-Control-Allow-Origin {allowed!r}, "
                           "so a browser blocks the fetch here")
        if status in _REDIRECTS and headers.get("Location"):
            url = urljoin(url, headers["Location"])
            continue
        if status != 200:
            raise Rejected(f"{url} answered {status}")
        try:
            rows = json.loads(body)
        except ValueError as exc:
            raise Rejected(f"{url} is not JSON: {exc}") from exc
        if not isinstance(rows, list) or not rows:
            raise Rejected(f"{url} is not a non-empty JSON array")
        return f"{len(rows)} rows after {hops} redirect(s)"
    raise Rejected(f"more than {MAX_HOPS} redirects")


def main(argv: list[str]) -> int:
    url = argv[1] if len(argv) > 1 else DEFAULT_URL
    for attempt in range(1, ATTEMPTS + 1):
        try:
            print(f"ok: {check(url)}")
            return 0
        except Rejected as exc:
            print(f"blocked: {exc}")
            return 1
        except OSError as exc:  # URLError, timeouts, resets and 5xx are retried
            if attempt == ATTEMPTS:
                print(f"unreachable: {url}: {exc}")
                return 2
            time.sleep(RETRY_DELAY_SECONDS)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
