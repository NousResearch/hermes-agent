#!/usr/bin/env python3
"""httpx.Client subclass that routes send() through curl_cffi for TLS impersonation.

The OpenAI SDK calls build_request() + send(request, stream=...), and checks
isinstance(client, httpx.Client). By subclassing httpx.Client and overriding
only send(), we pass all checks while curl_cffi handles the actual TLS.
"""

from __future__ import annotations

import httpx
from curl_cffi import requests as curl_requests

# httpx auto-headers that identify us as a Python bot — we strip these and
# let curl_cffi (with browser impersonation) set its own headers instead.
_CFFI_STRIP_REQUEST_HEADERS = {
    "user-agent",       # httpx sets "python-httpx/X.Y.Z" — instant Cloudflare 403
    "accept-encoding",  # curl_cffi handles its own encoding
    "connection",       # curl manages keep-alive natively
    "accept",           # let curl set its browser-like Accept header
}


class CurlCffiClient(httpx.Client):
    """httpx.Client that routes actual HTTP requests through curl_cffi.

    build_request() still creates normal httpx.Request objects (using httpx's
    URL normalization and header merging). send() converts the request to a
    curl_cffi call and returns an httpx.Response.

    This means:
    - isinstance(curl_client, httpx.Client) → True (OpenAI SDK gate passes)
    - The TLS handshake uses Chrome 124's fingerprint → Cloudflare allows it
    - httpx auto-headers (python-httpx UA, accept-encoding, etc.) are stripped
      so curl_cffi can send its own browser-like headers
    """

    def __init__(
        self,
        impersonate: str = "chrome124",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._curl_session = curl_requests.Session(impersonate=impersonate)

    def send(self, request: httpx.Request, *, stream: bool = False, **kwargs) -> httpx.Response:
        """Execute an httpx.Request through curl_cffi.

        Strips httpx's auto-headers (User-Agent, Accept-Encoding, etc.) so
        curl_cffi sends its own browser-impersonating headers instead.
        """
        # Read body
        content: bytes | None = None
        if request.content:
            if isinstance(request.content, bytes):
                content = request.content
            else:
                content = request.content.encode("utf-8")

        # Filter headers: pass through everything EXCEPT httpx auto-headers
        filtered_headers: list[tuple[str, str]] = []
        for k, v in request.headers.items():
            if k.lower() not in _CFFI_STRIP_REQUEST_HEADERS:
                filtered_headers.append((k, v))

        # Make the curl_cffi request with browser impersonation
        curl_resp = self._curl_session.request(
            method=request.method,
            url=str(request.url),
            headers=filtered_headers,
            data=content,
            stream=stream,
        )

        # Convert response headers — strip Content-Encoding since
        # curl_cffi already decompressed the response body
        resp_headers: list[tuple[str, str]] = []
        for k, v in curl_resp.headers.multi_items():
            if k.lower() == "content-encoding":
                continue
            resp_headers.append((k, v))

        resp_body = curl_resp.content if not stream else b""

        return httpx.Response(
            status_code=curl_resp.status_code,
            headers=resp_headers,
            content=resp_body,
            request=request,
        )

    def close(self) -> None:
        self._curl_session.close()
        super().close()


def build_curl_cffi_http_client(
    impersonate: str = "chrome124",
    proxy: str | None = None,
) -> CurlCffiClient:
    """Build a CurlCffiClient with browser TLS impersonation.

    Can be passed as http_client= to the OpenAI SDK directly.
    """
    client = CurlCffiClient(
        impersonate=impersonate,
        timeout=httpx.Timeout(connect=15.0, read=600.0, write=30.0, pool=30.0),
    )

    if proxy:
        client._curl_session.proxies = {"http": proxy, "https": proxy}

    return client
