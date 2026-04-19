"""httpx transport that E2EE-encrypts chat requests in-process.

Attach via `openai.OpenAI(http_client=httpx.Client(transport=E2EETransport(...)))`.
Crypto primitives live in `hermes_cli.e2ee_proxy`.
"""
import json
import logging
from typing import Iterable, Iterator, Optional

import httpx

from hermes_cli.e2ee_proxy import _generate_keypair, _encrypt_content, _decrypt_content

logger = logging.getLogger(__name__)

_DECRYPT_FIELDS = ("content", "reasoning_content", "reasoning")


class _ChainedByteStream(httpx.SyncByteStream):
    """Wrap a decrypting generator; close the upstream response when done."""
    def __init__(self, it: Iterable[bytes], upstream: httpx.Response):
        self._it = it
        self._upstream = upstream

    def __iter__(self) -> Iterator[bytes]:
        try:
            yield from self._it
        finally:
            self._upstream.close()

    def close(self) -> None:
        self._upstream.close()


class E2EETransport(httpx.BaseTransport):
    def __init__(
        self,
        signing_public_key: str,
        signing_algo: str,
        inner: Optional[httpx.BaseTransport] = None,
    ):
        self._pub = signing_public_key
        self._algo = signing_algo
        self._inner = inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/chat/completions"):
            return self._handle_chat(request)
        return self._inner.handle_request(request)

    def _handle_chat(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content or b"{}")
        priv_key, client_pub_hex = _generate_keypair(self._algo)

        for msg in payload.get("messages", []):
            if isinstance(msg.get("content"), str) and msg["content"]:
                msg["content"] = _encrypt_content(msg["content"], self._pub, self._algo)

        new_body = json.dumps(payload).encode()
        headers = httpx.Headers(request.headers)
        headers["X-Signing-Algo"] = self._algo
        headers["X-Client-Pub-Key"] = client_pub_hex
        headers["X-Model-Pub-Key"] = self._pub
        headers["Content-Length"] = str(len(new_body))

        upstream_req = httpx.Request(
            method=request.method,
            url=request.url,
            headers=headers,
            content=new_body,
            extensions=request.extensions,
        )
        upstream = self._inner.handle_request(upstream_req)

        is_stream = bool(payload.get("stream"))
        if is_stream:
            decrypt_stream = _ChainedByteStream(
                self._decrypt_sse(upstream.iter_bytes(), priv_key), upstream,
            )
            return httpx.Response(
                status_code=upstream.status_code,
                headers=upstream.headers,
                stream=decrypt_stream,
                extensions=upstream.extensions,
                request=request,
            )

        body = upstream.read()
        upstream.close()
        try:
            data = json.loads(body)
        except Exception:
            return httpx.Response(
                status_code=upstream.status_code,
                headers=upstream.headers,
                content=body,
                extensions=upstream.extensions,
                request=request,
            )
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            for field in _DECRYPT_FIELDS:
                if msg.get(field):
                    msg[field] = _decrypt_content(msg[field], priv_key, self._algo)
        out = json.dumps(data).encode()
        new_headers = httpx.Headers(upstream.headers)
        new_headers["Content-Length"] = str(len(out))
        return httpx.Response(
            status_code=upstream.status_code,
            headers=new_headers,
            content=out,
            extensions=upstream.extensions,
            request=request,
        )

    def _decrypt_sse(self, chunks: Iterable[bytes], priv_key) -> Iterator[bytes]:
        pending = b""
        for chunk in chunks:
            pending += chunk
            while b"\n" in pending:
                raw, pending = pending.split(b"\n", 1)
                line = raw.decode("utf-8", errors="replace")
                if line.startswith("data: {"):
                    try:
                        data = json.loads(line[6:])
                    except Exception:
                        yield (line + "\n").encode()
                        continue
                    for choice in data.get("choices", []):
                        delta = choice.get("delta", {})
                        for field in _DECRYPT_FIELDS:
                            if delta.get(field):
                                delta[field] = _decrypt_content(delta[field], priv_key, self._algo)
                    line = "data: " + json.dumps(data)
                yield (line + "\n").encode()
        if pending:
            yield pending
