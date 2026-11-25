"""Codex request identity helpers shared by agent client builders.

Leaf module with no dependency on the large auxiliary-client router, so a
long-lived process can import a newly added client builder without resolving a
new symbol from an older cached ``auxiliary_client`` module.
"""

from __future__ import annotations

import base64
import functools
import json
import logging
import os
from typing import Any, Dict
from urllib.parse import urlparse


CODEX_AUX_BASE_URL = "https://chatgpt.com/backend-api/codex"

# Response fields Hermes never reads, so the official backend may leave them out of the SSE
# stream: the echoed request tool catalog (repeated on every lifecycle frame), per-item usage
# attribution, delta sequence numbers and the reasoning ciphertext on ``output_item.added`` (replay
# takes it from ``output_item.done``). Same defaults as openai/codex#62863/#63219/#63461/#63479. A
# user ``model.default_headers`` value wins; an empty one opts out.
CODEX_STREAM_EXCLUSIONS = {
    "x-codex-exclude": "tools,usage.attribution",
    "x-codex-exclude-stream-fields": "delta.sequence_number,reasoning.added.encrypted_content",
}
_EXCLUSION_NAMES = frozenset(CODEX_STREAM_EXCLUSIONS)

logger = logging.getLogger(__name__)


def is_official_codex_base_url(base_url: str) -> bool:
    """Identify OpenAI's Codex endpoint without matching custom proxies."""
    try:
        parsed = urlparse(base_url)
        path = parsed.path.rstrip("/")
        return (
            parsed.scheme == "https"
            and parsed.hostname == "chatgpt.com"
            and parsed.port in (None, 443)
            and (path == "/backend-api/codex" or path.startswith("/backend-api/codex/"))
        )
    except (TypeError, ValueError):
        return False


def codex_cloudflare_headers(access_token: str, *, base_url: str = CODEX_AUX_BASE_URL,
                             stream_exclusions: bool = True) -> dict[str, str]:
    """Identity and account headers for chatgpt.com/backend-api/codex.

    OpenAI requires third-party harnesses to identify themselves: the official
    endpoint gets Hermes' originator and version, custom endpoints keep the
    codex_cli_rs compatibility identity. The account headers come from the
    OAuth JWT (see :func:`codex_account_headers`). ``stream_exclusions`` adds
    :data:`CODEX_STREAM_EXCLUSIONS` (Responses clients); non-Responses callers pass False.
    """
    if is_official_codex_base_url(base_url):
        from hermes_cli.version_info import get_version_info
        headers = {
            "User-Agent": f"HermesAgent/{get_version_info().base_version}",
            "originator": "hermes-agent",
            **(CODEX_STREAM_EXCLUSIONS if stream_exclusions else {}),
        }
    else:
        headers = {"User-Agent": "codex_cli_rs/0.0.0 (Hermes Agent)", "originator": "codex_cli_rs"}
    headers.update(codex_account_headers(access_token))
    return headers


def codex_account_headers(access_token: str) -> dict[str, str]:
    """Workspace headers the Codex backend derives from the OAuth JWT.

    ``ChatGPT-Account-ID`` (canonical casing, from codex-rs ``auth.rs``) comes from
    ``chatgpt_account_id``; ``x-openai-internal-codex-residency`` from
    ``chatgpt_data_residency`` (fallback ``chatgpt_compute_residency``) — without
    it residency-enforced workspaces answer 401 "Workspace is not authorized in
    this region". A malformed token drops the headers rather than raising, so it
    surfaces as a 401 instead of a crash at client construction.
    """
    headers: dict[str, str] = {}
    if not isinstance(access_token, str) or not access_token.strip():
        return headers
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return headers
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        auth = json.loads(base64.urlsafe_b64decode(payload_b64)).get("https://api.openai.com/auth", {})
        acct_id = auth.get("chatgpt_account_id")
        if isinstance(acct_id, str) and acct_id:
            headers["ChatGPT-Account-ID"] = acct_id
        residency = auth.get("chatgpt_data_residency") or auth.get("chatgpt_compute_residency")
        if isinstance(residency, str) and residency.strip():
            headers["x-openai-internal-codex-residency"] = residency.strip()
    except Exception:
        pass
    return headers


def _codex_compression_enabled() -> bool:
    """``agent.codex_request_compression`` (default on); built-in default under ``--ignore-user-config``."""
    if os.environ.get("HERMES_IGNORE_USER_CONFIG") == "1":
        return True
    from hermes_cli.config_backend import ConfigBackendUnavailable
    try:
        from hermes_cli.config import cfg_get, load_config_readonly
        return cfg_get(load_config_readonly(), "agent", "codex_request_compression", default=True) is not False
    # The caller drops its whole HTTP client on an exception, and a refused backend read is a
    # SystemExit that would end the gateway turn worker; plain JSON is the safe answer for both.
    except (Exception, ConfigBackendUnavailable):  # health: allow BLE001 -- unreadable config keeps plain JSON
        logger.warning("Could not read agent.codex_request_compression; sending plain JSON", exc_info=True)
        return False


_RAW_BODY_EXT = "hermes_codex_raw_body"
_PLAIN_BODY_EXT = "hermes_codex_plain_body"


def codex_request_compression_hooks(base_url: str, *, async_mode: bool = False) -> dict[str, list] | None:
    """httpx ``event_hooks`` that zstd-compress Responses request bodies for the official Codex backend.

    Parity with the official Codex client (zstd on ``/responses`` with ChatGPT auth, default since
    openai/codex#10102). The backend rejects some valid frames with a bare 400 (#148898), so the
    raw body rides along for :func:`codex_zstd_replay_client_cls` to resend once as plain JSON.
    Uses the stdlib codec (Python 3.14+); custom endpoint URLs always get plain JSON.
    """
    if not is_official_codex_base_url(base_url) or not _codex_compression_enabled():
        return None
    try:
        from compression import zstd
        import httpx
    except ImportError:
        return None

    def compress(request: httpx.Request) -> None:
        if (request.method != "POST" or not request.url.path.endswith("/responses")
                or "content-encoding" in request.headers or request.extensions.get(_PLAIN_BODY_EXT)):
            return
        raw = request.read()
        body = zstd.compress(raw)
        request.stream = httpx.ByteStream(body)
        request._content = body
        request.extensions[_RAW_BODY_EXT] = raw
        request.headers["Content-Encoding"] = "zstd"
        request.headers["Content-Length"] = str(len(body))

    if not async_mode:
        return {"request": [compress]}

    async def compress_async(request: httpx.Request) -> None:
        compress(request)

    return {"request": [compress_async]}


def _is_bare_bad_request(response: Any) -> bool:
    try:
        return response.status_code == 400 and json.loads(response.content) == {"detail": "Bad Request"}
    except ValueError:
        return False


def _plain_json_request(request: Any, raw: bytes) -> Any:
    import httpx
    headers = httpx.Headers(request.headers)
    del headers["content-encoding"]
    del headers["content-length"]
    return httpx.Request(request.method, request.url, headers=headers, content=raw,
                         extensions={**request.extensions, _PLAIN_BODY_EXT: True})


def _log_plain_json_replay(raw: bytes) -> None:
    logger.warning("Codex answered a zstd request body with a bare 400; resending it once as plain JSON "
                   "(%d bytes, #148898)", len(raw))


@functools.cache
def codex_zstd_replay_client_cls(async_mode: bool = False) -> type:
    """httpx client class that resends a zstd body once as plain JSON on the bare 400 (#148898).

    Only requests the compression hook encoded carry the raw body, so nothing else is resent. The
    raw body is dropped from the request as soon as the response arrives, so a long SSE stream does
    not keep a second copy alive. A rejection of the plain body is returned like any other 400.
    The budget is one resend per ``send``: an SDK-level retry (Hermes runs the agent client with
    ``max_retries=0``) builds a new request and may resend once more.
    """
    import httpx

    if async_mode:
        class _CodexZstdReplayAsyncClient(httpx.AsyncClient):
            async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
                response = await super().send(request, **kwargs)
                raw = request.extensions.pop(_RAW_BODY_EXT, None)
                if raw is None or response.status_code != 400:
                    return response
                try:
                    await response.aread()
                except BaseException:
                    await response.aclose()
                    raise
                if not _is_bare_bad_request(response):
                    return response
                await response.aclose()
                _log_plain_json_replay(raw)
                return await super().send(_plain_json_request(request, raw), **kwargs)

        return _CodexZstdReplayAsyncClient

    class _CodexZstdReplayClient(httpx.Client):
        def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
            response = super().send(request, **kwargs)
            raw = request.extensions.pop(_RAW_BODY_EXT, None)
            if raw is None or response.status_code != 400:
                return response
            try:
                response.read()
            except BaseException:
                response.close()
                raise
            if not _is_bare_bad_request(response):
                return response
            response.close()
            _log_plain_json_replay(raw)
            return super().send(_plain_json_request(request, raw), **kwargs)

    return _CodexZstdReplayClient


def apply_required_codex_headers(client_kwargs: dict[str, Any], *, access_token: str, base_url: str) -> None:
    """Keep required Codex identity after user/provider header overrides."""
    if not is_official_codex_base_url(base_url):
        return
    existing = client_kwargs.get("default_headers") or {}
    overridden = {str(name).lower() for name in existing} & _EXCLUSION_NAMES
    required = {name: value for name, value in codex_cloudflare_headers(access_token, base_url=base_url).items()
                if name not in overridden}
    required_names = {name.lower() for name in required}
    client_kwargs["default_headers"] = {
        **{name: value for name, value in existing.items() if str(name).lower() not in required_names},
        **required,
    }
