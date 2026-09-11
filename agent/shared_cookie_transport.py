"""Shared-cookie-jar transport for cookie-based LB sticky routing.

Some OpenAI-compatible deployments sit behind nginx / Cloudflare with
cookie-based sticky sessions (``Set-Cookie: route=...``).  Hermes builds a
fresh ``httpx.Client`` per request (see ``AIAgent._create_request_openai_client``
and the #10933 closed-transport invariants), so a default httpx cookie jar —
which lives on the *client* — is discarded after every turn and the LB treats
each request as a brand-new session, breaking prompt-cache locality.

``SharedCookieTransport`` moves cookie persistence to the transport layer: it
wraps the normal ``HTTPTransport`` machinery but reads the ``Cookie`` request
header from, and writes ``Set-Cookie`` responses into, a single
``http.cookiejar.CookieJar`` shared across every per-request client rebuild
for the life of the agent.

Opt-in per provider via ``providers.<name>.cookie_jar: true`` in config.yaml
(resolved in ``hermes_cli.config_providers.get_custom_provider_cookie_jar``
and threaded through ``create_openai_client``).
"""

from __future__ import annotations

import httpx
import threading
from http.cookiejar import CookieJar
from typing import Any, Optional


class SharedCookieTransport(httpx.BaseTransport):
    """Transport that persists cookies in a shared jar.

    Subclasses ``httpx.BaseTransport`` so it can be mounted directly.  Uses
    ``httpx.Cookies`` for the RFC 6265 header math (``set_cookie_header`` /
    ``extract_cookies``) over the caller-supplied jar, guarded by a lock
    because per-request clients are rebuilt concurrently from multiple
    threads.
    """

    def __init__(
        self,
        jar: CookieJar,
        *,
        verify: Any = True,
        limits: Any = None,
    ) -> None:
        self._jar = jar
        self._cookies = httpx.Cookies()
        # httpx.Cookies wraps its own jar; swap in the shared one so every
        # transport instance (and therefore every rebuilt client) reads and
        # writes the SAME underlying cookies.
        self._cookies.jar = jar
        self._lock = threading.Lock()
        self._transport = httpx.HTTPTransport(
            verify=verify,
            limits=limits or httpx.Limits(max_connections=100),
        )

    def handle_request(self, request) -> Any:
        with self._lock:
            self._cookies.set_cookie_header(request)
            response = self._transport.handle_request(request)
            # httpx's extract_cookies needs ``response.request`` set (the raw
            # transport does not do this — it happens later in Client.send).
            response.request = request
            self._cookies.extract_cookies(response)
        return response


class _SharedCookieTransportCompat(SharedCookieTransport):
    """Back-compat alias shim: keep ``__getattr__`` forwarding so any SDK or
    plugin code reaching through the mounted transport's attributes (close,
    stream handling) resolves against the inner HTTPTransport."""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._transport, name)


# The public name used by build_shared_cookie_http_client / tests.
SharedCookieTransport = _SharedCookieTransportCompat  # noqa: F811


def build_shared_cookie_http_client(
    *,
    jar: CookieJar,
    proxy: Optional[str] = None,
    verify: Any = True,
    limits: Any = None,
    timeout: Any = None,
) -> Any:
    """Build a real ``httpx.Client`` whose transport persists cookies in *jar*.

    The OpenAI SDK requires ``http_client`` to be an ``httpx.Client`` (it
    checks attributes like ``.build_request``), so the shared-jar transport
    is mounted here rather than handed to the SDK directly.  ``trust_env``
    is disabled and the resolved proxy is passed explicitly — the same
    env-only proxy policy as ``build_keepalive_http_client`` — because a
    mounted transport owns the request pipeline and would otherwise ignore
    client-level proxy settings.

    Each client gets its OWN transport instance (fresh sockets, closed with
    the client per the #10933 invariants) but the SAME cookie jar, which is
    the whole point: connection pools are per-request, cookies are not.
    """
    limits = limits or httpx.Limits(max_connections=100)
    mounts = {
        "http://": _SharedCookieTransportCompat(
            jar=jar, verify=verify, limits=limits,
        ),
        "https://": _SharedCookieTransportCompat(
            jar=jar, verify=verify, limits=limits,
        ),
    }
    return httpx.Client(
        limits=limits,
        timeout=timeout,
        proxy=proxy,
        mounts=mounts,
        verify=verify,
        trust_env=False,
    )
