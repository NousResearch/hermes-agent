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

Concurrency: one lock must guard a jar across ALL clients that share it —
per-transport locks are wrong because every rebuild mints fresh transports.
The jar and its lock travel together in a :class:`SharedCookieJar`; callers
that bypass the bundle (hand-rolled ``CookieJar``) get a per-call lock, which
is only safe for single-client use.

Opt-in per provider via ``providers.<name>.cookie_jar: true`` in config.yaml
(resolved in ``hermes_cli.config_providers.get_custom_provider_cookie_jar``
and threaded through ``create_openai_client``).
"""

from __future__ import annotations

import asyncio
import threading
from http.cookiejar import CookieJar
from typing import Any, Optional

import httpx


class SharedCookieJar:
    """A ``CookieJar`` bundled with the lock that guards it process-wide.

    The bundle is the concurrency unit: every client sharing a jar MUST share
    its lock.  Created alongside the jar in the config registry so all
    per-request client rebuilds lock against the same primitive.
    """

    __slots__ = ("jar", "lock")

    def __init__(self) -> None:
        self.jar = CookieJar()
        self.lock = threading.Lock()

    @classmethod
    def _from(cls, jar: CookieJar, lock: Optional[threading.Lock] = None) -> "SharedCookieJar":
        """Wrap a caller-supplied jar (e.g. hand-rolled ``CookieJar`` in tests).

        ``lock=None`` mints a fresh lock — correct only when a single client
        uses the jar; production callers pass the registry's bundle.
        """
        bundle = cls.__new__(cls)
        bundle.jar = jar
        bundle.lock = lock if lock is not None else threading.Lock()
        return bundle


class _SyncCookieTransport(httpx.BaseTransport):
    """Sync transport that persists cookies in a shared jar (with its lock)."""

    def __init__(
        self,
        bundle: SharedCookieJar,
        *,
        verify: Any = True,
        limits: Any = None,
    ) -> None:
        self._bundle = bundle
        self._cookies = httpx.Cookies()
        # httpx.Cookies wraps its own jar; swap in the shared one so every
        # transport instance (and therefore every rebuilt client) reads and
        # writes the SAME underlying cookies.
        self._cookies.jar = bundle.jar
        self._transport = httpx.HTTPTransport(
            verify=verify,
            limits=limits or httpx.Limits(max_connections=100),
        )

    def handle_request(self, request) -> Any:
        with self._bundle.lock:
            self._cookies.set_cookie_header(request)
            response = self._transport.handle_request(request)
            # httpx's extract_cookies needs ``response.request`` set (the raw
            # transport does not do this — it happens later in Client.send).
            response.request = request
            self._cookies.extract_cookies(response)
        return response

    def __getattr__(self, name: str) -> Any:
        # close(), stream handling, and any other httpx.Client-level API the
        # SDK expects from its http_client's transport chain.
        return getattr(self._transport, name)


class _AsyncCookieTransport(httpx.AsyncBaseTransport):
    """Async twin of :class:`_SyncCookieTransport`.

    Uses an ``asyncio.Lock`` bound to the running loop (async clients are
    loop-bound per #2681, so per-client-loop locking is correct here); the
    jar itself remains guarded against sync readers by ``bundle.lock``.
    """

    def __init__(
        self,
        bundle: "SharedCookieJar",
        *,
        verify: Any = True,
        limits: Any = None,
    ) -> None:
        self._bundle = bundle
        self._cookies = httpx.Cookies()
        self._cookies.jar = bundle.jar
        self._alock = asyncio.Lock()
        self._transport = httpx.AsyncHTTPTransport(
            verify=verify,
            limits=limits or httpx.Limits(max_connections=100),
        )

    async def handle_async_request(self, request) -> Any:
        async with self._alock:
            self._cookies.set_cookie_header(request)
            response = await self._transport.handle_async_request(request)
            response.request = request
            self._cookies.extract_cookies(response)
        return response


# Back-compat alias: the pre-review name pointed at the sync transport.
SharedCookieTransport = _SyncCookieTransport


def build_shared_cookie_http_client(
    *,
    jar: Any,
    lock: Optional[threading.Lock] = None,
    async_mode: bool = False,
    proxy: Optional[str] = None,
    verify: Any = True,
    limits: Any = None,
    timeout: Any = None,
) -> Any:
    """Build an ``httpx.Client``/``AsyncClient`` persisting cookies in *jar*.

    The OpenAI SDK requires ``http_client`` to be an ``httpx`` client (it
    checks attributes like ``.build_request``), so the shared-jar transport
    is mounted here rather than handed to the SDK directly.  ``trust_env``
    is disabled and the resolved proxy is passed explicitly — the same
    env-only proxy policy as ``build_keepalive_http_client`` — because a
    mounted transport owns the request pipeline and would otherwise ignore
    client-level proxy settings.

    ``jar`` may be a plain ``CookieJar`` (lock defaults to a per-call lock —
    only safe for single-client use) or a :class:`SharedCookieJar` whose
    lock is shared across every client rebuild.  ``async_mode=True`` returns
    an ``httpx.AsyncClient`` with the async transport twin.

    Each client gets its OWN transport instance (fresh sockets, closed with
    the client per the #10933 invariants) but the SAME cookie jar, which is
    the whole point: connection pools are per-request, cookies are not.
    """
    bundle = jar if isinstance(jar, SharedCookieJar) else SharedCookieJar._from(jar, lock)
    limits = limits or httpx.Limits(max_connections=100)
    if async_mode:
        mounts = {
            f"{scheme}://": _AsyncCookieTransport(
                bundle, verify=verify, limits=limits,
            )
            for scheme in ("http", "https")
        }
        return httpx.AsyncClient(
            limits=limits,
            timeout=timeout,
            proxy=proxy,
            mounts=mounts,
            verify=verify,
            trust_env=False,
        )
    mounts = {
        f"{scheme}://": _SyncCookieTransport(
            bundle, verify=verify, limits=limits,
        )
        for scheme in ("http", "https")
    }
    return httpx.Client(
        limits=limits,
        timeout=timeout,
        proxy=proxy,
        mounts=mounts,
        verify=verify,
        trust_env=False,
    )
