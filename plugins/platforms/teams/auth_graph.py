"""MSAL-backed Microsoft Graph token provider.

Provides :class:`GraphTokenProvider`, a thin async-friendly wrapper around
``msal.ConfidentialClientApplication`` that:

* Caches access tokens **per scope** (keyed on the scope string), so the
  same provider can mint tokens for both Bot Framework (``...default``)
  and Graph (``https://graph.microsoft.com/.default``) without stomping on
  itself.
* Uses a **per-scope** ``asyncio.Lock`` so concurrent ``get_token`` callers
  collapse onto a single MSAL round-trip rather than racing.
* Refreshes proactively when the cached token is within
  ``refresh_if_within_seconds`` of expiry (default 60s).
* Defers the ``import msal`` until ``_build_msal_app()`` actually runs so
  the module is cheap to import even when msal isn't installed (the
  plugin's lazy-deps mechanism will install it on first use).

MSAL itself is synchronous; calls are dispatched to the default executor
via ``loop.run_in_executor``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import msal  # noqa: F401


__all__ = ["AuthError", "GraphTokenProvider"]

log = logging.getLogger(__name__)


class AuthError(Exception):
    """Raised when MSAL returns an error response for token acquisition."""


class GraphTokenProvider:
    """Async-safe MSAL client-credential token provider with per-scope cache.

    Parameters
    ----------
    client_id:
        Azure AD application (client) ID.
    tenant_id:
        Azure AD tenant ID. Used to build the authority URL
        ``https://login.microsoftonline.com/{tenant_id}``.
    client_secret:
        Application client secret (the "Value" from the Azure portal's
        Certificates & secrets blade — *not* the secret ID).
    """

    def __init__(self, client_id: str, tenant_id: str, client_secret: str) -> None:
        self.client_id = client_id
        self.tenant_id = tenant_id
        self.client_secret = client_secret
        # Lazily built — see _get_app(). Tests monkeypatch _build_msal_app.
        self._msal_app: object | None = None
        # scope -> (access_token, expires_at_epoch_seconds)
        self._cache: dict[str, tuple[str, float]] = {}
        # scope -> asyncio.Lock guarding refresh of that scope's entry.
        self._locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # MSAL plumbing
    # ------------------------------------------------------------------

    def _build_msal_app(self):  # type: ignore[no-untyped-def]
        """Construct the underlying ``ConfidentialClientApplication``.

        ``msal`` is imported here (not at module top) so that the plugin
        can be loaded even when msal is not yet installed — the lazy-deps
        loader will install it before this method is invoked.
        """
        import msal  # local import: keeps module import cheap & lazy

        authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        return msal.ConfidentialClientApplication(
            self.client_id,
            authority=authority,
            client_credential=self.client_secret,
        )

    def _get_app(self):  # type: ignore[no-untyped-def]
        if self._msal_app is None:
            self._msal_app = self._build_msal_app()
        return self._msal_app

    def _get_lock(self, scope: str) -> asyncio.Lock:
        # setdefault is the conventional safe pattern: it atomically returns
        # the existing lock or installs a new one. asyncio is single-threaded
        # so a get-then-set sequence is safe-by-accident today, but using
        # setdefault makes the per-scope-lock invariant robust to future
        # refactors that might introduce an await between check and set.
        return self._locks.setdefault(scope, asyncio.Lock())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_token(
        self,
        scope: str,
        *,
        refresh_if_within_seconds: int = 60,
    ) -> str:
        """Return a valid access token for ``scope``, refreshing if needed.

        Parameters
        ----------
        scope:
            OAuth2 scope to request (e.g. ``"https://graph.microsoft.com/.default"``).
        refresh_if_within_seconds:
            Refresh proactively if the cached token expires within this many
            seconds (default 60). Bumping this value above the token's
            remaining lifetime forces a refresh on the next call.

        Raises
        ------
        AuthError
            If MSAL returns an ``error`` field instead of an ``access_token``.
        """
        # Fast path: cached and not near expiry → no lock needed.
        cached = self._cache.get(scope)
        if cached is not None:
            token, expires_at = cached
            if expires_at - time.time() > refresh_if_within_seconds:
                return token

        # Slow path: take the per-scope lock, double-check, then refresh.
        lock = self._get_lock(scope)
        async with lock:
            cached = self._cache.get(scope)
            if cached is not None:
                token, expires_at = cached
                if expires_at - time.time() > refresh_if_within_seconds:
                    return token

            app = self._get_app()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: app.acquire_token_for_client([scope]),
            )

            if "error" in result:
                err = result["error"]
                desc = result.get("error_description", "")
                log.warning("graph token error scope=%s err=%s", scope, err)
                raise AuthError(f"{err}: {desc}")

            token = result["access_token"]
            # Floor expires_in at 60s so a malformed/zero/negative value from
            # the IdP can't poison the fast path or trigger a refresh thrash.
            expires_in = max(int(result.get("expires_in", 3600)), 60)
            # Wall-clock time matches MSAL's expires_in semantics. A backwards
            # system-clock jump could over-extend cache freshness; 401 retries
            # elsewhere handle that, and Azure rotates infrequently enough
            # that switching to time.monotonic() isn't worth the complexity.
            self._cache[scope] = (token, time.time() + expires_in)
            log.debug(
                "graph token refresh scope=%s expires_in=%ss", scope, expires_in
            )
            return token
