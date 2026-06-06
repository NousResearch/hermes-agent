"""
Email OAuth2 (XOAUTH2) stubs for the Hermes gateway.

Provides OAuth2 token management and XOAUTH2 string generation for
IMAP and SMTP authentication, plus device-code-based authorization
flows for interactive OAuth consent.

TODO (child tasks T2-T10): replace all ``NotImplementedError`` stubs
with real implementations.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OAuthError(RuntimeError):
    """Base exception for OAuth2-related failures in email authentication."""


class DeviceCodeExpiredError(OAuthError):
    """Raised when the device code has expired before the user completed auth."""


class DeviceCodeDeniedError(OAuthError):
    """Raised when the user denied the device-code authorization request."""


# ---------------------------------------------------------------------------
# Token manager
# ---------------------------------------------------------------------------


class OAuthTokenManager:
    """Manages OAuth2 access tokens for email (IMAP/SMTP) authentication.

    Handles token acquisition, caching, refresh, and disk persistence.
    Supports both **client-credentials** (app-only, no user interaction)
    and **device-code** (user-consent via browser) flows.

    Parameters
    ----------
    tenant_id:
        Microsoft Entra ID (Azure AD) tenant ID.
    client_id:
        Application (client) ID registered in Entra ID.
    client_secret:
        Client secret for the application.
    scope:
        OAuth2 scope for the token request.
        Defaults to ``https://outlook.office.com/.default``.
    """

    EMAIL_OAUTH_SCOPE = "https://outlook.office.com/.default"

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        scope: str = EMAIL_OAUTH_SCOPE,
    ) -> None:
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self._cached_token: str | None = None
        self._cached_expires_at: float | None = None
        logger.debug(
            "[EmailOAuth] OAuthTokenManager initialized for tenant=%s client=%s",
            tenant_id,
            client_id[:8],
        )

    # -- Public API -------------------------------------------------------

    def get_token(self, *, force_refresh: bool = False) -> str:
        """Return a valid access token, acquiring or refreshing as needed.

        Parameters
        ----------
        force_refresh:
            If ``True``, bypass the in-memory cache and fetch a fresh token.

        Returns
        -------
        A bearer access token string.

        Raises
        ------
        OAuthError
            If token acquisition fails.
        """
        raise NotImplementedError("T2: implement client-credentials token acquisition")

    def clear_cache(self) -> None:
        """Drop the cached token so the next call fetches a fresh one."""
        self._cached_token = None
        self._cached_expires_at = None
        logger.debug("[EmailOAuth] Token cache cleared")

    def save_disk_cache(self) -> None:
        """Persist the current token to disk atomically."""
        # TODO: implement atomic JSON write to ~/.hermes/oauth_tokens.json
        raise NotImplementedError("T3: implement disk token persistence")

    @classmethod
    def load_disk_cache(cls) -> str | None:
        """Load a cached token from disk, or ``None`` if missing/expired."""
        raise NotImplementedError("T3: implement disk token loading")

    def invalidate_disk_cache(self) -> None:
        """Remove the on-disk token cache."""
        raise NotImplementedError("T3: implement disk cache invalidation")

    def inspect_health(self) -> dict[str, Any]:
        """Return a snapshot of the token manager's state for diagnostics."""
        return {
            "tenant_id": self.tenant_id,
            "client_id": self.client_id[:8] + "...",
            "scope": self.scope,
            "cached": self._cached_token is not None,
            "expires_at": self._cached_expires_at,
        }

    # -- Device code flow -------------------------------------------------

    def begin_device_code_flow(self) -> dict[str, Any]:
        """Start a device-authorization (device code) flow.

        Returns
        -------
        A dict with keys ``device_code``, ``user_code``, ``verification_uri``,
        ``verification_url``, ``expires_in``, and ``interval`` (polling
        interval in seconds).
        """
        raise NotImplementedError("T4: implement device-code flow initiation")

    def poll_device_code(self, device_code: str) -> str | None:
        """Poll for device-code authorization completion.

        Parameters
        ----------
        device_code:
            The ``device_code`` returned by ``begin_device_code_flow``.

        Returns
        -------
        The access token string if the user completed authorization,
        or ``None`` if the user hasn't responded yet.

        Raises
        ------
        DeviceCodeExpiredError
            If the device code has expired.
        DeviceCodeDeniedError
            If the user denied the request.
        OAuthError
            On other OAuth failures.
        """
        raise NotImplementedError("T4: implement device-code poll")


# ---------------------------------------------------------------------------
# XOAUTH2 string builders
# ---------------------------------------------------------------------------


def build_imap_xoauth2_bytes(address: str, access_token: str) -> bytes:
    """Return the raw XOAUTH2 SASL bytes for IMAP authentication.

    ``imaplib.IMAP4.authenticate('XOAUTH2', callback)`` expects the
    callback to return *raw bytes* (not base64); imaplib itself handles
    the base64 wire encoding.

    The SASL payload follows the XOAUTH2 format:

        user=<email>\\x01auth=Bearer <token>\\x01\\x01

    where ``\\x01`` is the SASL delimiter (SOH character).

    Parameters
    ----------
    address:
        The email address (IMAP user name).
    access_token:
        The OAuth2 bearer access token.

    Returns
    -------
    Raw ASCII-encoded bytes for the SASL exchange.
    """
    return (
        b"user=" + address.encode("ascii")
        + b"\x01auth=Bearer " + access_token.encode("ascii")
        + b"\x01\x01"
    )


def build_smtp_xoauth2_str(address: str, access_token: str) -> str:
    """Return the raw XOAUTH2 SASL string for SMTP authentication.

    ``smtplib.SMTP.auth('XOAUTH2', callback)`` expects the callback to
    return a *raw string* (not base64); smtplib itself encodes it with
    ``encode_base64`` before sending.

    Delegates to ``build_imap_xoauth2_bytes`` and decodes the result.

    Parameters
    ----------
    address:
        The email address (SMTP user name).
    access_token:
        The OAuth2 bearer access token.

    Returns
    -------
    Raw ASCII string for the SASL exchange.
    """
    raw = build_imap_xoauth2_bytes(address, access_token)
    return raw.decode("ascii")


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def begin_device_code_flow(
    tenant_id: str,
    client_id: str,
    scope: str = OAuthTokenManager.EMAIL_OAUTH_SCOPE,
) -> dict[str, Any]:
    """Start a device-authorization flow without instantiating a manager.

    Shortcut that creates a temporary ``OAuthTokenManager`` and delegates
    to its ``begin_device_code_flow``.

    Returns
    -------
    A dict with keys ``device_code``, ``user_code``, ``verification_uri``,
    ``verification_url``, ``expires_in``, and ``interval``.
    """
    raise NotImplementedError("T4: implement device-code flow initiation")


def poll_device_code(
    tenant_id: str,
    client_id: str,
    device_code: str,
    scope: str = OAuthTokenManager.EMAIL_OAUTH_SCOPE,
) -> str | None:
    """Poll for device-code authorization completion without a manager.

    Shortcut that creates a temporary ``OAuthTokenManager`` and delegates
    to its ``poll_device_code``.

    Parameters
    ----------
    device_code:
        The ``device_code`` from ``begin_device_code_flow``.

    Returns
    -------
    The access token or ``None`` if the user hasn't responded yet.
    """
    raise NotImplementedError("T4: implement device-code poll")
