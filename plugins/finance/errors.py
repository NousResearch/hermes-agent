"""Shared error types for the finance plugin.

These are provider-agnostic so the CLI, tools, and sync engine can map any
backend failure (Plaid today, Teller/SnapTrade/etc. later) onto a small set
of actionable categories without importing a specific provider's module.
"""

from __future__ import annotations

from typing import Optional


class FinanceError(RuntimeError):
    """Base error for all finance plugin failures."""


class FinanceConfigError(FinanceError):
    """Raised when required configuration or credentials are missing."""


class FinanceAuthRequiredError(FinanceError):
    """Raised when no account is linked yet (the user must connect first)."""


class FinanceProviderError(FinanceError):
    """Structured failure returned by a financial-data provider's API."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        error_type: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.error_type = error_type
        self.request_id = request_id


class FinanceReauthRequiredError(FinanceProviderError):
    """Raised when a linked item needs the user to re-authenticate.

    Maps to Plaid's ``ITEM_LOGIN_REQUIRED`` and similar states where the
    stored access token is still valid but the institution requires the
    user to refresh their login via Link update mode.
    """
