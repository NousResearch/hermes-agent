"""Minimal Plaid REST client built on the core ``httpx`` dependency.

Only the endpoints the finance plugin needs are wrapped. Every call posts the
``client_id``/``secret`` pair in the JSON body (Plaid's auth scheme) and maps
Plaid's structured error envelope onto the plugin's error hierarchy so callers
get actionable, provider-agnostic failures.

Reference: https://plaid.com/docs/api/
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from plugins.finance.errors import (
    FinanceConfigError,
    FinanceProviderError,
    FinanceReauthRequiredError,
)

# Plaid retired the "development" environment; sandbox and production remain.
_ENV_BASE_URLS = {
    "sandbox": "https://sandbox.plaid.com",
    "production": "https://production.plaid.com",
}

# Error codes that mean "the stored access token is fine, but the user must
# re-authenticate via Link update mode".
_REAUTH_CODES = {"ITEM_LOGIN_REQUIRED", "PENDING_EXPIRATION", "ITEM_LOGIN_REQUIRED_UPDATE"}

# Error codes that indicate the configured API credentials are wrong/missing.
_CONFIG_CODES = {"INVALID_API_KEYS", "INVALID_CLIENT", "INVALID_SECRET", "UNAUTHORIZED_ENVIRONMENT"}

DEFAULT_COUNTRY_CODES = ["US"]
DEFAULT_LANGUAGE = "en"


class PlaidClient:
    """Thin synchronous wrapper around the Plaid REST API."""

    def __init__(
        self,
        client_id: str,
        secret: str,
        environment: str = "sandbox",
        *,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        if not client_id or not secret:
            raise FinanceConfigError(
                "Plaid credentials are not configured. Set PLAID_CLIENT_ID and "
                "PLAID_SECRET in your .env."
            )
        self.client_id = client_id
        self.secret = secret
        self.environment = (environment or "sandbox").strip().lower()
        self.base_url = (base_url or _ENV_BASE_URLS.get(self.environment, _ENV_BASE_URLS["sandbox"])).rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = {"client_id": self.client_id, "secret": self.secret}
        if payload:
            body.update(payload)
        try:
            response = httpx.post(url, json=body, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise FinanceProviderError(f"Plaid request to {path} failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError:
            data = {}

        if response.status_code >= 400 or (isinstance(data, dict) and data.get("error_code")):
            self._raise_for_error(path, response.status_code, data if isinstance(data, dict) else {})
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _raise_for_error(path: str, status_code: int, data: Dict[str, Any]) -> None:
        error_code = data.get("error_code")
        error_type = data.get("error_type")
        message = (
            data.get("error_message")
            or data.get("display_message")
            or f"Plaid request to {path} failed with HTTP {status_code}"
        )
        request_id = data.get("request_id")
        if error_code in _REAUTH_CODES:
            raise FinanceReauthRequiredError(
                message, status_code=status_code, error_code=error_code,
                error_type=error_type, request_id=request_id,
            )
        if error_code in _CONFIG_CODES:
            raise FinanceConfigError(message)
        raise FinanceProviderError(
            message, status_code=status_code, error_code=error_code,
            error_type=error_type, request_id=request_id,
        )

    # ------------------------------------------------------------------
    # Link / onboarding
    # ------------------------------------------------------------------

    def link_token_create(
        self,
        *,
        user_id: str,
        client_name: str = "Hermes",
        products: Optional[List[str]] = None,
        country_codes: Optional[List[str]] = None,
        language: str = DEFAULT_LANGUAGE,
        hosted_link: Optional[Dict[str, Any]] = None,
        webhook: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "user": {"client_user_id": user_id},
            "client_name": client_name,
            "country_codes": country_codes or DEFAULT_COUNTRY_CODES,
            "language": language,
        }
        # Update mode (re-auth) omits products and passes the existing token.
        if access_token:
            payload["access_token"] = access_token
        else:
            payload["products"] = products or ["transactions"]
        if hosted_link is not None:
            payload["hosted_link"] = hosted_link
        if webhook:
            payload["webhook"] = webhook
        if redirect_uri:
            payload["redirect_uri"] = redirect_uri
        return self._post("/link/token/create", payload)

    def link_token_get(self, link_token: str) -> Dict[str, Any]:
        return self._post("/link/token/get", {"link_token": link_token})

    def item_public_token_exchange(self, public_token: str) -> Dict[str, Any]:
        return self._post("/item/public_token/exchange", {"public_token": public_token})

    def sandbox_public_token_create(
        self, *, institution_id: str, initial_products: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        return self._post(
            "/sandbox/public_token/create",
            {"institution_id": institution_id, "initial_products": initial_products or ["transactions"]},
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def accounts_balance_get(self, access_token: str) -> Dict[str, Any]:
        return self._post("/accounts/balance/get", {"access_token": access_token})

    def transactions_sync(
        self, access_token: str, *, cursor: Optional[str] = None, count: int = 500
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"access_token": access_token, "count": count}
        if cursor:
            payload["cursor"] = cursor
        return self._post("/transactions/sync", payload)

    def investments_holdings_get(self, access_token: str) -> Dict[str, Any]:
        return self._post("/investments/holdings/get", {"access_token": access_token})

    def liabilities_get(self, access_token: str) -> Dict[str, Any]:
        return self._post("/liabilities/get", {"access_token": access_token})

    def item_get(self, access_token: str) -> Dict[str, Any]:
        return self._post("/item/get", {"access_token": access_token})

    def item_remove(self, access_token: str) -> Dict[str, Any]:
        return self._post("/item/remove", {"access_token": access_token})

    def institutions_get_by_id(
        self, institution_id: str, *, country_codes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        return self._post(
            "/institutions/get_by_id",
            {"institution_id": institution_id, "country_codes": country_codes or DEFAULT_COUNTRY_CODES},
        )
