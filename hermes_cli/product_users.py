from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from hermes_cli.product_stack import _api_headers, resolve_product_urls
from hermes_cli.product_config import load_product_config


_PLACEHOLDER_EMAIL_DOMAIN = "users.local.invalid"
_DEFAULT_SIGNUP_TOKEN_TTL = 7 * 24 * 60 * 60
_DEFAULT_SIGNUP_TOKEN_USAGE_LIMIT = 1


class ProductUser(BaseModel):
    id: str
    username: str
    display_name: str
    email: str | None = None
    email_is_placeholder: bool = False
    is_admin: bool = False
    disabled: bool = False


class ProductSignupToken(BaseModel):
    token: str
    signup_url: str
    ttl_seconds: int
    usage_limit: int


class PocketIdUserRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    username: str
    email: str | None = None
    email_verified: bool = Field(default=False, alias="emailVerified")
    first_name: str = Field(default="", alias="firstName")
    last_name: str = Field(default="", alias="lastName")
    display_name: str = Field(default="", alias="displayName")
    is_admin: bool = Field(default=False, alias="isAdmin")
    disabled: bool = False
    locale: str | None = None
    custom_claims: list[Any] = Field(default_factory=list, alias="customClaims")
    user_groups: list[Any] = Field(default_factory=list, alias="userGroups")
    ldap_id: str | None = Field(default=None, alias="ldapId")


def _client(config: dict[str, Any] | None = None) -> httpx.Client:
    product_config = config or load_product_config()
    base_url = resolve_product_urls(product_config)["issuer_url"]
    return httpx.Client(
        base_url=base_url,
        headers=_api_headers(product_config),
        timeout=15.0,
    )


def _request_json(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    expected_status: int,
    **kwargs: Any,
) -> dict[str, Any]:
    response = client.request(method, path, **kwargs)
    if response.status_code != expected_status:
        raise RuntimeError(f"{method} {path} failed with {response.status_code}: {response.text}")
    return response.json() if response.content else {}


def _is_internal_user(record: PocketIdUserRecord) -> bool:
    return record.username.startswith("static-api-user-")


def _placeholder_email(username: str) -> str:
    return f"{username}@{_PLACEHOLDER_EMAIL_DOMAIN}"


def _normalize_email(email: str | None) -> tuple[str | None, bool]:
    normalized = (email or "").strip()
    if not normalized:
        return None, False
    if normalized.endswith(f"@{_PLACEHOLDER_EMAIL_DOMAIN}"):
        return None, True
    return normalized, False


def _normalize_user(record: PocketIdUserRecord) -> ProductUser:
    email, email_is_placeholder = _normalize_email(record.email)
    display_name = record.display_name.strip() or record.username
    return ProductUser(
        id=record.id,
        username=record.username,
        display_name=display_name,
        email=email,
        email_is_placeholder=email_is_placeholder,
        is_admin=record.is_admin,
        disabled=record.disabled,
    )


def _split_display_name(display_name: str, username: str) -> tuple[str, str]:
    normalized = display_name.strip() or username
    parts = normalized.split(maxsplit=1)
    first_name = parts[0]
    last_name = parts[1] if len(parts) > 1 else "user"
    return first_name[:50], last_name[:50]


def list_product_users(config: dict[str, Any] | None = None) -> list[ProductUser]:
    with _client(config) as client:
        payload = _request_json(client, "GET", "/api/users", expected_status=200)
    records = [
        PocketIdUserRecord.model_validate(item)
        for item in payload.get("data", [])
    ]
    visible = [_normalize_user(item) for item in records if not _is_internal_user(item)]
    return sorted(visible, key=lambda item: (item.disabled, item.username.lower()))


def get_product_user_by_id(user_id: str, config: dict[str, Any] | None = None) -> ProductUser | None:
    with _client(config) as client:
        response = client.get(f"/api/users/{user_id}")
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise RuntimeError(f"GET /api/users/{user_id} failed with {response.status_code}: {response.text}")
    record = PocketIdUserRecord.model_validate(response.json())
    if _is_internal_user(record):
        return None
    return _normalize_user(record)


def create_product_user(
    username: str,
    display_name: str,
    *,
    email: str | None = None,
    config: dict[str, Any] | None = None,
) -> ProductUser:
    normalized_username = username.strip()
    if not normalized_username:
        raise ValueError("Username must not be empty")
    normalized_display_name = display_name.strip() or normalized_username
    first_name, last_name = _split_display_name(normalized_display_name, normalized_username)
    normalized_email = (email or "").strip() or _placeholder_email(normalized_username)
    payload = {
        "username": normalized_username,
        "firstName": first_name,
        "lastName": last_name,
        "displayName": normalized_display_name,
        "email": normalized_email,
        "emailsVerified": False,
        "isAdmin": False,
        "disabled": False,
    }
    with _client(config) as client:
        response = _request_json(client, "POST", "/api/users", expected_status=200, json=payload)
    return _normalize_user(PocketIdUserRecord.model_validate(response))


def deactivate_product_user(user_id: str, config: dict[str, Any] | None = None) -> ProductUser:
    with _client(config) as client:
        get_response = client.get(f"/api/users/{user_id}")
        if get_response.status_code == 404:
            raise ValueError("User not found")
        if get_response.status_code != 200:
            raise RuntimeError(
                f"GET /api/users/{user_id} failed with {get_response.status_code}: {get_response.text}"
            )
        record = PocketIdUserRecord.model_validate(get_response.json())
        if _is_internal_user(record):
            raise ValueError("Internal service users cannot be managed from the product UI")
        payload = {
            "username": record.username,
            "email": record.email,
            "firstName": record.first_name,
            "lastName": record.last_name,
            "displayName": record.display_name,
            "isAdmin": record.is_admin,
            "disabled": True,
            "locale": record.locale,
        }
        response = _request_json(client, "PUT", f"/api/users/{user_id}", expected_status=200, json=payload)
    return _normalize_user(PocketIdUserRecord.model_validate(response))


def create_product_signup_token(config: dict[str, Any] | None = None) -> ProductSignupToken:
    product_config = config or load_product_config()
    payload = {
        "ttl": _DEFAULT_SIGNUP_TOKEN_TTL,
        "usageLimit": _DEFAULT_SIGNUP_TOKEN_USAGE_LIMIT,
        "userGroupIds": [],
    }
    with _client(product_config) as client:
        response = _request_json(client, "POST", "/api/signup-tokens", expected_status=200, json=payload)
    token = str(response.get("token", "")).strip()
    if not token:
        raise RuntimeError("Pocket ID did not return a signup token")
    issuer_url = resolve_product_urls(product_config)["issuer_url"]
    return ProductSignupToken(
        token=token,
        signup_url=f"{issuer_url}/st/{token}",
        ttl_seconds=_DEFAULT_SIGNUP_TOKEN_TTL,
        usage_limit=_DEFAULT_SIGNUP_TOKEN_USAGE_LIMIT,
    )
