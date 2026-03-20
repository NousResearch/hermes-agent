"""Authenticated product app surface for hermes-core."""

from __future__ import annotations

import hashlib
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from hermes_cli.product_config import load_product_config
from hermes_cli.product_oidc import (
    create_oidc_login_request,
    discover_product_oidc_provider_metadata,
    exchange_product_oidc_code,
    fetch_product_oidc_userinfo,
    load_product_oidc_client_settings,
)
from hermes_cli.product_runtime import delete_product_runtime, get_product_runtime_session, stream_product_runtime_turn
from hermes_cli.product_stack import resolve_product_urls
from hermes_cli.product_users import (
    ProductSignupToken,
    ProductUser,
    create_product_signup_token,
    create_product_user,
    deactivate_product_user,
    get_product_user_by_id,
    list_product_users,
)
from hermes_cli.product_web import build_product_index_html


class ProductHealthResponse(BaseModel):
    status: str = "ok"
    auth_provider: str
    issuer_url: str
    app_base_url: str


class ProductSessionResponse(BaseModel):
    authenticated: bool
    user: dict[str, Any] | None = None


class ProductAdminUsersResponse(BaseModel):
    users: list[ProductUser]


class ProductCreateUserRequest(BaseModel):
    username: str
    display_name: str
    email: str | None = None


class ProductChatMessage(BaseModel):
    role: str
    content: str


class ProductChatSessionResponse(BaseModel):
    session_id: str
    messages: list[ProductChatMessage]


class ProductChatTurnRequest(BaseModel):
    user_message: str


def _session_secret() -> str:
    settings = load_product_oidc_client_settings()
    digest = hashlib.sha256(settings.client_secret.encode("utf-8")).hexdigest()
    return f"hermes-product-session-{digest}"


def _session_user_payload(userinfo: dict[str, Any]) -> dict[str, Any]:
    product_config = load_product_config()
    bootstrap = product_config.get("bootstrap", {})
    first_admin_username = (
        str(bootstrap.get("first_admin_username", "admin")).strip() or "admin"
    )
    preferred_username = str(userinfo.get("preferred_username") or "").strip()
    email = userinfo.get("email")
    if isinstance(email, str) and email.endswith("@users.local.invalid"):
        email = None
    return {
        "id": userinfo.get("sub", ""),
        "sub": userinfo.get("sub", ""),
        "email": email,
        "name": userinfo.get("name") or userinfo.get("preferred_username") or userinfo.get("sub", ""),
        "preferred_username": userinfo.get("preferred_username"),
        "email_verified": userinfo.get("email_verified"),
        "is_admin": preferred_username == first_admin_username,
    }


def _refresh_session_user(user: dict[str, Any]) -> dict[str, Any] | None:
    user_id = str(user.get("sub") or "").strip()
    if not user_id:
        return None
    provider_user = get_product_user_by_id(user_id)
    if provider_user is None or provider_user.disabled:
        return None
    refreshed = dict(user)
    refreshed["id"] = provider_user.id
    refreshed["email"] = provider_user.email
    refreshed["name"] = provider_user.display_name or refreshed.get("name")
    refreshed["preferred_username"] = provider_user.username
    return refreshed


def _require_product_user(request: Request) -> dict[str, Any]:
    user = request.session.get("user")
    if not isinstance(user, dict):
        raise HTTPException(status_code=401, detail="Not authenticated")
    refreshed = _refresh_session_user(user)
    if refreshed is None:
        request.session.clear()
        raise HTTPException(status_code=401, detail="Not authenticated")
    request.session["user"] = refreshed
    return refreshed


def _require_admin_user(request: Request) -> dict[str, Any]:
    user = _require_product_user(request)
    if not bool(user.get("is_admin")):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def create_product_app() -> FastAPI:
    product_config = load_product_config()
    urls = resolve_product_urls(product_config)
    auth_provider = str(product_config.get("auth", {}).get("provider", "unknown")).strip() or "unknown"
    issuer_url = str(product_config.get("auth", {}).get("issuer_url", "")).strip() or urls["issuer_url"]

    app = FastAPI(title="Hermes Core Product App", version="0.1.0")
    app.add_middleware(
        SessionMiddleware,
        secret_key=_session_secret(),
        session_cookie="hermes_product_session",
        same_site="lax",
        https_only=False,
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        product_name = (
            str(product_config.get("product", {}).get("brand", {}).get("name", "Hermes Core")).strip()
            or "Hermes Core"
        )
        return HTMLResponse(build_product_index_html(product_name=product_name))

    @app.get("/healthz", response_model=ProductHealthResponse)
    def healthz() -> ProductHealthResponse:
        return ProductHealthResponse(
            auth_provider=auth_provider,
            issuer_url=issuer_url,
            app_base_url=urls["app_base_url"],
        )

    @app.get("/api/auth/login")
    def auth_login(request: Request) -> RedirectResponse:
        settings = load_product_oidc_client_settings()
        metadata = discover_product_oidc_provider_metadata(settings)
        login_request = create_oidc_login_request(settings, metadata)
        request.session["oidc_pending"] = {
            "state": login_request["state"],
            "nonce": login_request["nonce"],
            "verifier": login_request["verifier"],
        }
        return RedirectResponse(login_request["authorization_url"], status_code=307)

    @app.get("/api/auth/oidc/callback")
    def auth_callback(request: Request, code: str, state: str) -> RedirectResponse:
        pending = request.session.get("oidc_pending")
        if not isinstance(pending, dict):
            raise HTTPException(status_code=400, detail="Missing OIDC login state")
        if state != pending.get("state"):
            raise HTTPException(status_code=400, detail="OIDC state mismatch")

        settings = load_product_oidc_client_settings()
        metadata = discover_product_oidc_provider_metadata(settings)
        token_response = exchange_product_oidc_code(
            settings,
            metadata,
            code=code,
            verifier=str(pending.get("verifier", "")),
        )
        access_token = str(token_response.get("access_token", "")).strip()
        if not access_token:
            raise HTTPException(status_code=502, detail="OIDC token response missing access_token")
        userinfo = fetch_product_oidc_userinfo(access_token, metadata)
        request.session.pop("oidc_pending", None)
        request.session["user"] = _session_user_payload(userinfo)
        return RedirectResponse(urls["app_base_url"], status_code=303)

    @app.get("/api/auth/session", response_model=ProductSessionResponse)
    def auth_session(request: Request) -> ProductSessionResponse:
        user = request.session.get("user")
        if not isinstance(user, dict):
            return ProductSessionResponse(authenticated=False)
        refreshed = _refresh_session_user(user)
        if refreshed is None:
            request.session.clear()
            return ProductSessionResponse(authenticated=False)
        request.session["user"] = refreshed
        return ProductSessionResponse(authenticated=True, user=refreshed)

    @app.post("/api/auth/logout", response_model=ProductSessionResponse)
    def auth_logout(request: Request) -> ProductSessionResponse:
        request.session.clear()
        return ProductSessionResponse(authenticated=False)

    @app.get("/api/chat/session", response_model=ProductChatSessionResponse)
    def chat_session(request: Request) -> ProductChatSessionResponse:
        user = _require_product_user(request)
        payload = get_product_runtime_session(user)
        return ProductChatSessionResponse(**payload)

    @app.post("/api/chat/turn/stream")
    def chat_turn_stream(request: Request, payload: ProductChatTurnRequest) -> StreamingResponse:
        user = _require_product_user(request)
        try:
            event_stream = stream_product_runtime_turn(user, payload.user_message)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return StreamingResponse(
            event_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/admin/users", response_model=ProductAdminUsersResponse)
    def admin_list_users(request: Request) -> ProductAdminUsersResponse:
        _require_admin_user(request)
        return ProductAdminUsersResponse(users=list_product_users())

    @app.post("/api/admin/users", response_model=ProductUser)
    def admin_create_user(request: Request, payload: ProductCreateUserRequest) -> ProductUser:
        _require_admin_user(request)
        try:
            return create_product_user(
                payload.username,
                payload.display_name,
                email=payload.email,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/admin/signup-tokens", response_model=ProductSignupToken)
    def admin_create_signup_token(request: Request) -> ProductSignupToken:
        _require_admin_user(request)
        try:
            return create_product_signup_token()
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/admin/users/{user_id}/deactivate", response_model=ProductUser)
    def admin_deactivate_user(request: Request, user_id: str) -> ProductUser:
        admin_user = _require_admin_user(request)
        if user_id == str(admin_user.get("sub") or ""):
            raise HTTPException(status_code=400, detail="Admins cannot deactivate their own account")
        try:
            updated = deactivate_product_user(user_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        normalized = ProductUser.model_validate(updated)
        delete_product_runtime(normalized.username)
        return normalized

    return app
