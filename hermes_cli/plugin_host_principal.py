"""Credential-free verified account context for allowlisted hosted HTTP routes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.parse import parse_qsl, urlencode

from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from hermes_cli.dashboard_auth.base import Session


@dataclass(frozen=True)
class VerifiedSessionPrincipal:
    provider: str
    issuer: str
    org_id: str
    user_id: str


_HISTORY_ROUTES = {
    ("GET", "/history"),
    ("POST", "/history/sync"),
    ("POST", "/history/import"),
    ("POST", "/history/context"),
}
_FIELDS = {"provider", "issuer", "org_id", "user_id"}


def _allowed(plugin: str, method: str, path: str) -> bool:
    return plugin == "hermes-locations" and (method, path) in _HISTORY_ROUTES


def principal_for_request(
    request: Request, plugin: str, path: str,
) -> VerifiedSessionPrincipal | None:
    """Only middleware's verified Session is authoritative, never HTTP input."""
    if not _allowed(plugin, request.method, path):
        return None
    session = getattr(request.state, "session", None)
    if not isinstance(session, Session):
        return None
    return VerifiedSessionPrincipal(
        provider=session.provider, issuer=session.issuer,
        org_id=session.org_id, user_id=session.user_id,
    )


def encode_principal(
    principal: VerifiedSessionPrincipal | None, plugin: str, method: str, path: str,
) -> dict[str, str] | None:
    if not _allowed(plugin, method, path) or principal is None:
        return None
    if not isinstance(principal, VerifiedSessionPrincipal):
        raise TypeError("hosted HTTP principal must be a VerifiedSessionPrincipal")
    return asdict(principal)


def history_http_input(
    plugin: str, method: str, path: str, query: str, headers: list,
) -> tuple[str, list]:
    """The history host needs identity, not gateway-wide admission credentials."""
    if not _allowed(plugin, method, path):
        return query, headers
    from hermes_cli.web_server_plugin_profiles import _GATEWAY_CREDENTIAL_HEADERS, _GATEWAY_QUERY_KEYS
    return (
        urlencode([(key, value) for key, value in parse_qsl(query, keep_blank_values=True)
                   if key not in _GATEWAY_QUERY_KEYS]),
        [(key, value) for key, value in headers if key.lower() not in _GATEWAY_CREDENTIAL_HEADERS],
    )


def asgi_with_principal(app: ASGIApp, params: dict) -> ASGIApp:
    """Build request-local state; never mutate the cached app or share a ContextVar."""
    raw = params.get("verified_session_principal")
    principal = None
    if _allowed(params["plugin"], params["method"], params["path"]) and raw is not None:
        if (not isinstance(raw, dict) or raw.keys() != _FIELDS
                or not all(isinstance(value, str) for value in raw.values())
                or not raw["provider"].strip() or not raw["user_id"].strip()):
            raise ValueError("invalid verified hosted HTTP principal")
        principal = VerifiedSessionPrincipal(**raw)

    async def scoped(scope: Scope, receive: Receive, send: Send) -> None:
        state = dict(scope.get("state") or {})
        state.pop("verified_session_principal", None)
        if principal is not None:
            state["verified_session_principal"] = asdict(principal)
        await app({**scope, "state": state}, receive, send)

    return scoped
