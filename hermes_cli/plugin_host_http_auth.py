"""Carry route-local admission over the private plugin-host channel, without credentials."""
from fastapi import HTTPException


def admitted_for_sensitive_route(request) -> bool:
    """Preserve the normal guard's decision without changing unguarded plugin routes."""
    from hermes_cli.web_server import _require_token
    try:
        _require_token(request)
    except HTTPException as exc:
        if exc.status_code != 401:
            raise
        return False
    return True


def admitted_for_native_http(request, home) -> bool:
    """Check the gateway's redeemed native HTTP grant against the worker's home."""
    from hermes_constants import hermes_home_key

    grant = getattr(request.state, "native_http_principal", None)
    if not isinstance(grant, dict) or grant.get("purpose") != "native-http" or "origin" in request.headers:
        return False
    profile_id = grant.get("profile_id")
    return (
        isinstance(profile_id, str)
        and bool(profile_id)
        and hermes_home_key(profile_id) == hermes_home_key(home)
    )


def with_hosted_admission(app, authenticated, native_authenticated):
    """Admission belongs to this IPC request, never to the cached plugin app."""
    async def scoped(scope, receive, send):
        state = dict(scope.get("state") or {})
        state["hosted_http_authenticated"] = authenticated is True
        state["hosted_native_http_authenticated"] = authenticated is True and native_authenticated is True
        await app({**scope, "state": state}, receive, send)
    return scoped
