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


def with_hosted_admission(app, authenticated):
    """Admission belongs to this IPC request, never to the cached plugin app."""
    async def scoped(scope, receive, send):
        state = dict(scope.get("state") or {})
        state["hosted_http_authenticated"] = authenticated is True
        await app({**scope, "state": state}, receive, send)
    return scoped
