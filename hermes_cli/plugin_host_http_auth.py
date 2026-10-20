"""Carry route-local admission over the private plugin-host channel, without credentials."""
from typing import Any, Optional

from fastapi import HTTPException

# The stable account identity of a verified dashboard Session: no tokens, contact data or expiry.
SESSION_PRINCIPAL_FIELDS = ("provider", "issuer", "org_id", "user_id")
SESSION_PRINCIPAL_STATE_KEY = "verified_session_principal"


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


def session_principal(value: Any) -> Optional[dict[str, str]]:
    """``value`` as an exact four-field account principal, or None: every field a ``str``, and a
    non-blank ``provider`` and ``user_id`` (``issuer`` / ``org_id`` may be empty, as on Session)."""
    if not isinstance(value, dict) or set(value) != set(SESSION_PRINCIPAL_FIELDS):
        return None
    if not all(type(value[k]) is str for k in SESSION_PRINCIPAL_FIELDS):
        return None
    if not value["provider"].strip() or not value["user_id"].strip():
        return None
    return {k: value[k] for k in SESSION_PRINCIPAL_FIELDS}


def hosted_admission(request) -> tuple[bool, Optional[dict[str, str]]]:
    """What a hosted plugin request may carry across the private channel: the guard's decision, and
    for an admitted request the gate verified as a dashboard account Session, that account's
    identity. Native tickets, the loopback token and service tokens admit without an account."""
    from hermes_cli.dashboard_auth.base import Session
    authenticated = admitted_for_sensitive_route(request)
    session = getattr(request.state, "session", None)
    if not authenticated or not isinstance(session, Session):
        return authenticated, None
    return True, session_principal({
        "provider": session.provider, "issuer": session.issuer,
        "org_id": session.org_id, "user_id": session.user_id})


def with_hosted_admission(app, authenticated, principal=None):
    """Admission and identity belong to this IPC request, never to the cached plugin app."""
    admitted = authenticated is True
    account = session_principal(principal) if admitted else None

    async def scoped(scope, receive, send):
        state = dict(scope.get("state") or {})
        state["hosted_http_authenticated"] = admitted
        state.pop(SESSION_PRINCIPAL_STATE_KEY, None)
        if account is not None:
            state[SESSION_PRINCIPAL_STATE_KEY] = dict(account)
        await app({**scope, "state": state}, receive, send)
    return scoped
