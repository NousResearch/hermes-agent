"""The backend process credential authorizes only the cross-process activation POST."""

from __future__ import annotations

import hmac

from fastapi import Request


def has_plugin_activation_token(request: Request) -> bool:
    """Keep machine activation separate from provider sessions and other machine routes."""
    if (request.method != "POST"
            or request.url.path != "/api/dashboard/agent-plugins/activate"
            or "origin" in request.headers):
        return False
    from hermes_cli import web_server

    tokens = request.headers.getlist(web_server._SESSION_HEADER_NAME)
    return (len(tokens) == 1 and bool(tokens[0]) and bool(web_server._SESSION_TOKEN)
            and hmac.compare_digest(tokens[0].encode(), web_server._SESSION_TOKEN.encode()))
