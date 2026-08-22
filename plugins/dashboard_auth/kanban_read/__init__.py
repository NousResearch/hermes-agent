"""KanbanReadSecretProvider — loopback-only read access to the board projection.

This provider exposes one fixed capability: ``kanban.read``. Its bearer
credential can authenticate only the exact ``GET /api/plugins/kanban/board``
route, and only when both the dashboard bind and request peer are loopback.
Browser requests without a service bearer continue through the native session
gate; the native Kanban handler remains the sole source of the response
envelope.
"""
from __future__ import annotations

import hmac
import logging
import os
from typing import Optional

from hermes_cli.dashboard_auth import (
    DashboardAuthProvider,
    LoginStart,
    Session,
    TokenPrincipal,
)
from hermes_cli.dashboard_auth.secret_strength import assess_secret_strength

logger = logging.getLogger(__name__)

KANBAN_BOARD_ROUTE_PATH = "/api/plugins/kanban/board"
KANBAN_READ_SCOPE = "kanban.read"
LAST_SKIP_REASON: str = ""


class KanbanReadSecretProvider(DashboardAuthProvider):
    """Non-interactive provider for the loopback Kanban board projection."""

    name = "kanban-read-secret"
    display_name = "Kanban Board Reader (service credential)"
    supports_token = True
    supports_session = False

    def __init__(self, *, secret: str) -> None:
        reason = assess_secret_strength(secret)
        if reason is not None:
            raise ValueError(f"kanban read secret rejected: {reason}")
        self._secret = secret

    def verify_token(self, *, token: str) -> Optional[TokenPrincipal]:
        """Return the fixed board-reader principal on a constant-time match."""
        if not token:
            return None
        if hmac.compare_digest(token.encode("utf-8"), self._secret.encode("utf-8")):
            return TokenPrincipal(
                principal="kanban-board-reader",
                provider=self.name,
                scopes=(KANBAN_READ_SCOPE,),
            )
        return None

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        raise NotImplementedError(
            "KanbanReadSecretProvider is a non-interactive service credential; "
            "there is no login flow."
        )

    def complete_login(
        self, *, code: str, state: str, code_verifier: str, redirect_uri: str
    ) -> Session:
        raise NotImplementedError(
            "KanbanReadSecretProvider is a non-interactive service credential."
        )

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        return None

    def refresh_session(self, *, refresh_token: str) -> Session:
        raise NotImplementedError(
            "KanbanReadSecretProvider is a non-interactive service credential."
        )

    def revoke_session(self, *, refresh_token: str) -> None:
        return None


def register(ctx) -> None:
    """Register the provider only when its strong env credential is present."""
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""

    secret = os.environ.get("HERMES_DASHBOARD_KANBAN_READ_SECRET", "").strip()
    if not secret:
        LAST_SKIP_REASON = (
            "HERMES_DASHBOARD_KANBAN_READ_SECRET is not set; the Kanban "
            "read service credential remains disabled."
        )
        logger.debug("dashboard-auth-kanban-read: %s", LAST_SKIP_REASON)
        return

    reason = assess_secret_strength(secret)
    if reason is not None:
        LAST_SKIP_REASON = (
            f"HERMES_DASHBOARD_KANBAN_READ_SECRET rejected — {reason}. "
            "The Kanban read route stays disabled (fail-closed)."
        )
        logger.warning("dashboard-auth-kanban-read: %s", LAST_SKIP_REASON)
        return

    try:
        provider = KanbanReadSecretProvider(secret=secret)
        from hermes_cli.dashboard_auth.token_auth import register_token_route

        register_token_route(
            KANBAN_BOARD_ROUTE_PATH,
            methods=("GET",),
            required_scope=KANBAN_READ_SCOPE,
            loopback_only=True,
            allow_session_fallback=True,
        )
    except (ValueError, TypeError) as exc:
        LAST_SKIP_REASON = f"Kanban read registration failed: {exc}"
        logger.warning("dashboard-auth-kanban-read: %s", LAST_SKIP_REASON)
        return

    ctx.register_dashboard_auth_provider(provider)
    logger.info(
        "dashboard-auth-kanban-read: registered board-reader provider "
        "(scope=%s, route=%s, method=GET, loopback-only)",
        KANBAN_READ_SCOPE,
        KANBAN_BOARD_ROUTE_PATH,
    )
