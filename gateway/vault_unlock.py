"""Owner-mediated vault unlock for messaging sessions (#108316).

The browser-vault tools prompt through a *surface* callback installed on the turn's thread by the
CLI panel, the TUI bridge and the Desktop. A messaging gateway (Telegram, Discord, …) installs none —
nobody may type a master password into a chat, and the tools must never accept one there — so a locked
Bitwarden/1Password vault used to be a dead end for those sessions: ``browser_vault_list`` reported
``unavailable_in_this_session`` and ``browser_vault_unlock`` refused.

Those sessions get the supported route here: the gateway's own control socket
(``gateway/control_socket.py``), the local-only ACL-authenticated channel every other owner-driven
gateway operation uses. The prompt leaves the chat. The tool mints a one-time code naming the backend,
the profile home and the session that asked; the owner runs ``hermes vault unlock <code>`` in a
terminal on the machine running the gateway; the CLI reads the master password with ``getpass`` and
hands it to the running gateway in ONE control request; the gateway then unlocks through the same
``agent.vault_backends`` API an interactive surface calls.

Guarantees are those of the interactive path, unchanged: the master password is consumed by the
manager CLI and dropped — never logged, returned, written to disk, or placed in argv or our process
environment; only the manager's session token survives, in this process's memory, scoped to the
profile and owned by the session that asked. Codes are single-use and expire.

Trust boundary: the password travels over the local socket (0600 file / per-user named pipe) inside
one request line — the same host-local channel the CLI already uses to reach ``bw``/``op``. Connecting
requires this user's filesystem/pipe rights, and a process with those rights can already read the
profile's other credentials.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# The owner has to walk to a terminal: long enough to be convenient, short enough that a code left in
# a chat transcript cannot be redeemed much later.
TICKET_TTL_S = 5 * 60

_TICKETS: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def owner_unlock_available() -> bool:
    """Can a local operator reach THIS process to answer the unlock?

    Both halves matter. A human must be on the other end of the session (the gateway binds the
    platform; cron/webhook/api_server sessions are refused by the same predicate the approval gates
    use, since nobody would see the code), and the process must be the gateway that bound the control
    socket — otherwise the command in the tool result could not reach the vault state it unlocks.
    Contexts that fail either half keep the honest ``unavailable_in_this_session`` answer.
    """
    from gateway.control_socket import get_local_control_server

    if get_local_control_server() is None:
        return False
    from tools.approval_context import _is_gateway_approval_context

    return bool(_is_gateway_approval_context())


def mint_unlock_ticket(backend_name: str) -> Optional[Dict[str, Any]]:
    """One-time code the owner redeems from a terminal, or None when no terminal can reach this process.

    Reuses a live code for the same backend+session so an agent that retries the unlock (or a fill
    that trips it again) does not churn through codes in the chat.
    """
    from gateway.session_context import get_session_env

    if not owner_unlock_available():
        return None
    backend_name = str(backend_name or "")
    if not backend_name or backend_name == "local":
        return None
    from hermes_constants import get_hermes_home

    home = str(get_hermes_home())
    session_id = get_session_env("HERMES_SESSION_ID", "") or None
    now = time.monotonic()
    with _lock:
        for code, ticket in list(_TICKETS.items()):
            if ticket["expires_at"] <= now:
                _TICKETS.pop(code, None)
            elif (ticket["backend"], ticket["home"], ticket["session_id"]) == (backend_name, home, session_id):
                return _ticket_payload(code, ticket, now)
        code = secrets.token_hex(4).upper()
        ticket = {"backend": backend_name, "home": home, "session_id": session_id,
                  "expires_at": now + TICKET_TTL_S}
        _TICKETS[code] = ticket
    return _ticket_payload(code, ticket, now)


def _ticket_payload(code: str, ticket: Dict[str, Any], now: float) -> Dict[str, Any]:
    return {"code": code, "backend": ticket["backend"], "home": ticket["home"],
            "expires_in_seconds": max(1, int(ticket["expires_at"] - now)),
            "command": f"hermes vault unlock {code}"}


def _redeem(code: str) -> Optional[Dict[str, Any]]:
    """Consume a code (single use, TTL-checked)."""
    now = time.monotonic()
    with _lock:
        ticket = _TICKETS.pop(code, None)
    return ticket if ticket and ticket["expires_at"] > now else None


def handle_vault_unlock(request: Dict[str, Any]) -> Dict[str, Any]:
    """``vault-unlock`` control verb: unlock a manager for the session that asked.

    Params: ``code`` (minted by ``browser_vault_unlock``; names the backend, profile home and
    session) — or ``backend`` for an owner-initiated unlock with no pending request — plus
    ``password``. Answers ``{unlocked, backend, session_id, home}`` or ``{unlocked: False,
    error_type, error}``. Never raises: the caller is a one-shot local CLI and the answer must not
    carry the password back.
    """
    params = request.get("params") if isinstance(request, dict) else None
    params = params if isinstance(params, dict) else {}
    password = str(params.get("password") or "")
    code = str(params.get("code") or "").strip().upper()
    ticket = _redeem(code) if code else None
    if code and ticket is None:
        return {"unlocked": False, "error_type": "code_invalid",
                "error": ("That unlock code is unknown or has expired. Ask the session that needs the "
                          "login to request an unlock again.")}
    backend_name = str(ticket["backend"]) if ticket else str(params.get("backend") or "")
    if not backend_name:
        return {"unlocked": False, "error_type": "bad_request",
                "error": "Pass the unlock code from your chat, or --backend NAME to unlock without one."}
    if not password:
        return {"unlocked": False, "error_type": "bad_request",
                "error": "The master password is required."}

    from agent.vault_backends import enabled_backends
    from agent.vault_backends.unlock import set_current_session_id
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    # A ticket names the home the REQUESTING session runs in: multiplex turns resolve their profile
    # by home override, and the token must land in the same slot that session reads (unlock._key()).
    home_token = set_hermes_home_override(ticket["home"]) if ticket else None
    set_current_session_id(ticket.get("session_id") if ticket else None)
    try:
        backend = next((b for b in enabled_backends() if b.name == backend_name and b.needs_unlock), None)
        if backend is None:
            return {"unlocked": False, "error_type": "unknown_backend",
                    "error": f"{backend_name!r} is not an enabled password manager."}
        backend.unlock(password)  # type: ignore[attr-defined] — the interactive surfaces' own API
    except Exception as exc:
        return {"unlocked": False, "error_type": "unlock_failed",
                "error": str(exc).replace(password, "[REDACTED]")[:300]}
    finally:
        password = ""
        set_current_session_id(None)  # pooled executor thread: never leave a session id behind
        if home_token is not None:
            reset_hermes_home_override(home_token)
    return {"unlocked": True, "backend": backend_name, "session_id": (ticket or {}).get("session_id"),
            "home": (ticket or {}).get("home")}


def vault_unlock_handlers() -> Dict[str, Any]:
    """The control-socket verb table this module contributes to ``GatewayControlServer``."""
    return {"vault-unlock": handle_vault_unlock}
