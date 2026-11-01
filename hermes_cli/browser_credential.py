"""The browser credential, and how every grant made under it ends when it is replaced.

The browser credential is the dashboard session token (``hermes_cli.web_server._SESSION_TOKEN``). A
loopback Webapp tab holds it; behind the sign-in gate each Webapp login is bound to it
(``hermes_cli.webapp_sign_in``). The gateway keeps the Webapp's credential in its home
(``gateway.webapp_listener.browser_session_token``) and sets it when its listener starts.

Every grant a browser gets under the credential records the credential's generation, a SHA-256 digest
of it, when the grant is made, and is honoured only while :func:`is_current` says that generation is
still the one in use: window and launch tickets (``web_routers.webapp``), socket tickets
(``web_session_sockets.ticket_refusal``), Bot Desktop viewer tickets (``web_routers.display``), file
tickets (``web_server_file_tickets``), open sockets (``web_session_sockets.track``) and a cookie
login's binding (``webapp_sign_in.check``). Stores keep the generation, never the credential.

:func:`rotate` (``hermes webapp --sign-out``) replaces the credential, then runs every revoker
registered with :func:`on_rotate`: grants that stay in use without being presented again (open
sockets) end at once, and stores drop what can no longer be redeemed.
"""
from __future__ import annotations

import hashlib
import hmac
import threading
from typing import Callable, Optional

_lock = threading.Lock()
#: Revoker name -> callable that ends the grants made under a replaced credential and returns how
#: many it ended.
_revokers: dict[str, Callable[[], int]] = {}


def current() -> str:
    """The browser credential in use now."""
    from hermes_cli import web_server
    return web_server._SESSION_TOKEN


def generation_of(credential: str) -> str:
    """The generation a grant made under *credential* records."""
    return hashlib.sha256(credential.encode()).hexdigest()


def generation() -> str:
    """The generation of the credential in use now."""
    return generation_of(current())


def is_current(recorded: Optional[str]) -> bool:
    """Whether a grant that recorded generation *recorded* was made under the credential in use now."""
    return current_if(recorded) is not None


def current_if(recorded: Optional[str]) -> Optional[str]:
    """The credential in use now when *recorded* is its generation, else None. For a grant that
    hands the credential over or binds a login to it: one read, so a rotation between the check
    and the use cannot pass on the credential that replaced the one the grant was made under."""
    credential = current()
    if recorded is None or not hmac.compare_digest(recorded, generation_of(credential)):
        return None
    return credential


def on_rotate(name: str, revoker: Callable[[], int]) -> None:
    """Run *revoker* each time the credential is replaced; registering a name again replaces it."""
    with _lock:
        _revokers[name] = revoker


def rotate(credential: str) -> dict[str, int]:
    """Make *credential* the browser credential and end every grant made under the one it replaces.
    Returns how many grants each registered revoker ended, by name. Safe from any thread."""
    from hermes_cli import web_server

    web_server._SESSION_TOKEN = credential
    with _lock:
        revokers = dict(_revokers)
    return {name: revoker() for name, revoker in revokers.items()}
