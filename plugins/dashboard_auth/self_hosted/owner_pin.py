"""Self-hosted OIDC sign-in is owner-only: one IdP account, the machine owner's.

Hermes is single-user, and a signed-in Webapp tab acts as the machine owner (gateway.owner_identity).
An OIDC provider can hold many accounts, so the sign-in admits exactly one: the ``sub`` named by
``dashboard.oauth.self_hosted.owner``, or, when that is unset, the first account that signs in, whose
``[issuer, sub]`` is pinned in ``dashboard-auth/self-hosted-owner.json`` under the Hermes root home.
Any other account is refused like an invalid token (the callback reports a failed sign-in).
"""
from __future__ import annotations

import json
import logging
import os
import threading

from hermes_cli.dashboard_auth import InvalidCodeError

logger = logging.getLogger(__name__)
_lock = threading.Lock()


def _pin_path():
    from hermes_constants import get_process_hermes_home
    return get_process_hermes_home() / "dashboard-auth" / "self-hosted-owner.json"


def require_owner(issuer: str, sub: str, configured: str) -> None:
    """Raise :class:`InvalidCodeError` unless *sub* is the owner's account."""
    if not sub:
        raise InvalidCodeError("ID token names no account")
    if configured:
        if sub != configured:
            raise InvalidCodeError("this account is not the owner of this Hermes")
        return
    path = _pin_path()
    with _lock:
        try:
            pinned = json.loads(path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            pinned = None
        if pinned is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump([issuer, sub], handle)
            logger.warning("dashboard-auth-self-hosted: pinned the first account to sign in as this Hermes's "
                           "owner (%s); set dashboard.oauth.self_hosted.owner to choose it", path)
            return
    if pinned != [issuer, sub]:
        raise InvalidCodeError("this account is not the owner of this Hermes")
