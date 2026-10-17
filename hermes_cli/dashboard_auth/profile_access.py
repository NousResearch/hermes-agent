"""Per-identity profile authorisation for the dashboard.

The login gate answers *who* is calling; this module answers *which profiles* that caller may
see and drive. Plugins register checkers through ``ctx.register_dashboard_profile_access``. A
checker receives the request identity and a profile name and returns ``True`` (allow), ``False``
(deny) or ``None`` (abstain). Any ``False`` denies; with no checker registered, or every checker
abstaining, access is allowed, so a dashboard without such a plugin behaves exactly as before.

The identity of the request is bound to a context variable by the HTTP gate
(``bind_request_identity``) and by the WebSocket gate (``bind_identity``), so the profile scope
helpers can check access without the request object.
"""
from __future__ import annotations

import contextvars
import logging
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from fastapi import HTTPException

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Identity:
    """The caller as seen by profile checkers. ``email``, ``display_name`` and ``org_id`` are
    empty on WebSocket upgrades, whose tickets only carry ``user_id`` and ``provider``."""
    user_id: str
    provider: str
    email: str = ""
    display_name: str = ""
    org_id: str = ""


Checker = Callable[[Optional[Identity], str], Optional[bool]]

_identity: contextvars.ContextVar[Optional[Identity]] = contextvars.ContextVar(
    "hermes_dashboard_identity", default=None)
_lock = threading.Lock()
_checkers: list[Checker] = []


def register_checker(checker: Checker) -> None:
    with _lock:
        if checker not in _checkers:
            _checkers.append(checker)


def unregister_checker(checker: Checker) -> None:
    with _lock:
        if checker in _checkers:
            _checkers.remove(checker)


def has_checkers() -> bool:
    with _lock:
        return bool(_checkers)


def identity_from_session(session) -> Optional[Identity]:
    if session is None:
        return None
    return Identity(
        user_id=session.user_id, provider=session.provider, email=session.email,
        display_name=session.display_name, org_id=session.org_id)


def identity_from_ws_stamp(stamp: Optional[dict]) -> Optional[Identity]:
    if not stamp or not stamp.get("user_id"):
        return None
    return Identity(user_id=str(stamp["user_id"]), provider=str(stamp.get("provider") or ""))


def bind_identity(identity: Optional[Identity]) -> contextvars.Token:
    return _identity.set(identity)


def reset_identity(token: contextvars.Token) -> None:
    _identity.reset(token)


def current_identity() -> Optional[Identity]:
    return _identity.get()


def is_allowed(profile: str, identity: Optional[Identity] = None) -> bool:
    """True unless a registered checker denies ``profile`` to the identity. A checker that
    raises denies: an authorisation layer must fail closed."""
    who = identity if identity is not None else _identity.get()
    with _lock:
        checkers = list(_checkers)
    for checker in checkers:
        try:
            verdict = checker(who, profile)
        except Exception:
            _log.exception("dashboard profile access checker failed; denying %r", profile)
            return False
        if verdict is False:
            return False
    return True


def require(profile: str) -> None:
    """Raise HTTP 403 when the bound identity may not use ``profile``. The message names no
    owner, so a denied caller learns nothing beyond the refusal."""
    if not is_allowed(profile):
        raise HTTPException(status_code=403, detail=f"Profile '{profile}' is not available to you.")


def filter_names(names: Iterable[str]) -> list[str]:
    return [name for name in names if is_allowed(name)]
