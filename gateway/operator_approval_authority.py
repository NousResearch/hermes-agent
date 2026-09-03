"""Authenticated dashboard identity boundary for specialist promotion.

The legacy loopback dashboard token proves only that a caller knows a
process-local bearer secret.  It cannot identify a human operator.  Promotion
therefore accepts an identity only from the verified dashboard-auth session in
gated mode and only when that identity appears in the operator's explicit
allowlist.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def dashboard_session_subject(session: Any, *, auth_required: bool) -> str | None:
    """Return the verified dashboard session's stable subject, if any."""
    if not auth_required or session is None:
        return None
    provider = getattr(session, "provider", None)
    user_id = getattr(session, "user_id", None)
    if not isinstance(provider, str) or not provider.strip():
        return None
    if not isinstance(user_id, str) or not user_id.strip():
        return None
    return f"{provider.strip()}:{user_id.strip()}"


def authenticated_operator_identity(
    session: Any,
    *,
    auth_required: bool,
    allowed_subjects: Iterable[str],
) -> str | None:
    """Return an allowlisted authenticated subject, otherwise fail closed."""
    subject = dashboard_session_subject(session, auth_required=auth_required)
    if subject is None:
        return None
    configured = {
        value.strip()
        for value in allowed_subjects
        if isinstance(value, str) and value.strip()
    }
    return subject if subject in configured else None
