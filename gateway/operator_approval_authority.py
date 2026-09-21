"""Authenticated dashboard identity boundary for specialist approval."""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import quote

from hermes_cli.dashboard_auth.base import Session


def canonical_dashboard_subject(*, provider: str, user_id: str) -> str:
    """Encode one provider/user tuple reversibly without delimiter ambiguity."""
    normalized_provider = provider.strip()
    normalized_user_id = user_id.strip()
    if not normalized_provider or not normalized_user_id:
        raise ValueError("provider and user_id must be non-empty strings")
    return f"dashboard:{quote(normalized_provider, safe='')}:{quote(normalized_user_id, safe='')}"


def dashboard_session_subject(session: Session | None, *, auth_required: bool) -> str | None:
    """Return a verified dashboard session's canonical subject, if available."""
    if not auth_required or not isinstance(session, Session):
        return None
    if not isinstance(session.provider, str) or not session.provider.strip():
        return None
    if not isinstance(session.user_id, str) or not session.user_id.strip():
        return None
    return canonical_dashboard_subject(provider=session.provider, user_id=session.user_id)


def authenticated_operator_identity(
    session: Session | None,
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
