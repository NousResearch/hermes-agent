"""Permission helpers for RecruitmentSystem queries."""

from __future__ import annotations

from .models import UserContext

_ADMIN_ROLES = {"admin", "hr", "recruitment_admin", "manager"}


def can_query_attendance(user: UserContext, *, target_user_id: str | None) -> bool:
    if not target_user_id:
        return True
    if str(target_user_id) == str(user.user_id):
        return True
    return bool(user.roles & _ADMIN_ROLES)
