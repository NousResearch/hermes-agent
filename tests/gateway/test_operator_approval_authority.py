"""Contracts for authenticated Hermes specialist-promotion approval."""

from __future__ import annotations

from types import SimpleNamespace

from gateway.operator_approval_authority import authenticated_operator_identity


def _session(*, user_id: str = "operator-1", provider: str = "portal"):
    return SimpleNamespace(user_id=user_id, provider=provider, org_id="org-1")


def test_only_gated_session_with_explicit_subject_allowlist_is_operator():
    identity = authenticated_operator_identity(
        _session(),
        auth_required=True,
        allowed_subjects=("portal:operator-1",),
    )

    assert identity == "portal:operator-1"


def test_loopback_token_or_unlisted_session_cannot_be_operator():
    assert authenticated_operator_identity(
        _session(), auth_required=False, allowed_subjects=("portal:operator-1",)
    ) is None
    assert authenticated_operator_identity(
        _session(), auth_required=True, allowed_subjects=("portal:someone-else",)
    ) is None
