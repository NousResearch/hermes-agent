"""Contracts for authenticated specialist-promotion approval identity."""

from __future__ import annotations

import pytest


def _approval_api():
    try:
        from gateway.operator_approval_authority import (
            authenticated_operator_identity,
            canonical_dashboard_subject,
        )
    except ImportError as exc:  # RED: the split starts without this boundary.
        pytest.fail(f"operator approval authority is unavailable: {exc}")
    return authenticated_operator_identity, canonical_dashboard_subject


def _session(*, user_id: str = "operator-1", provider: str = "portal"):
    from hermes_cli.dashboard_auth.base import Session

    return Session(
        user_id=user_id,
        email="",
        display_name="",
        org_id="",
        provider=provider,
        expires_at=0,
        access_token="",
        refresh_token="",
    )


def test_gated_allowlisted_session_yields_canonical_operator_subject():
    authenticated_operator_identity, canonical_dashboard_subject = _approval_api()
    allowed = canonical_dashboard_subject(provider="portal", user_id="operator-1")

    identity = authenticated_operator_identity(
        _session(),
        auth_required=True,
        allowed_subjects=(allowed,),
    )

    assert identity == allowed


def test_subject_encoding_cannot_collide_on_provider_user_delimiter():
    _, canonical_dashboard_subject = _approval_api()

    first = canonical_dashboard_subject(provider="portal:team", user_id="operator")
    second = canonical_dashboard_subject(provider="portal", user_id="team:operator")

    assert first != second
    assert first == '{"provider":"portal:team","user_id":"operator"}'
    assert second == '{"provider":"portal","user_id":"team:operator"}'


def test_legacy_delimited_allowlist_value_does_not_authorize():
    authenticated_operator_identity, _ = _approval_api()

    assert authenticated_operator_identity(
        _session(provider="portal:team", user_id="operator"),
        auth_required=True,
        allowed_subjects=("portal:team:operator",),
    ) is None


def test_loopback_mode_and_unlisted_subject_fail_closed():
    authenticated_operator_identity, canonical_dashboard_subject = _approval_api()
    allowed = canonical_dashboard_subject(provider="portal", user_id="operator-1")

    assert authenticated_operator_identity(
        _session(), auth_required=False, allowed_subjects=(allowed,)
    ) is None
    assert authenticated_operator_identity(
        _session(),
        auth_required=True,
        allowed_subjects=(canonical_dashboard_subject(provider="portal", user_id="someone-else"),),
    ) is None


@pytest.mark.parametrize(
    "session",
    (None, _session(user_id=""), _session(provider="")),
)
def test_incomplete_session_identity_fails_closed(session):
    authenticated_operator_identity, canonical_dashboard_subject = _approval_api()

    assert authenticated_operator_identity(
        session,
        auth_required=True,
        allowed_subjects=(canonical_dashboard_subject(provider="portal", user_id="operator-1"),),
    ) is None
