"""Contract test for DashboardAuthProvider implementations.

Every provider plugin should call ``assert_protocol_compliance`` on its
provider class in its own unit test. This module tests the abstract base
itself: dataclass fields, ABC rejection of partial impls, and the
protocol-compliance helper.
"""
from __future__ import annotations

import pytest

from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider,
    Session,
    LoginStart,
    assert_protocol_compliance,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


def test_session_has_required_fields():
    s = Session(
        user_id="u1",
        email="a@b.com",
        display_name="A",
        org_id="org_1",
        provider="test",
        expires_at=1234567890,
        access_token="at",
        refresh_token="rt",
    )
    assert s.user_id == "u1"
    assert s.provider == "test"
    assert s.expires_at == 1234567890


def test_login_start_has_redirect_and_state():
    ls = LoginStart(
        redirect_url="https://portal/authorize?...",
        cookie_payload={"hermes_session_pkce": "verifier=abc;state=xyz"},
    )
    assert ls.redirect_url.startswith("https://")
    assert "hermes_session_pkce" in ls.cookie_payload


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------


def test_abstract_provider_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DashboardAuthProvider()  # type: ignore[abstract]


class _BrokenProvider(DashboardAuthProvider):
    name = "broken"
    display_name = "Broken"
    # Deliberately missing all the methods.


def test_assert_protocol_compliance_rejects_partial_impl():
    with pytest.raises(TypeError):
        assert_protocol_compliance(_BrokenProvider)


class _CompliantProvider(DashboardAuthProvider):
    name = "ok"
    display_name = "OK"

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        return LoginStart(redirect_url="x", cookie_payload={})

    def complete_login(self, *, code, state, code_verifier, redirect_uri) -> Session:
        return Session(
            user_id="u", email="x", display_name="x", org_id="o",
            provider=self.name, expires_at=0,
            access_token="a", refresh_token="r",
        )

    def verify_session(self, *, access_token: str):
        return None

    def refresh_session(self, *, refresh_token: str) -> Session:
        return Session(
            user_id="u", email="x", display_name="x", org_id="o",
            provider=self.name, expires_at=0,
            access_token="a", refresh_token="r",
        )

    def revoke_session(self, *, refresh_token: str) -> None:
        return None


def test_assert_protocol_compliance_accepts_full_impl():
    # Returns None on success; the helper raises on failure.
    assert assert_protocol_compliance(_CompliantProvider) is None


def test_assert_protocol_compliance_rejects_missing_name_attr():
    class NoName(_CompliantProvider):
        name = ""  # empty is treated as missing

    with pytest.raises(TypeError, match="name"):
        assert_protocol_compliance(NoName)


def test_assert_protocol_compliance_rejects_missing_display_name():
    class NoDisplay(_CompliantProvider):
        display_name = ""

    with pytest.raises(TypeError, match="display_name"):
        assert_protocol_compliance(NoDisplay)


# ---------------------------------------------------------------------------
# Registry (Task 1.2)
# ---------------------------------------------------------------------------


from hermes_cli.dashboard_auth import (  # noqa: E402  (after-imports for clarity)
    register_provider,
    get_provider,
    list_providers,
    clear_providers,
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Every test starts with an empty registry and leaves it empty."""
    clear_providers()
    yield
    clear_providers()


def test_registry_register_and_get():
    p = _CompliantProvider()
    register_provider(p)
    assert get_provider("ok") is p


def test_registry_get_missing_returns_none():
    assert get_provider("nope") is None


def test_registry_lists_in_registration_order():
    class A(_CompliantProvider):
        name = "a"
        display_name = "A"

    class B(_CompliantProvider):
        name = "b"
        display_name = "B"

    register_provider(A())
    register_provider(B())
    names = [p.name for p in list_providers()]
    assert names == ["a", "b"]


def test_registry_rejects_non_compliant_provider():
    with pytest.raises(TypeError):
        register_provider(_BrokenProvider())  # type: ignore[abstract]


def test_registry_rejects_duplicate_name():
    register_provider(_CompliantProvider())
    with pytest.raises(ValueError, match="already registered"):
        register_provider(_CompliantProvider())


def test_registry_clear_drops_all():
    register_provider(_CompliantProvider())
    assert get_provider("ok") is not None
    clear_providers()
    assert get_provider("ok") is None
    assert list_providers() == []
