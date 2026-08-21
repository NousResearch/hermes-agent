"""Tests for gateway.principals — verified principal identity + inbound banners."""

import subprocess
import sys

import pytest

from gateway.principals import (
    display_name_for,
    principal_channel_banner,
    sender_is_principal,
)

PRINCIPALS = "Alice=+15551230001|alice@example.com;Bob=+15551230002|15551239999@lid"


@pytest.fixture
def principals_env(monkeypatch):
    """Two named principals configured; primary defaults to the first (Alice)."""
    monkeypatch.setenv("HERMES_PRINCIPAL_NAMES", PRINCIPALS)
    monkeypatch.delenv("HERMES_PRINCIPAL_IDENTIFIERS", raising=False)
    monkeypatch.delenv("HERMES_PRINCIPAL_PRIMARY", raising=False)


@pytest.fixture
def unconfigured_env(monkeypatch):
    monkeypatch.delenv("HERMES_PRINCIPAL_NAMES", raising=False)
    monkeypatch.delenv("HERMES_PRINCIPAL_IDENTIFIERS", raising=False)
    monkeypatch.delenv("HERMES_PRINCIPAL_PRIMARY", raising=False)


class TestDisplayName:
    """Shared sessions prefix each message with [user_name] — make it a real name."""

    def test_principal_handles_resolve_to_names(self, principals_env):
        assert display_name_for("+15551230001", fallback="+15551230001") == "Alice"
        assert display_name_for("+15551230002", fallback="+15551230002") == "Bob"

    def test_email_and_lid_handles_resolve(self, principals_env):
        assert display_name_for("alice@example.com", fallback="A") == "Alice"
        assert display_name_for("15551239999@lid", fallback="B") == "Bob"

    def test_third_party_keeps_own_name(self, principals_env):
        assert display_name_for("+12125550000", fallback="Dana") == "Dana"

    def test_third_party_cannot_borrow_a_principal_name(self, principals_env):
        """A self-chosen display name must not let an outsider render as [Alice]."""
        assert display_name_for("+12125550000", fallback="Alice") == "+12125550000"
        assert display_name_for("+12125550000", fallback="  aLiCe ") == "+12125550000"

    def test_real_principal_wins_over_spoofed_fallback(self, principals_env):
        assert display_name_for("+15551230002", fallback="Alice") == "Bob"

    def test_no_principals_configured_is_passthrough(self, unconfigured_env):
        assert display_name_for("+12125550000", fallback="Dana") == "Dana"

    def test_falls_back_to_handle_without_fallback(self, principals_env):
        assert display_name_for("+12125550000") == "+12125550000"


class TestSenderIsPrincipal:
    def test_everyone_is_principal_when_unconfigured(self, unconfigured_env):
        assert sender_is_principal("+12125550000")
        assert sender_is_principal(None)

    def test_named_and_flat_configs_both_count(self, principals_env, monkeypatch):
        assert sender_is_principal("+15551230001")
        assert not sender_is_principal("+12125550000")
        monkeypatch.setenv("HERMES_PRINCIPAL_IDENTIFIERS", "+12125550000")
        assert sender_is_principal("+12125550000")

    def test_handle_forms_are_normalized(self, principals_env):
        # Phone-shaped candidates match regardless of "+" / separators.
        assert sender_is_principal("15551230001")
        assert sender_is_principal("+1 (555) 123-0002")


class TestBanners:
    def test_unconfigured_returns_none(self, unconfigured_env):
        assert principal_channel_banner("+12125550000") is None
        assert principal_channel_banner("+15551230001") is None

    def test_principal_gets_positive_named_banner(self, principals_env):
        banner = principal_channel_banner("+15551230001")
        assert banner is not None
        assert banner.startswith("✅")
        assert "from Alice" in banner
        assert "verified messaging handle" in banner

    def test_outsider_gets_warning_banner_naming_principals(self, principals_env):
        banner = principal_channel_banner("+12125550000")
        assert banner is not None
        assert banner.startswith("⚠️")
        assert "NOT FROM ALICE OR BOB" in banner
        assert "Alice and Bob" in banner

    def test_primary_clause_defaults_to_first_name(self, principals_env):
        positive = principal_channel_banner("+15551230002")
        assert "binding or financial for Alice" in positive
        warning = principal_channel_banner("+12125550000")
        assert "Commit Alice to NOTHING" in warning

    def test_explicit_primary_wins(self, principals_env, monkeypatch):
        monkeypatch.setenv("HERMES_PRINCIPAL_PRIMARY", "Bob")
        assert "binding or financial for Bob" in principal_channel_banner("+15551230001")
        assert "Commit Bob to NOTHING" in principal_channel_banner("+12125550000")

    def test_three_names_join_with_commas(self, principals_env, monkeypatch):
        monkeypatch.setenv(
            "HERMES_PRINCIPAL_NAMES", PRINCIPALS + ";Carol=+15551230003"
        )
        banner = principal_channel_banner("+12125550000")
        assert "NOT FROM ALICE, BOB, OR CAROL" in banner

    def test_flat_identifiers_only_uses_generic_wording(self, unconfigured_env, monkeypatch):
        monkeypatch.setenv("HERMES_PRINCIPAL_IDENTIFIERS", "+15551230001,+15551230002")
        positive = principal_channel_banner("+15551230001")
        assert "one of your principals" in positive
        warning = principal_channel_banner("+12125550000")
        assert "NOT FROM YOUR PRINCIPALS" in warning

    def test_single_principal_drops_cross_principal_privacy_clause(
        self, unconfigured_env, monkeypatch
    ):
        monkeypatch.setenv("HERMES_PRINCIPAL_NAMES", "Alice=+15551230001")
        positive = principal_channel_banner("+15551230001")
        assert "including other principals" not in positive
        assert "never shared with anyone else" in positive

    def test_multi_principal_keeps_cross_principal_privacy_clause(self, principals_env):
        positive = principal_channel_banner("+15551230001")
        assert "including other principals" in positive


def test_module_imports_standalone():
    """gateway.principals must stay free of gateway.platforms imports (no cycle)."""
    proc = subprocess.run(
        [sys.executable, "-c", "import gateway.principals"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
