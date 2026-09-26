"""Audit log for dashboard-auth events.

Profile-aware location: ``$HERMES_HOME/logs/dashboard-auth.log``.
Format: one JSON object per line. Token-like kwargs are dropped before
serialisation so we never leak refresh tokens or JWTs to disk.
"""
from __future__ import annotations

import json
import pytest

from hermes_cli.dashboard_auth.audit import audit_log, AuditEvent


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    """Redirect $HERMES_HOME and ~ to a tmp dir for the duration of the test."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Some code paths fall back to Path.home() — patch that too.
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    return home


def test_audit_writes_jsonlines(profile_home):
    audit_log(AuditEvent.LOGIN_START, provider="nous", ip="1.2.3.4")
    audit_log(
        AuditEvent.LOGIN_SUCCESS,
        provider="nous", user_id="u1",
        email="a@b.com", ip="1.2.3.4",
    )

    path = profile_home / "logs" / "dashboard-auth.log"
    assert path.exists(), f"audit log not created at {path}"
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2

    second = json.loads(lines[1])
    assert second["event"] == "login_success"
    assert second["provider"] == "nous"
    assert second["user_id"] == "u1"
    assert second["email"] == "a@b.com"
    assert "ts" in second  # ISO-8601 timestamp


def test_audit_redacts_token_like_fields(profile_home):
    audit_log(
        AuditEvent.LOGIN_SUCCESS,
        provider="nous", access_token="should-not-appear",
        refresh_token="also-not", code="not-this", state="nope",
    )
    raw = (profile_home / "logs" / "dashboard-auth.log").read_text()
    for forbidden in ("should-not-appear", "also-not", "not-this", "nope"):
        assert forbidden not in raw, f"token-like value leaked into audit log: {forbidden}"


# ---------------------------------------------------------------------------
# ws_ticket_minted sampling (#57749)
# ---------------------------------------------------------------------------

@pytest.fixture
def reset_ticket_sampler():
    """Reset the module-level mint counter so tests are order-independent."""
    import hermes_cli.dashboard_auth.audit as audit_mod
    audit_mod._ticket_mint_count = 0
    audit_mod._ticket_mint_since_last_logged = 0
    yield
    audit_mod._ticket_mint_count = 0
    audit_mod._ticket_mint_since_last_logged = 0


def test_ticket_mints_are_sampled_not_flooded(profile_home, reset_ticket_sampler):
    import hermes_cli.dashboard_auth.audit as audit_mod

    n = audit_mod._TICKET_MINT_SAMPLE_EVERY
    for _ in range(n * 2 + 1):
        audit_log(AuditEvent.WS_TICKET_MINTED, provider="basic", user_id="hermes")

    lines = (profile_home / "logs" / "dashboard-auth.log").read_text().splitlines()
    # First of every batch survives: mints 1, N+1, 2N+1.
    assert len(lines) == 3


def test_other_audit_events_are_never_sampled(profile_home, reset_ticket_sampler):
    import hermes_cli.dashboard_auth.audit as audit_mod

    for _ in range(audit_mod._TICKET_MINT_SAMPLE_EVERY * 3):
        audit_log(AuditEvent.WS_TICKET_REJECTED, provider="basic", reason="bad ticket")

    lines = (profile_home / "logs" / "dashboard-auth.log").read_text().splitlines()
    assert len(lines) == audit_mod._TICKET_MINT_SAMPLE_EVERY * 3


def test_sampling_preserves_event_payload(profile_home, reset_ticket_sampler):
    audit_log(AuditEvent.WS_TICKET_MINTED, provider="basic", user_id="hermes")
    line = json.loads(
        (profile_home / "logs" / "dashboard-auth.log").read_text().splitlines()[0]
    )
    assert line["event"] == "ws_ticket_minted"
    assert line["provider"] == "basic"
    assert "ts" in line
    assert "ticket" not in line  # redaction still applies to sampled events


def test_sampling_includes_suppressed_count(profile_home, reset_ticket_sampler):
    """Each logged mint includes count of suppressed mints since last logged mint."""
    import hermes_cli.dashboard_auth.audit as audit_mod

    n = audit_mod._TICKET_MINT_SAMPLE_EVERY
    # Fire N+1 mints: first logs (suppressed=0), next N-1 suppressed, (N+1)th logs (suppressed=N-1)
    for i in range(n + 1):
        audit_log(AuditEvent.WS_TICKET_MINTED, provider="basic", user_id="hermes")

    lines = (profile_home / "logs" / "dashboard-auth.log").read_text().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["suppressed_mints_since_last_logged"] == 0

    second = json.loads(lines[1])
    assert second["suppressed_mints_since_last_logged"] == n - 1


def test_suppressed_count_resets_after_logged(profile_home, reset_ticket_sampler):
    """Suppressed counter resets to 0 after a mint is logged."""
    import hermes_cli.dashboard_auth.audit as audit_mod

    n = audit_mod._TICKET_MINT_SAMPLE_EVERY
    # Fire 2N+1 mints: logged at 1, N+1, 2N+1
    for i in range(2 * n + 1):
        audit_log(AuditEvent.WS_TICKET_MINTED, provider="basic", user_id="hermes")

    lines = (profile_home / "logs" / "dashboard-auth.log").read_text().splitlines()
    assert len(lines) == 3

    # All logged mints should have suppressed count = N-1 (except first which is 0)
    first = json.loads(lines[0])
    assert first["suppressed_mints_since_last_logged"] == 0

    second = json.loads(lines[1])
    assert second["suppressed_mints_since_last_logged"] == n - 1

    third = json.loads(lines[2])
    assert third["suppressed_mints_since_last_logged"] == n - 1
