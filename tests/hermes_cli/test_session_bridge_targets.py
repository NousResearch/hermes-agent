from __future__ import annotations

import sqlite3
import uuid

import pytest

from hermes_cli.session_bridge_targets import (
    SessionTargetObservation, TargetUnavailable, ensure_schema,
    record_provider_observation, resolve_ticket_target,
)


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def _observation(**overrides) -> SessionTargetObservation:
    values = {
        "observation_id": str(uuid.uuid4()), "ticket_ref": "ESN-0407",
        "assignment_id": "assignment-7", "attempt_id": "attempt-3",
        "assignment_active": True, "attempt_active": True,
        "backend": "claude", "native_session_id": str(uuid.uuid4()),
        "delivery_handle": "uds:/owner-only/session.sock", "agent_id": "agent-9",
        "team_name": "team-a", "transport": "claude-local-uds",
        "owner_pid": 123, "owner_started_at": 45.0, "ownership_kind": "local-process",
        "identity_verified": True, "transport_authenticated": True,
        "hermes_callable": True,
        "reply_capable": True, "cancel_capable": True, "non_interrupting": True,
        "answer_only": True, "authority": "claude-native-adapter",
        "provenance": {"source": "provider-session-registry", "schema": "v1"},
        "observed_at": 100.0, "expires_at": 150.0,
    }
    values.update(overrides)
    return SessionTargetObservation(**values)


def test_ticket_resolves_exact_attempt_without_manual_native_id():
    conn = _db()
    expected = _observation()
    record_provider_observation(conn, expected)
    actual = resolve_ticket_target(conn, "esn-0407", now=110, liveness=lambda pid, started: True)
    assert actual.assignment_id == "assignment-7" and actual.attempt_id == "attempt-3"
    assert actual.native_session_id == expected.native_session_id
    assert actual.delivery_handle == "uds:/owner-only/session.sock"
    assert actual.delivery_handle != actual.native_session_id


@pytest.mark.parametrize("field", [
    "assignment_active", "attempt_active",
    "identity_verified", "transport_authenticated", "hermes_callable", "reply_capable",
    "cancel_capable", "non_interrupting",
])
def test_answer_only_wording_cannot_manufacture_transport_or_identity(field: str):
    conn = _db()
    record_provider_observation(conn, _observation(**{field: False, "answer_only": True}))
    with pytest.raises(TargetUnavailable, match=field):
        resolve_ticket_target(conn, "ESN-0407", now=110, liveness=lambda pid, started: True)


def test_answer_authority_must_be_provider_attested():
    conn = _db()
    record_provider_observation(conn, _observation(answer_only=False))
    with pytest.raises(TargetUnavailable, match="answer_only"):
        resolve_ticket_target(conn, "ESN-0407", now=110, liveness=lambda pid, started: True)


def test_stale_or_dead_observation_fails_closed():
    conn = _db()
    record_provider_observation(conn, _observation(expires_at=500))
    with pytest.raises(TargetUnavailable, match="stale or expired"):
        resolve_ticket_target(conn, "ESN-0407", now=200, liveness=lambda pid, started: True)
    with pytest.raises(TargetUnavailable, match="owner is not provably live"):
        resolve_ticket_target(conn, "ESN-0407", now=110, liveness=lambda pid, started: False)


def test_multiple_active_attempts_are_not_guessed():
    conn = _db()
    record_provider_observation(conn, _observation())
    record_provider_observation(conn, _observation(attempt_id="attempt-4"))
    with pytest.raises(TargetUnavailable, match="ambiguous"):
        resolve_ticket_target(conn, "ESN-0407", now=110, liveness=lambda pid, started: True)


def test_newer_refresh_for_same_attempt_supersedes_old_observation():
    conn = _db()
    record_provider_observation(conn, _observation(observed_at=100, expires_at=150))
    expected = _observation(observed_at=105, expires_at=155, delivery_handle="uds:/rotated.sock")
    record_provider_observation(conn, expected)
    actual = resolve_ticket_target(conn, "ESN-0407", now=110, liveness=lambda pid, started: True)
    assert actual.observation_id == expected.observation_id
    assert actual.delivery_handle == "uds:/rotated.sock"


def test_registry_has_no_public_cli_ingress():
    from hermes_cli.subcommands.session_bridge import build_session_bridge_parser
    import argparse

    parser = argparse.ArgumentParser()
    build_session_bridge_parser(parser.add_subparsers(dest="command"))
    bridge = parser.parse_args(["session-bridge", "capabilities"])
    assert bridge.bridge_action == "capabilities"
    assert "register-target" not in parser.format_help()

