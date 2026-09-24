"""ESTOP allowlist + deadman TTL — who keeps working while the fleet is paused.

`hermes pause --allow-user <id> --ttl 45m` is single-user mode: the pause holds cron,
kanban and new gateway turns, but the operator's own authenticated id is exempt, and the
sentinel lifts itself if the window job dies before it releases. These tests pin the
BEHAVIOUR of the gateway turn gate — an exempt identity is SERVED, everyone else gets the
pause notice — against a real `GatewayRunner` gate call and a real temp HERMES_HOME.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from agent import estop

OPERATOR = "operator-uid-7"
PEER = "bot-peer-1"


def _stamp(delta_seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=delta_seconds)).isoformat()


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir and reset estop module log state."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    estop._logged_components.clear()
    estop._expired_logged.clear()
    return tmp_path


class _FakeSource:
    platform = None
    chat_id = "c1"
    user_name = "user"
    chat_type = "dm"

    def __init__(self, user_id=None, profile=None):
        self.user_id = user_id
        self.profile = profile


class _FakeEvent:
    internal = False
    text = "hello"

    def __init__(self, source):
        self.source = source


def _gate(user_id=None, profile=None):
    """The real gate on a bare runner: notice str when the turn is REFUSED, None when served."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    source = _FakeSource(user_id=user_id, profile=profile)
    return runner._hm_estop_gate(_FakeEvent(source), source, is_internal=False)


def test_operator_identity_is_served_while_a_peer_is_refused(hermes_home):
    """The whole point of single-user mode: the pause holds the fleet, never the operator."""
    estop.engage(reason="update window", allow={"user_ids": [OPERATOR]}, ttl="30m")

    assert _gate(user_id=PEER) is not None, "a non-allowlisted peer must get the pause notice"
    assert _gate(user_id=OPERATOR) is None, "the allowlisted operator turn must be served"


def test_allowlist_matches_identity_not_profile_when_both_are_present(hermes_home):
    """Identity is primary: an allowlisted id passes even on an unlisted (or absent) profile."""
    estop.engage(allow={"user_ids": [OPERATOR]}, reason="window")

    assert _gate(user_id=OPERATOR, profile="some-other-profile") is None
    assert _gate(user_id=PEER, profile="other-lane") is not None


def test_profile_key_is_the_secondary_fallback_for_a_lane(hermes_home):
    """A maintenance lane can be admitted by profile when its user id is not the operator's."""
    estop.engage(allow={"profiles": ["primary-lane"]}, reason="window")

    assert _gate(user_id=PEER, profile="primary-lane") is None
    assert _gate(user_id=PEER, profile="other-lane") is not None


def test_missing_allowlist_admits_nobody(hermes_home):
    """Default must stay fail-closed: a plain `hermes pause` holds every turn, operator included."""
    estop.engage(reason="plain pause")

    assert _gate(user_id=OPERATOR) is not None


def test_allowlist_ids_are_compared_as_strings(hermes_home):
    """A numeric id from config and a string id from the platform are the same identity."""
    estop.engage(allow={"user_ids": [424242]}, reason="window")

    assert _gate(user_id="424242") is None


def test_expired_deadman_serves_every_turn(hermes_home):
    """Past expires_at the pause is gone, so the gate stops refusing — the fleet wakes itself."""
    estop.engage(reason="window", allow={"user_ids": [OPERATOR]})
    (hermes_home / "ESTOP").write_text(
        json.dumps({"reason": "window", "expires_at": _stamp(-60), "allow": {"user_ids": [OPERATOR]}}),
        encoding="utf-8")

    assert estop.is_engaged() is False
    assert _gate(user_id=PEER) is None


def test_no_sentinel_means_no_gate(hermes_home):
    assert _gate(user_id=PEER) is None
