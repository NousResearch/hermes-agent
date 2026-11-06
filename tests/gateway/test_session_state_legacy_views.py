"""Ratchet for the SessionState migration: migrated per-session dict names stay gone.

Each name below used to be a legacy dict view on GatewayRunner; its state now lives only on
``SessionState`` (``runner._session_state(key)``), so there is one way to reach it.
"""

import pytest

from gateway import session_state
from gateway.run import GatewayRunner

MIGRATED = (
    "_active_session_leases",
    "_turn_lease_tokens",
    "_pending_one_turn_model_restores",
    "_session_service_tier_overrides",
    "_pending_turn_sidecar_notes",
    "_pending_native_image_paths_by_session",
    "_session_ephemeral_pin",
    "_session_vc_last",
)


@pytest.mark.parametrize("name", MIGRATED)
def test_migrated_legacy_view_is_gone(name):
    assert not hasattr(GatewayRunner, name)
    assert name not in session_state.LEGACY_FIELD_SPECS


def test_lease_token_view_helpers_are_gone():
    assert not hasattr(session_state, "legacy_lease_token_property")
    assert not hasattr(session_state, "TurnLeaseTokenView")


def test_remaining_legacy_view_still_reads_session_state():
    runner = object.__new__(GatewayRunner)
    runner._session_state("k").turn.agent = sentinel = object()
    assert runner._running_agents["k"] is sentinel
