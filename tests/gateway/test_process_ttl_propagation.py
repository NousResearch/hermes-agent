"""Tests for gateway/run.py — finished-process TTL propagation into the
process registry.

The process registry is a module-level singleton. The gateway pushes
``session_reset.finished_process_ttl_minutes`` into it once at startup via
``_apply_finished_process_ttl()``. Scope contract (documented at the call
site): the TTL is PROCESS-WIDE — multiplexed profiles share the registry's
prune cadence; only the primary profile's default reset policy is applied.
"""

import pytest

from gateway.config import SessionResetPolicy
from tools.process_registry import get_finished_ttl_seconds, set_finished_ttl_seconds


@pytest.fixture()
def reset_ttl():
    original = get_finished_ttl_seconds()
    yield
    set_finished_ttl_seconds(original)


class TestApplyFinishedProcessTtl:
    def test_propagates_policy_value_to_registry(self, reset_ttl):
        """The gateway's startup helper must push the configured minutes into
        the registry as seconds."""
        from gateway.run import _apply_finished_process_ttl

        policy = SessionResetPolicy(finished_process_ttl_minutes=15)
        _apply_finished_process_ttl(policy)

        assert get_finished_ttl_seconds() == 15 * 60

    def test_default_when_policy_lacks_ttl(self, reset_ttl):
        """Policies that predate the field (or omit it) fall back to the
        registry default rather than clobbering with 0."""
        from gateway.run import _apply_finished_process_ttl

        policy = SessionResetPolicy()  # finished_process_ttl_minutes == 10
        _apply_finished_process_ttl(policy)

        assert get_finished_ttl_seconds() == 10 * 60

    def test_nonpositive_value_keeps_previous_ttl(self, reset_ttl):
        """A 0/negative value is ignored (the registry setter's guard) so a
        malformed config cannot zero out the prune TTL."""
        from gateway.run import _apply_finished_process_ttl

        policy = SessionResetPolicy(finished_process_ttl_minutes=0)
        before = get_finished_ttl_seconds()
        _apply_finished_process_ttl(policy)

        assert get_finished_ttl_seconds() == before


class TestMultiplexScope:
    def test_registry_is_process_wide_singleton(self):
        """The registry is not partitioned by profile — this is the documented
        process-wide scope. A multiplex profile's own policy would have to go
        through the same singleton, so per-profile TTLs are not supported."""
        from tools.process_registry import process_registry

        assert hasattr(process_registry, "_finished")
        assert hasattr(process_registry, "_running")
