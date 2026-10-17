"""Invariant tests for the gateway-state test-isolation guard.

The guard exists because a gateway test writing the LIVE ``~/.hermes/gateway_state.json``
leaves a dead pytest pid as the live gateway's platform writer (the feishu leak) and
could ``--replace`` the live daemon. Production side (``gateway/status.py``) fails closed
when a test-context process resolves the production home; the harness side
(``tests/gateway/conftest.py``) refuses a live ``HERMES_HOME`` and verifies the live state
is untouched. These tests pin the production-guard behaviour contract.
"""

import pytest

from gateway.status import _get_process_hermes_home
from hermes_state_guard import _real_platform_state_root


def _live_home():
    home = _real_platform_state_root()
    assert home is not None, "could not resolve the real production home"
    return home


@pytest.mark.parametrize("suffix", ["", "profiles/some-profile"])
def test_refuses_production_or_profile_home_under_test(monkeypatch, suffix):
    """A test-context process resolving the live home (or a profile under it) for a
    gateway identity file must fail closed — that is the feishu-writer leak."""
    live = _live_home()
    target = live if not suffix else live / suffix
    monkeypatch.setenv("HERMES_HOME", str(target))
    with pytest.raises(RuntimeError, match="Refusing to resolve the production Hermes home"):
        _get_process_hermes_home()


def test_allows_scratch_home_under_test(monkeypatch, tmp_path):
    """A scratch HERMES_HOME is not the production home, so the guard passes it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert _get_process_hermes_home() == tmp_path


def test_bypass_env_disarms_the_guard(monkeypatch):
    """HERMES_GATEWAY_STATE_GUARD_BYPASS=1 is the explicit opt-out."""
    live = _live_home()
    monkeypatch.setenv("HERMES_HOME", str(live))
    monkeypatch.setenv("HERMES_GATEWAY_STATE_GUARD_BYPASS", "1")
    assert _get_process_hermes_home() == live
