"""A profile's external start and terminal retirement use the same S6 fence."""

import sys
from unittest.mock import Mock

import pytest

from hermes_cli.container_boot import _read_desired_state
from hermes_cli.s6_profile_lifecycle import (
    GatewayRetiredError,
    reconcile_profile_gateway,
)
from hermes_cli.service_manager import S6ServiceManager

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="S6 uses POSIX file locks"
)


@pytest.fixture
def gateway(tmp_path):
    profile_dir = tmp_path / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    slot = tmp_path / "gateway-worker"
    slot.mkdir()
    (slot / "down").touch()
    manager = Mock(spec=S6ServiceManager)
    manager.scandir = tmp_path
    manager.gateway_supervisor_state.return_value = (False, False)
    manager.gateway_fully_down.return_value = True
    return profile_dir, manager


def test_retirement_waits_for_accepted_start_then_supervisor_shutdown(gateway):
    profile_dir, manager = gateway

    def reconcile(desired):
        return reconcile_profile_gateway(manager, "worker", profile_dir, desired)

    assert not reconcile("running")
    manager.start.assert_called_once()
    assert isinstance(manager.start.call_args.kwargs["lock_fd"], int)

    # The initial down sample may predate the queued 'up' command.
    assert not reconcile("retired")
    manager.stop.assert_not_called()
    manager.gateway_fully_down.assert_not_called()

    manager.gateway_supervisor_state.return_value = (True, True)
    assert not reconcile("retired")
    manager.stop.assert_called_once()
    assert manager.stop.call_args.kwargs["persistent"] is True
    assert _read_desired_state(profile_dir) == "stopped"

    # A fresh request reads the same persisted fence without process-local state.
    with pytest.raises(GatewayRetiredError):
        reconcile("running")
    manager.start.assert_called_once()

    manager.gateway_supervisor_state.return_value = (False, False)
    manager.gateway_fully_down.return_value = False
    assert not reconcile("retired")
    manager.gateway_fully_down.return_value = True
    assert reconcile("retired")


def test_intent_and_lock_do_not_cross_profile_boundaries(tmp_path, gateway):
    import fcntl

    profile_dir, manager = gateway
    intent_lock = profile_dir / ".gateway-lifecycle-intent.lock"
    other = tmp_path / "profiles" / "other"
    other.mkdir()
    slot = tmp_path / "gateway-other"
    slot.mkdir()
    (slot / "down").touch()
    with intent_lock.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert not reconcile_profile_gateway(manager, "worker", profile_dir, "retired")
        assert reconcile_profile_gateway(manager, "other", other, "retired")
    assert _read_desired_state(other) == "stopped"
    assert _read_desired_state(profile_dir) is None
    assert reconcile_profile_gateway(manager, "worker", profile_dir, "retired")
