"""Lifecycle guard: a supervised gateway may manage a SIBLING profile's gateway service.

Branches B/C match any gateway label, but only this process's own launchd job / systemd unit can
SIGTERM it; a break-glass profile must be able to kickstart a wedged sibling. Identity comes only
from the supervisor (launchd XPC_SERVICE_NAME, systemd cgroup unit); without it nothing changes.
"""
import pytest

import cron.lifecycle_guard as guard
from cron.lifecycle_guard import contains_gateway_lifecycle_command_or_referenced_script as blocked

SIB = "ai.hermes.gateway-aegis"


@pytest.fixture
def as_default_gateway(monkeypatch):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.setenv("XPC_SERVICE_NAME", "ai.hermes.gateway")


@pytest.fixture
def no_identity(monkeypatch):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)


@pytest.mark.parametrize("command", [
    f"launchctl kickstart -k gui/501/{SIB}",
    f"launchctl bootout gui/501/{SIB}",
    "systemctl --user restart hermes-gateway-aegis.service",
])
def test_sibling_service_is_allowed_for_a_supervised_gateway(as_default_gateway, command):
    assert not blocked(command), command


@pytest.mark.parametrize("command", [
    "launchctl kickstart -k gui/501/ai.hermes.gateway",
    "systemctl --user restart hermes-gateway",  # same profile, other supervisor namespace
    f"launchctl kickstart -k gui/501/{SIB}; launchctl kickstart -k gui/501/ai.hermes.gateway",
    f"L={SIB}; launchctl kickstart -k gui/501/$L",
    f"pkill -f {SIB}",
    f"launchctl kickstart -k gui/501/{SIB} && hermes gateway restart",
    f"launchctl submit -l {SIB} -- /bin/sh restart.sh",
    "launchctl kickstart -k gui/501/ai.hermes.gatewayx",
])
def test_self_mixed_or_ambiguous_targets_stay_blocked(as_default_gateway, command):
    assert blocked(command), command


def test_sibling_view_is_symmetric(monkeypatch):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.setenv("XPC_SERVICE_NAME", SIB)
    assert not blocked("launchctl kickstart -k gui/501/ai.hermes.gateway")
    assert blocked(f"launchctl kickstart -k gui/501/{SIB}")


def test_without_supervisor_identity_every_gateway_label_stays_blocked(no_identity):
    assert guard._self_gateway_service_names() == frozenset()
    assert blocked(f"launchctl kickstart -k gui/501/{SIB}")
