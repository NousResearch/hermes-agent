from __future__ import annotations

import logging

from agent import relay_runtime
from agent.relay_runtime import NoopRelayRuntime, RelayHostRegistry


def _raise(exc: Exception):
    def _init(self, *, profile_key: str | None = None) -> None:
        raise exc

    return _init


def test_missing_nemo_relay_wheel_falls_back_quietly(monkeypatch, caplog):
    missing = ModuleNotFoundError("No module named 'nemo_relay'")
    missing.name = "nemo_relay"
    monkeypatch.setattr(relay_runtime, "RelayRuntime", type("_R", (), {"__init__": _raise(missing)}))

    registry = RelayHostRegistry()
    with caplog.at_level(logging.DEBUG, logger="agent.relay_runtime"):
        host = registry.for_profile("intel-mac-profile")

    assert isinstance(host, NoopRelayRuntime)
    assert host.reason == str(missing)
    assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)
    assert registry.for_profile("intel-mac-profile") is host


def test_genuine_relay_init_failure_stays_a_warning_with_traceback(monkeypatch, caplog):
    failure = RuntimeError("Relay plugin initialization exploded")
    monkeypatch.setattr(relay_runtime, "RelayRuntime", type("_R", (), {"__init__": _raise(failure)}))

    registry = RelayHostRegistry()
    with caplog.at_level(logging.DEBUG, logger="agent.relay_runtime"):
        host = registry.for_profile("broken-init-profile")

    assert isinstance(host, NoopRelayRuntime)
    assert host.reason == str(failure)
    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None


def test_transitive_module_not_found_inside_installed_nemo_relay_stays_a_warning(monkeypatch, caplog):
    transitive = ModuleNotFoundError("No module named 'nemo_relay._native'")
    transitive.name = "nemo_relay._native"
    monkeypatch.setattr(relay_runtime, "RelayRuntime", type("_R", (), {"__init__": _raise(transitive)}))

    registry = RelayHostRegistry()
    with caplog.at_level(logging.DEBUG, logger="agent.relay_runtime"):
        host = registry.for_profile("partial-install-profile")

    assert isinstance(host, NoopRelayRuntime)
    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None
