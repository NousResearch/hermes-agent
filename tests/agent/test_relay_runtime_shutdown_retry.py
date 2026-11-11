"""A Relay runtime shutdown that could not run to the end can be requested again."""

from __future__ import annotations

import threading

import pytest

from agent import relay_runtime
from tests.agent.test_relay_runtime_plugins import _FakeRelay


@pytest.fixture(autouse=True)
def _static_plugin_config(tmp_path, monkeypatch):
    config = tmp_path / "plugins.toml"
    config.write_text(
        'version = 1\n\n[[components]]\nkind = "observability"\nenabled = true\n\n'
        "[components.config]\nversion = 4\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, str(config))
    relay_runtime._reset_for_tests()
    yield
    relay_runtime._reset_for_tests()


def _refuse_thread_start(self):
    raise RuntimeError("can't start new thread")


def test_shutdown_whose_deferred_close_could_not_start_leaves_the_runtime_open(monkeypatch, caplog):
    relay = _FakeRelay()
    host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")
    assert host.ensure_session({"session_id": "session"}) is not None
    lease = host.acquire_operation_lease()

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", _refuse_thread_start)
        host.shutdown()
    lease.release()
    failures = [record for record in caplog.records if "shutdown could not start" in record.getMessage()]
    assert len(failures) == 1 and failures[0].exc_info is not None
    assert ("plugin.activation.close",) not in relay.events
    # The runtime keeps serving: the open session survives and a new one starts.
    assert host.get_session("session") is not None
    assert host.ensure_session({"session_id": "next"}) is not None

    host.shutdown()
    assert ("plugin.activation.close",) in relay.events
    assert any(event[0] == "scope.pop" for event in relay.events)


def test_completed_shutdown_ignores_later_requests():
    relay = _FakeRelay()
    host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")
    assert host.ensure_session({"session_id": "session"}) is not None

    host.shutdown()
    events_after_first = list(relay.events)
    assert ("plugin.activation.close",) in events_after_first

    host.shutdown()
    assert relay.events == events_after_first
    assert host.ensure_session({"session_id": "late"}) is None
