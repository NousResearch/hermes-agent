"""Supplementary unit guards. Real transport evidence lives in the integration harness.
These mocks are NOT evidence that a transport can recover.
"""

from types import SimpleNamespace
from unittest.mock import Mock
import threading
import pytest
from cron import scheduler
from tools import (
    mcp_tool as core,
    mcp_tool_discovery as discovery,
    mcp_tool_loop as loop,
)
from tools import mcp_tool_registration as registration
from tools.registry import registry


def server(*, session=None, ready=False, names=(), ever=True, park=None):
    s = SimpleNamespace(
        session=session,
        _ever_connected=ever,
        _park_reason=park,
        _registered_tool_names=list(names),
        _tools=[],
        _ready=threading.Event(),
        _reconnect_event=threading.Event(),
        _task=SimpleNamespace(done=lambda: False),
    )
    if ready:
        s._ready.set()
    return s


def setup(monkeypatch, servers, configs=None):
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(scheduler.time, "monotonic", lambda: clock[0])

    def sleep(n):
        clock[0] += n
        sleeps.append(n)

    monkeypatch.setattr(scheduler.time, "sleep", sleep)
    monkeypatch.setattr(core, "_servers", servers)
    monkeypatch.setattr("tools.mcp_tool_scope._resolve_server_key", lambda n: n)
    monkeypatch.setattr(
        discovery._config,
        "_load_mcp_config",
        lambda: configs or {n: {} for n in servers},
    )
    discover = Mock(return_value=[])
    monkeypatch.setattr(discovery, "discover_mcp_tools", discover)
    signal = Mock(side_effect=AssertionError("must not reconnect a second time"))
    monkeypatch.setattr(loop, "_signal_reconnect_and_wait", signal)
    monkeypatch.setattr(registration, "_existing_tool_names", list)
    return clock, sleeps, discover, signal


def test_waits_for_registered_handler_not_session(monkeypatch):
    s = server(session=object(), ready=True, names=["mcp__ink__hello"])
    clock, _sleeps, discover, signal = setup(monkeypatch, {"ink": s})
    monkeypatch.setattr(
        registry,
        "get_entry",
        lambda n: SimpleNamespace(handler=lambda: None) if clock[0] >= 0.2 else None,
    )
    scheduler._init_cron_mcp_tools("test")
    assert 0.2 <= clock[0] < 0.3
    assert discover.call_count == 1 and not signal.called


def test_waits_for_tools_list_ready(monkeypatch):
    s = server(session=object())
    clock, _sleeps, discover, _signal = setup(monkeypatch, {"ink": s})
    original = scheduler.time.sleep

    def sleep(n):
        original(n)
        if clock[0] >= 0.2:
            s._ready.set()

    monkeypatch.setattr(scheduler.time, "sleep", sleep)
    scheduler._init_cron_mcp_tools("test")
    assert 0.2 <= clock[0] < 0.3 and discover.call_count == 1


def test_multiple_servers_share_deadline(monkeypatch):
    clock, sleeps, discover, signal = setup(
        monkeypatch, {str(i): server() for i in range(8)}
    )
    scheduler._init_cron_mcp_tools("test")
    assert clock[0] == pytest.approx(15) and sum(sleeps) == pytest.approx(15)
    assert discover.call_count == 1 and not signal.called


def test_discovery_consumes_same_budget(monkeypatch):
    clock, sleeps, discover, _signal = setup(monkeypatch, {"ink": server()})
    discover.side_effect = lambda: clock.__setitem__(0, 12)
    scheduler._init_cron_mcp_tools("test")
    assert clock[0] == pytest.approx(15) and sum(sleeps) == pytest.approx(3)


@pytest.mark.parametrize(
    "state", ["healthy", "permanent", "never", "disabled", "finished", "zero-filtered"]
)
def test_no_unnecessary_wait(monkeypatch, state):
    s = server(
        session=object() if state in ("healthy", "zero-filtered") else None,
        ready=state in ("healthy", "zero-filtered"),
        ever=state != "never",
        park="permanent error" if state == "permanent" else None,
    )
    if state == "finished":
        s._task = SimpleNamespace(done=lambda: True)
    configs = {"ink": {"enabled": state != "disabled"}}
    if state == "zero-filtered":
        s._tools = [SimpleNamespace(name="hello")]
        configs["ink"]["tools"] = {"include": []}
    _clock, sleeps, discover, signal = setup(monkeypatch, {"ink": s}, configs)
    scheduler._init_cron_mcp_tools("test")
    assert not sleeps and discover.call_count == 1 and not signal.called


def test_stale_ready_session_with_deregistered_tools_is_not_ready(monkeypatch):
    s = server(session=object(), ready=True)
    s._tools = [SimpleNamespace(name="hello")]
    clock, _sleeps, discover, _signal = setup(monkeypatch, {"ink": s})
    scheduler._init_cron_mcp_tools("test")
    assert clock[0] == pytest.approx(15) and discover.call_count == 1


def test_profile_resolution_does_not_wait_on_foreign_server(monkeypatch):
    healthy = server(session=object(), ready=True)
    _clock, sleeps, _discover, _signal = setup(
        monkeypatch, {"ink": server(), ("current", "ink"): healthy}
    )
    monkeypatch.setattr(
        "tools.mcp_tool_scope._resolve_server_key", lambda n: ("current", n)
    )
    monkeypatch.setattr(discovery._config, "_load_mcp_config", lambda: {"ink": {}})
    scheduler._init_cron_mcp_tools("test")
    assert not sleeps
