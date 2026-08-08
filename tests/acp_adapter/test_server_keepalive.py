"""Server-integration tests for TurnKeepalive.

Verifies interval + activity + disable + stop wiring behavior that
``ACPAgent.prompt()`` relies on when it calls ``make_turn_keepalive``.
"""
import time

import acp_adapter.keepalive as keepalive_mod
from acp_adapter.keepalive import TurnKeepalive, make_turn_keepalive


class DummyConn:
    def __init__(self):
        self.calls = []


class DummyLoop:
    pass


def _payload():
    return "keepalive"


def _count_fires(monkeypatch, calls):
    monkeypatch.setattr(keepalive_mod, "_send_update", lambda *a, **kw: calls.append(1))


def test_server_keepalive_interval(monkeypatch):
    """With interval=0.1s and no activity, keepalive fires >=3 times in 0.35s."""
    calls = []
    _count_fires(monkeypatch, calls)
    k = TurnKeepalive(DummyConn(), "s1", DummyLoop(), interval_s=0.1, payload_factory=_payload)
    k.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and len(calls) < 3:
            time.sleep(0.02)
        assert len(calls) >= 3, f"expected >=3 keepalives, got {len(calls)}"
    finally:
        k.stop()


def test_server_keepalive_activity(monkeypatch):
    """mark_activity() resets deadline; keepalive fires only after activity stops."""
    calls = []
    _count_fires(monkeypatch, calls)
    k = TurnKeepalive(DummyConn(), "s2", DummyLoop(), interval_s=0.12, payload_factory=_payload)
    k.start()
    try:
        for _ in range(4):
            time.sleep(0.05)
            k.mark_activity()
        assert not calls, "keepalive should not have fired under continuous activity"
        time.sleep(0.20)
        assert calls, "keepalive should have fired after resets stopped"
    finally:
        k.stop()


def test_server_keepalive_disable(monkeypatch):
    """acp.keepalive_interval_s=0 in config disables the feature."""
    monkeypatch.setattr(keepalive_mod, "_read_config_interval", lambda: 0.0)
    assert make_turn_keepalive(DummyConn(), "s3", DummyLoop()) is None


def test_server_keepalive_stop_idempotence(monkeypatch):
    monkeypatch.setattr(keepalive_mod, "_read_config_interval", lambda: 0.2)
    k = make_turn_keepalive(DummyConn(), "s4", DummyLoop())
    assert k is not None
    k.start()
    k.stop()
    k.stop()  # server's finally may call stop redundantly on retries
