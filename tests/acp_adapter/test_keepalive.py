"""Unit tests for acp_adapter.keepalive.TurnKeepalive."""
import time
from unittest.mock import MagicMock

import pytest

from acp_adapter.keepalive import (
    TurnKeepalive,
    get_keepalive_interval,
    make_turn_keepalive,
)


class DummyConn:
    def __init__(self):
        self.calls = []

    def session_update(self, session_id, update):
        self.calls.append((session_id, update))


class DummyLoop:
    pass


def _payload():
    return "payload"


def test_fires_after_interval():
    mock = MagicMock()
    k = TurnKeepalive(DummyConn(), "s1", DummyLoop(), interval_s=0.1, payload_factory=_payload)
    import acp_adapter.keepalive as mod
    orig = mod._send_update
    mod._send_update = lambda *a, **kw: mock()
    try:
        k.start()
        # Poll up to 2s for >=3 fires instead of a tight sleep — robust on
        # slow CI where a bare time.sleep(0.35) races the scheduler.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and mock.call_count < 3:
            time.sleep(0.02)
        assert mock.call_count >= 3, f"expected >=3 calls, got {mock.call_count}"
    finally:
        k.stop()
        mod._send_update = orig


def test_mark_activity_resets_timer():
    import acp_adapter.keepalive as mod
    calls = []
    orig = mod._send_update
    mod._send_update = lambda *a, **kw: calls.append(time.time())
    k = TurnKeepalive(DummyConn(), "s1", DummyLoop(), interval_s=0.1, payload_factory=_payload)
    try:
        k.start()
        # Repeatedly extend the deadline before it can fire.
        for _ in range(4):
            time.sleep(0.05)
            k.mark_activity()
        assert not calls, "should not have fired while activity kept resetting"
        time.sleep(0.20)
        assert calls, "should have fired after resets stopped"
    finally:
        k.stop()
        mod._send_update = orig


def test_stop_prevents_further_fires():
    import acp_adapter.keepalive as mod
    calls = []
    orig = mod._send_update
    mod._send_update = lambda *a, **kw: calls.append(time.time())
    k = TurnKeepalive(DummyConn(), "s1", DummyLoop(), interval_s=0.1, payload_factory=_payload)
    try:
        k.start()
        time.sleep(0.15)
        k.stop()
        c1 = len(calls)
        time.sleep(0.3)
        assert len(calls) == c1, "no further fires after stop"
    finally:
        k.stop()
        mod._send_update = orig


def test_stop_is_idempotent():
    k = TurnKeepalive(DummyConn(), "sess", DummyLoop(), interval_s=0.1)
    k.start()
    k.stop()
    k.stop()  # must not raise


def test_start_is_idempotent():
    import acp_adapter.keepalive as mod
    calls = []
    orig = mod._send_update
    mod._send_update = lambda *a, **kw: calls.append(1)
    k = TurnKeepalive(DummyConn(), "s", DummyLoop(), interval_s=0.1, payload_factory=_payload)
    try:
        k.start()
        k.start()  # second start must be a no-op — single worker thread
        assert k._thread is not None
        thread_id = k._thread.ident
        k.start()
        assert k._thread.ident == thread_id, "start() must not spawn extra threads"
        time.sleep(0.25)
        assert 1 <= len(calls) <= 3, "multiple uncoordinated timers present?"
    finally:
        k.stop()
        mod._send_update = orig


def test_config_disable(monkeypatch):
    """acp.keepalive_interval_s=0 in config.yaml disables the feature."""
    import acp_adapter.keepalive as mod
    monkeypatch.setattr(mod, "_read_config_interval", lambda: 0.0)
    assert make_turn_keepalive(DummyConn(), "sess", DummyLoop()) is None


def test_config_overrides_default(monkeypatch):
    """acp.keepalive_interval_s in config.yaml wins over the hardcoded default."""
    import acp_adapter.keepalive as mod
    monkeypatch.setattr(mod, "_read_config_interval", lambda: 12.5)
    assert get_keepalive_interval() == pytest.approx(12.5)


def test_default_when_no_config(monkeypatch):
    """With no config entry, falls back to the provided default."""
    import acp_adapter.keepalive as mod
    monkeypatch.setattr(mod, "_read_config_interval", lambda: None)
    assert get_keepalive_interval(default=45.0) == 45.0


def test_explicit_arg_wins_over_config(monkeypatch):
    """Explicit constructor arg takes precedence over config.yaml."""
    import acp_adapter.keepalive as mod
    monkeypatch.setattr(mod, "_read_config_interval", lambda: 12.5)
    k = TurnKeepalive(DummyConn(), "s", DummyLoop(), interval_s=0.1)
    assert k.interval_s == pytest.approx(0.1)
    k.stop()


def test_default_payload_is_valid_agent_message_chunk():
    from acp.schema import AgentMessageChunk, TextContentBlock

    payload = TurnKeepalive._default_payload_factory()
    assert isinstance(payload, AgentMessageChunk)
    assert payload.session_update == "agent_message_chunk"
    assert isinstance(payload.content, TextContentBlock)
    assert payload.content.text == ""


# --- _coerce_float hardening (addresses Copilot review on PR #75124) ---


def test_coerce_float_rejects_bool_true():
    """`true` in YAML must not silently become 1.0s keepalive."""
    from acp_adapter.keepalive import _coerce_float

    assert _coerce_float(True) is None


def test_coerce_float_rejects_bool_false():
    """`false` in YAML must not silently become 0.0 (disable)."""
    from acp_adapter.keepalive import _coerce_float

    assert _coerce_float(False) is None


def test_coerce_float_rejects_nan():
    """YAML `.nan` would break the keepalive loop (`remaining > 0` is False)."""
    from acp_adapter.keepalive import _coerce_float

    assert _coerce_float(float("nan")) is None


def test_coerce_float_rejects_inf():
    """YAML `.inf` would produce an infinite sleep — disable feature instead."""
    from acp_adapter.keepalive import _coerce_float

    assert _coerce_float(float("inf")) is None
    assert _coerce_float(float("-inf")) is None


def test_coerce_float_accepts_valid_numbers():
    """Sanity check: real ints and floats still coerce."""
    from acp_adapter.keepalive import _coerce_float

    assert _coerce_float(45) == 45.0
    assert _coerce_float(45.0) == 45.0
    assert _coerce_float("45") == 45.0
    assert _coerce_float(0) == 0.0  # 0 is a valid disable signal
    assert _coerce_float(None) is None
    assert _coerce_float("not-a-number") is None
