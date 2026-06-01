"""Tests for the previous-session bridge feature."""
from datetime import datetime, timezone

import pytest

from gateway.session import SessionEntry


def test_session_entry_carries_previous_session_id():
    entry = SessionEntry(
        session_key="signal:dm:user-abc",
        session_id="20260531_120000_aaaa",
        created_at=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
        previous_session_id="20260530_150000_bbbb",
    )
    assert entry.previous_session_id == "20260530_150000_bbbb"


def test_session_entry_round_trip_preserves_previous_session_id():
    entry = SessionEntry(
        session_key="k",
        session_id="new",
        created_at=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
        previous_session_id="old",
    )
    restored = SessionEntry.from_dict(entry.to_dict())
    assert restored.previous_session_id == "old"


def test_session_entry_default_previous_session_id_is_none():
    entry = SessionEntry(
        session_key="k",
        session_id="s",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    assert entry.previous_session_id is None


# ---------------------------------------------------------------------------
# Task 2: SessionStore.get_or_create_session populates previous_session_id
# ---------------------------------------------------------------------------

from pathlib import Path
from gateway.config import GatewayConfig, SessionResetPolicy
from gateway.session import SessionStore, SessionSource, Platform


def _src(uid="u1"):
    return SessionSource(
        platform=Platform.SIGNAL, chat_id="c1", user_id=uid, chat_type="dm"
    )


def test_auto_reset_populates_previous_session_id(tmp_path, monkeypatch):
    cfg = GatewayConfig()
    cfg.default_reset_policy = SessionResetPolicy(mode="idle", idle_minutes=1)

    store = SessionStore(sessions_dir=tmp_path / "sessions", config=cfg)
    src = _src()
    first = store.get_or_create_session(src)
    first.total_tokens = 500  # simulate activity
    # Force expiry by backdating updated_at (gateway uses naive local datetimes)
    from datetime import datetime, timedelta
    first.updated_at = datetime.now() - timedelta(hours=2)

    second = store.get_or_create_session(src)
    assert second.session_id != first.session_id
    assert second.was_auto_reset is True
    assert second.previous_session_id == first.session_id


def test_first_session_has_no_previous_session_id(tmp_path):
    cfg = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=cfg)
    entry = store.get_or_create_session(_src())
    assert entry.previous_session_id is None


def test_empty_session_rotation_does_not_set_previous_session_id(tmp_path):
    """If the prior session had zero activity, don't bridge — nothing useful to carry."""
    cfg = GatewayConfig()
    cfg.default_reset_policy = SessionResetPolicy(mode="idle", idle_minutes=1)
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=cfg)
    src = _src()
    first = store.get_or_create_session(src)
    # No total_tokens bump — simulates an empty session
    from datetime import datetime, timedelta
    first.updated_at = datetime.now() - timedelta(hours=2)

    second = store.get_or_create_session(src)
    assert second.was_auto_reset is True
    assert second.previous_session_id is None


# ---------------------------------------------------------------------------
# Task 3: PreviousSessionBridge config dataclass
# ---------------------------------------------------------------------------

def test_previous_session_bridge_defaults():
    from gateway.config import PreviousSessionBridge
    bridge = PreviousSessionBridge()
    assert bridge.enabled is True
    assert bridge.max_exchanges == 3
    assert bridge.max_chars == 4000


def test_gateway_config_has_previous_session_bridge():
    from gateway.config import GatewayConfig, PreviousSessionBridge
    cfg = GatewayConfig()
    assert isinstance(cfg.previous_session_bridge, PreviousSessionBridge)
    assert cfg.previous_session_bridge.enabled is True


def test_previous_session_bridge_round_trip():
    from gateway.config import PreviousSessionBridge
    bridge = PreviousSessionBridge(enabled=False, max_exchanges=5, max_chars=2000)
    restored = PreviousSessionBridge.from_dict(bridge.to_dict())
    assert restored.enabled is False
    assert restored.max_exchanges == 5
    assert restored.max_chars == 2000


def test_gateway_config_round_trip_preserves_bridge_settings():
    from gateway.config import GatewayConfig, PreviousSessionBridge
    cfg = GatewayConfig()
    cfg.previous_session_bridge = PreviousSessionBridge(
        enabled=False, max_exchanges=10, max_chars=8000
    )
    data = cfg.to_dict()
    restored = GatewayConfig.from_dict(data)
    assert restored.previous_session_bridge.enabled is False
    assert restored.previous_session_bridge.max_exchanges == 10
    assert restored.previous_session_bridge.max_chars == 8000


# ---------------------------------------------------------------------------
# Task 4: render_previous_session_tail
# ---------------------------------------------------------------------------

def _msg(role, content):
    return {"role": role, "content": content}


def test_render_tail_returns_empty_on_no_messages():
    from gateway.session import render_previous_session_tail
    assert render_previous_session_tail([], max_exchanges=3, max_chars=4000) == ""


def test_render_tail_only_includes_user_and_assistant():
    from gateway.session import render_previous_session_tail
    msgs = [
        _msg("system", "You are a helpful agent."),
        _msg("user", "hi"),
        _msg("assistant", "hello"),
        _msg("tool", "..."),
    ]
    out = render_previous_session_tail(msgs, max_exchanges=3, max_chars=4000)
    assert "You are a helpful agent" not in out
    assert "User: hi" in out
    assert "Assistant: hello" in out
    # The bare role "tool" shouldn't be labelled in the body
    assert "Tool:" not in out


def test_render_tail_skips_assistant_tool_calls_without_content():
    from gateway.session import render_previous_session_tail
    msgs = [
        _msg("user", "what's the weather?"),
        {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
        _msg("assistant", "It's 72 and sunny."),
    ]
    out = render_previous_session_tail(msgs, max_exchanges=5, max_chars=4000)
    assert "User: what's the weather?" in out
    assert "Assistant: It's 72 and sunny." in out


def test_render_tail_keeps_only_last_n_exchanges():
    from gateway.session import render_previous_session_tail
    msgs = []
    for i in range(10):
        msgs.append(_msg("user", f"q{i}"))
        msgs.append(_msg("assistant", f"a{i}"))
    out = render_previous_session_tail(msgs, max_exchanges=2, max_chars=10_000)
    assert "q9" in out and "a9" in out
    assert "q8" in out and "a8" in out
    assert "q7" not in out
    assert "q0" not in out


def test_render_tail_truncates_to_max_chars():
    from gateway.session import render_previous_session_tail
    msgs = [_msg("user", "x" * 5000), _msg("assistant", "y" * 5000)]
    out = render_previous_session_tail(msgs, max_exchanges=10, max_chars=200)
    # Header + intro + truncation marker, so the total cap is a bit over 200
    # but the *body* portion that came from messages must respect the cap.
    assert "truncated" in out.lower()


def test_render_tail_has_header():
    from gateway.session import render_previous_session_tail
    msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
    out = render_previous_session_tail(msgs, max_exchanges=3, max_chars=4000)
    assert out.startswith("## Previous Session Tail")
