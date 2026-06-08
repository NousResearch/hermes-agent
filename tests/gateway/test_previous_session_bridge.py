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


def test_config_yaml_previous_session_bridge_mapping(tmp_path, monkeypatch):
    """config.yaml -> GatewayConfig.previous_session_bridge round-trips."""
    import yaml
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    config_yaml = hermes_home / "config.yaml"
    config_yaml.write_text(yaml.safe_dump({
        "previous_session_bridge": {
            "enabled": False,
            "max_exchanges": 7,
            "max_chars": 1234,
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    cfg = load_gateway_config()
    assert cfg.previous_session_bridge.enabled is False
    assert cfg.previous_session_bridge.max_exchanges == 7
    assert cfg.previous_session_bridge.max_chars == 1234


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


# ---------------------------------------------------------------------------
# Task 5: build_session_context_prompt accepts previous_session_tail
# ---------------------------------------------------------------------------

def test_build_session_context_prompt_includes_tail_when_provided():
    from gateway.session import (
        SessionContext, SessionSource, Platform,
        build_session_context_prompt,
    )
    ctx = SessionContext(
        source=SessionSource(
            platform=Platform.SIGNAL, chat_id="c1", user_id="u1", chat_type="dm",
            user_name="Eugene",
        ),
        connected_platforms=[Platform.SIGNAL],
        home_channels={},
    )
    out = build_session_context_prompt(
        ctx, redact_pii=False,
        previous_session_tail="## Previous Session Tail\n\nintro\n\nUser: hi\n\nAssistant: hello",
    )
    assert "## Previous Session Tail" in out
    assert "User: hi" in out
    assert "Assistant: hello" in out


def test_build_session_context_prompt_works_without_tail():
    from gateway.session import (
        SessionContext, SessionSource, Platform,
        build_session_context_prompt,
    )
    ctx = SessionContext(
        source=SessionSource(
            platform=Platform.SIGNAL, chat_id="c1", user_id="u1", chat_type="dm",
            user_name="Eugene",
        ),
        connected_platforms=[Platform.SIGNAL],
        home_channels={},
    )
    out = build_session_context_prompt(ctx, redact_pii=False)
    assert "## Previous Session Tail" not in out


def test_build_session_context_prompt_ignores_empty_tail():
    from gateway.session import (
        SessionContext, SessionSource, Platform,
        build_session_context_prompt,
    )
    ctx = SessionContext(
        source=SessionSource(
            platform=Platform.SIGNAL, chat_id="c1", user_id="u1", chat_type="dm",
            user_name="Eugene",
        ),
        connected_platforms=[Platform.SIGNAL],
        home_channels={},
    )
    out = build_session_context_prompt(ctx, redact_pii=False, previous_session_tail="")
    assert "## Previous Session Tail" not in out


# ---------------------------------------------------------------------------
# Task 6: End-to-end integration — SessionDB → bridge → context prompt
# ---------------------------------------------------------------------------

def test_end_to_end_bridge(tmp_path, monkeypatch):
    """
    Full path: write messages to SessionDB → rotate the session in SessionStore
    → render_previous_session_tail surfaces the prior turns.
    """
    from datetime import datetime, timedelta

    monkeypatch.setattr("hermes_state.get_hermes_home", lambda: tmp_path)

    from hermes_state import SessionDB
    from gateway.config import GatewayConfig, SessionResetPolicy
    from gateway.session import (
        SessionStore, SessionSource, Platform,
        render_previous_session_tail,
    )

    db = SessionDB()
    cfg = GatewayConfig()
    cfg.default_reset_policy = SessionResetPolicy(mode="idle", idle_minutes=1)

    store = SessionStore(sessions_dir=tmp_path / "sessions", config=cfg)
    src = SessionSource(
        platform=Platform.SIGNAL, chat_id="c1", user_id="u1", chat_type="dm",
    )

    first = store.get_or_create_session(src)
    db.create_session(session_id=first.session_id, source="signal", user_id="u1")
    db.append_message(first.session_id, "user", "what's the weather tomorrow?")
    db.append_message(
        first.session_id, "assistant", "Should I send the draft? Reply yes/no.",
    )
    # Simulate activity + expiry
    first.total_tokens = 500
    first.updated_at = datetime.now() - timedelta(hours=2)

    second = store.get_or_create_session(src)
    assert second.was_auto_reset is True
    assert second.previous_session_id == first.session_id

    tail = render_previous_session_tail(
        db.get_messages(first.session_id),
        max_exchanges=3, max_chars=4000,
    )
    assert "## Previous Session Tail" in tail
    assert "weather tomorrow" in tail
    assert "Reply yes/no" in tail


# ---------------------------------------------------------------------------
# Issue 1, 2, 5: should_bridge_previous_session predicate
# ---------------------------------------------------------------------------

def test_predicate_returns_true_when_all_conditions_met():
    from gateway.session import should_bridge_previous_session
    assert should_bridge_previous_session(
        bridge_enabled=True,
        was_auto_reset=True,
        previous_session_id="20260530_120000_abc",
        has_session_db=True,
        shared_multi_user_session=False,
        redact_pii=False,
    ) is True


def test_predicate_skips_when_bridge_disabled():
    from gateway.session import should_bridge_previous_session
    assert should_bridge_previous_session(
        bridge_enabled=False, was_auto_reset=True, previous_session_id="x",
        has_session_db=True, shared_multi_user_session=False, redact_pii=False,
    ) is False


def test_predicate_skips_when_not_auto_reset():
    from gateway.session import should_bridge_previous_session
    assert should_bridge_previous_session(
        bridge_enabled=True, was_auto_reset=False, previous_session_id="x",
        has_session_db=True, shared_multi_user_session=False, redact_pii=False,
    ) is False


def test_predicate_skips_when_previous_session_id_none():
    from gateway.session import should_bridge_previous_session
    assert should_bridge_previous_session(
        bridge_enabled=True, was_auto_reset=True, previous_session_id=None,
        has_session_db=True, shared_multi_user_session=False, redact_pii=False,
    ) is False


def test_predicate_skips_when_no_session_db():
    from gateway.session import should_bridge_previous_session
    assert should_bridge_previous_session(
        bridge_enabled=True, was_auto_reset=True, previous_session_id="x",
        has_session_db=False, shared_multi_user_session=False, redact_pii=False,
    ) is False


def test_predicate_skips_shared_multi_user_session():
    """Issue 1: shared sessions leak user A's turns into user B's prompt."""
    from gateway.session import should_bridge_previous_session
    assert should_bridge_previous_session(
        bridge_enabled=True, was_auto_reset=True, previous_session_id="x",
        has_session_db=True, shared_multi_user_session=True, redact_pii=False,
    ) is False


def test_predicate_skips_when_redact_pii_enabled():
    """Issue 2: raw message bodies in SessionDB bypass PII redaction layer."""
    from gateway.session import should_bridge_previous_session
    assert should_bridge_previous_session(
        bridge_enabled=True, was_auto_reset=True, previous_session_id="x",
        has_session_db=True, shared_multi_user_session=False, redact_pii=True,
    ) is False


# ---------------------------------------------------------------------------
# Issue 4: max_chars total-length bound
# ---------------------------------------------------------------------------

def test_render_tail_total_length_bounded():
    """Total rendered output stays within max_chars + a small framing overhead."""
    from gateway.session import render_previous_session_tail
    msgs = [_msg("user", "x" * 10_000), _msg("assistant", "y" * 10_000)]
    out = render_previous_session_tail(msgs, max_exchanges=10, max_chars=500)
    # Body cap is 500; header + intro + truncation marker < 200 chars.
    # Total should not exceed 700.
    assert len(out) < 700, f"rendered tail is {len(out)} chars, expected < 700"


# ---------------------------------------------------------------------------
# Issue 7: bridge no-ops when prior session messages were deleted from SQLite
# ---------------------------------------------------------------------------

def test_render_tail_handles_empty_message_list_from_db():
    """Simulates: prior session_id is stamped on the entry, but its messages
    were deleted from SessionDB before the next turn arrived (e.g. via
    hermes sessions delete or manual cleanup)."""
    from gateway.session import render_previous_session_tail
    # An empty list — what SessionDB.get_messages returns for a deleted session
    assert render_previous_session_tail([], max_exchanges=3, max_chars=4000) == ""


# ---------------------------------------------------------------------------
# Issue 3: cache invariant — bridge fires once, not on subsequent turns
# ---------------------------------------------------------------------------

def test_session_entry_was_auto_reset_clears_to_false_for_subsequent_turns():
    """The gateway must clear was_auto_reset after the first turn so the
    bridge predicate returns False on turn 2+. SessionEntry exposes the
    field as a normal attribute; verify it's mutable for that purpose."""
    from datetime import datetime
    from gateway.session import SessionEntry, should_bridge_previous_session
    entry = SessionEntry(
        session_key="k", session_id="s",
        created_at=datetime.now(), updated_at=datetime.now(),
        was_auto_reset=True, previous_session_id="prior",
    )
    # Turn 1: predicate sees was_auto_reset=True
    assert should_bridge_previous_session(
        bridge_enabled=True, was_auto_reset=entry.was_auto_reset,
        previous_session_id=entry.previous_session_id,
        has_session_db=True, shared_multi_user_session=False, redact_pii=False,
    ) is True

    # Gateway clears the flag after building the prompt
    entry.was_auto_reset = False

    # Turn 2: predicate now returns False, no bridge re-emission
    assert should_bridge_previous_session(
        bridge_enabled=True, was_auto_reset=entry.was_auto_reset,
        previous_session_id=entry.previous_session_id,
        has_session_db=True, shared_multi_user_session=False, redact_pii=False,
    ) is False
