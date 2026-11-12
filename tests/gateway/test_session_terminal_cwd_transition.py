"""Non-boundary chat route repoints retain the persisted local terminal cwd (#41128)."""

from unittest.mock import patch

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


def test_nonboundary_repoint_keeps_cwd_but_new_discards_it(tmp_path, monkeypatch):
    import hermes_state

    def no_sqlite():
        raise RuntimeError("SQLite disabled in test")

    monkeypatch.setattr(hermes_state, "SessionDB", no_sqlite)
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    store._loaded = True

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm",
                           user_id="user", user_name="tester")
    entry = store.get_or_create_session(source)
    project = tmp_path / "project"
    project.mkdir()
    assert store.set_session_metadata(entry.session_key, "terminal_cwd", str(project))

    # A compression-tip/ownership repoint is not a new conversation.
    switched = store.switch_session(entry.session_key, "continuation-id")
    assert switched is not None
    assert switched.metadata.get("terminal_cwd") == str(project)

    # /resume (and handoff) opt out: the conversation it lands on doesn't inherit this one's cwd.
    resumed = store.switch_session(entry.session_key, "other-conversation", preserve_prompt_pin=False)
    assert resumed is not None and "terminal_cwd" not in resumed.metadata
    assert store.set_session_metadata(entry.session_key, "terminal_cwd", str(project))

    # /new is a deliberate conversation boundary; the older cwd must not leak.
    fresh = store.reset_session(entry.session_key)
    assert fresh is not None
    assert "terminal_cwd" not in fresh.metadata
