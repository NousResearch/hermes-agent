from types import SimpleNamespace

from agent.cross_channel_context import build_cross_channel_context_block
from hermes_state import SessionDB


def test_cross_channel_context_disabled_returns_empty():
    agent = SimpleNamespace(_cross_channel_context_config={"enabled": False})

    assert build_cross_channel_context_block(agent) == ""


def test_cross_channel_context_reads_sibling_profile_state(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    active_home = root / "profiles" / "sknerus"
    other_home = root / "profiles" / "diodak"
    active_home.mkdir(parents=True)
    other_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(active_home))

    active_db = SessionDB(active_home / "state.db")
    other_db = SessionDB(other_home / "state.db")
    try:
        active_db.create_session(
            session_id="sknerus-active",
            source="telegram",
            chat_id="shopping",
            chat_type="group",
        )
        active_db.append_message("sknerus-active", "user", "current follow-up")

        other_db.create_session(
            session_id="diodak-research",
            source="telegram",
            chat_id="shopping",
            chat_type="group",
        )
        other_db.append_message("diodak-research", "user", "research product from TikTok")
        other_db.append_message(
            "diodak-research",
            "assistant",
            "Added Insta360 Wave and Govee floor lamp to the Hub shopping list.",
        )

        agent = SimpleNamespace(
            session_id="sknerus-active",
            _session_db=active_db,
            _cross_channel_context_config={
                "enabled": True,
                "lookback_seconds": 3600,
                "max_sessions": 3,
                "max_messages_per_session": 3,
                "max_chars_per_message": 200,
                "include_profiles": True,
            },
        )

        block = build_cross_channel_context_block(agent)
    finally:
        active_db.close()
        other_db.close()

    assert "profile diodak" in block
    assert "research product from TikTok" in block
    assert "Insta360 Wave and Govee floor lamp" in block
    assert "current follow-up" not in block

