from types import SimpleNamespace

from agent.cross_channel_context import build_cross_channel_context_block
from hermes_state import SessionDB


def test_cross_channel_context_disabled_returns_empty():
    agent = SimpleNamespace(_cross_channel_context_config={"enabled": False})

    assert build_cross_channel_context_block(agent) == ""


def test_cross_channel_context_does_not_read_sibling_profile_state(
    tmp_path, monkeypatch
):
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
        active_db.create_session(
            session_id="sknerus-local",
            source="telegram",
            chat_id="shopping",
            chat_type="group",
        )
        active_db.append_message("sknerus-local", "user", "same profile context")

        other_db.create_session(
            session_id="diodak-research",
            source="telegram",
            chat_id="shopping",
            chat_type="group",
        )
        other_db.append_message(
            "diodak-research", "user", "research product from TikTok"
        )
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

    assert "same profile context" in block
    assert "research product from TikTok" not in block
    assert "Insta360 Wave and Govee floor lamp" not in block
    assert "current follow-up" not in block


def test_cross_channel_context_block_is_stable_for_session(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    db = SessionDB(home / "state.db")
    try:
        db.create_session(session_id="active", source="cli")
        db.create_session(session_id="other", source="telegram")
        db.append_message("other", "user", "first sibling context")

        agent = SimpleNamespace(
            session_id="active",
            _session_db=db,
            _cross_channel_context_cache_key=None,
            _cross_channel_context_cache_block=None,
            _cross_channel_context_config={
                "enabled": True,
                "lookback_seconds": 3600,
                "max_sessions": 2,
                "max_messages_per_session": 4,
                "max_chars_per_message": 200,
            },
        )

        first = build_cross_channel_context_block(agent)
        db.append_message("other", "assistant", "newer sibling context")
        second = build_cross_channel_context_block(agent)

        agent.session_id = "fresh"
        third = build_cross_channel_context_block(agent)
    finally:
        db.close()

    assert first == second
    assert "first sibling context" in second
    assert "newer sibling context" not in second
    assert "newer sibling context" in third


def test_cross_channel_context_cache_is_scoped_to_active_home(tmp_path, monkeypatch):
    first_home = tmp_path / "hermes" / "profiles" / "first"
    second_home = tmp_path / "hermes" / "profiles" / "second"
    first_home.mkdir(parents=True)
    second_home.mkdir(parents=True)
    first_db = SessionDB(first_home / "state.db")
    second_db = SessionDB(second_home / "state.db")
    try:
        for db, content in (
            (first_db, "first profile context"),
            (second_db, "second profile context"),
        ):
            db.create_session(session_id="active", source="cli")
            db.create_session(session_id="other", source="telegram")
            db.append_message("other", "user", content)

        monkeypatch.setenv("HERMES_HOME", str(first_home))
        agent = SimpleNamespace(
            session_id="active",
            _session_db=first_db,
            _cross_channel_context_config={"enabled": True},
        )
        first = build_cross_channel_context_block(agent)

        monkeypatch.setenv("HERMES_HOME", str(second_home))
        agent._session_db = second_db
        second = build_cross_channel_context_block(agent)
    finally:
        first_db.close()
        second_db.close()

    assert "first profile context" in first
    assert "second profile context" in second
    assert "first profile context" not in second
