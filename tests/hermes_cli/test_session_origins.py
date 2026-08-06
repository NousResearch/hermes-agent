import json

from hermes_cli.session_origins import enrich_sessions_with_origins, session_origin_metadata
from hermes_state import SessionDB


def test_session_origin_metadata_keeps_dm_labels_private():
    metadata = session_origin_metadata(
        {
            "source": "telegram",
            "chat_type": "dm",
            "display_name": "+4712345678",
            "origin_json": json.dumps({"platform": "telegram", "chat_type": "dm", "chat_name": "Alice"}),
        }
    )

    assert metadata == {"platform": "telegram", "chat_type": "dm", "display_label": "Telegram DM"}


def test_session_origin_metadata_uses_non_identifier_room_label():
    metadata = session_origin_metadata(
        {
            "source": "discord",
            "chat_type": "channel",
            "display_name": "Production",
            "chat_id": "123",
            "origin_json": json.dumps({"platform": "discord", "chat_type": "channel", "chat_id": "123"}),
        }
    )

    assert metadata == {"platform": "discord", "chat_type": "channel", "display_label": "Production"}


def test_enrich_sessions_reads_gateway_origins_from_state_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("gateway-session", "telegram")
        db.record_gateway_session_peer(
            "gateway-session",
            source="telegram",
            session_key="agent:main:telegram:group:42",
            chat_id="42",
            chat_type="group",
            display_name="Lighting crew",
            origin_json=json.dumps({"platform": "telegram", "chat_type": "group", "chat_id": "42"}),
        )
        sessions = [{"id": "gateway-session"}, {"id": "local-session"}]

        enrich_sessions_with_origins(sessions, db)

        assert sessions[0]["origin"] == {
            "platform": "telegram",
            "chat_type": "group",
            "display_label": "Lighting crew",
        }
        assert "origin" not in sessions[1]
    finally:
        db.close()
