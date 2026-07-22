from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


def test_historical_media_delivery_marker_survives_session_store_reload(tmp_path):
    config = GatewayConfig()
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C_TEST",
        chat_type="group",
        user_id="U_TEST",
        thread_id="1000.0",
        scope_id="T_TEST",
    )

    store = SessionStore(tmp_path / "sessions", config)
    entry = store.get_or_create_session(source)

    assert not store.has_historical_media_delivered(entry.session_key)
    assert store.mark_historical_media_delivered(entry.session_key)
    assert store.has_historical_media_delivered(entry.session_key)

    reloaded = SessionStore(tmp_path / "sessions", config)
    assert reloaded.has_historical_media_delivered(entry.session_key)
