"""Recovered UTC routes coexist with legacy local-time routes."""
from datetime import datetime, timedelta, timezone

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


def test_list_sessions_orders_and_filters_mixed_clock_routes(tmp_path):
    store = SessionStore(sessions_dir=tmp_path / 'sessions', config=GatewayConfig())
    recent = store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id='recent'))
    old = store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id='old'))
    recent.updated_at = datetime.now(timezone.utc)
    old.updated_at = datetime.now() - timedelta(hours=2)

    assert store.list_sessions() == [recent, old]
    assert store.list_sessions(active_minutes=60) == [recent]
