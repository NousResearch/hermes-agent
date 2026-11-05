"""``/model reset`` in a messaging chat stores the default route on the session row the dashboard reads."""

import json
import threading
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner, _SESSION_DB_UNPINNED
from gateway.session import SessionSource, SessionStore, build_session_key
from hermes_constants import get_hermes_home


@pytest.mark.asyncio
async def test_reset_stores_the_default_model_on_the_session_row(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML; CI ships no pyyaml for test fixtures.
    (home / "config.yaml").write_text(json.dumps({"model": {"default": "cfg-model", "provider": "cfg-prov"}}))
    monkeypatch.setattr(gateway_run, "_hermes_home", home)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1", chat_type="dm")
    key = build_session_key(source)
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    try:
        entry = store.get_or_create_session(source)
        store.set_model_override(key, {"model": "pinned-model", "provider": "pinned-prov"})
        # The switch that planted the override stored its route on the row.
        store.persistence.sessions.update_model(entry.session_id, "pinned-model", provider="pinned-prov")

        runner = object.__new__(GatewayRunner)
        runner.config = SimpleNamespace(multiplex_profiles=False)
        runner.session_store = store
        runner._session_db_pinned = _SESSION_DB_UNPINNED
        runner._session_db_handles = {}
        runner._session_db_handles_lock = threading.Lock()
        runner._evict_cached_agent = lambda k, **_kw: None
        runner._normalize_source_for_session_key = lambda s: s
        runner._session_key_for_source = lambda s: build_session_key(s)

        reply = await runner._handle_model_command(MessageEvent(text="/model reset", message_type=MessageType.TEXT, source=source))

        assert "cfg-model" in reply
        row = store.persistence.sessions.get(entry.session_id)
        assert (row.model, row.model_config.get("provider")) == ("cfg-model", "cfg-prov")
    finally:
        store.close_all_db_handles()
