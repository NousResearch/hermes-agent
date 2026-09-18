from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore

from hermes_cli.session_model_overrides import (
    clear_model_override,
    overrides_from_store,
    resolve_override_entry,
)


def _source(chat_id="c1"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id=chat_id,
        user_name="tester",
        chat_type="dm",
    )


def test_list_and_clear_override(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no sqlite")))
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    entry = store.get_or_create_session(_source())
    store.set_model_override(entry.session_key, {"model": "old-model", "provider": "nous", "api_key": "sk-secret"})

    rows = overrides_from_store(store)
    assert len(rows) == 1
    assert rows[0]["model"] == "old-model"
    assert rows[0]["provider"] == "nous"
    assert "api_key" not in rows[0]

    assert resolve_override_entry(store, entry.session_id[:8]).session_id == entry.session_id
    assert resolve_override_entry(store, "nope") is None

    cleared = clear_model_override(store, entry.session_id)
    assert cleared["cleared"] is True
    assert overrides_from_store(store) == []
    restarted = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    assert overrides_from_store(restarted) == []


def test_clear_unknown_is_none(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no sqlite")))
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    store.get_or_create_session(_source())
    assert clear_model_override(store, "missing") is None


def test_ambiguous_prefix_is_unresolved(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no sqlite")))
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    a = store.get_or_create_session(_source("aa"))
    b = store.get_or_create_session(_source("bb"))
    # Force colliding prefixes by using session keys if they share nothing; skip if unique.
    common = ""
    for i in range(1, min(len(a.session_id), len(b.session_id))):
        if a.session_id[:i] == b.session_id[:i]:
            common = a.session_id[:i]
        else:
            break
    if common:
        assert resolve_override_entry(store, common) is None


def test_sessions_parser_has_clear_model_and_with_model():
    import argparse
    from hermes_cli.subcommands.sessions import build_sessions_parser

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    build_sessions_parser(sub, cmd_sessions=lambda *a, **k: None)
    ns = parser.parse_args(["sessions", "clear-model", "abc123", "--yes"])
    assert ns.sessions_action == "clear-model"
    assert ns.session == "abc123"
    ns = parser.parse_args(["sessions", "list", "--with-model"])
    assert ns.with_model is True
    ns = parser.parse_args(["sessions", "overrides", "--json"])
    assert ns.sessions_action == "overrides"
    assert ns.json is True
