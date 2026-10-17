"""Route anchors survive gateway background/subprocess producers without stale fallback."""
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from gateway.session_context import get_session_env


def test_session_route_anchors_reach_metadata_and_context_then_clear(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {}
    source = SessionSource(platform=Platform.DISCORD, chat_id="post", chat_type="thread",
                           thread_id="post", scope_id="scope", guild_id="guild",
                           parent_chat_id="parent", profile="yuki")
    context = SessionContext(source=source, connected_platforms=[], home_channels={}, session_key="origin")
    expected = {key: getattr(source, key) for key in ("scope_id", "parent_chat_id")}
    metadata = runner._thread_metadata_for_source(source)
    assert {key: metadata.get(key) for key in expected} == expected
    keys = {key: "HERMES_SESSION_" + key.upper() for key in ("scope_id", "parent_chat_id")}
    for name in keys.values():
        monkeypatch.setenv(name, "stale")
    tokens = runner._set_session_env(context)
    try:
        assert {key: get_session_env(name) for key, name in keys.items()} == {key: expected[key] for key in keys}
    finally:
        runner._clear_session_env(tokens)
    assert all(get_session_env(name) == "" for name in keys.values())


def test_slash_subscription_keeps_the_routed_source_owner(tmp_path, monkeypatch):
    import asyncio
    from gateway.platforms.event import MessageEvent
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn

    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._kanban_notifier_profile = "default"
    source = SessionSource(platform=Platform.DISCORD, chat_id="post", chat_type="thread",
                           thread_id="post", scope_id="guild", parent_chat_id="parent", profile="yuki")
    with kbc.connect() as conn:
        task = kb.create_task(conn, title="slash-created")
    assert asyncio.run(runner._kanban_auto_subscribe(MessageEvent(text="/kanban create", source=source), task, None))
    with kbc.connect() as conn:
        sub = kbn.list_notify_subs(conn, task)[0]
    assert sub["notifier_profile"] == source.profile
    assert all(sub["delivery_metadata"][key] == getattr(source, key)
               for key in ("scope_id", "parent_chat_id"))
