"""Messaging sessions carry ``origin_url``: the session lists' "Open in Slack/Discord/Telegram"
link, built by the owning platform plugin's ``thread_url`` from the row's stored origin."""

import json

import pytest

from hermes_cli import isolated_scope

_SLACK = {"platform": "slack", "chat_id": "C0123ABCD", "chat_type": "group",
          "thread_id": "1700000000.000100", "scope_id": "T0ABC"}
_DISCORD = {"platform": "discord", "chat_id": "222", "chat_type": "group", "scope_id": "111",
            "message_id": "333"}
_TELEGRAM_TOPIC = {"platform": "telegram", "chat_id": "-1001234567890", "chat_type": "forum",
                   "thread_id": "7", "message_id": "42"}
_TELEGRAM_DM = {"platform": "telegram", "chat_id": "5550001", "chat_type": "dm", "message_id": "42"}


@pytest.fixture(autouse=True)
def _reset_scope(monkeypatch):
    from hermes_cli.web_routers import profiles as profiles_routes

    monkeypatch.setattr(profiles_routes, "_SIDEBAR_CACHE_TTL_SECONDS", 0.0)
    yield
    isolated_scope._LAUNCH_PROFILE = None


@pytest.fixture
def client(_isolate_hermes_home):
    from starlette.testclient import TestClient

    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    root = get_hermes_home()
    worker = root / "profiles" / "worker"
    # The worker disables Slack: its Slack session must get no link even though the launch
    # profile (root) loads the same bundled plugin and links its own Slack session.
    rows = {root: {"local-cli": None, "slack-thread": _SLACK},
            worker: {"discord-chan": _DISCORD, "tg-topic": _TELEGRAM_TOPIC, "tg-dm": _TELEGRAM_DM,
                     "worker-slack": _SLACK}}
    configs = {root: "{}\n", worker: "plugins:\n  disabled: [slack-platform]\n"}
    for home, sessions in rows.items():
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(configs[home], encoding="utf-8")
        db = SessionDB(db_path=home / "state.db")
        try:
            for sid, origin in sessions.items():
                db.create_session(sid, source=origin["platform"] if origin else "cli",
                                  origin_json=json.dumps(origin) if origin else None)
                db.append_message(session_id=sid, role="user", content="hi")
        finally:
            db.close()
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_session_lists_link_each_messaging_session_to_its_conversation(client):
    expected = {
        "local-cli": None,
        "slack-thread": "https://app.slack.com/client/T0ABC/C0123ABCD/thread/C0123ABCD-1700000000.000100",
        "discord-chan": "https://discord.com/channels/111/222/333",
        "tg-topic": "https://t.me/c/1234567890/7/42",
        "tg-dm": None,  # Telegram gives a private chat no link
        "worker-slack": None,  # Slack is disabled in the owning profile
    }
    listed = {s["id"]: s["origin_url"] for s in client.get("/api/profiles/sessions?limit=50").json()["sessions"]}
    assert listed == expected

    sidebar = client.get("/api/profiles/sessions/sidebar").json()
    sidebar_rows = {s["id"]: s["origin_url"] for key in ("recents", "messaging") for s in sidebar[key]["sessions"]}
    assert sidebar_rows == {sid: expected[sid] for sid in sidebar_rows}
    assert {"discord-chan", "slack-thread", "worker-slack"} <= set(sidebar_rows)

    own = {s["id"]: s["origin_url"] for s in client.get("/api/sessions?limit=50").json()["sessions"]}
    assert own == {"local-cli": None, "slack-thread": expected["slack-thread"]}
    detail = client.get("/api/sessions/discord-chan", params={"profile": "worker"}).json()
    assert detail["origin_url"] == expected["discord-chan"]
    worker_slack = client.get("/api/sessions/worker-slack", params={"profile": "worker"}).json()
    assert worker_slack["origin_url"] is None

    # Search results carry the same link as the list rows.
    found = {r["id"]: r.get("origin_url") for r in client.get("/api/sessions/search", params={"q": "hi"}).json()["results"]}
    assert found == {"local-cli": None, "slack-thread": expected["slack-thread"]}


def test_platform_without_thread_url_gets_no_link(_isolate_hermes_home):
    from gateway.platform_registry import PlatformEntry, platform_registry
    from hermes_cli.session_origin_link import stamp_origin_urls
    from hermes_constants import get_hermes_home

    platform_registry.register(PlatformEntry(
        name="linkless", label="Linkless", adapter_factory=lambda cfg: None, check_fn=lambda: True))
    try:
        rows = [{"origin_json": json.dumps({"platform": "linkless", "chat_id": "room-1"})},
                {"origin_json": "{not json"}]
        stamp_origin_urls(rows, get_hermes_home())
        assert [r["origin_url"] for r in rows] == [None, None]
    finally:
        platform_registry.unregister("linkless")
