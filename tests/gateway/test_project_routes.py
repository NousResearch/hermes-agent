import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import projects_db
from gateway import project_routes
from gateway.config import Platform
from gateway.project_routes import (
    InvalidProjectRouteError,
    apply_gateway_session_route,
    apply_cron_session_route,
    bind_gateway_session,
    bind_inbound_session,
    claim_mirror_delivery,
    complete_mirror_delivery,
    get_session_binding,
    release_mirror_delivery,
    resolve_event_route,
    set_project_route,
    sync_route_table,
    mirror_desktop_turn,
)
from gateway.run import GatewayRunner


def _project(conn, path: Path) -> str:
    path.mkdir()
    return projects_db.create_project(
        conn,
        name="Routing project",
        slug="routing-project",
        primary_path=str(path),
    )


def test_exact_routes_resolve_for_each_supported_origin(tmp_path):
    conn = projects_db.connect(tmp_path / "projects.db")
    cwd = tmp_path / "workspace"
    project_id = _project(conn, cwd)
    for kind, key in (
        ("telegram", "-1001:44"),
        ("cron", "job-123"),
        ("webhook", "github-push"),
    ):
        set_project_route(
            conn,
            origin_kind=kind,
            origin_key=key,
            project_id=project_id,
            cwd=str(cwd),
            telegram_chat_id="-1001",
            telegram_thread_id="44",
        )

    assert resolve_event_route(conn, "telegram", "-1001:44").project_id == project_id
    assert resolve_event_route(conn, "cron", "job-123").cwd == str(cwd.resolve())
    assert resolve_event_route(conn, "webhook", "github-push").telegram_thread_id == "44"
    assert resolve_event_route(conn, "cron", "unknown") is None


def test_route_write_rejects_missing_project_and_cwd_outside_project(tmp_path):
    conn = projects_db.connect(tmp_path / "projects.db")
    cwd = tmp_path / "workspace"
    project_id = _project(conn, cwd)
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(InvalidProjectRouteError, match="project does not exist"):
        set_project_route(
            conn, origin_kind="cron", origin_key="job", project_id="p_missing"
        )
    with pytest.raises(InvalidProjectRouteError, match="does not belong"):
        set_project_route(
            conn,
            origin_kind="cron",
            origin_key="job",
            project_id=project_id,
            cwd=str(outside),
        )


def test_matching_corrupt_route_fails_closed_but_absence_is_normal(tmp_path):
    conn = projects_db.connect(tmp_path / "projects.db")
    cwd = tmp_path / "workspace"
    project_id = _project(conn, cwd)
    set_project_route(
        conn,
        origin_kind="webhook",
        origin_key="route-a",
        project_id=project_id,
        cwd=str(cwd),
    )
    cwd.rmdir()

    with pytest.raises(InvalidProjectRouteError, match="working directory"):
        resolve_event_route(conn, "webhook", "route-a")
    assert resolve_event_route(conn, "webhook", "route-b") is None


def test_telegram_session_always_gets_exact_mirror_destination_without_project_route(tmp_path):
    conn = projects_db.connect(tmp_path / "projects.db")

    binding = bind_inbound_session(
        conn,
        session_id="telegram-session",
        origin_kind="telegram",
        origin_key="-1001:44",
        telegram_chat_id="-1001",
        telegram_thread_id="44",
    )

    assert binding.project_id is None
    assert get_session_binding(conn, "telegram-session").telegram_chat_id == "-1001"
    assert get_session_binding(conn, "telegram-session").telegram_thread_id == "44"


def test_telegram_new_session_keeps_route_but_never_reuses_session_identity(tmp_path):
    conn = projects_db.connect(tmp_path / "projects.db")
    cwd = tmp_path / "workspace"
    project_id = _project(conn, cwd)
    set_project_route(
        conn,
        origin_kind="telegram",
        origin_key="-1001:44",
        project_id=project_id,
        cwd=str(cwd),
    )

    first = bind_inbound_session(
        conn,
        session_id="before-new",
        origin_kind="telegram",
        origin_key="-1001:44",
        telegram_chat_id="-1001",
        telegram_thread_id="44",
    )
    second = bind_inbound_session(
        conn,
        session_id="after-new",
        origin_kind="telegram",
        origin_key="-1001:44",
        telegram_chat_id="-1001",
        telegram_thread_id="44",
    )

    assert first.session_id != second.session_id
    assert first.project_id == second.project_id == project_id
    assert first.cwd == second.cwd == str(cwd.resolve())
    assert first.telegram_thread_id == second.telegram_thread_id == "44"


def test_cron_and_webhook_sessions_remain_distinct_even_in_same_project(tmp_path):
    conn = projects_db.connect(tmp_path / "projects.db")
    cwd = tmp_path / "workspace"
    project_id = _project(conn, cwd)
    for kind, key in (("cron", "nightly"), ("webhook", "push")):
        set_project_route(
            conn,
            origin_kind=kind,
            origin_key=key,
            project_id=project_id,
            cwd=str(cwd),
            telegram_chat_id="-1001",
            telegram_thread_id="44",
        )
        bind_inbound_session(
            conn,
            session_id=f"{kind}-session",
            origin_kind=kind,
            origin_key=key,
        )

    assert get_session_binding(conn, "cron-session").session_id == "cron-session"
    assert get_session_binding(conn, "webhook-session").session_id == "webhook-session"
    assert get_session_binding(conn, "cron-session").project_id == project_id
    assert get_session_binding(conn, "webhook-session").project_id == project_id


def test_mirror_delivery_claim_is_idempotent_and_failure_can_retry(tmp_path):
    conn = projects_db.connect(tmp_path / "projects.db")

    assert claim_mirror_delivery(conn, "s1", "desktop-user:m1") is True
    assert claim_mirror_delivery(conn, "s1", "desktop-user:m1") is False
    release_mirror_delivery(conn, "s1", "desktop-user:m1")
    assert claim_mirror_delivery(conn, "s1", "desktop-user:m1") is True
    complete_mirror_delivery(conn, "s1", "desktop-user:m1")
    assert claim_mirror_delivery(conn, "s1", "desktop-user:m1") is False


def test_gateway_source_binding_uses_exact_origin_identity(tmp_path):
    conn = projects_db.connect(tmp_path / "projects.db")
    cwd = tmp_path / "workspace"
    project_id = _project(conn, cwd)
    set_project_route(
        conn,
        origin_kind="telegram",
        origin_key="-1001:44",
        project_id=project_id,
        cwd=str(cwd),
    )
    telegram = SimpleNamespace(
        platform=SimpleNamespace(value="telegram"),
        chat_id="-1001",
        thread_id="44",
        user_name="Thibaut",
    )

    binding = bind_gateway_session(conn, "s-telegram", telegram)

    assert binding.project_id == project_id
    assert binding.telegram_chat_id == "-1001"
    assert binding.telegram_thread_id == "44"


def test_gateway_webhook_binding_routes_by_subscription_not_delivery_chat(tmp_path):
    conn = projects_db.connect(tmp_path / "projects.db")
    cwd = tmp_path / "workspace"
    project_id = _project(conn, cwd)
    set_project_route(
        conn,
        origin_kind="webhook",
        origin_key="github-push",
        project_id=project_id,
        cwd=str(cwd),
    )
    webhook = SimpleNamespace(
        platform=SimpleNamespace(value="webhook"),
        chat_id="webhook:github-push:delivery-1",
        thread_id=None,
        user_name="github-push",
    )

    assert bind_gateway_session(conn, "s-webhook", webhook).project_id == project_id


def test_gateway_ignores_ordinary_desktop_source(tmp_path):
    conn = projects_db.connect(tmp_path / "projects.db")
    desktop = SimpleNamespace(
        platform=SimpleNamespace(value="desktop"),
        chat_id="local",
        thread_id=None,
        user_name="user",
    )

    assert bind_gateway_session(conn, "s-desktop", desktop) is None
    assert get_session_binding(conn, "s-desktop") is None


def test_gateway_uses_served_named_profile_when_source_profile_is_empty(
    tmp_path, monkeypatch
):
    default_home = tmp_path / "default"
    named_home = tmp_path / "profiles" / "named"
    default_home.mkdir(parents=True)
    named_home.mkdir(parents=True)

    def configure(home, folder_name):
        conn = projects_db.connect(home / "projects.db")
        cwd = home / folder_name
        project_id = _project(conn, cwd)
        conn.close()
        (home / "session_project_routes.json").write_text(
            json.dumps(
                {
                    "version": 2,
                    "routes": [
                        {
                            "origin": {
                                "source": "telegram",
                                "chat_id": "-1001",
                                "thread_id": "44",
                            },
                            "project_id": project_id,
                            "cwd": str(cwd),
                        }
                    ],
                }
            )
        )
        return cwd

    default_cwd = configure(default_home, "wrong-default")
    named_cwd = configure(named_home, "right-named")

    class SessionDB:
        def __init__(self):
            self.updated = []

        def update_session_cwd(self, session_id, cwd):
            self.updated.append((session_id, cwd))
            return 1

    runner = GatewayRunner.__new__(GatewayRunner)
    db = SessionDB()
    runner.session_store = SimpleNamespace(_db=db)
    monkeypatch.setattr(
        runner, "_resolve_profile_home_for_source", lambda _source: named_home
    )
    source = SimpleNamespace(
        platform=SimpleNamespace(value="telegram"),
        chat_id="-1001",
        thread_id="44",
        profile=None,
    )
    context = SimpleNamespace(session_id="s-named", cwd=None, source=source)

    binding = runner._apply_project_session_route(context)

    assert binding.cwd == str(named_cwd.resolve())
    assert binding.cwd != str(default_cwd.resolve())
    assert context.cwd == str(named_cwd.resolve())


def test_gateway_route_is_applied_before_turn_and_persisted_on_session(tmp_path):
    conn = projects_db.connect(tmp_path / "projects.db")
    cwd = tmp_path / "workspace"
    project_id = _project(conn, cwd)
    set_project_route(
        conn,
        origin_kind="telegram",
        origin_key="-1001:",
        project_id=project_id,
        cwd=str(cwd),
    )
    context = SimpleNamespace(
        session_id="s-telegram",
        cwd=None,
        source=SimpleNamespace(
            platform=SimpleNamespace(value="telegram"),
            chat_id="-1001",
            thread_id=None,
            user_name="Thibaut",
        ),
    )

    class FakeSessionDB:
        def __init__(self):
            self.updated = []

        def update_session_cwd(self, session_id, route_cwd):
            self.updated.append((session_id, route_cwd))
            return 1

    session_db = FakeSessionDB()
    binding = apply_gateway_session_route(conn, session_db, context)

    assert binding.project_id == project_id
    assert context.cwd == str(cwd.resolve())
    assert session_db.updated == [("s-telegram", str(cwd.resolve()))]


def test_v2_route_table_syncs_all_origins_idempotently(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    conn = projects_db.connect(tmp_path / "projects.db")
    cwd = tmp_path / "workspace"
    project_id = _project(conn, cwd)
    table = {
        "version": 2,
        "routes": [
            {
                "origin": {"source": "telegram", "chat_id": "-1001", "thread_id": "44"},
                "project_id": project_id,
                "cwd": str(cwd),
                "deliver": "telegram:-1001:44",
            },
            {
                "origin": {"source": "cron", "job_id": "job-1"},
                "project_id": project_id,
                "cwd": str(cwd),
                "deliver": "telegram:-1001:44",
            },
            {
                "origin": {"source": "webhook", "subscription": "github-push"},
                "project_id": project_id,
                "cwd": str(cwd),
                "deliver_extra": {"chat_id": "-1001", "thread_id": "44"},
            },
        ],
    }
    (tmp_path / "session_project_routes.json").write_text(json.dumps(table))

    assert sync_route_table(conn, tmp_path) == 3
    assert sync_route_table(conn, tmp_path) == 3
    assert resolve_event_route(conn, "telegram", "-1001:44").project_id == project_id
    assert resolve_event_route(conn, "cron", "job-1").telegram_thread_id == "44"
    assert resolve_event_route(conn, "webhook", "github-push").telegram_chat_id == "-1001"


def test_v2_route_table_fails_closed_without_partial_sync(tmp_path):
    conn = projects_db.connect(tmp_path / "projects.db")
    cwd = tmp_path / "workspace"
    project_id = _project(conn, cwd)
    set_project_route(
        conn,
        origin_kind="cron",
        origin_key="existing",
        project_id=project_id,
        cwd=str(cwd),
    )
    table = {
        "version": 2,
        "routes": [
            {
                "origin": {"source": "cron", "job_id": "valid"},
                "project_id": project_id,
                "cwd": str(cwd),
            },
            {
                "origin": {"source": "cron", "job_id": "invalid"},
                "project_id": "missing",
                "cwd": str(cwd),
            },
        ],
    }
    (tmp_path / "session_project_routes.json").write_text(json.dumps(table))

    with pytest.raises(InvalidProjectRouteError, match="project does not exist"):
        sync_route_table(conn, tmp_path)
    assert resolve_event_route(conn, "cron", "existing") is not None
    assert resolve_event_route(conn, "cron", "valid") is None


def test_cron_route_only_updates_session_classification_cwd(tmp_path, monkeypatch):
    conn = projects_db.connect(tmp_path / "projects.db")
    cwd = tmp_path / "classification"
    project_id = _project(conn, cwd)
    set_project_route(
        conn,
        origin_kind="cron",
        origin_key="job-1",
        project_id=project_id,
        cwd=str(cwd),
    )
    monkeypatch.setenv("TERMINAL_CWD", "/job/runtime")

    class FakeSessionDB:
        def __init__(self):
            self.updated = []

        def update_session_cwd(self, session_id, route_cwd):
            self.updated.append((session_id, route_cwd))
            return 1

    session_db = FakeSessionDB()
    binding = apply_cron_session_route(conn, session_db, "cron-session", "job-1")

    assert binding.cwd == str(cwd.resolve())
    assert session_db.updated == [("cron-session", str(cwd.resolve()))]
    assert os.environ["TERMINAL_CWD"] == "/job/runtime"


def test_matching_cron_route_fails_closed_when_session_row_is_missing(tmp_path):
    conn = projects_db.connect(tmp_path / "projects.db")
    cwd = tmp_path / "classification"
    project_id = _project(conn, cwd)
    set_project_route(
        conn,
        origin_kind="cron",
        origin_key="job-1",
        project_id=project_id,
        cwd=str(cwd),
    )

    class MissingSessionDB:
        def update_session_cwd(self, _session_id, _route_cwd):
            return None

    with pytest.raises(InvalidProjectRouteError, match="missing session"):
        apply_cron_session_route(
            conn, MissingSessionDB(), "missing-cron-session", "job-1"
        )


def test_matching_routes_fail_closed_without_durable_updater(tmp_path):
    conn = projects_db.connect(tmp_path / "projects.db")
    cwd = tmp_path / "classification"
    project_id = _project(conn, cwd)
    for kind, key in (("telegram", "-1001:44"), ("cron", "job-1")):
        set_project_route(
            conn,
            origin_kind=kind,
            origin_key=key,
            project_id=project_id,
            cwd=str(cwd),
        )
    source = SimpleNamespace(
        platform=SimpleNamespace(value="telegram"),
        chat_id="-1001",
        thread_id="44",
    )
    context = SimpleNamespace(session_id="telegram-session", cwd=None, source=source)
    with pytest.raises(InvalidProjectRouteError, match="no durable updater"):
        apply_gateway_session_route(conn, object(), context)
    with pytest.raises(InvalidProjectRouteError, match="no durable updater"):
        apply_cron_session_route(conn, object(), "cron-session", "job-1")


def test_desktop_turn_mirrors_only_telegram_inherited_session_in_order(tmp_path):
    conn = projects_db.connect(tmp_path / "projects.db")
    bind_inbound_session(
        conn,
        session_id="telegram-session",
        origin_kind="telegram",
        origin_key="-1001:44",
        telegram_chat_id="-1001",
        telegram_thread_id="44",
    )
    conn.close()
    sent = []

    def send(chat_id, thread_id, text):
        sent.append((chat_id, thread_id, text))

    result = mirror_desktop_turn(
        tmp_path,
        "telegram-session",
        user_text="Desktop question",
        assistant_text="Final answer",
        user_message_key="row:10",
        assistant_message_key="row:11",
        send=send,
    )

    assert result == {"user": True, "assistant": True}
    assert sent == [
        ("-1001", "44", "Desktop question"),
        ("-1001", "44", "Final answer"),
    ]
    assert mirror_desktop_turn(
        tmp_path,
        "telegram-session",
        user_text="Desktop question",
        assistant_text="Final answer",
        user_message_key="row:10",
        assistant_message_key="row:11",
        send=send,
    ) == {"user": False, "assistant": False}
    assert len(sent) == 2


def test_exact_telegram_send_runs_inside_requested_profile_scope(tmp_path, monkeypatch):
    entered = []

    @contextlib.contextmanager
    def fake_scope(profile_home):
        entered.append(Path(profile_home))
        yield

    fake_platform = SimpleNamespace(enabled=True)
    fake_config = SimpleNamespace(platforms={Platform.TELEGRAM: fake_platform})
    sent = []
    monkeypatch.setattr("gateway.run._profile_runtime_scope", fake_scope)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: fake_config)
    monkeypatch.setattr(
        "tools.send_message_tool._send_to_platform",
        lambda *args, **kwargs: sent.append((args, kwargs)) or object(),
    )
    monkeypatch.setattr("model_tools._run_async", lambda _awaitable: {"success": True})

    project_routes._send_exact_telegram(tmp_path, "-1001", "44", "hello")

    assert entered == [tmp_path]
    assert sent[0][0][1] is fake_platform


def test_desktop_turn_does_not_mirror_ordinary_desktop_session(tmp_path):
    projects_db.connect(tmp_path / "projects.db").close()
    sent = []

    assert mirror_desktop_turn(
        tmp_path,
        "desktop-session",
        user_text="question",
        assistant_text="answer",
        user_message_key="row:1",
        assistant_message_key="row:2",
        send=lambda *args: sent.append(args),
    ) == {"user": False, "assistant": False}
    assert sent == []


def test_desktop_mirror_network_failure_never_blocks_retry(tmp_path):
    conn = projects_db.connect(tmp_path / "projects.db")
    bind_inbound_session(
        conn,
        session_id="telegram-session",
        origin_kind="telegram",
        origin_key="-1001:",
        telegram_chat_id="-1001",
    )
    conn.close()

    def fail(*_args):
        raise OSError("network down")

    assert mirror_desktop_turn(
        tmp_path,
        "telegram-session",
        user_text="question",
        assistant_text="answer",
        user_message_key="row:1",
        assistant_message_key="row:2",
        send=fail,
    ) == {"user": False, "assistant": False}

    sent = []
    assert mirror_desktop_turn(
        tmp_path,
        "telegram-session",
        user_text="question",
        assistant_text="answer",
        user_message_key="row:1",
        assistant_message_key="row:2",
        send=lambda chat, thread, text: sent.append((chat, thread, text)),
    ) == {"user": True, "assistant": True}
    assert [item[2] for item in sent] == ["question", "answer"]
