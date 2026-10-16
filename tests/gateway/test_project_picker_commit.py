"""`/new -project` through the real runner + Telegram adapter: ordering, stability, serialization (#138777)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import projects_db
from hermes_state import SessionDB
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture
def rig(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    runner = GatewayRunner(GatewayConfig(platforms={
        Platform.TELEGRAM: PlatformConfig(enabled=True, token="test-token")}))
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._send_control_message = AsyncMock(return_value=SimpleNamespace(message_id=701))
    adapter.gateway_runner = runner
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._wire_adapter_handlers(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm")
    key = runner.session_store._generate_session_key(source)
    old = runner.session_store.get_or_create_session(source)
    db = SessionDB(home / "state.db")
    runner.session_store._db = db
    db.create_session(old.session_id, "telegram", session_key=key, chat_id="42", user_id="42")

    def make_project(name, *folders, slug=None):
        paths = []
        for folder in folders:
            (tmp_path / folder).mkdir(exist_ok=True)
            paths.append(str(tmp_path / folder))
        with projects_db.connect_closing(home / "projects.db") as conn:
            return projects_db.create_project(conn, name=name, slug=slug, folders=paths)

    def tap(token, action):
        return SimpleNamespace(callback_query=SimpleNamespace(
            data=f"pj:{token}:{action}", from_user=SimpleNamespace(id=42, first_name="t"),
            message=SimpleNamespace(message_id=701, chat_id=42, message_thread_id=None,
                                    chat=SimpleNamespace(type="private")),
            answer=AsyncMock(), edit_message_text=AsyncMock()))

    yield SimpleNamespace(runner=runner, adapter=adapter, source=source, key=key, old=old, db=db,
                          home=home, tmp=tmp_path, make_project=make_project, tap=tap)
    db.close()
    runner._shutdown_executor()


async def _open(rig):
    await rig.runner._handle_message(MessageEvent(text="/new -project", source=rig.source))
    return next(iter(rig.adapter._project_picker_state.items()))


@pytest.mark.asyncio
async def test_topic_root_lobby_is_refused_before_the_picker_opens(rig, monkeypatch):
    rig.make_project("P", "w1")
    monkeypatch.setattr(rig.runner, "_is_telegram_topic_root_lobby", lambda _s: True)
    reply = await rig.runner._handle_message(MessageEvent(text="/new -project", source=rig.source))
    assert reply == rig.runner._telegram_topic_root_new_message()
    assert not rig.adapter.__dict__.get("_project_picker_state")
    rig.adapter._send_control_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_archived_project_is_not_listed_and_listing_stays_stable_across_back(rig):
    keep = rig.make_project("Keep", "k1", "k2", slug="keep")
    gone = rig.make_project("Gone", "g1", slug="gone")
    with projects_db.connect_closing(rig.home / "projects.db") as conn:
        projects_db.archive_project(conn, gone)
    token, state = await _open(rig)
    assert [p["id"] for p in state["projects"]] == [keep]
    snapshot = list(state["projects"])
    await rig.adapter._handle_callback_query(rig.tap(token, "select_0"), None)
    assert state["view"] == "folders" and state["projects"] == snapshot
    await rig.adapter._handle_callback_query(rig.tap(token, "back"), None)
    assert [p["id"] for p in state["projects"]] == [keep]
    assert rig.runner.session_store.peek_session_id(rig.key) == rig.old.session_id


@pytest.mark.asyncio
async def test_project_archived_after_listing_is_refused_at_selection(rig):
    pid = rig.make_project("Soon gone", "s1", slug="soon")
    token, _ = await _open(rig)
    with projects_db.connect_closing(rig.home / "projects.db") as conn:
        projects_db.archive_project(conn, pid)
    await rig.adapter._handle_callback_query(rig.tap(token, "select_0"), None)
    assert rig.runner.session_store.peek_session_id(rig.key) == rig.old.session_id
    assert token not in rig.adapter._project_picker_state


@pytest.mark.asyncio
async def test_workspace_deleted_before_commit_keeps_the_previous_session(rig):
    rig.make_project("P", "w1")
    token, _ = await _open(rig)
    (rig.tmp / "w1").rmdir()
    await rig.adapter._handle_callback_query(rig.tap(token, "select_0"), None)
    assert rig.runner.session_store.peek_session_id(rig.key) == rig.old.session_id
    assert rig.db.session_count() == 1


@pytest.mark.asyncio
async def test_validation_failures_do_not_stop_the_running_turn(rig, monkeypatch):
    rig.make_project("P", "w1")
    stop = AsyncMock()
    monkeypatch.setattr(rig.runner, "_hm_stop_owner_turn", stop)
    token, _ = await _open(rig)
    monkeypatch.setenv("TERMINAL_ENV", "docker")  # host path cannot be honored
    await rig.adapter._handle_callback_query(rig.tap(token, "select_0"), None)
    stop.assert_not_awaited()
    assert rig.runner.session_store.peek_session_id(rig.key) == rig.old.session_id


@pytest.mark.asyncio
async def test_storage_failure_after_stop_keeps_old_route_and_reports(rig, monkeypatch):
    rig.make_project("P", "w1")
    token, _ = await _open(rig)
    monkeypatch.setattr(rig.runner.session_store, "_persist_routing_data",
                        lambda *a: (_ for _ in ()).throw(OSError("disk")))
    query = rig.tap(token, "select_0")
    await rig.adapter._handle_callback_query(query, None)
    assert rig.runner.session_store.peek_session_id(rig.key) == rig.old.session_id
    assert rig.db.session_count() == 1
    shown = query.callback_query.edit_message_text.await_args.kwargs.get("text", "")
    assert "not changed" in shown.lower()


@pytest.mark.asyncio
async def test_ordinary_new_waits_for_an_in_flight_project_commit(rig, monkeypatch):
    rig.make_project("P", "w1")
    token, _ = await _open(rig)
    order, entered, release = [], asyncio.Event(), asyncio.Event()
    real_stop = rig.runner._stop_and_clear_route_for_reset

    async def slow_stop(route, *, reason, **kw):
        order.append(f"start:{reason}")
        entered.set()
        await release.wait()
        result = await real_stop(route, reason=reason, **kw)
        order.append(f"end:{reason}")
        return result

    monkeypatch.setattr(rig.runner, "_stop_and_clear_route_for_reset", slow_stop)
    commit = asyncio.create_task(rig.adapter._handle_callback_query(rig.tap(token, "select_0"), None))
    await entered.wait()
    plain = asyncio.create_task(rig.runner._handle_reset_command_serialized(
        MessageEvent(text="/new", source=rig.source)))
    for _ in range(20):  # let the plain /new run as far as it can; it must park on the route lock
        await asyncio.sleep(0)
    assert order == ["start:project_session_reset"] and not plain.done()  # waiting on the route lock
    release.set()
    await commit
    await plain
    assert order == ["start:project_session_reset", "end:project_session_reset",
                     "start:session_reset", "end:session_reset"]
    # The plain /new then replaced the project session; the route ends on a session with no project pin.
    final = rig.runner.session_store.peek_session_id(rig.key)
    assert final not in {rig.old.session_id}
    assert rig.db.get_session_model_config_value(final, "project_session") is None


@pytest.mark.asyncio
async def test_turn_on_a_project_session_whose_folder_vanished_fails_closed(rig):
    rig.make_project("P", "w1")
    token, _ = await _open(rig)
    await rig.adapter._handle_callback_query(rig.tap(token, "select_0"), None)
    entry = rig.runner.session_store.lookup_by_session_key(rig.key)
    assert rig.db.get_session_model_config_value(entry.session_id, "project_session")["cwd"] == str(rig.tmp / "w1")
    (rig.tmp / "w1").rmdir()
    event = MessageEvent(text="hello", source=rig.source)
    prepared, tokens = await rig.runner._hmwa_prepare_turn(event, rig.source, entry, rig.key, rig.key, 0)
    assert isinstance(prepared, str) and "no longer available" in prepared
    assert tokens == []


@pytest.mark.asyncio
async def test_transport_edit_failure_still_releases_busy_and_answers_callback(rig):
    rig.make_project("P", "w1", "w2")
    token, state = await _open(rig)
    query = rig.tap(token, "select_0")
    query.callback_query.edit_message_text = AsyncMock(side_effect=RuntimeError("telegram down"))
    await rig.adapter._handle_callback_query(query, None)
    assert state["busy"] is False
    query.callback_query.answer.assert_awaited()
    # The picker is still usable afterwards.
    await rig.adapter._handle_callback_query(rig.tap(token, "back"), None)
    assert state["view"] == "projects"


@pytest.mark.asyncio
async def test_commit_interrupts_busy_turn_only_after_publish_and_not_when_stale(rig, monkeypatch):
    rig.make_project("P", "w1")
    calls = []

    async def interrupt(route, source, **kw):
        calls.append(("interrupt", rig.runner.session_store.peek_session_id(route)))

    monkeypatch.setattr(rig.runner, "_interrupt_and_clear_session", interrupt)
    token, _ = await _open(rig)
    rig.runner.session_store.reset_session(rig.key)  # route moves: the picker is stale
    await rig.adapter._handle_callback_query(rig.tap(token, "select_0"), None)
    assert calls == []  # stale CAS: the running turn was never touched

    token, _ = await _open(rig)
    await rig.adapter._handle_callback_query(rig.tap(token, "select_0"), None)
    assert len(calls) == 1
    assert calls[0][1] == rig.runner.session_store.peek_session_id(rig.key)  # published before teardown
