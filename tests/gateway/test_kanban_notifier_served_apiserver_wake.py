"""A served profile's multiplexed ``api_server`` Kanban wake delivers in-process, in its own scope.

A subscription whose destination is a RAW session id on the shared ``api_server`` cannot be
anchored by ``gateway.profile_routes`` (there is no chat/thread/guild discriminator), and a
platform-wide ``api_server`` route would deny the default profile's own ``api_server``
destinations. The notifier therefore authorized nothing for a served secondary profile, and — even
had it authorized — the HTTP wake self-post targets the unprefixed listener with the PRIMARY
adapter's key, so a served profile's wake turn would have resumed the session in the DEFAULT
profile's store (``/p/<profile>/`` would need that profile's own ``API_SERVER_KEY``, which a
route-only profile legitimately does not have).

Invariant: a completion for a shared-``api_server`` session wakes the served profile that owns
that exact session, in that profile's scope, without a second listener, without a secondary
credential, and without granting the profile the shared ``api_server`` generally.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.kanban_watchers_notifier import _adapter_for_subscription
from gateway.profile_routing import parse_profile_routes
from gateway.run import GatewayRunner, _profile_runtime_scope
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from hermes_constants import get_hermes_home
from hermes_state import SessionDB

SESSION = "20260918_033413_0665eb"      # the originating (Relay/web-UI) session id
WORKER_SESSION = "20260918_034333_945a5a"  # the dispatcher-spawned worker's own session


class RecordingApiServerAdapter:
    """Non-push (stateless) adapter double: records in-process wake turns, never sends."""

    supports_async_delivery = False

    def __init__(self):
        self.turns = []
        self.homes = []
        self.profiles = []

    async def run_internal_session_turn(self, *, session_id, text, profile, notification_category="result"):
        self.homes.append(str(get_hermes_home()))
        self.profiles.append(profile)
        self.turns.append({"session_id": session_id, "text": text, "category": notification_category})

    async def send(self, chat_id, text, metadata=None):
        from gateway.platforms.base import SendResult
        return SendResult(success=False, error="API server uses HTTP request/response, not send()")


class _FakeResponse:
    status = 200

    async def text(self):
        return ""

    async def read(self):
        return b""


class _FakePostCtx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class _FakeHttpSession:
    """Records wake self-posts instead of sending them (proves which transport was used)."""

    calls: list = []

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, json=None, headers=None):
        type(self).calls.append({"url": url, "json": json, "headers": headers})
        return _FakePostCtx(_FakeResponse())


@pytest.fixture
def served(tmp_path, monkeypatch):
    """Default multiplex home serving a route-only secondary ``builder`` (no adapters, no key)."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "builder").mkdir(parents=True)
    (root / "profiles" / "atlas").mkdir(parents=True)
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n", encoding="utf-8")
    (root / ".env").write_text("", encoding="utf-8")
    (root / "profiles" / "builder" / "config.yaml").write_text("{}\n", encoding="utf-8")
    # A route-only profile owns no API-server credential of its own.
    (root / "profiles" / "builder" / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    return SimpleNamespace(root=root, builder=root / "profiles" / "builder", atlas=root / "profiles" / "atlas")


def _own_session(home: Path, session_id: str, profile_name: str) -> None:
    db = SessionDB(home / "state.db")
    try:
        db.create_session(session_id, source="webui", profile_name=profile_name)
    finally:
        db.close()


def _make_runner(*, adapter=None, routes=None, builder_adapters=None):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.API_SERVER: adapter or RecordingApiServerAdapter()}
    runner._profile_adapters = {"builder": builder_adapters or {}, "atlas": {}}
    runner._profile_failed_platforms = {}
    runner._primary_profile_name = "default"
    runner._kanban_notifier_profile = "default"
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    runner.config = SimpleNamespace(
        multiplex_profiles=True, profile_routes=parse_profile_routes(routes or []))
    return runner


def _subscription(*, chat_id=SESSION, profile="builder"):
    """One completed card whose origin is the raw session id, as the Relay origin creates it."""
    with kbc.connect() as conn:
        task = kb.create_task(conn, title="relay origin", assignee="builder", session_id=WORKER_SESSION)
        kbn.add_notify_sub(conn, task_id=task, platform="api_server", chat_id=chat_id,
                           chat_type="dm", notifier_profile=profile, delivery_mode="notify+wake")
        kb.complete_task(conn, task, summary="done once")
    return task


def _unseen(task, chat_id=SESSION):
    with kbc.connect() as conn:
        return kbn.unseen_events_for_sub(conn, task_id=task, platform="api_server", chat_id=chat_id,
                                         kinds=["completed"])[1]


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _api_sub(chat_id=SESSION, profile="builder"):
    return {"task_id": "t_owned", "platform": "api_server", "chat_id": chat_id, "chat_type": "dm",
            "thread_id": "", "notifier_profile": profile, "delivery_metadata": None}


def test_served_profile_wake_runs_in_process_only_for_the_session_it_owns(served, monkeypatch):
    """Owned session → in-process wake in the owner's scope; everything else fails closed; the
    default profile keeps its HTTP self-post."""
    import aiohttp

    _own_session(served.builder, SESSION, "builder")
    _FakeHttpSession.calls = []
    monkeypatch.setattr(aiohttp, "ClientSession", _FakeHttpSession)
    kb.init_db()
    adapter = RecordingApiServerAdapter()
    runner = _make_runner(adapter=adapter)
    task = _subscription()

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert [turn["session_id"] for turn in adapter.turns] == [SESSION]  # the origin, not the worker
    assert task in adapter.turns[0]["text"] and "done once" in adapter.turns[0]["text"]
    # Under the OWNING profile's scope, told which profile the authorization proved.
    assert adapter.homes == [str(served.builder)]
    assert adapter.profiles == ["builder"]
    # No HTTP self-post (so no secondary API_SERVER_KEY), one shared adapter, no borrowing.
    assert _FakeHttpSession.calls == []
    assert runner.adapters[Platform.API_SERVER] is adapter
    assert runner._profile_adapters["builder"] == {}
    assert _unseen(task) == []

    # Ownership of the exact session — not the platform — is the only proof.
    resolve = lambda sub, profile="builder", **kw: _adapter_for_subscription(  # noqa: E731
        _make_runner(**kw), Platform.API_SERVER, sub, profile)
    assert resolve(_api_sub(chat_id="20260918_051500_deadbe")) is None           # unknown session
    assert resolve(_api_sub(), profile="atlas") is None                          # not the owner
    assert resolve(_api_sub(), profile="ghost") is None                          # unserved profile
    # An adapter on ANOTHER platform is not a boundary for this one (#115460); an own api_server
    # adapter is — the primary never stands in for a credential the profile holds itself.
    assert resolve(_api_sub(), builder_adapters={Platform.DISCORD: object()}) is not None
    own_api = object()
    assert resolve(_api_sub(), builder_adapters={Platform.API_SERVER: own_api}) is own_api
    _own_session(served.atlas, "stamped-elsewhere", "builder")                   # foreign store
    assert resolve(_api_sub(chat_id="stamped-elsewhere")) is None
    # A row in the served store stamped for another profile is not ownership either.
    _own_session(served.builder, "atlas-row", "atlas")
    assert resolve(_api_sub(chat_id="atlas-row")) is None

    # Control: the default profile's own api_server subscription still HTTP self-posts.
    _own_session(served.root, "default-origin", "default")
    default_adapter = RecordingApiServerAdapter()
    default_adapter._api_key, default_adapter._host = "k" * 20, "127.0.0.1"
    default_adapter._port, default_adapter._model_name = 8642, "hermes"
    runner = _make_runner(adapter=default_adapter)
    task = _subscription(chat_id="default-origin", profile="default")
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert default_adapter.turns == []
    assert [c["headers"]["X-Hermes-Session-Id"] for c in _FakeHttpSession.calls] == ["default-origin"]
    assert _FakeHttpSession.calls[0]["url"].endswith("/v1/chat/completions")
    assert _unseen(task, chat_id="default-origin") == []


def test_internal_session_turn_targets_the_live_session_under_the_owner_profile(served, monkeypatch):
    """Adapter-level: the in-process turn binds the proven profile, adopts the compression tip,
    and never runs against a session the owner's store does not hold."""
    from gateway.platforms.api_server import APIServerAdapter, _api_request_profile

    parent, tip = "20260918_010000_parent", "20260918_020000_tip"
    db = SessionDB(served.builder / "state.db")
    try:
        db.create_session(parent, source="webui", profile_name="builder")
        db.append_message(parent, "user", "old turn")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", profile_name="builder", parent_session_id=parent)
        db.append_message(tip, "user", "live turn")
        assert db.resolve_resume_session_id(parent) == tip, "fixture must produce a live tip"
    finally:
        db.close()

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    # A model_routes alias for the virtual model must apply to the wake turn as it does to the
    # HTTP self-post's /v1/chat/completions turn.
    aliased_route = {"provider": "custom", "model": "wake-model", "base_url": "http://r"}
    adapter._model_routes = {adapter._model_name: aliased_route}
    seen = {}

    async def fake_run_agent(**kwargs):
        seen.update(kwargs)
        seen["home"] = str(get_hermes_home())
        seen["request_profile"] = _api_request_profile.get()
        return {}, {}

    monkeypatch.setattr(adapter, "_run_agent", fake_run_agent)

    with _profile_runtime_scope(served.builder):
        asyncio.run(adapter.run_internal_session_turn(session_id=parent, text="wake", profile="builder"))

    assert seen["session_id"] == tip  # the live continuation, not the retired parent slice
    assert "live turn" in str(seen["conversation_history"])
    assert seen["user_message"] == "wake"
    assert seen["session_history_delivery"] == "1"
    assert seen["route"] == aliased_route  # resolved like the HTTP wake path, not route=None
    assert seen["home"] == str(served.builder)
    assert seen["request_profile"] == "builder"
    assert _api_request_profile.get() is None  # binding restored, never leaked

    seen.clear()
    with _profile_runtime_scope(served.builder):
        with pytest.raises(RuntimeError):  # not in the owner's store → caller rewinds
            asyncio.run(adapter.run_internal_session_turn(session_id="not-a-session", text="wake",
                                                          profile="builder"))
        with pytest.raises(ValueError):  # the proof is the caller's; never derived here
            asyncio.run(adapter.run_internal_session_turn(session_id=tip, text="wake", profile=""))
    assert seen == {}
