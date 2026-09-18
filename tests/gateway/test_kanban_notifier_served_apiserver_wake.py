"""A served profile's multiplexed ``api_server`` Kanban wake delivers in-process, in its own scope.

A subscription whose destination is a RAW session id on the shared ``api_server`` cannot be
anchored by ``gateway.profile_routes`` (there is no chat/thread/guild discriminator), and a
platform-wide ``api_server`` route would deny the default profile's own ``api_server``
destinations. The notifier therefore authorized nothing for a served secondary profile, and — even
had it authorized — the HTTP wake self-post targets the unprefixed listener with the PRIMARY
adapter's key, so a served profile's wake turn would have resumed the session in the DEFAULT
profile's store (``/p/<profile>/`` would need that profile's own ``API_SERVER_KEY``, which a
route-only profile legitimately does not have).

The invariant: a completion for a multiplexed shared-``api_server`` session wakes the served
profile that canonically owns that exact originating session, without granting that profile
ownership of the shared ``api_server`` generally, without a second listener and without a
secondary credential.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.kanban_watchers_notifier import _adapter_for_subscription, _notifier_collect
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

    def __init__(self, *, fail_first: bool = False):
        self.turns = []
        self.homes = []
        self._fail_first = fail_first

    async def run_internal_session_turn(self, *, session_id, text, notification_category="result"):
        self.homes.append(str(get_hermes_home()))
        if self._fail_first:
            self._fail_first = False
            raise RuntimeError("simulated wake failure")
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


def _make_runner(served, *, adapter=None, routes=None, builder_adapters=None):
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


def _subscription(*, chat_id=SESSION, profile="builder", mode="notify+wake", session_id=WORKER_SESSION):
    """One completed card whose origin is the raw session id, as the Relay origin creates it."""
    with kbc.connect() as conn:
        task = kb.create_task(conn, title="relay origin", assignee="builder", session_id=session_id)
        kbn.add_notify_sub(conn, task_id=task, platform="api_server", chat_id=chat_id,
                           chat_type="dm", notifier_profile=profile, delivery_mode=mode)
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


def test_served_profile_api_server_subscription_wakes_in_process(served, monkeypatch):
    """Served secondary + session it owns → in-process wake in that profile, cursor advances."""
    import aiohttp

    _own_session(served.builder, SESSION, "builder")
    _FakeHttpSession.calls = []
    monkeypatch.setattr(aiohttp, "ClientSession", _FakeHttpSession)
    kb.init_db()
    adapter = RecordingApiServerAdapter()
    runner = _make_runner(served, adapter=adapter)
    task = _subscription()

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert [turn["session_id"] for turn in adapter.turns] == [SESSION]
    assert WORKER_SESSION not in [turn["session_id"] for turn in adapter.turns]
    assert task in adapter.turns[0]["text"] and "done once" in adapter.turns[0]["text"]
    # The wake ran under the OWNING profile's runtime scope, not the launch profile's.
    assert adapter.homes == [str(served.builder)]
    # No HTTP self-post: no shared-listener request, so no secondary API_SERVER_KEY is involved.
    assert _FakeHttpSession.calls == []
    assert _unseen(task) == []
    # One shared listener/adapter, no duplicate and no credential borrowing.
    assert runner.adapters[Platform.API_SERVER] is adapter
    assert runner._profile_adapters["builder"] == {}


def test_default_profile_api_server_subscription_still_self_posts(served, monkeypatch):
    """Default-profile behaviour is unchanged: the HTTP self-post is still the delivery."""
    import aiohttp

    _own_session(served.root, SESSION, "default")
    _FakeHttpSession.calls = []
    monkeypatch.setattr(aiohttp, "ClientSession", _FakeHttpSession)
    kb.init_db()
    adapter = RecordingApiServerAdapter()
    adapter._api_key = "k" * 20
    adapter._host, adapter._port, adapter._model_name = "127.0.0.1", 8642, "hermes"
    runner = _make_runner(served, adapter=adapter)
    task = _subscription(profile="default")

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.turns == []  # in-process path is for served profiles only
    assert len(_FakeHttpSession.calls) == 1
    call = _FakeHttpSession.calls[0]
    assert call["url"].endswith("/v1/chat/completions")
    assert call["headers"]["X-Hermes-Session-Id"] == SESSION
    assert _unseen(task) == []


def test_foreign_session_claim_fails_closed(served, monkeypatch):
    """Subscription says ``builder`` but the session lives in the default store → no delivery."""
    import aiohttp

    _own_session(served.root, SESSION, "default")
    _FakeHttpSession.calls = []
    monkeypatch.setattr(aiohttp, "ClientSession", _FakeHttpSession)
    kb.init_db()
    adapter = RecordingApiServerAdapter()
    runner = _make_runner(served, adapter=adapter)
    task = _subscription()

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.turns == []
    assert _FakeHttpSession.calls == []
    assert _unseen(task)  # retryable, never misdelivered


def test_wake_failure_after_claim_rewinds_and_retries(served, monkeypatch):
    """A failed served-profile wake leaves the event retryable; the next tick delivers once."""
    _own_session(served.builder, SESSION, "builder")
    kb.init_db()
    adapter = RecordingApiServerAdapter(fail_first=True)
    runner = _make_runner(served, adapter=adapter)
    task = _subscription()

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert adapter.turns == []
    assert _unseen(task)

    runner._running = True
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert [turn["session_id"] for turn in adapter.turns] == [SESSION]
    assert _unseen(task) == []


def test_resolver_authorizes_only_the_owned_session(served):
    """Direct resolver: ownership, not the platform, authorizes the shared adapter."""
    _own_session(served.builder, SESSION, "builder")
    runner = _make_runner(served)
    resolver = lambda sub, profile="builder": _adapter_for_subscription(  # noqa: E731
        runner, Platform.API_SERVER, sub, profile)

    assert resolver(_api_sub()) is runner.adapters[Platform.API_SERVER]
    assert resolver(_api_sub(chat_id="20260918_051500_deadbe")) is None       # unknown session
    assert resolver(_api_sub(), profile="atlas") is None                     # wrong owner
    assert resolver(_api_sub(), profile="ghost") is None                     # unserved profile
    assert resolver(_api_sub(chat_id="")) is None                            # no destination


def test_session_stamped_for_another_profile_fails_closed(served):
    """A row inside the served store that is stamped for another profile is not ownership."""
    _own_session(served.builder, SESSION, "atlas")
    runner = _make_runner(served)
    assert _adapter_for_subscription(runner, Platform.API_SERVER, _api_sub(), "builder") is None


def test_static_routes_and_credential_boundaries_are_unchanged(served):
    """Explicit ``profile_routes`` and a connected secondary adapter still win."""
    _own_session(served.builder, SESSION, "builder")
    # A route that claims this destination for another profile denies it (route beats ownership).
    routed = _make_runner(served, routes=[
        {"name": "api-atlas", "platform": "api_server", "chat_id": SESSION, "profile": "atlas"}])
    assert _adapter_for_subscription(routed, Platform.API_SERVER, _api_sub(), "builder") is None
    # A route naming the owner authorizes it exactly as before.
    own_route = _make_runner(served, routes=[
        {"name": "api-builder", "platform": "api_server", "chat_id": SESSION, "profile": "builder"}])
    assert _adapter_for_subscription(own_route, Platform.API_SERVER, _api_sub(), "builder") \
        is own_route.adapters[Platform.API_SERVER]
    # A profile that owns a credential boundary never borrows the primary listener.
    owned = _make_runner(served, builder_adapters={Platform.DISCORD: object()})
    assert _adapter_for_subscription(owned, Platform.API_SERVER, _api_sub(), "builder") is None


def test_internal_session_turn_binds_profile_and_targets_the_session(served, monkeypatch):
    """Adapter-level: the in-process turn resumes the exact session under the owner's scope."""
    from gateway.platforms.api_server import APIServerAdapter, _api_request_profile

    _own_session(served.builder, SESSION, "builder")
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    seen = {}

    async def fake_run_agent(**kwargs):
        seen.update(kwargs)
        seen["home"] = str(get_hermes_home())
        seen["request_profile"] = _api_request_profile.get()
        return {}, {}

    monkeypatch.setattr(adapter, "_run_agent", fake_run_agent)

    with _profile_runtime_scope(served.builder):
        asyncio.run(adapter.run_internal_session_turn(session_id=SESSION, text="wake"))

    assert seen["session_id"] == SESSION
    assert seen["user_message"] == "wake"
    assert seen["session_history_delivery"] == "1"
    assert seen["home"] == str(served.builder)
    assert seen["request_profile"] == "builder"
    # The request-scoped profile binding is restored, never leaked to later callers.
    assert _api_request_profile.get() is None

    # A session the active profile's store does not own fails closed.
    with _profile_runtime_scope(served.builder):
        with pytest.raises(RuntimeError):
            asyncio.run(adapter.run_internal_session_turn(session_id="not-a-session", text="wake"))

    # The concurrent-run cap defers the wake instead of bypassing it (caller rewinds the cursor).
    adapter._max_concurrent_runs = 1
    adapter._inflight_agent_runs = 1
    with _profile_runtime_scope(served.builder):
        with pytest.raises(RuntimeError):
            asyncio.run(adapter.run_internal_session_turn(session_id=SESSION, text="wake"))
