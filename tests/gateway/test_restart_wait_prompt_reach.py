"""A graceful restart's after-turn wait stays reachable for a turn parked on a human prompt (#139041).

The draining owner still owns its in-flight turn, so a client must be able to attach and answer the
pending approval/clarify. New input is committed queued for the next owner and never claimed here.
"""
import asyncio
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch, list_session_admissions


@pytest.mark.platforms("linux")
def test_ensure_hands_out_the_restart_wait_owner_as_draining_with_its_endpoint(tmp_path):
    from tests.hermes_cli.test_gateway_runtime import control_peer
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    from hermes_cli.gateway_runtime_cli import ensure_exit_code

    async def probe():
        home = tmp_path / "profile"
        identity = str(home.resolve())
        payload = {"protocol": 1, "runtime_protocol": 1, "instance_id": "owner", "authority_epoch": 1,
                   "hermes_home": identity, "state": "draining", "drain_reason": "restart_wait",
                   "api_origin": "http://127.0.0.1:41817", "supervisor": "none",
                   "served_profiles": [{"profile_id": identity, "home": identity}],
                   "capabilities": ["session-authority-v1"]}
        async with control_peer(home, payload):
            wait = await asyncio.to_thread(discover_gateway_endpoint, home, timeout=1)
            # Still draining (exit 6, never a fresh-start target) but dialable for answering prompts.
            assert (wait.state, wait.reason_code) == ("draining", "restart_wait")
            assert wait.endpoint is not None and wait.endpoint.instance_id == "owner"
            assert ensure_exit_code(wait) == 6
            # Once stop() begins the reason is withdrawn: a plain refusal, nothing to dial.
            del payload["drain_reason"]
            stopping = await asyncio.to_thread(discover_gateway_endpoint, home, timeout=1)
            assert stopping.state == "draining" and stopping.endpoint is None

    asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_update_fence_hands_out_only_the_draining_owners_restart_wait(tmp_path):
    """`hermes update`'s drain (pause-for-update) is a restart wait: while the update marker is live,
    ensure still hands that owner out so a parked prompt can be answered; anything else (a ready owner,
    a stopping one, none at all) stays ``update_paused`` and nothing is ever started."""
    import os
    import time
    from tests.hermes_cli.test_gateway_runtime import control_peer
    from hermes_cli.gateway_runtime import ensure_gateway_runtime
    from hermes_cli.update_lock import MARKER_NAME, process_create_time

    home = tmp_path / "profile"
    identity = str(home.resolve())
    payload = {"protocol": 1, "runtime_protocol": 1, "instance_id": "owner", "authority_epoch": 1,
               "hermes_home": identity, "state": "draining", "drain_reason": "restart_wait",
               "api_origin": "http://127.0.0.1:41817", "supervisor": "none",
               "served_profiles": [{"profile_id": identity, "home": identity}],
               "capabilities": ["session-authority-v1"]}

    def ensure():
        result = ensure_gateway_runtime(home, timeout=5)
        return result.state, result.reason_code, None if result.endpoint is None else result.endpoint.instance_id

    async def probe():
        async with control_peer(home, payload):
            (home / MARKER_NAME).write_text(
                f"{os.getpid()}\n{int(time.time())}\nct:{process_create_time():.3f}\n", encoding="utf-8")
            assert await asyncio.to_thread(ensure) == ("draining", "restart_wait", "owner")
            del payload["drain_reason"]  # stop() began
            assert await asyncio.to_thread(ensure) == ("draining", "update_paused", None)
            payload["state"] = "ready"
            assert await asyncio.to_thread(ensure) == ("draining", "update_paused", None)
        assert await asyncio.to_thread(ensure) == ("draining", "update_paused", None)
        assert not (home / "logs").exists()  # no start was attempted while fenced

    asyncio.run(probe())


@pytest.mark.asyncio
async def test_restart_wait_commits_new_input_queued_and_never_claims_it(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import Principal, SessionRef, Submission

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="telegram")
    ran = []

    async def handle(event):
        ran.append(event)
        return "ran"

    runner = SimpleNamespace(_draining=True, _running=True, config=SimpleNamespace(multiplex_profiles=False),
                             session_runtime_descriptor={"state": "draining", "drain_reason": "restart_wait"},
                             _adapter_for_source=lambda source: None, _handle_message=handle)
    authority = SessionAuthority(runner, profile_id="p", instance_id="owner", db=db,
                                 epoch=begin_runtime_epoch(db, instance_id="owner"))
    authority.sessions["s"] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id="c", user_id="human"), "route")
    actor = Principal("human", "p", frozenset({"session:submit"}), "t")
    try:
        receipt = await authority.submit(actor, Submission("r1", SessionRef("p", "s"), {"text": "next"}, "queue"),
                                         queue_while_restarting=True)
        task = authority.sessions["s"].task
        if task is not None:
            await asyncio.wait_for(task, 10)
        rows = list_session_admissions(db, session_id="s")
        assert [(r["admission_id"], r["status"]) for r in rows] == [(receipt.admission_id, "queued")]
        assert ran == []  # the draining owner never runs it; the next owner's FIFO does
        # stop() withdraws the window: input is refused, as before.
        del runner.session_runtime_descriptor["drain_reason"]
        with pytest.raises(RuntimeStoreError, match="runtime_draining"):
            await authority.submit(actor, Submission("r2", SessionRef("p", "s"), {"text": "late"}, "queue"),
                                   queue_while_restarting=True)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_restart_wait_serves_only_the_reads_that_open_the_parked_session():
    """Desktop finds and opens the parked conversation over HTTP (sidebar, detail, transcript); every
    write and unrelated route keeps the drain's 503, and stop() withdraws the reads too."""
    from gateway.run_api import GatewayRuntimeAPI

    reached = []

    async def app(scope, receive, send):
        reached.append((scope['method'], scope['path']))
        await send({'type': 'http.response.start', 'status': 200, 'headers': []})
        await send({'type': 'http.response.body', 'body': b'{}'})

    runner = SimpleNamespace(_draining=True, _running=True,
                             session_runtime_descriptor={'state': 'draining', 'drain_reason': 'restart_wait'})
    api = GatewayRuntimeAPI(app, runner, SimpleNamespace(state=None))

    async def status(method, path):
        sent = []

        async def receive():
            return {'type': 'http.request', 'body': b''}

        async def send(message):
            sent.append(message)
        await api({'type': 'http', 'method': method, 'path': path, 'headers': [], 'query_string': b'',
                   'client': ('127.0.0.1', 1)}, receive, send)
        return sent[0]['status']

    reads = ['/api/profiles/sessions/sidebar', '/api/profiles/sessions', '/api/sessions/s1',
             '/api/sessions/s1/messages', '/api/sessions/s1/messages/around', '/api/sessions/s1/timeline',
             '/api/health', '/api/status']
    assert [await status('GET', path) for path in reads] == [200] * len(reads)
    refused = [('PATCH', '/api/sessions/s1'), ('DELETE', '/api/sessions/s1'), ('POST', '/api/sessions'),
               ('GET', '/api/config'), ('GET', '/api/sessions/s1/export/x'), ('POST', '/api/model/set')]
    assert [await status(m, path) for m, path in refused] == [503] * len(refused)
    del runner.session_runtime_descriptor['drain_reason']  # stop() began: nothing is served anymore
    assert await status('GET', '/api/profiles/sessions/sidebar') == 503
    assert reached == [('GET', path) for path in reads]


@pytest.mark.asyncio
async def test_restart_wait_ws_serves_the_parked_turn_and_refuses_every_other_verb(tmp_path):
    """The interactive socket a restart's after-turn wait admits may resume, read and answer the
    parked prompt; a write verb (config.set, session.save, process.kill, reload.*, ...) is refused
    with runtime_draining before its handler runs on the shutting-down owner."""
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_controls import AuthorityConnection

    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="test")
    runner = SimpleNamespace(_draining=True, _running=True,
                             session_runtime_descriptor={"state": "draining", "drain_reason": "restart_wait"})
    authority = SessionAuthority(runner, profile_id="p", instance_id="owner", db=db,
                                 epoch=begin_runtime_epoch(db, instance_id="owner"))
    authority.sessions["s"] = LiveSession(None, "route")
    viewer = AuthorityConnection(authority, object(), {"user_id": "human"}, operator=True)

    async def call(method, **params):
        return await viewer.dispatch({"id": 1, "method": method, "params": {"session_id": "s", **params}})

    try:
        assert "result" in await call("session.resume")
        answered = await call("approval.respond", execution_generation=1, prompt_id="gone", choice="once")
        # Reached the authority: a stale prompt is its own verdict, never the drain's refusal.
        assert answered.get("error", {}).get("message") != "runtime_draining"
        for method, params in [("config.set", {"key": "model", "value": "x"}), ("session.save", {}),
                               ("process.kill", {"pid": 1}), ("reload.mcp", {}), ("rollback.restore", {})]:
            refused = await call(method, **params)
            assert refused["error"]["message"] == "runtime_draining", method
    finally:
        await viewer.close()
        db.close()


@pytest.mark.asyncio
async def test_restart_wait_queues_only_interactive_input(tmp_path, monkeypatch):
    """Only the interactive WS submit commits queued input during the wait; a2a, kanban and hosted
    rooms call ``authority.submit`` directly and keep the plain refusal, so none of them can read a
    released waiter as failure and resubmit the same input under a new request id."""
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import Principal, SessionRef, Submission

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="telegram")
    runner = SimpleNamespace(_draining=True, _running=True, config=SimpleNamespace(multiplex_profiles=False),
                             session_runtime_descriptor={"state": "draining", "drain_reason": "restart_wait"},
                             _adapter_for_source=lambda source: None, _handle_message=None)
    authority = SessionAuthority(runner, profile_id="p", instance_id="owner", db=db,
                                 epoch=begin_runtime_epoch(db, instance_id="owner"))
    authority.sessions["s"] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id="c", user_id="peer"), "route")
    peer = Principal("peer", "p", frozenset({"session:submit"}), "t")
    try:
        with pytest.raises(RuntimeStoreError, match="runtime_draining"):
            await authority.submit(peer, Submission("a2a:1", SessionRef("p", "s"), {"text": "hi"}, "queue"))
        assert list_session_admissions(db, session_id="s") == []
    finally:
        db.close()


def test_update_drain_hint_names_the_draining_profile(tmp_path, monkeypatch):
    """The updater's hint opens the profile whose gateway holds the parked turn: a bare
    ``hermes --tui`` opens the default profile, where that session isn't listed (#139041)."""
    from hermes_cli.update_cmd_drain_report import format_drain_report

    root = tmp_path / ".hermes"
    (root / "profiles" / "work").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root))
    work = [{"kind": "chat", "session": "s1"}]
    assert "`hermes -p work --tui`" in format_drain_report(work, remaining_s=60, home=root / "profiles" / "work")
    assert "`hermes --tui`" in format_drain_report(work, remaining_s=60, home=root)
