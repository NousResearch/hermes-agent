"""A closed local conversation nobody views leaves the authority's memory, as a restart leaves it.

The last viewer's close stamps the row and runs the close lifecycle; the LiveSession (event stream,
replay ring key, locks) then left memory only at restart, so a long-lived gateway held one per
conversation ever closed. Retirement must never strand input admitted while the close ran, must
keep a closed branch's shared launch key while the branch can still resume, and a later delete or
prune must still forget what the local adapter keeps for the retired conversation.
"""
import asyncio

import pytest

from tests.waiting import async_wait_until


async def _owner(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    db = SessionDB(tmp_path / "state.db")
    runner.session_store._db = runner._session_db = db
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner", db=db)
    return runner, authority, db


def _viewer():
    from gateway.session_contract import Principal
    return Principal("test-owner", "default", frozenset(
        {"session:create", "session:submit", "session:read", "session:control"}), "viewer")


def _create(authority, request_id="r"):
    from gateway.session_local import create_local_session
    return create_local_session(authority, _viewer(), dict(request_id=request_id, source="tui", model="fixture",
                                                           toolsets=[]))


async def _close(authority, ref):
    subscription = (await authority.attach(_viewer(), ref)).subscription_id
    await authority.detach(_viewer(), subscription, end_reason="tui_shutdown")


@pytest.mark.asyncio
async def test_closed_local_session_leaves_memory_and_resumes_with_its_source(tmp_path, monkeypatch):
    runner, authority, db = await _owner(tmp_path, monkeypatch)
    replies = []

    async def turn(event):
        replies.append(event.text)
        return "ok"

    monkeypatch.setattr(runner, "_handle_message", turn)
    try:
        ref = _create(authority)
        source = authority.sessions[ref.session_id].source
        await _close(authority, ref)
        await async_wait_until(lambda: ref.session_id not in authority.sessions, timeout=10,
                               what="the closed session to leave memory")
        assert db.get_session(ref.session_id)["end_reason"] == "tui_shutdown"
        # Resume: the next viewer gets it back with the very source its background work holds,
        # and its next prompt runs.
        await authority.attach(_viewer(), ref)
        live = authority.sessions[ref.session_id]
        assert live.source is source
        assert runner._delivery_adapter_for(source).authorize_source(source)
        from gateway.session_contract import Submission
        await authority.submit(_viewer(), Submission("again", ref, {"text": "again"}, "queue"))
        await async_wait_until(lambda: replies == ["again"], timeout=10, what="the resumed turn")
    finally:
        db.close()


@pytest.mark.asyncio
async def test_prompt_admitted_while_the_close_runs_is_never_stranded(tmp_path, monkeypatch):
    """A submit lands while the close lifecycle awaits (its drain is scheduled, and may clear
    ``rescan`` and start claiming, before the lifecycle returns): the turn runs and the session
    stays, whichever point of the claim the lifecycle returns at."""
    from gateway.session_contract import Submission
    from hermes_state_runtime import list_session_admissions

    runner, authority, db = await _owner(tmp_path, monkeypatch)
    claim_started, release_claim = asyncio.Event(), asyncio.Event()
    replies = []

    async def turn(event):
        replies.append(event.text)
        return "ok"

    original_claim = authority._claim_next

    async def held_claim(ref, live):
        if replies == [] and not release_claim.is_set():
            async with live.claim_gate:  # parked mid-claim, as a drain awaiting the claim's write is
                claim_started.set()
                await release_claim.wait()
        return await original_claim(ref, live)

    async def finalize(**_kwargs):
        # The lifecycle's await: a prompt arrives and its drain starts claiming meanwhile.
        await authority.submit(_viewer(), Submission("racing", ref, {"text": "racing"}, "queue"))
        await asyncio.wait_for(claim_started.wait(), 10)
        return []

    monkeypatch.setattr(runner, "_handle_message", turn)
    monkeypatch.setattr(runner, "_finalize_session_off_loop", finalize, raising=False)
    monkeypatch.setattr(authority, "_claim_next", held_claim)
    try:
        ref = _create(authority)
        await _close(authority, ref)
        await asyncio.wait_for(claim_started.wait(), 10)
        await asyncio.sleep(0)
        assert ref.session_id in authority.sessions  # never retired under a claim in progress
        release_claim.set()
        await async_wait_until(lambda: replies == ["racing"], timeout=10, what="the racing turn")
        await async_wait_until(
            lambda: all(row["status"] == "terminal" for row in list_session_admissions(
                db, session_id=ref.session_id, pending_only=False)), timeout=10, what="its settlement")
        # The turn reopened the row, so the conversation is no longer the closed one: it stays.
        assert ref.session_id in authority.sessions
    finally:
        db.close()


@pytest.mark.asyncio
async def test_retired_session_is_forgotten_by_delete_and_prune(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.session_mutations import retire_live_sessions

    runner, authority, db = await _owner(tmp_path, monkeypatch)
    try:
        ref = _create(authority)
        await _close(authority, ref)
        await async_wait_until(lambda: ref.session_id not in authority.sessions, timeout=10,
                               what="the closed session to leave memory")
        local = runner._delivery_adapter_for(type("S", (), {"platform": Platform.LOCAL})())
        assert ref.session_id in local.sources and ref.session_id in local.policies
        retire_live_sessions(authority, [ref.session_id])  # the bulk prune's path: no restore first
        assert ref.session_id not in local.sources
        assert ref.session_id not in local.policies
        assert ref.session_id not in authority.launch_sources
    finally:
        db.close()


@pytest.mark.asyncio
async def test_prompt_parked_behind_a_busy_head_while_the_close_runs_is_never_stranded(tmp_path, monkeypatch):
    """A submit lands while the close lifecycle awaits, and its drain parks it behind a busy head
    (a worker holds the FIFO) and exits before the lifecycle returns. The session must stay, so the
    worker's end wakes the FIFO (``wake_after_worker`` only wakes sessions in memory) and it runs."""
    from gateway.session_contract import Submission

    runner, authority, db = await _owner(tmp_path, monkeypatch)
    replies, parked = [], asyncio.Event()

    async def turn(event):
        replies.append(event.text)
        return "ok"

    original_claim = authority._claim_next

    async def busy_claim(ref, live):
        if not parked.is_set():  # the head is held by a worker: the drain pauses ``session_busy``
            live.rescan = False
            parked.set()
            return None, {"admission_id": "held"}
        return await original_claim(ref, live)

    async def finalize(**_kwargs):
        await authority.submit(_viewer(), Submission("parked", ref, {"text": "parked"}, "queue"))
        await asyncio.wait_for(parked.wait(), 10)
        live = authority.sessions[ref.session_id]
        await async_wait_until(lambda: live.task is None or live.task.done(), timeout=10,
                               what="the parked drain to exit")
        return []

    monkeypatch.setattr(runner, "_handle_message", turn)
    monkeypatch.setattr(runner, "_finalize_session_off_loop", finalize, raising=False)
    monkeypatch.setattr(authority, "_claim_next", busy_claim)
    try:
        ref = _create(authority)
        await _close(authority, ref)
        await asyncio.wait_for(parked.wait(), 10)
        live = authority.sessions[ref.session_id]
        # ``closed_stamp`` is set by the retirement check, in the loop step that would drop it.
        await async_wait_until(lambda: live.closed_stamp is not None, timeout=10, what="the close to settle")
        assert ref.session_id in authority.sessions  # its queued prompt pins it
        authority.wake_after_worker(ref.session_id)  # the worker ended
        await async_wait_until(lambda: replies == ["parked"], timeout=10, what="the parked turn")
    finally:
        db.close()


async def _branching_owner(tmp_path, monkeypatch):
    """An owner whose local sessions carry a launch key and can branch (test_launch_secret_retirement)."""
    from types import SimpleNamespace

    from gateway import run, session_mutations
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    config = {'platform_toolsets': {'cli': []}, 'model': {
        'provider': 'custom', 'api_key': 'config-secret', 'base_url': 'http://127.0.0.1:9/v1'}}
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a, **k: config)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _evict_cached_agent=lambda route: None,
                             _set_session_reasoning_override=lambda route, value: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    owner = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='fixture')
    connection = AuthorityConnection(owner, object(), {'user_id': 'owner'})

    async def mutate(ref, operation):
        handle = owner._handle(ref)
        return await session_mutations.mutate_session(owner, connection.actor, ref, dict(
            session_id=ref.session_id, request_id=operation + ref.session_id, operation=operation,
            payload={}, expected_revision=handle.revision, expected_generation=handle.execution_generation))
    return owner, connection, mutate


@pytest.mark.asyncio
async def test_closed_branch_keeps_its_parents_launch_key_after_the_parent_is_deleted(tmp_path, monkeypatch):
    """A branch copies its parent's launch-key slot. Closed (and so retired from memory), it still
    resumes on that key after the parent is deleted, as an open branch does."""
    from gateway.config import Platform
    from gateway.session_contract import SessionRef
    from gateway.session_local import create_local_session
    from gateway.session_policy import launch_key

    owner, connection, mutate = await _branching_owner(tmp_path, monkeypatch)
    try:
        parent = create_local_session(owner, connection.actor, {'request_id': 'parent', 'source': 'cli',
            'cwd': str(tmp_path), 'model': 'm', 'toolsets': [], 'api_key': 'launch-parent'})
        branch = SessionRef(parent.profile_id, (await mutate(parent, 'branch'))['branched_session_id'])
        subscription = (await owner.attach(connection.actor, branch)).subscription_id
        await owner.detach(connection.actor, subscription, end_reason='tui_shutdown')
        await async_wait_until(lambda: branch.session_id not in owner.sessions, timeout=10,
                               what='the closed branch to leave memory')
        await mutate(parent, 'delete')
        assert 'launch-parent' in owner._local_launch_keys.values()
        await owner.attach(connection.actor, branch)  # resume
        local = owner.runner.adapters[Platform.LOCAL]
        assert launch_key(owner, local.policies[branch.session_id]) == 'launch-parent'
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_retired_conversation_a_rename_brings_back_retires_again(tmp_path, monkeypatch):
    """A rename (any authorize: a read RPC, archive, pin) restores a retired conversation to run
    its checks; once it sits idle again it leaves memory again instead of staying until restart."""
    from gateway import session_retirement
    from gateway.session_mutations import mutate_session
    monkeypatch.setattr(session_retirement, 'RESTORED_IDLE_S', 0.05)
    _runner, authority, db = await _owner(tmp_path, monkeypatch)
    try:
        ref = _create(authority)
        await _close(authority, ref)
        await async_wait_until(lambda: ref.session_id not in authority.sessions, timeout=10,
                               what="the closed session to leave memory")
        handle = authority._handle(ref)
        await mutate_session(authority, _viewer(), ref, dict(
            session_id=ref.session_id, request_id='rename1', operation='rename', payload={'title': 'Renamed'},
            expected_revision=handle.revision, expected_generation=handle.execution_generation))
        assert db.get_session(ref.session_id)['title'] == 'Renamed'
        await async_wait_until(lambda: ref.session_id not in authority.sessions, timeout=10,
                               what="the renamed conversation to retire again")
        await authority.attach(_viewer(), ref)  # and it still resumes
        assert ref.session_id in authority.sessions
    finally:
        db.close()


@pytest.mark.asyncio
async def test_closed_conversation_with_a_running_refine_review_stays(tmp_path, monkeypatch):
    """A cold /refine's review run token lives on the LiveSession: closing mid-review keeps it, so a
    reopen's second /refine joins that review and the next turn cancels it."""
    import threading
    from types import SimpleNamespace

    from gateway import session_retirement
    _runner, authority, db = await _owner(tmp_path, monkeypatch)
    checked = []
    retire_closed = session_retirement.retire_closed

    def recorded(*args):
        checked.append(retire_closed(*args))
        return checked[-1]
    monkeypatch.setattr(session_retirement, 'retire_closed', recorded)
    try:
        ref = _create(authority)
        live = authority.sessions[ref.session_id]
        live.review._background_review_run = SimpleNamespace(request_done=threading.Event())
        await _close(authority, ref)
        await async_wait_until(lambda: checked, timeout=10, what="the close's retirement check")
        assert checked == [False] and authority.sessions.get(ref.session_id) is live
    finally:
        db.close()
