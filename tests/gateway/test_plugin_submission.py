"""Plugin receipts identify durable work, including lost acknowledgements and session rotation."""
import asyncio
import functools
import threading

import hermes_yaml as yaml
import pytest
import pytest_asyncio

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tests.gateway.test_automation_compression_media import _admit_human, _discord_owner

pytestmark = pytest.mark.usefixtures("authority_cwd_in_tmp")


@pytest_asyncio.fixture
async def submission(tmp_path, monkeypatch):
    from gateway.platforms.event import MessageEvent
    from hermes_constants import get_hermes_home

    runner, _, authority, source = await _discord_owner(tmp_path, monkeypatch)
    human = await _admit_human(runner, authority, MessageEvent(
        text="human", source=source, message_id="human"))
    home = get_hermes_home()
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"entries": {
        "receipt-probe": {"allow_gateway_injection": True}}}}), encoding="utf-8")
    manager = PluginManager(scope_key=str(home))
    ctx = PluginContext(PluginManifest(name="receipt-probe", key="receipt-probe", source="user"), manager)
    runner._gateway_loop = asyncio.get_running_loop()
    runner._running = True
    manager.set_gateway_message_injector(runner, runner._schedule_plugin_message_injection)
    identity = {"session_key": runner.session_store._generate_session_key(source),
                "expected_session_id": human.ref.session_id, "request_id": "stable-request"}
    yield ctx, runner, authority, source, identity
    runner._running = False
    manager.clear_gateway_message_injector(runner)


@pytest.mark.asyncio
async def test_receipt_replay_survives_provenance_change_and_observes_terminal_result(submission):
    from gateway.platforms.event import MessageEvent

    ctx, runner, authority, source, identity = submission
    first = await asyncio.to_thread(ctx.submit_message, "do the work", **identity)
    assert first["ok"] and first["status"] == "queued"
    assert first["result"] is None
    assert first["admission_id"]
    await _admit_human(runner, authority, MessageEvent(text="another human turn", source=source,
                                                     message_id="human-2"))
    repeated = await asyncio.to_thread(ctx.submit_message, "do the work", **identity)
    assert repeated["admission_id"] == first["admission_id"]
    conflict = await asyncio.to_thread(ctx.submit_message, "different work", **identity)
    assert conflict["status"] == "refused" and conflict["error"] == "admission_conflict"

    ledger = authority.persistence.admissions
    rows = ledger.list(session_id=identity["expected_session_id"], pending_only=False)
    assert len([row for row in rows if row.payload.get("text") == "do the work"]) == 1
    # Claim and settle the preceding human turn through the same durable FIFO.
    human = ledger.claim(epoch=authority.epoch, session_id=identity["expected_session_id"])
    ledger.settle(epoch=authority.epoch, admission_id=human.admission_id,
                  generation=human.generation, outcome="completed")
    work = ledger.claim(epoch=authority.epoch, session_id=identity["expected_session_id"])
    assert work.admission_id == first["admission_id"]
    active = await asyncio.to_thread(ctx.submission_status, **identity)
    assert active["status"] == "started" and active["result"] is None
    ledger.settle(epoch=authority.epoch, admission_id=work.admission_id, generation=work.generation,
                  outcome="completed", result={"result": {"final_response": "finished", "messages": []}})
    done = await asyncio.to_thread(ctx.submission_status, **identity)
    assert done["status"] == "terminal" and done["outcome"] == "completed"
    assert done["result"] == {"final_response": "finished", "truncated": False}


@pytest.mark.asyncio
async def test_admitted_work_executes_once_and_publishes_its_result(submission, monkeypatch):
    from gateway.platforms.base import SendResult
    from gateway.session_contract import SessionRef
    from gateway.session_results import execution_result

    ctx, runner, authority, source, identity = submission
    executed = []
    started, release = asyncio.Event(), asyncio.Event()

    async def send(*args, **kwargs):
        return SendResult(success=True, message_id="sent")

    async def turn(event, source, quick_key, run_generation):
        executed.append(event.text)
        if event.text == "work":
            started.set()
            await release.wait()
        execution_result.get()["result"] = {"final_response": "result: " + event.text, "messages": []}
        return "result: " + event.text

    monkeypatch.setattr(runner._delivery_adapter_for(source), "send", send)
    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    receipt = await asyncio.to_thread(ctx.submit_message, "work", **identity)
    assert receipt["status"] == "queued" and executed == []
    task = asyncio.create_task(authority._drain(SessionRef(authority.profile_id, identity["expected_session_id"])))
    try:
        await asyncio.wait_for(started.wait(), 10)
        status = await asyncio.to_thread(ctx.submission_status, **identity)
        assert status["status"] == "started" and status["result"] is None
        repeated = await asyncio.to_thread(ctx.submit_message, "work", **identity)
        assert repeated["admission_id"] == receipt["admission_id"]
    finally:
        release.set()
        await asyncio.wait_for(task, 10)
    done = await asyncio.to_thread(ctx.submission_status, **identity)
    assert done["outcome"] == "completed"
    assert done["result"]["final_response"] == "result: work"
    assert executed == ["human", "work"]


@pytest.mark.asyncio
@pytest.mark.parametrize("rotation", ["reset", "compression"])
async def test_pinned_submission_distinguishes_reset_from_compression(submission, rotation):
    from persistence.conversation.lineage import CompressionChild
    from persistence.conversation.messages import NewMessage
    from tests.fixtures.storage_seed import seed_messages

    ctx, runner, authority, _, identity = submission
    root = identity["expected_session_id"]
    if rotation == "reset":
        runner.session_store.reset_session(identity["session_key"])
    else:
        seed_messages(authority.persistence, root, {"role": "user", "content": "human"})
        child = root + "-compressed"
        authority.persistence.lineage.publish_compression_child(
            root, CompressionChild(session_id=child, source="discord"),
            [NewMessage.from_live({"role": "user", "content": "summary"})],
            require_compression_lease=False)
        runner.session_store.advance_compression_session(identity["session_key"], root, child)
    result = await asyncio.to_thread(ctx.submit_message, "pinned work", **identity)
    rows = authority.persistence.admissions.list(session_id=root, pending_only=False)
    work = [row for row in rows if row.payload.get("text") == "pinned work"]
    if rotation == "reset":
        assert result["status"] == "refused" and not work
        current = runner.session_store.lookup_by_session_key(identity["session_key"])
        assert authority.persistence.admissions.list(session_id=current.session_id, pending_only=False) == []
    else:
        assert result["ok"] and len(work) == 1
        assert result["session_id"] == root


@pytest.mark.asyncio
async def test_gateway_loop_refusal_does_not_enqueue(submission):
    ctx, _, authority, _, identity = submission
    result = ctx.submit_message("must not run", **identity)
    assert result["status"] == "refused" and result["error"] == "requires_worker_thread"
    assert len(authority.persistence.admissions.list(
        session_id=identity["expected_session_id"], pending_only=False)) == 1


@pytest.mark.asyncio
async def test_timeout_can_be_recovered_without_duplicate_admission(submission, monkeypatch):
    from gateway.plugin_submission import service

    ctx, _, authority, _, identity = submission
    entered, release = threading.Event(), threading.Event()
    ledger = authority.persistence.admissions
    original = type(ledger).admit

    def held(self, **kwargs):
        entered.set()
        assert release.wait(10), "test did not release the admission writer"
        return original(self, **kwargs)

    monkeypatch.setattr(type(ledger), "admit", held)
    monkeypatch.setattr(service, "call", functools.partial(service.call, timeout=2))
    try:
        result = await asyncio.to_thread(ctx.submit_message, "slow admission", **identity)
        assert entered.is_set()
        assert result["status"] == "unknown" and result["error"] == "timeout"
    finally:
        release.set()
    monkeypatch.setattr(type(ledger), "admit", original)
    repeated = await asyncio.to_thread(ctx.submit_message, "slow admission", **identity)
    assert repeated["ok"]
    rows = ledger.list(session_id=identity["expected_session_id"], pending_only=False)
    assert len([row for row in rows if row.payload.get("text") == "slow admission"]) == 1


@pytest.mark.asyncio
async def test_reset_fence_during_commit_refuses_the_old_request(submission, monkeypatch):
    ctx, runner, authority, _, identity = submission
    entered, release = threading.Event(), threading.Event()
    ledger = authority.persistence.admissions
    original = type(ledger).admit

    def held(self, **kwargs):
        entered.set()
        assert release.wait(10), "test did not release the admission writer"
        return original(self, **kwargs)

    monkeypatch.setattr(type(ledger), "admit", held)
    pending = asyncio.create_task(asyncio.to_thread(ctx.submit_message, "stale work", **identity))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        authority.fence_route(identity["session_key"])
        runner.session_store.reset_session(identity["session_key"])
    finally:
        release.set()
    result = await asyncio.wait_for(pending, 10)
    assert result["status"] == "refused" and result["error"] == "admission_conflict"
    rows = ledger.list(session_id=identity["expected_session_id"], pending_only=False)
    assert not any(row.payload.get("text") == "stale work" for row in rows)


@pytest.mark.asyncio
async def test_other_plugin_cannot_observe_a_receipt_with_the_same_request_id(submission):
    ctx, _, _, _, identity = submission
    receipt = await asyncio.to_thread(ctx.submit_message, "owned work", **identity)
    assert receipt["ok"]
    home = ctx._manager.home_path
    config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    config["plugins"]["entries"]["other-plugin"] = {"allow_gateway_injection": True}
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    other = PluginContext(PluginManifest(name="other-plugin", key="other-plugin", source="user"), ctx._manager)
    result = await asyncio.to_thread(other.submission_status, **identity)
    assert result["status"] == "refused" and result["error"] == "not_found"
    assert result["admission_id"] is None and result["result"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [None, "", "   "])
async def test_invalid_submission_cannot_be_mistaken_for_status(submission, content):
    ctx, _, authority, _, identity = submission
    first = await asyncio.to_thread(ctx.submit_message, "owned work", **identity)
    assert first["ok"]
    invalid = await asyncio.to_thread(ctx.submit_message, content, **identity)
    assert invalid["status"] == "refused" and invalid["error"] == "invalid_params"
    assert invalid["admission_id"] is None
    rows = authority.persistence.admissions.list(session_id=identity["expected_session_id"], pending_only=False)
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_plugin_cancels_only_its_queued_admission(submission):
    ctx, _, authority, _, identity = submission
    queued = await asyncio.to_thread(ctx.submit_message, 'cancel this work', **identity)
    assert queued['controls'] == ['cancel']
    stale = await asyncio.to_thread(ctx.control_submission, 'cancel', **identity,
                                   authority_epoch=queued['authority_epoch'] + 1)
    assert stale['error'] == 'stale_generation'
    cancelled = await asyncio.to_thread(ctx.control_submission, 'cancel', **identity,
                                       authority_epoch=queued['authority_epoch'])
    assert cancelled['status'] == 'terminal'
    assert cancelled['control'] == {'action': 'cancel', 'status': 'accepted', 'delivery': 'none'}
    rows = authority.persistence.admissions.list(session_id=identity['expected_session_id'], pending_only=True)
    assert [row.payload['text'] for row in rows] == ['human']


@pytest.mark.asyncio
async def test_plugin_controls_running_admission_and_refuses_old_generation(submission, monkeypatch):
    from gateway.platforms.base import SendResult
    from gateway.session_contract import SessionRef
    from gateway.session_results import execution_result

    ctx, runner, authority, source, identity = submission
    started, release = asyncio.Event(), asyncio.Event()
    received = []
    interrupted = []

    class Agent:
        def steer(self, text):
            received.append(text)
            return True

        def hard_interrupt(self):
            interrupted.append(True)

    async def send(*args, **kwargs):
        return SendResult(success=True, message_id='sent')

    async def turn(event, source, quick_key, run_generation):
        if event.text == 'work':
            ref = SessionRef(authority.profile_id, identity['expected_session_id'])
            authority.adopt_agent(ref.session_id, run_generation, Agent())
            authority.sessions[ref.session_id].event_stream.note_progress(
                run_generation, 'tool.start', {'tool_id': 'one', 'name': 'terminal'})
            started.set()
            await release.wait()
        execution_result.get()['result'] = {'final_response': 'done', 'messages': []}
        return 'done'

    monkeypatch.setattr(runner._delivery_adapter_for(source), 'send', send)
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    await asyncio.to_thread(ctx.submit_message, 'work', **identity)
    task = asyncio.create_task(authority._drain(SessionRef(authority.profile_id, identity['expected_session_id'])))
    try:
        await asyncio.wait_for(started.wait(), 10)
        observed = await asyncio.to_thread(ctx.submission_status, **identity)
        assert observed['controls'] == ['steer', 'interrupt']
        assert observed['progress']['tool'] == 'terminal'
        args = dict(authority_epoch=observed['authority_epoch'],
                    execution_generation=observed['execution_generation'])
        stale = await asyncio.to_thread(ctx.control_submission, 'steer', **identity,
                                       authority_epoch=args['authority_epoch'],
                                       execution_generation=args['execution_generation'] + 1, text='wrong')
        assert stale['error'] == 'stale_generation' and received == []
        response = await asyncio.to_thread(ctx.control_submission, 'steer', **identity, **args, text='focus tests')
        assert response['control'] == {'action': 'steer', 'status': 'accepted', 'delivery': 'agent'}
        assert received == ['focus tests']
        stopped = await asyncio.to_thread(ctx.control_submission, 'interrupt', **identity, **args)
        assert stopped['control'] == {'action': 'interrupt', 'status': 'accepted', 'delivery': 'agent'}
        assert interrupted == [True]
    finally:
        release.set()
        await asyncio.wait_for(task, 10)
    late = await asyncio.to_thread(ctx.control_submission, 'steer', **identity, **args, text='too late')
    assert late['error'] == 'stale_generation' and received == ['focus tests']


@pytest.mark.asyncio
async def test_control_request_cannot_borrow_another_plugin_receipt(submission):
    ctx, _, _, _, identity = submission
    queued = await asyncio.to_thread(ctx.submit_message, 'owned work', **identity)
    home = ctx._manager.home_path
    config = yaml.safe_load((home / 'config.yaml').read_text())
    config['plugins']['entries']['other-plugin'] = {'allow_gateway_injection': True}
    (home / 'config.yaml').write_text(yaml.safe_dump(config))
    other = PluginContext(PluginManifest(name='other-plugin', key='other-plugin', source='user'), ctx._manager)
    response = await asyncio.to_thread(other.control_submission, 'cancel', **identity,
                                       authority_epoch=queued['authority_epoch'])
    assert response['error'] == 'not_found'
    assert (await asyncio.to_thread(ctx.submission_status, **identity))['status'] == 'queued'


@pytest.mark.asyncio
async def test_claim_winning_cancel_commit_does_not_interrupt_the_claim(submission, monkeypatch):
    ctx, _, authority, _, identity = submission
    receipt = await asyncio.to_thread(ctx.submit_message, 'work', **identity)
    ledger = authority.persistence.admissions
    human = ledger.claim(epoch=authority.epoch, session_id=identity['expected_session_id'])
    ledger.settle(epoch=authority.epoch, admission_id=human.admission_id,
                  generation=human.generation, outcome='completed')
    cancel = type(ledger).cancel

    def claim_first(self, **kwargs):
        claimed = self.claim(epoch=authority.epoch, session_id=identity['expected_session_id'])
        assert claimed.admission_id == receipt['admission_id']
        return cancel(self, **kwargs)

    monkeypatch.setattr(type(ledger), 'cancel', claim_first)
    response = await asyncio.to_thread(ctx.control_submission, 'cancel', **identity,
                                       authority_epoch=receipt['authority_epoch'])
    assert response['error'] == 'stale_generation'
    assert ledger.get(admission_id=receipt['admission_id']).status == 'started'
    assert not authority.pending_stops and not authority.delivered_stops


@pytest.mark.asyncio
async def test_settled_but_unpublished_turn_has_no_controls(submission, monkeypatch):
    from gateway import session_results
    from gateway.platforms.base import SendResult
    from gateway.session_contract import SessionRef

    ctx, runner, authority, source, identity = submission
    entered, release = threading.Event(), threading.Event()
    finish = session_results.finish_result

    def held_finish(store, **kwargs):
        result = finish(store, **kwargs)
        if kwargs['row']['payload']['text'] == 'work':
            entered.set()
            assert release.wait(10)
        return result

    async def send(*args, **kwargs):
        return SendResult(success=True, message_id='sent')

    async def turn(event, *args):
        session_results.execution_result.get()['result'] = {'final_response': 'done', 'messages': []}
        return 'done'

    monkeypatch.setattr(session_results, 'finish_result', held_finish)
    monkeypatch.setattr(runner._delivery_adapter_for(source), 'send', send)
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    await asyncio.to_thread(ctx.submit_message, 'work', **identity)
    task = asyncio.create_task(authority._drain(SessionRef(authority.profile_id, identity['expected_session_id'])))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        observed = await asyncio.to_thread(ctx.submission_status, **identity)
        assert observed['status'] == 'started' and observed['result'] is None
        assert observed['controls'] == [] and observed['progress'] == {}
    finally:
        release.set()
        await asyncio.wait_for(task, 10)
    assert (await asyncio.to_thread(ctx.submission_status, **identity))['status'] == 'terminal'
