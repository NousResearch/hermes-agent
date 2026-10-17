"""A background delegate child keeps durable authority after its dispatching worker settles.

Production symptom: background ``delegate_task`` reviewers made tool calls, then later calls failed
with ``Durable tool observation refused``. The child's pool thread copies the dispatcher's context,
so it carried the parent's chat-execution identity, and every observation or descendant
reservation demanded that the parent worker still be ``pending`` and current. The parent turn
ending, or the next turn (often the one an earlier child's completion wakes) reserving a newer
generation, refused every later child tool call.

A background unit is now a reserved and started ``async_delegation_thread`` resource of the
dispatching generation, reserved only once the unit is admitted to the pool; while it is started,
its subtree observes and reserves (never another detached unit) against it. A fence or superseded
worker rejects the dispatch, as a retiring backend does at admission (retirement never stops a
running child; its prepare waits the child out). A superseded identity with no resource, and an
unreserved, orphaned, stolen, forged or finished resource still refuse.
"""
import asyncio
from concurrent.futures import Future
from contextvars import copy_context
from dataclasses import replace
import threading

import pytest

from agent.chat_execution_context import (
    bind_chat_execution, bind_concurrent_tool_resource, concurrent_tool_resource_for_submit,
    observe_tool_before_execution,
)
from gateway.chat_execution_recovery import ChatExecutionStore
from tests.agent.test_tool_continuation_resource import producer
from tests.gateway.test_chat_execution_tool_dispatch import dispatch, started
from tests.gateway.test_chat_execution_worker import reserved_worker
from tests.waiting import wait_until
from tools import async_delegation as ad
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _delegation_registry():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def observed(home, generation, key="key"):
    with ChatExecutionStore(home)._db() as db:
        row = db.execute("SELECT tool_observed, effects FROM executions WHERE session_key=? AND generation=?",
                         (key, generation)).fetchone()
    return tuple(row)


def resource_states(home, key="key"):
    return sorted((row["kind"], row["state"]) for row in ChatExecutionStore(home).inspect_resources(key))


def read_only_child(monkeypatch, home, release, *, after=None, key="key"):
    """A disposable read-only reviewer: a tool round, a sequential leaf and a middleware call."""
    reads = []

    def leaf():
        reads.append(observed(home, 1, key))
        return "read"

    sequential = dispatch(monkeypatch, "registry", leaf)
    middleware = producer(monkeypatch, lambda name, args, next_call, **kw: next_call(args), leaf)

    def runner():
        assert release.wait(10)
        observe_tool_before_execution()  # every tool round enters through this (execute_tool_calls_*)
        sequential.execute({})
        middleware()
        if after is not None:
            after()
        return {"status": "completed", "summary": "review done"}

    return runner, reads


def dispatch_background(runner, session_key="key"):
    return ad.dispatch_async_delegation(
        goal="read-only review", context=None, toolsets=None, role="leaf", model=None,
        session_key=session_key, runner=runner)


def completion(delegation_id):
    def find():
        while not process_registry.completion_queue.empty():
            event = process_registry.completion_queue.get_nowait()
            if event.get("delegation_id") == delegation_id:
                return event
        return None
    return wait_until(find, what=f"completion of {delegation_id}")


class QueuedPool:
    """Holds the unit's pool submission so its physical start can be ordered after the parent."""
    def __init__(self, monkeypatch):
        self.jobs = []
        monkeypatch.setattr(ad, "_get_executor", lambda workers: self)

    def submit(self, fn):
        self.jobs.append((fn, Future()))
        return self.jobs[-1][1]

    def start(self):
        fn, future = self.jobs.pop(0)
        fn()
        future.set_result(None)


def child_started(home):
    return wait_until(lambda: ("async_delegation_thread", "started") in resource_states(home),
                      what="background child resource started")


@pytest.mark.asyncio
async def test_background_child_tools_survive_parent_worker_completion(tmp_path, monkeypatch):
    runner, _state, ctx, store = reserved_worker(tmp_path, monkeypatch)
    release = threading.Event()
    child, reads = read_only_child(monkeypatch, tmp_path, release, key=ctx.session_key)
    handles = []
    worker = runner._run_agent_start_turn_worker(
        ctx, lambda: handles.append(dispatch_background(child, ctx.session_key)))
    await worker.executor_task
    assert await asyncio.to_thread(worker.worker_done.wait, 5)
    assert handles[0]["status"] == "dispatched"
    assert store.inspect(ctx.session_key, 1)["worker"] == "completed"
    release.set()
    event = await asyncio.to_thread(completion, handles[0]["delegation_id"])
    assert (event["status"], event["error"]) == ("completed", None)
    assert reads == [(1, "unknown"), (1, "unknown")]
    assert resource_states(tmp_path, ctx.session_key) == [
        ("async_delegation_thread", "finished"), ("middleware_tool_continuation", "finished")]
    proof = store.inspect(ctx.session_key, 1)
    assert (proof["worker"], proof["effects"]) == ("completed", "unknown")


def test_background_child_keeps_its_generation_after_a_successor_turn(tmp_path, monkeypatch):
    store = started(tmp_path)
    release = threading.Event()
    child, reads = read_only_child(monkeypatch, tmp_path, release)
    with bind_chat_execution(tmp_path, "key", 1):
        handle = dispatch_background(child)
    child_started(tmp_path)
    assert store.worker_finished("key", 1)
    assert store.reserve("key", 0, 0, predecessor=(1, "session")) == 2
    assert store.bind_worker_started("key", 2, "successor-session")
    # A new call still bound to the superseded generation has no resource: refused.
    with bind_chat_execution(tmp_path, "key", 1):
        with pytest.raises(RuntimeError, match="Durable tool observation refused"):
            observe_tool_before_execution()
        with pytest.raises(RuntimeError, match="reservation refused"):
            concurrent_tool_resource_for_submit("stale-new-call")
    release.set()
    event = completion(handle["delegation_id"])
    assert (event["status"], event["error"]) == ("completed", None)
    assert reads == [(1, "unknown"), (1, "unknown")]
    assert observed(tmp_path, 2) == (0, "unknown")  # the child never claims the successor's turn
    assert store.inspect("key", 2)["effects"] == "unknown"  # older unresolved work stays relevant


@pytest.mark.parametrize("settled", ["completed", "successor"])
def test_queued_child_start_after_parent_settles_is_admitted(tmp_path, monkeypatch, settled):
    store = started(tmp_path)
    pool = QueuedPool(monkeypatch)
    release = threading.Event()
    release.set()
    child, reads = read_only_child(monkeypatch, tmp_path, release)
    with bind_chat_execution(tmp_path, "key", 1):
        handle = dispatch_background(child)
    assert resource_states(tmp_path) == [("async_delegation_thread", "reserved")]
    assert store.worker_finished("key", 1)
    if settled == "successor":
        assert store.reserve("key", 0, 0, predecessor=(1, "session")) == 2
    pool.start()
    event = completion(handle["delegation_id"])
    assert (event["status"], event["error"]) == ("completed", None)
    assert reads == [(1, "unknown"), (1, "unknown")]


@pytest.mark.parametrize("when", ["running", "queued"])
def test_fenced_session_refuses_background_child_tools(tmp_path, monkeypatch, when):
    store = started(tmp_path)
    pool = QueuedPool(monkeypatch) if when == "queued" else None
    release = threading.Event()
    child, reads = read_only_child(monkeypatch, tmp_path, release)
    with bind_chat_execution(tmp_path, "key", 1):
        handle = dispatch_background(child)
    if when == "running":
        child_started(tmp_path)
    assert store.worker_finished("key", 1)
    assert store.begin_recovery("key", 1, "stop-fence")
    release.set()
    if pool is not None:
        pool.start()
    event = completion(handle["delegation_id"])
    assert event["status"] == "error"
    assert "refused" in event["error"]
    assert reads == []
    assert store.recover("key", 1, "stop-fence", lambda proof: pytest.fail("transfer invoked"))["status"] == "blocked"


def test_retirement_at_dispatch_admission_reserves_and_runs_nothing(tmp_path, monkeypatch):
    from hermes_cli import backend_retirement
    store = started(tmp_path)
    retirement = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", retirement)
    token = retirement.prepare()["token"]
    with bind_chat_execution(tmp_path, "key", 1):
        assert dispatch_background(lambda: pytest.fail("child ran"))["status"] == "rejected"
    assert retirement.cancel(token) == {"ok": True}
    assert store.inspect_resources("key") == []
    assert ad.active_count() == 0


@pytest.mark.parametrize("refusal", ["fence", "superseded"])
def test_journal_refusal_rejects_dispatch_instead_of_raising(tmp_path, refusal):
    # Callers (delegate_task's parent-stop restore, cronjob run's claim fallback) handle only the dict.
    store = started(tmp_path)
    if refusal == "fence":
        assert store.begin_recovery("key", 1, "stop-fence")
    else:
        assert store.reserve("key", 0, 0) == 2  # ordinary admission can supersede a pending worker
    with bind_chat_execution(tmp_path, "key", 1):
        handle = dispatch_background(lambda: pytest.fail("child ran"))
    assert handle["status"] == "rejected"
    assert "chat-execution journal" in handle["error"]
    assert store.inspect_resources("key") == []
    assert ad.active_count() == 0


def ghost(home, resource_id, generation=1):
    """A handle rebuilt from journal rows alone: what a stray holder of a leaked id could present."""
    from agent.chat_execution_context import ConcurrentToolResource
    return ConcurrentToolResource(str(home.resolve()), "key", generation, "session", resource_id, "async_delegation_thread")


def test_capacity_rejected_dispatch_reserves_nothing(tmp_path):
    store = started(tmp_path)
    gate = threading.Event()
    with bind_chat_execution(tmp_path, "key", 1):
        first = ad.dispatch_async_delegation(
            goal="holds the only slot", context=None, toolsets=None, role="leaf", model=None,
            session_key="key", runner=lambda: (gate.wait(10), {"summary": "done"})[1], max_async_children=1)
        second = ad.dispatch_async_delegation(
            goal="over capacity", context=None, toolsets=None, role="leaf", model=None,
            session_key="key", runner=lambda: pytest.fail("child ran"), max_async_children=1)
    try:
        assert (first["status"], second["status"]) == ("dispatched", "rejected")
        rows = store.inspect_resources("key")
        assert [row["resource_id"] for row in rows] == [f"async_delegation:{first['delegation_id']}"]
    finally:
        gate.set()
    assert completion(first["delegation_id"])["status"] == "completed"
    assert resource_states(tmp_path) == [("async_delegation_thread", "finished")]


@pytest.mark.parametrize("failure", ["submit", "cancelled"])
def test_unrun_submission_leaves_no_startable_reservation(tmp_path, monkeypatch, failure):
    store = started(tmp_path)
    pool = QueuedPool(monkeypatch)
    if failure == "submit":
        def refuse(fn):
            raise RuntimeError("injected pool submit failure")
        pool.submit = refuse
    with bind_chat_execution(tmp_path, "key", 1):
        handle = dispatch_background(lambda: pytest.fail("child ran"))
    if failure == "cancelled":
        assert handle["status"] == "dispatched"
        assert pool.jobs[0][1].cancel()
    else:
        assert handle["status"] == "rejected"
    rows = store.inspect_resources("key")
    assert [(row["kind"], row["state"]) for row in rows] == [("async_delegation_thread", "reserved")]
    assert store.worker_finished("key", 1)
    assert store.reserve("key", 0, 0, predecessor=(1, "session")) == 2
    with bind_chat_execution(tmp_path, "key", 1), bind_concurrent_tool_resource(ghost(tmp_path, rows[0]["resource_id"])) as admitted:
        assert not admitted
        with pytest.raises(RuntimeError, match="Durable tool observation refused"):
            observe_tool_before_execution()
    # Never-run work is not settlement proof: the orphaned row keeps both generations unresolved.
    assert store.inspect("key", 1)["effects"] == "unknown"
    assert store.inspect("key", 2)["effects"] == "unknown"


def test_rebuilt_handle_cannot_steal_a_queued_unit_start(tmp_path, monkeypatch):
    store = started(tmp_path)
    pool = QueuedPool(monkeypatch)
    release = threading.Event()
    release.set()
    child, reads = read_only_child(monkeypatch, tmp_path, release)
    with bind_chat_execution(tmp_path, "key", 1):
        handle = dispatch_background(child)
    assert store.worker_finished("key", 1)
    assert store.reserve("key", 0, 0, predecessor=(1, "session")) == 2
    stolen = ghost(tmp_path, f"async_delegation:{handle['delegation_id']}")
    with bind_chat_execution(tmp_path, "key", 1), bind_concurrent_tool_resource(stolen) as admitted:
        assert not admitted
    pool.start()  # the dispatched owner still starts its own unit
    event = completion(handle["delegation_id"])
    assert (event["status"], event["error"]) == ("completed", None)
    assert reads == [(1, "unknown"), (1, "unknown")]


def test_detached_authority_cannot_mint_a_detached_descendant(tmp_path):
    store = started(tmp_path)
    with bind_chat_execution(tmp_path, "key", 1):
        child = concurrent_tool_resource_for_submit("async_delegation:child", kind="async_delegation_thread")
    assert store.worker_finished("key", 1)
    assert store.reserve("key", 0, 0, predecessor=(1, "session")) == 2
    with bind_chat_execution(tmp_path, "key", 1), bind_concurrent_tool_resource(child) as admitted:
        assert admitted
        assert concurrent_tool_resource_for_submit("child-tool-thread") is not None  # ordinary descendant
        with pytest.raises(RuntimeError, match="reservation refused"):
            concurrent_tool_resource_for_submit("async_delegation:grandchild", kind="async_delegation_thread")
    assert ("async_delegation_thread", "reserved") not in resource_states(tmp_path)


@pytest.mark.parametrize("forgery", ["never_reserved", "session_id", "kind", "home", "generation", "finished"])
def test_unreserved_forged_or_finished_child_resource_admits_nothing(tmp_path, forgery):
    store = started(tmp_path)
    with bind_chat_execution(tmp_path, "key", 1):
        handle = concurrent_tool_resource_for_submit("deleg_probe", kind="async_delegation_thread")
    if forgery == "finished":
        with bind_chat_execution(tmp_path, "key", 1), bind_concurrent_tool_resource(handle) as admitted:
            assert admitted
    forged = {
        "never_reserved": replace(handle, resource_id="never-reserved"),
        "session_id": replace(handle, session_id="other-session"),
        "kind": replace(handle, kind="concurrent_tool_thread"),
        "home": replace(handle, home=str((tmp_path / "other").resolve())),
        "generation": replace(handle, generation=2),
        "finished": handle,
    }[forgery]
    assert store.worker_finished("key", 1)
    bodies = []
    with bind_chat_execution(tmp_path, "key", 1), bind_concurrent_tool_resource(forged) as admitted:
        if admitted:
            bodies.append("body")
        with pytest.raises(RuntimeError, match="Durable tool observation refused"):
            observe_tool_before_execution()
        with pytest.raises(RuntimeError, match="reservation refused"):
            concurrent_tool_resource_for_submit("descendant")
    assert bodies == []
    assert observed(tmp_path, 1) == (1, "unknown")  # only the genuine reservation marked it


def test_straggler_outliving_its_child_is_refused(tmp_path, monkeypatch):
    store = started(tmp_path)
    release = threading.Event()
    contexts = []
    child, _reads = read_only_child(monkeypatch, tmp_path, release, after=lambda: contexts.append(copy_context()))
    with bind_chat_execution(tmp_path, "key", 1):
        handle = dispatch_background(child)
    child_started(tmp_path)
    assert store.worker_finished("key", 1)
    release.set()
    assert completion(handle["delegation_id"])["status"] == "completed"
    with pytest.raises(RuntimeError, match="Durable tool observation refused"):
        contexts[0].run(observe_tool_before_execution)
    with pytest.raises(RuntimeError, match="reservation refused"):
        contexts[0].run(concurrent_tool_resource_for_submit, "late-descendant")
