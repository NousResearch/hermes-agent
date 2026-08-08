"""Contract tests for the plugin durable external background-task API.

Covers the public surface on ``PluginContext.background_tasks``:
registration bound to the ACTIVE host parent, plugin-owned unguessable
tamper-evident handles, idempotent registration / terminal events, bounded
payloads, atomic single terminal transitions, durable cancel intent, restart
restore, and temp ``HERMES_HOME`` isolation.
"""

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.background_tasks import (
    BackgroundTaskError,
    _ExternalBackgroundTasksService,
    _create_external_background_tasks_service,
    ExternalTaskHandle,
    ExternalTaskResult,
    ExternalTaskState,
    ExternalTaskStatus,
)
from agent.background_tasks_store import connect, load_or_create_hmac_key
from agent.host_context import bind_host_parent, get_active_host_parent
from hermes_constants import get_hermes_home
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _clean_queue():
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _parent(session_id="parent-1"):
    return SimpleNamespace(session_id=session_id)


def _service(plugin_id="test-plugin", resolver=None):
    return _create_external_background_tasks_service(
        plugin_id=plugin_id,
        parent_agent_resolver=resolver or get_active_host_parent,
    )


def _load_hmac_key_from_process(hermes_home: str) -> bytes:
    os.environ["HERMES_HOME"] = hermes_home
    connection = connect()
    try:
        return load_or_create_hmac_key(connection)
    finally:
        connection.close()


def _drain_one():
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not process_registry.completion_queue.empty():
            return process_registry.completion_queue.get_nowait()
        time.sleep(0.01)
    return None


# ---------------------------------------------------------------------------
# PluginContext exposure
# ---------------------------------------------------------------------------


def test_plugin_context_lazily_exposes_service_and_captures_identity():
    from hermes_cli.plugins import PluginContext, PluginManifest

    mgr = MagicMock()
    manifest = PluginManifest(
        name="ext-tasks-plugin", key="ext-tasks-plugin", source="user"
    )
    ctx = PluginContext(manifest, mgr)

    assert ctx._background_tasks is None
    svc = ctx.background_tasks
    assert isinstance(svc, _ExternalBackgroundTasksService)
    assert svc is ctx.background_tasks  # cached, built once
    assert svc.plugin_id == "ext-tasks-plugin"

    # The service resolves the ACTIVE host parent supplied by Hermes.
    assert svc._parent_agent_resolver is get_active_host_parent


def test_plugin_context_service_is_per_plugin_identity():
    from hermes_cli.plugins import PluginContext, PluginManifest

    mgr = MagicMock()
    a = PluginContext(PluginManifest(name="plugin-a", key="plugin-a"), mgr)
    b = PluginContext(PluginManifest(name="plugin-b", key="plugin-b"), mgr)

    assert a.background_tasks.plugin_id == "plugin-a"
    assert b.background_tasks.plugin_id == "plugin-b"


def test_plugin_context_snapshots_identity_before_lazy_service_access():
    from hermes_cli.plugins import PluginContext, PluginManifest

    manifest = PluginManifest(name="plugin-a", key="plugin-a")
    ctx = PluginContext(manifest, MagicMock())
    manifest.key = "plugin-b"

    assert ctx.background_tasks.plugin_id == "plugin-a"


def test_direct_service_construction_is_host_only():
    with pytest.raises(BackgroundTaskError, match="host-owned"):
        _ExternalBackgroundTasksService(
            plugin_id="plugin-b",
            parent_agent_resolver=get_active_host_parent,
        )


def test_service_implementation_is_not_a_public_module_export():
    import agent.background_tasks as background_tasks

    assert "ExternalBackgroundTasksService" not in background_tasks.__all__
    assert not hasattr(background_tasks, "ExternalBackgroundTasksService")


def test_plugin_service_identity_is_read_only():
    service = _service("plugin-a")

    with pytest.raises(AttributeError):
        service.plugin_id = "plugin-b"  # type: ignore[misc]

    assert service.plugin_id == "plugin-a"


def test_hmac_key_initialization_is_atomic_across_processes(tmp_path):
    hermes_home = str(tmp_path / "shared-home")

    with ProcessPoolExecutor(max_workers=4) as executor:
        keys = list(executor.map(_load_hmac_key_from_process, [hermes_home] * 8))

    assert len(set(keys)) == 1


# ---------------------------------------------------------------------------
# Parent binding
# ---------------------------------------------------------------------------


def test_register_outside_active_parent_fails():
    svc = _service()
    with pytest.raises(BackgroundTaskError):
        svc.register_external(external_id="run-1")


def test_register_binds_only_the_active_host_parent():
    svc = _service()
    with bind_host_parent(_parent("parent-a")):
        handle = svc.register_external(external_id="run-1", payload={"q": 1})
    assert handle.parent_session_id == "parent-a"

    with bind_host_parent(_parent("parent-b")):
        handle_b = svc.register_external(external_id="run-1", payload={"q": 1})
    assert handle_b.parent_session_id == "parent-b"
    assert handle_b.task_id != handle.task_id
    assert handle.plugin_id == "test-plugin"


def test_public_contract_results_are_json_serializable_mappings():
    svc = _service()
    with bind_host_parent(_parent("parent-a")):
        handle = svc.register_external(external_id="run-1")

    status = svc.list_pending()[0]
    completed = svc.complete(handle, event_id="event-1", summary="done")

    assert json.loads(json.dumps(status.to_dict()))["state"] == "registered"
    assert json.loads(json.dumps(completed.to_dict()))["state"] == "completed"


def test_public_contract_rejects_non_finite_json_numbers():
    svc = _service()
    with bind_host_parent(_parent("parent-a")):
        with pytest.raises(BackgroundTaskError):
            svc.register_external(
                external_id="run-nan", payload={"value": float("nan")}
            )

        handle = svc.register_external(external_id="run-1")

    with pytest.raises(BackgroundTaskError):
        svc.complete(
            handle,
            event_id="event-nan",
            summary="done",
            result_payload={"value": float("inf")},
        )


def test_external_failure_message_uses_generic_background_task_wording():
    from tools.process_registry import _format_async_delegation

    svc = _service()
    with bind_host_parent(_parent("parent-a")):
        handle = svc.register_external(external_id="run-1", label="code run")
    assert svc.fail(handle, event_id="event-1", error="provider failed").accepted
    event = _drain_one()
    assert event is not None

    rendered = _format_async_delegation(event)
    assert "The background task did not complete successfully" in rendered
    assert "The subagent did not complete successfully" not in rendered


def test_register_cannot_be_forged_through_method_parameters():
    """The public surface has no parent/session parameters at all."""
    import inspect

    sig = inspect.signature(_ExternalBackgroundTasksService.register_external)
    params = set(sig.parameters)
    assert params == {
        "self",
        "external_id",
        "payload",
        "idempotency_key",
        "label",
    }
    with bind_host_parent(_parent("parent-a")):
        handle = _service().register_external(external_id="run-1")
    assert handle.parent_session_id == "parent-a"


def test_registration_requires_a_durable_parent_session():
    svc = _service()
    with bind_host_parent(_parent("")):
        with pytest.raises(BackgroundTaskError):
            svc.register_external(external_id="run-1")


def test_register_captures_gateway_session_routing_from_parent_context():
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(
        platform="telegram",
        source="telegram",
        session_key="agent:main:telegram:dm:12345:678",
        ui_session_id="desktop-sid-9",
    )
    try:
        svc = _service()
        with bind_host_parent(_parent("parent-telegram")):
            handle = svc.register_external(
                external_id="run-1", label="classify the image"
            )
        svc.complete(handle, event_id="e1", summary="done")
    finally:
        clear_session_vars(tokens)

    evt = _drain_one()
    assert evt is not None
    assert evt["session_key"] == "agent:main:telegram:dm:12345:678"
    assert evt["origin_ui_session_id"] == "desktop-sid-9"
    assert evt["parent_session_id"] == "parent-telegram"
    assert evt["type"] == "async_delegation"


# ---------------------------------------------------------------------------
# Plugin ownership / handle security
# ---------------------------------------------------------------------------


def test_cross_plugin_operations_rejected_without_existence_leak():
    owner = _service("owner-plugin")
    intruder = _service("intruder-plugin")
    with bind_host_parent(_parent("parent-1")):
        handle = owner.register_external(external_id="run-1")

    assert intruder.list_pending() == []

    res = intruder.complete(handle, event_id="e1", summary="x")
    assert res.unknown_handle is True
    assert res.accepted is False

    res = intruder.fail(handle, event_id="e1", error="x")
    assert res.unknown_handle is True

    res = intruder.request_cancel(handle)
    assert res.unknown_handle is True

    # A forged handle claiming the intruder's own plugin id but the owner's
    # task_id must also be unknown (signature mismatch) and still leak nothing.
    forged = ExternalTaskHandle(
        contract_version=1,
        task_id=handle.task_id,
        plugin_id="intruder-plugin",
        parent_session_id="parent-1",
        created_at=handle.created_at,
        signature="0" * 64,
    )
    res = intruder.complete(forged, event_id="e1", summary="x")
    assert res.unknown_handle is True
    assert intruder.list_pending() == []


def test_handle_tampering_is_rejected():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        handle = svc.register_external(external_id="run-1")

    cases = []
    d = handle.to_dict()
    d2 = dict(d)
    d2["signature"] = "0" * 64
    cases.append(d2)
    d3 = dict(d)
    d3["parent_session_id"] = "parent-forged"
    cases.append(d3)
    d4 = dict(d)
    d4["task_id"] = "deadbeef" * 4
    cases.append(d4)
    d5 = dict(d)
    d5["plugin_id"] = "other-plugin"
    cases.append(d5)

    for forged in cases:
        res = svc.complete(forged, event_id="e1", summary="x")
        assert res.unknown_handle is True
        assert res.accepted is False


def test_malformed_handle_is_unknown_not_crash():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        handle = svc.register_external(external_id="run-1")
    res = svc.complete(handle.to_dict(), event_id="e1", summary="x")
    assert res.accepted is True
    # Unknown/garbage handles are rejected cleanly.
    res = svc.complete("garbage", event_id="e1", summary="x")
    assert res.unknown_handle is True


# ---------------------------------------------------------------------------
# Registration idempotency
# ---------------------------------------------------------------------------


def test_registration_replay_returns_same_handle():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        a = svc.register_external(external_id="run-1", payload={"q": 1})
        b = svc.register_external(external_id="run-1", payload={"q": 1})
    assert a == b
    assert a.task_id == b.task_id


def test_registration_conflicting_replay_fails():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        svc.register_external(external_id="run-1", payload={"q": 1})
        with pytest.raises(BackgroundTaskError):
            svc.register_external(external_id="run-1", payload={"q": 2})


def test_registration_idempotency_key_defaults_to_external_id():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        a = svc.register_external(external_id="run-1", idempotency_key="k-1")
        b = svc.register_external(external_id="run-1", idempotency_key="k-1")
    assert a == b


def test_distinct_external_ids_are_distinct_tasks():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        a = svc.register_external(external_id="run-1")
        b = svc.register_external(external_id="run-2")
    assert a.task_id != b.task_id


# ---------------------------------------------------------------------------
# Terminal transitions: idempotency + atomicity
# ---------------------------------------------------------------------------


def test_complete_event_replay_is_harmless():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        handle = svc.register_external(external_id="run-1")

    first = svc.complete(
        handle, event_id="e1", summary="done", result_payload={"k": "v"}
    )
    assert first.accepted is True
    assert first.state == ExternalTaskState.COMPLETED.value
    evt1 = _drain_one()
    assert evt1 is not None
    assert evt1["summary"] == "done"

    replay = svc.complete(
        handle, event_id="e1", summary="done", result_payload={"k": "v"}
    )
    assert replay.accepted is True
    assert replay.already_terminal is True
    assert replay.conflict is False
    assert _drain_one() is None  # no second completion event


def test_complete_same_event_id_different_payload_conflicts():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        handle = svc.register_external(external_id="run-1")

    assert svc.complete(handle, event_id="e1", summary="done").accepted is True
    assert _drain_one() is not None
    res = svc.complete(handle, event_id="e1", summary="DIFFERENT")
    assert res.accepted is False
    assert res.conflict is True
    assert _drain_one() is None


def test_completed_task_cannot_fail_later():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        handle = svc.register_external(external_id="run-1")

    svc.complete(handle, event_id="e1", summary="done")
    assert _drain_one() is not None

    res = svc.fail(handle, event_id="e2", error="late failure")
    assert res.accepted is False
    assert res.already_terminal is True
    assert _drain_one() is None  # no second completion event


def test_fail_event_replay_and_conflict():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        handle = svc.register_external(external_id="run-1")

    assert svc.fail(handle, event_id="f1", error="boom").accepted is True
    evt = _drain_one()
    assert evt["status"] == "failed"
    assert evt["error"] == "boom"

    replay = svc.fail(handle, event_id="f1", error="boom")
    assert replay.accepted is True and replay.already_terminal is True

    conflict = svc.fail(handle, event_id="f1", error="different boom")
    assert conflict.conflict is True
    assert _drain_one() is None


def test_atomic_single_terminal_completion_under_concurrency():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        handle = svc.register_external(external_id="run-1")

    n = 8
    results = []

    def worker(idx):
        res = svc.complete(handle, event_id=f"e-{idx}", summary=f"summary {idx}")
        results.append(res)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    accepted = [r for r in results if r.accepted and not r.already_terminal]
    already_terminal = [r for r in results if r.already_terminal]
    conflicts = [r for r in results if r.conflict]
    assert len(accepted) == 1
    assert len(already_terminal) + len(conflicts) == n - 1

    events = []
    while not process_registry.completion_queue.empty():
        events.append(process_registry.completion_queue.get_nowait())
    assert len(events) == 1  # exactly one parent completion emitted
    assert events[0]["summary"].startswith("summary ")


def test_concurrent_register_same_key_is_idempotent():
    svc = _service()
    handles = []
    lock = threading.Lock()

    def worker():
        with bind_host_parent(_parent("parent-1")):
            h = svc.register_external(external_id="run-x", payload={"n": 1})
        with lock:
            handles.append(h)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len({h.task_id for h in handles}) == 1


# ---------------------------------------------------------------------------
# Size / schema limits
# ---------------------------------------------------------------------------


def test_oversized_payload_fails_before_persistence():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        with pytest.raises(BackgroundTaskError):
            svc.register_external(external_id="run-1", payload={"blob": "x" * 40_000})
    assert svc.list_pending() == []


def test_oversized_summary_fails_before_queue_side_effect():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        handle = svc.register_external(external_id="run-1")

    with pytest.raises(BackgroundTaskError):
        svc.complete(handle, event_id="e1", summary="x" * 40_000)
    assert _drain_one() is None  # nothing persisted or queued


def test_oversized_error_fails_before_queue_side_effect():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        handle = svc.register_external(external_id="run-1")

    with pytest.raises(BackgroundTaskError):
        svc.fail(handle, event_id="e1", error="x" * 40_000)
    assert _drain_one() is None

    # The task is still completable after the rejected attempt.
    assert svc.complete(handle, event_id="e1", summary="ok").accepted is True
    assert _drain_one() is not None


def test_oversized_result_payload_fails_before_queue_side_effect():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        handle = svc.register_external(external_id="run-1")

    with pytest.raises(BackgroundTaskError):
        svc.complete(
            handle,
            event_id="e1",
            summary="ok",
            result_payload={"blob": "y" * 40_000},
        )
    assert _drain_one() is None


def test_non_mapping_payload_rejected():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        with pytest.raises(BackgroundTaskError):
            svc.register_external(external_id="run-1", payload="not-a-mapping")


def test_missing_event_id_rejected():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        handle = svc.register_external(external_id="run-1")
    with pytest.raises(BackgroundTaskError):
        svc.complete(handle, event_id="", summary="x")
    with pytest.raises(BackgroundTaskError):
        svc.fail(handle, event_id="", error="x")


# ---------------------------------------------------------------------------
# Cancel intent
# ---------------------------------------------------------------------------


def test_cancel_intent_is_durable_but_distinct_from_cancel_success():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        handle = svc.register_external(external_id="run-1")

    res = svc.request_cancel(handle)
    assert res.accepted is True
    assert res.state == ExternalTaskState.CANCEL_REQUESTED.value
    assert _drain_one() is None  # no completion emitted for a cancel request

    # Idempotent.
    again = svc.request_cancel(handle)
    assert again.accepted is True
    assert again.cancel_already_requested is True

    # Still listed as non-terminal work, and can still be finalized with the
    # REAL outcome (the plugin performs the external cancellation itself).
    statuses = svc.list_pending()
    assert len(statuses) == 1
    assert statuses[0].state is ExternalTaskState.CANCEL_REQUESTED

    assert (
        svc.complete(handle, event_id="e1", summary="cancelled externally").accepted
        is True
    )
    evt = _drain_one()
    assert evt is not None and evt["summary"] == "cancelled externally"


def test_cancel_on_terminal_task_is_rejected():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        handle = svc.register_external(external_id="run-1")
    svc.complete(handle, event_id="e1", summary="done")
    _drain_one()
    res = svc.request_cancel(handle)
    assert res.accepted is False
    assert res.already_terminal is True


# ---------------------------------------------------------------------------
# list_pending
# ---------------------------------------------------------------------------


def test_list_pending_scope_and_lifecycle():
    from tools.async_delegation import (
        claim_event_delivery,
        complete_event_delivery,
    )

    svc = _service("owner")
    other = _service("other")
    with bind_host_parent(_parent("parent-1")):
        running = svc.register_external(external_id="run-1")
        other.register_external(external_id="run-2")

    statuses = svc.list_pending()
    assert [s.handle for s in statuses] == [running]
    assert statuses[0].state is ExternalTaskState.REGISTERED

    assert svc.complete(running, event_id="e1", summary="done").accepted is True
    evt = _drain_one()
    claim = claim_event_delivery(evt, "test-consumer")
    assert claim is not None
    complete_event_delivery(evt, claim)
    assert svc.list_pending() == []


def test_list_pending_includes_undelivered_terminal_until_delivered():
    svc = _service()
    with bind_host_parent(_parent("parent-1")):
        handle = svc.register_external(external_id="run-1")

    svc.complete(handle, event_id="e1", summary="done")
    # Terminal but NOT yet delivered: still listed.
    assert len(svc.list_pending()) == 1
    evt = _drain_one()

    # Ack delivery the way a drain consumer does, then it drops off the list.
    from tools.async_delegation import (
        claim_event_delivery,
        complete_event_delivery,
    )

    claim = claim_event_delivery(evt, "test-consumer")
    assert claim is not None
    complete_event_delivery(evt, claim)
    assert svc.list_pending() == []


def test_list_pending_returns_stable_handles_after_restart(tmp_path, monkeypatch):
    """A fresh store over the same profile DB returns operable handles."""
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))
    svc = _service("restart-plugin")
    with bind_host_parent(_parent("parent-1")):
        handle = svc.register_external(external_id="run-1")

    # Simulate process restart: a NEW service instance over the SAME profile.
    svc2 = _service("restart-plugin")
    statuses = svc2.list_pending()
    assert len(statuses) == 1
    assert statuses[0].handle == handle

    res = svc2.complete(statuses[0].handle, event_id="e1", summary="done")
    assert res.accepted is True


# ---------------------------------------------------------------------------
# Restart restore (real subprocess)
# ---------------------------------------------------------------------------


def test_real_restart_restores_undelivered_completion_and_pending_state(tmp_path):
    """A fresh interpreter restores a pending external completion + handle."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env = {**os.environ, "HERMES_HOME": str(tmp_path), "PYTHONPATH": repo}

    producer = r"""
import json
from types import SimpleNamespace
from agent.background_tasks import _create_external_background_tasks_service
from agent.host_context import bind_host_parent

svc = _create_external_background_tasks_service(
    plugin_id="restart-plugin",
    parent_agent_resolver=lambda: SimpleNamespace(session_id="parent-1"),
)
with bind_host_parent(SimpleNamespace(session_id="parent-1")):
    handle = svc.register_external(external_id="run-1", label="durable run")
    print(json.dumps(handle.to_dict()))
    res = svc.complete(handle, event_id="e1", summary="completed before restart")
    assert res.accepted is True
"""
    first = subprocess.run(
        [sys.executable, "-c", producer],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )
    handle_json = first.stdout.strip().splitlines()[-1]

    consumer = (
        r"""
import json
import sys
from types import SimpleNamespace
from agent.background_tasks import _create_external_background_tasks_service
from tools.process_registry import process_registry, format_process_notification

svc = _create_external_background_tasks_service(
    plugin_id="restart-plugin",
    parent_agent_resolver=lambda: SimpleNamespace(session_id="parent-1"),
)
pending = svc.list_pending()
assert len(pending) == 1, pending
handle = pending[0].handle
assert handle.to_dict()["task_id"] == json.loads(%r)["task_id"]

# The undelivered completion is restored onto the shared queue by the fresh
# process's registry, exactly like a durable async-delegation completion.
deadline = 0
evt = None
while True:
    import time
    if not process_registry.completion_queue.empty():
        evt = process_registry.completion_queue.get_nowait()
        break
    if deadline > 5.0:
        break
    time.sleep(0.05)
    deadline += 0.05
assert evt is not None
assert evt["type"] == "async_delegation"
assert evt["status"] == "completed"
assert evt["summary"] == "completed before restart"
text = format_process_notification(evt)
assert "durable run" in text
print("OK")
"""
        % handle_json
    )
    second = subprocess.run(
        [sys.executable, "-c", consumer],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )
    assert second.stdout.strip().splitlines()[-1] == "OK"


# ---------------------------------------------------------------------------
# HERMES_HOME isolation
# ---------------------------------------------------------------------------


def test_tests_never_touch_the_real_profile(tmp_path):
    from pathlib import Path

    real_default = Path.home() / ".hermes"
    home = Path(get_hermes_home())
    assert home != real_default
    assert str(home).startswith(str(tmp_path)) or home != real_default
    # Registration state lands under the temp HERMES_HOME, never the profile.
    with bind_host_parent(_parent("parent-1")):
        handle = _service().register_external(external_id="run-1")
    state_db = home / "state.db"
    assert state_db.exists()
