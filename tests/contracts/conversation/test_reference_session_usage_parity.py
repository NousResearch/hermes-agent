"""Session and usage regressions shared by the reference and SQLite backends."""

from datetime import datetime, UTC

import pytest

from persistence import AuthorityRefusal, SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession, SessionLifecycleStatus
from persistence.conversation.usage import BillingRoute, TokenCounts, UsageDelta
from tests.contracts.conversation.test_usage import accounting
from tests.contracts._scopes import MEDIA_ROOT


@pytest.mark.parametrize("initial", ["missing", None, "", "initial"])
@pytest.mark.parametrize("fields,routed,accounted", [({}, True, False), ({"estimated_cost_usd": 0.0}, True, False), ({"tokens": TokenCounts(input_tokens=2)}, True, True), ({"api_call_count": 1}, True, True), ({"estimated_cost_usd": 0.5}, True, True), ({"actual_cost_usd": 0.5}, True, True), ({"tokens": TokenCounts(input_tokens=2)}, False, True)])
def test_usage_projects_model_to_session(backend, scope_a, initial, fields, routed, accounted):
    store_a = backend.open(scope_a)
    sid = SessionId("usage-model")
    if initial != "missing":
        store_a.sessions.create(NewSession(session_id=sid, source="cli", model=initial))
    route = BillingRoute(model="m", billing_provider="p", billing_base_url="url", billing_mode="api") if routed else None
    store_a.usage.queue_usage(sid, UsageDelta(source="telegram", route=route, **fields))
    row = store_a.sessions.get(sid)
    expected = None if initial == "missing" else initial
    if routed and (expected is None or accounted):
        expected = "m"
    assert row.model == expected
    assert row.source == ("telegram" if initial == "missing" else "cli")


@pytest.mark.parametrize("initial", [None, "", "initial"])
def test_zero_call_usage_replaces_session_model_until_first_call(backend, scope_a, initial):
    store_a = backend.open(scope_a)
    sid = SessionId("usage-model-sequence")
    store_a.sessions.create(NewSession(session_id=sid, source="cli", model=initial))
    for model, calls, expected in [("a", 0, "a"), ("b", 0, "b"), ("c", 1, "c"), ("d", 1, "c")]:
        route = BillingRoute(model=model, billing_provider="p" + model, billing_base_url="url", billing_mode="api")
        store_a.usage.queue_usage(sid, UsageDelta(route=route, api_call_count=calls, estimated_cost_usd=0.5))
        assert (store_a.sessions.get(sid)).model == expected
        assert (store_a.sessions.get(sid)).billing_provider == "p" + expected


def test_usage_keeps_route_columns_when_session_model_already_matches(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = SessionId("usage-same-model")
    store_a.sessions.create(NewSession(session_id=sid, source="cli", model="m"))
    store_a.usage.set_billing_route(sid, provider="p", base_url="keep", billing_mode="keep")
    store_a.usage.queue_usage(sid, UsageDelta(api_call_count=1, route=BillingRoute(model="m", billing_provider="p", billing_base_url="url", billing_mode="api")))
    assert (store_a.sessions.get(sid)).model == "m"
    route = store_a.sessions.get(sid)
    assert (route.billing_provider, route.billing_base_url, route.billing_mode) == ("p", "keep", "keep")


@pytest.mark.parametrize("estimate,actual,attributed", [(0.5, None, True), (-0.5, None, True), (0.0, None, False), (None, 0.5, False), (None, 0.0, False)])
def test_cost_only_usage_attribution(backend, scope_a, estimate, actual, attributed):
    store_a = backend.open(scope_a)
    sid = SessionId("cost-only")
    store_a.sessions.create(NewSession(session_id=sid, source="cli"))
    route = BillingRoute(model="m", billing_provider="p", billing_base_url="url", billing_mode="api")
    store_a.usage.queue_usage(sid, UsageDelta(estimated_cost_usd=estimate, actual_cost_usd=actual, route=route))
    observed = store_a.usage.recent_route(sid)
    if attributed:
        assert observed.route == route
        assert observed.api_call_count == 0
    else:
        assert observed is None
    usage = accounting(store_a, sid)
    assert usage.tokens == TokenCounts()
    assert usage.api_call_count == 0
    assert usage.actual_cost_usd == actual
    assert usage.estimated_cost_usd == (estimate or 0.0)


def test_usage_delta_initializes_estimate_without_initializing_actual(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = SessionId("cost-defaults")
    store_a.sessions.create(NewSession(session_id=sid, source="cli"))
    assert (accounting(store_a, sid)).estimated_cost_usd is None
    for delta, expected in [(UsageDelta(), 0.0), (UsageDelta(estimated_cost_usd=0.25), 0.25), (UsageDelta(), 0.25)]:
        store_a.usage.queue_usage(sid, delta)
        usage = accounting(store_a, sid)
        assert usage.estimated_cost_usd == expected
        assert usage.actual_cost_usd is None


@pytest.mark.parametrize("verb", ["delete", "delete_many", "delete_many_skip_guarded", "delete_if_empty", "delete_empty"])
def test_delete_folds_terminal_admissions_and_workers(backend, scope_a, verb):
    store_a = backend.open(scope_a)
    from persistence import RetiredSessionError
    from persistence.workers.executions import WorkerKind
    from tests.contracts.workers._support import begin_epoch

    sid = SessionId("terminal-evidence")
    store_a.sessions.create(NewSession(session_id=sid, source="cli"))
    epoch = begin_epoch(backend, scope_a)
    request = dict(epoch=epoch, principal_id="principal", session_id=sid, request_id="request", payload={"text": "private input"})
    admission = store_a.admissions.admit(**request)[0]
    store_a.admissions.cancel(epoch=epoch, admission_id=admission.admission_id)
    store_a.worker_executions.register(epoch=epoch, execution_id="worker", session_id=sid, generation=0, kind=WorkerKind.COMPUTE, adoption_secret="probe-adoption")
    worker = dict(epoch=epoch, execution_id="worker", session_id=sid, generation=0)
    store_a.worker_executions.record_usage(**worker, sequence=1, input_tokens=1)
    store_a.worker_executions.finish(**worker, sequence=2)
    if verb == "delete_empty":
        store_a.sessions.end(sid, "cli_close")
        assert store_a.sessions.delete_empty(media_root=MEDIA_ROOT) == (sid,)
    elif verb.startswith("delete_many"):
        result = store_a.sessions.delete_many([sid], skip_guarded=verb == "delete_many_skip_guarded", media_root=MEDIA_ROOT)
        assert (result.deleted, result.skipped_guarded, result.removed_ids) == (1, (), (sid,))
    else:
        assert getattr(backend.open(scope_a).sessions, verb)(sid, media_root=MEDIA_ROOT) == (sid,)
    assert store_a.sessions.get(sid) is None
    with pytest.raises(RetiredSessionError):
        store_a.sessions.create(NewSession(session_id=sid, source="cli"))
    replayed = store_a.admissions.admit(**request)[0]
    assert replayed.admission_id == admission.admission_id
    assert replayed.payload == {}
    assert (store_a.admissions.get(admission_id=admission.admission_id)).payload == {}
    assert store_a.admissions.list(session_id=sid, pending_only=False) == []
    with pytest.raises(AuthorityRefusal, match="^admission_conflict$"):
        store_a.admissions.admit(**dict(request, payload={"text": "changed"}))
    with pytest.raises(AuthorityRefusal, match="^not_found$"):
        store_a.admissions.admit(**dict(request, request_id="new-request"))
    # The execution and its receipts left the live tables with the row: neither receipt replays any more.
    with pytest.raises(AuthorityRefusal, match="^not_found$"):
        store_a.worker_executions.finish(**worker, sequence=2)
    with pytest.raises(AuthorityRefusal, match="^not_found$"):
        store_a.worker_executions.record_usage(**worker, sequence=1, input_tokens=1)
    assert store_a.worker_executions.execution_session(execution_id="worker") is None


@pytest.mark.parametrize("role,reason,expected", [("assistant", "error", SessionLifecycleStatus.ERROR), ("assistant", " ERROR ", SessionLifecycleStatus.ERROR), ("user", "error", SessionLifecycleStatus.ERROR), ("assistant", "stop", SessionLifecycleStatus.COMPLETE), ("user", None, SessionLifecycleStatus.INTERRUPTED)])
def test_lifecycle_reads_stored_finish_reason(backend, scope_a, role, reason, expected):
    store_a = backend.open(scope_a)
    sid = SessionId("finish-reason")
    store_a.sessions.create(NewSession(session_id=sid, source="cli"))
    store_a.messages.append(sid, NewMessage(role=role, content="x", timestamp=datetime(2023, 11, 14, tzinfo=UTC), finish_reason=reason))
    assert store_a.sessions.lifecycle_statuses([sid, SessionId("absent")]) == {sid: expected, SessionId("absent"): SessionLifecycleStatus.EMPTY}
