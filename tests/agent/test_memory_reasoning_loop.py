"""P2a: the restricted runner must execute the real AIAgent loop."""

from collections import defaultdict
from types import SimpleNamespace
import json
import hashlib
import os
from pathlib import Path
import sqlite3
import threading
import time
from uuid import uuid4

import pytest

from agent.memory_reasoning import (
    ReasoningTask, BudgetLimits, Evidence, HostHandlers, RestrictedReasoningRunner, Route,
    ScopedSessionEvidenceReader, ToolBudgetExceeded,
)
from agent.memory_reasoning.transport import CoreSingleAttemptTransport, PriceQuote, CompletedUsageInterrupted


def test_contract_imports():
    assert ReasoningTask and BudgetLimits and RestrictedReasoningRunner


class Ledger:
    def __init__(self):
        self.values = defaultdict(float)
        self.lock = threading.Lock()
        self.attempts = {}
        self.next_attempt = 0

    def reserve(self, task, dimension, amount, ceiling):
        with self.lock:
            key = (task, dimension)
            if self.values[key] + amount > ceiling:
                return False
            self.values[key] += amount
            return True

    def record(self, task, dimension, amount):
        with self.lock:
            self.values[(task, dimension)] += amount

    def begin_attempt(self, task):
        with self.lock:
            self.next_attempt += 1
            self.attempts[self.next_attempt] = (task, {})
            self.values[(task, "model_attempts_prepared")] += 1
            return self.next_attempt

    def settle_actual(self, attempt_id, dimension, amount):
        with self.lock:
            task, actual = self.attempts[attempt_id]
            if dimension in actual:
                assert actual[dimension] == amount
                return False
            actual[dimension] = amount
            key = "actual_" + dimension + ("_unknown" if amount is None else "")
            self.values[(task, key)] += 1 if amount is None else amount
            return True

    def snapshot(self, task):
        with self.lock:
            return {dimension: value for (key, dimension), value in self.values.items() if key == task}


class DurableTestLedger:
    """Test host adapter: SQLite reservation remains atomic across runner reconstruction."""

    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS budget (task TEXT, dimension TEXT, amount REAL, PRIMARY KEY(task, dimension))")
        self.db.execute("CREATE TABLE IF NOT EXISTS attempt (id INTEGER PRIMARY KEY, task TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS actual (attempt_id INTEGER, dimension TEXT, amount REAL, PRIMARY KEY(attempt_id, dimension))")

    def reserve(self, task, dimension, amount, ceiling):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO budget VALUES (?, ?, 0)", (task, dimension))
            updated = self.db.execute(
                "UPDATE budget SET amount = amount + ? WHERE task = ? AND dimension = ? AND amount + ? <= ?",
                (amount, task, dimension, amount, ceiling),
            )
            return updated.rowcount == 1

    def record(self, task, dimension, amount):
        with self.db:
            self.db.execute(
                "INSERT INTO budget VALUES (?, ?, ?) ON CONFLICT(task, dimension) DO UPDATE SET amount = amount + excluded.amount",
                (task, dimension, amount),
            )

    def begin_attempt(self, task):
        with self.db:
            attempt_id = self.db.execute("INSERT INTO attempt(task) VALUES (?)", (task,)).lastrowid
            self.record(task, "model_attempts_prepared", 1)
            return attempt_id

    def settle_actual(self, attempt_id, dimension, amount):
        with self.db:
            row = self.db.execute("SELECT amount FROM actual WHERE attempt_id=? AND dimension=?",
                                  (attempt_id, dimension)).fetchone()
            if row:
                assert row[0] == amount
                return False
            self.db.execute("INSERT INTO actual VALUES (?,?,?)", (attempt_id, dimension, amount))
            task = self.db.execute("SELECT task FROM attempt WHERE id=?", (attempt_id,)).fetchone()[0]
            self.record(task, "actual_" + dimension + ("_unknown" if amount is None else ""),
                        1 if amount is None else amount)
            return True

    def snapshot(self, task):
        return dict(self.db.execute("SELECT dimension, amount FROM budget WHERE task = ?", (task,)))

    def close(self):
        self.db.close()


def call(name, args, i=1):
    return SimpleNamespace(id=f"call_{i}", type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def response(calls=(), content=None):
    message = SimpleNamespace(content=content, reasoning=None, tool_calls=list(calls))
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if calls else "stop")],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10),
    )


def test_transport_completed_response_cancel_keeps_numeric_usage(monkeypatch):
    stopped = {"value": False}
    def complete(**kwargs):
        stopped["value"] = True
        return response(content="private response body")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    agent = SimpleNamespace(provider="openai", model="test-model", api_mode="chat_completions",
        _create_request_openai_client=lambda **kwargs: client,
        _abort_request_openai_client=lambda *args, **kwargs: None,
        _close_request_openai_client=lambda *args, **kwargs: None)
    monkeypatch.setattr("agent.memory_reasoning.transport.bypass_chat_sdk_request_transform",
                        lambda wire, client: wire)
    transport = CoreSingleAttemptTransport(input_bound=lambda *args: 100,
        price=PriceQuote("v1", "openai", "test-model", 1.0, 2.0),
        cancelled=lambda: stopped["value"], deadline_monotonic=time.monotonic() + 30)
    with pytest.raises(CompletedUsageInterrupted) as caught:
        transport.complete({"model": "test-model"}, agent)
    assert caught.value.usage == {"input_tokens": 20, "output_tokens": 10,
                                  "cost_usd": 0.00004}
    assert "private response body" not in repr(caught.value)


class ScriptedTransport:
    single_attempt = True

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def supports_effort(self, route):
        return True

    def effective_effort(self, request, route):
        return route.effort

    def input_token_upper_bound(self, request, route):
        return 100

    def cost_upper_bound(self, input_tokens, output_tokens, route):
        return 0.01

    def actual_cost(self, response, route):
        return 0.001

    def complete(self, request, resolved_agent):
        assert resolved_agent.model == "test-model"
        assert {t["function"]["name"] for t in request["tools"]} == {
            "reason_memory_search", "reason_exact_read", "reason_source_trace",
            "reason_document_read", "reason_proposal_submit",
        }
        self.requests.append(request)
        return self.script.pop(0)


def evidence(kind, ref, status="complete"):
    return Evidence(kind, ref, "sha256:test", "line:1", "2026-09-30T00:00:00Z",
                    status, "bounded exact read", {"text": ref})


def runner(script, ledger, events, *, limits=None, cancelled=None):
    task = ReasoningTask("task-1", "owner-1", "rev-1", "User corrected project A to B")
    handlers = HostHandlers(
        memory_search=lambda q, meter: meter() or events.append(("search", q)) or evidence("memory", "memory:1"),
        exact_read=lambda r, meter: meter() or events.append(("exact", r)) or evidence("message", "message:2"),
        source_trace=lambda r, meter: meter() or events.append(("trace", r)) or evidence("source", "source:3"),
        document_read=lambda r, meter: meter() or events.append(("document", r)) or evidence("document", "document:4"),
        submit_proposal=lambda s: events.append(("proposal", s.proposal_revision,
                                                  json.loads(s.proposal_json), len(s.evidence),
                                                  s.proposal_digest, s.evidence_digest)) or "receipt:1",
    )
    return RestrictedReasoningRunner(
        task=task, route=Route("openai", "test-model", "high", max_output_tokens=40),
        limits=limits or BudgetLimits(5, 8, 8, 1000, 200, 60, 0.1),
        handlers=handlers, ledger=ledger, transport=ScriptedTransport(script),
        cancelled=cancelled or (lambda: False), deadline_monotonic=time.monotonic() + 60,
    )


@pytest.fixture
def fake_route(monkeypatch):
    monkeypatch.setattr(
        "agent.auxiliary_client.resolve_provider_client",
        lambda *args, **kwargs: (SimpleNamespace(api_key="synthetic-test-only", base_url="http://localhost/v1"), "test-model"),
    )
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **kwargs: SimpleNamespace(close=lambda: None))


def test_real_loop_search_read_submit(fake_route):
    events, ledger = [], Ledger()
    job = runner([
        response([call("reason_memory_search", {"query": "project A"})]),
        response([call("reason_exact_read", {"reference": "memory:1"})]),
        response([call("reason_proposal_submit", {"proposal": {
            "action": "UPDATE", "evidence_refs": ["message:2"], "new_value": "B",
        }})]),
        response(content="Submitted for deterministic review"),
    ], ledger, events)
    result = job.run()
    assert result.status == "proposal_submitted"
    assert [e[0] for e in events] == ["search", "exact", "proposal"]
    assert events[-1][1].startswith("rev-1:")
    assert events[-1][-1].startswith("sha256:")
    assert result.proposal_receipt == "receipt:1"
    assert result.usage["model_requests"] == 4
    assert result.usage["logical_tools"] == 3
    assert result.usage["underlying_reads"] == 2
    assert result.usage["related_items"] == 1
    assert result.auto_eligible is False
    assert not (Path(os.environ["HERMES_HOME"]) / "state.db").exists()
    assert not (Path(os.environ["HERMES_HOME"]) / "MEMORY.md").exists()


def test_forbidden_tool_and_malformed_wrapper_never_dispatch(fake_route):
    for name, args in [
        ("terminal", {"command": "echo unsafe"}),
        ("tool_search", {"name": "terminal"}),
        ("reason_exact_read", {"reference": "session:ok", "profile": "other"}),
    ]:
        events, ledger = [], Ledger()
        job = runner([response([call(name, args)])], ledger, events)
        result = job.run()
        assert result.status == "restricted_failure"
        assert not events
        assert result.usage["model_requests"] == 1


def test_budget_reuse_and_cancel_blocks_new_send(fake_route):
    ledger = Ledger()
    limits = BudgetLimits(1, 2, 2, 300, 40, 60, 0.02)
    first = runner([response(content="need evidence")], ledger, [], limits=limits)
    assert first.run().usage["model_requests"] == 1
    second = runner([response(content="should not send")], ledger, [], limits=limits)
    outcome = second.run()
    assert outcome.status == "budget_exhausted"
    assert not second.transport.requests
    cancelled = runner([response(content="should not send")], Ledger(), [], cancelled=lambda: True)
    assert cancelled.run().status == "cancelled"


def test_scoped_session_reader_reuses_read_shape_without_cross_scope(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    writer = SessionDB(db_path=db_path)
    try:
        writer.create_session("allowed", source="cli")
        writer.create_session("other", source="cli")
        writer.create_session("long", source="cli")
        writer.append_message("allowed", "user", "Alpha plan")
        writer.append_message("other", "user", "Secret other owner")
        writer.append_message("long", "user", "x" * 3000)
        with ScopedSessionEvidenceReader(
            db_path=db_path, owner_key="owner-1",
            allowed_session_ids=frozenset({"allowed", "missing", "long"}),
        ) as reader:
            with pytest.raises(PermissionError):
                reader.read("session:other")
            with pytest.raises(ValueError):
                reader.read(str(tmp_path / "elsewhere"))
            found = reader.search("Alpha")
            assert found.status == "missing"  # one approved session does not exist
            assert found.content["hits"][0]["session_id"] == "allowed"
            assert not reader.search("Secret").content["hits"]
            exact = reader.read("session:allowed")
            assert exact.status == "complete"
            assert exact.exact_reference == "session:allowed"
            assert exact.source_kind == "session" and exact.digest.startswith("sha256:")
            assert exact.location and exact.read_at and exact.coverage
            assert reader.read("session:missing").status == "missing"
            long_before = reader.read("session:long")
            assert long_before.status == "truncated"
            assert reader.read("session:long").digest == long_before.digest
            writer.append_message("long", "user", "new row beyond the first clipped message")
            assert reader.read("session:long").digest != long_before.digest
            monkeypatch.setattr(reader._db, "_read_all", lambda *a: (_ for _ in ()).throw(RuntimeError("DB failed")))
            assert reader.read("session:allowed").status == "error"
    finally:
        writer.close()


def test_concurrent_invoke_route_uses_same_handlers_and_budget(fake_route, monkeypatch):
    monkeypatch.setattr("agent.tool_dispatch_helpers._plan_tool_batch_segments",
                        lambda calls, **kwargs: [("parallel", list(calls))])
    barrier = threading.Barrier(2, timeout=5)
    events, ledger = [], Ledger()
    job = runner([
        response([
            call("reason_memory_search", {"query": "project"}, 1),
            call("reason_exact_read", {"reference": "memory:1"}, 2),
        ]),
        response(content="Evidence collected"),
    ], ledger, events)

    def searched(query, meter):
        meter()
        barrier.wait()
        events.append(("search", query))
        return evidence("memory", "memory:1")

    def exact(ref, meter):
        meter()
        barrier.wait()
        events.append(("exact", ref))
        return evidence("message", "message:2")

    job.handlers = HostHandlers(searched, exact, job.handlers.source_trace,
                                job.handlers.document_read, job.handlers.submit_proposal)
    result = job.run()
    assert {e[0] for e in events} == {"search", "exact"}
    assert result.usage["underlying_reads"] == 2
    assert result.usage["peak_parallelism"] == 2


def test_cancel_after_read_cannot_submit_or_send_again(fake_route):
    cancelled = {"value": False}
    events, ledger = [], Ledger()
    job = runner([
        response([call("reason_memory_search", {"query": "project"})]),
        response([call("reason_proposal_submit", {"proposal": {
            "action": "UPDATE", "evidence_refs": ["memory:1"],
        }})]),
    ], ledger, events, cancelled=lambda: cancelled["value"])

    def cancel_during_read(query, meter):
        meter()
        cancelled["value"] = True
        events.append(("search", query))
        return evidence("memory", "memory:1")

    job.handlers = HostHandlers(cancel_during_read, job.handlers.exact_read,
                                job.handlers.source_trace, job.handlers.document_read,
                                job.handlers.submit_proposal)
    result = job.run()
    assert result.status == "cancelled"
    assert len(job.transport.requests) == 1
    assert events == [("search", "project")]


def test_unsupported_effort_is_rejected_at_construction():
    events, ledger = [], Ledger()
    job = runner([response(content="unused")], ledger, events)
    job.transport.supports_effort = lambda route: False
    with pytest.raises(ValueError, match="effort unsupported"):
        RestrictedReasoningRunner(
            task=job.task, route=job.route, limits=job.limits, handlers=job.handlers,
            ledger=ledger, transport=job.transport, cancelled=lambda: False,
            deadline_monotonic=time.monotonic() + 60,
        )


def test_fallback_route_is_visible_and_stops_before_second_send(fake_route):
    events, ledger = [], Ledger()
    job = runner([
        response([call("reason_memory_search", {"query": "project"})]),
        response(content="must not send"),
    ], ledger, events)
    original_complete = job.transport.complete

    def switch_route(request, resolved_agent):
        result = original_complete(request, resolved_agent)
        job._agent.model = "fallback-model"
        return result

    job.transport.complete = switch_route
    result = job.run()
    assert result.status == "restricted_failure"
    assert result.fallback is True
    assert result.actual_route.model == "fallback-model"
    assert result.auto_eligible is False
    assert len(job.transport.requests) == 1


def test_bad_host_envelope_stops_following_actions(fake_route):
    events, ledger = [], Ledger()
    job = runner([
        response([call("reason_memory_search", {"query": "project"})]),
        response([call("reason_proposal_submit", {"proposal": {
            "action": "UPDATE", "evidence_refs": ["memory:1"],
        }})]),
    ], ledger, events)
    job.handlers = HostHandlers(lambda query, meter: meter() or {"status": "complete"},
                                job.handlers.exact_read, job.handlers.source_trace,
                                job.handlers.document_read, job.handlers.submit_proposal)
    result = job.run()
    assert result.status == "restricted_failure"
    assert not events
    assert len(job.transport.requests) == 1


def test_parent_client_is_not_closed_or_rebound(fake_route):
    from run_agent import AIAgent

    parent = AIAgent(provider="openai", model="test-model", api_mode="chat_completions",
                     enabled_toolsets=["__memory_reasoning_none__"], skip_memory=True,
                     skip_context_files=True, quiet_mode=True)
    original_client = parent.client
    try:
        job = runner([response(content="No proposal")], Ledger(), [])
        job.run()
        assert parent.client is original_client
    finally:
        parent.close()


def test_restricted_agent_skips_configured_context_engine_before_selection(fake_route, monkeypatch):
    from agent import agent_init
    from agent.context_compressor import ContextCompressor
    from agent.context_engine import ContextEngine
    from run_agent import AIAgent

    events = []

    class ProbeEngine(ContextEngine):
        @property
        def name(self):
            return "probe"

        def update_model(self, *args, **kwargs):
            events.append("update_model")
            return super().update_model(*args, **kwargs)

        def bind_session_state(self, **kwargs):
            events.append("bind_session_state")

        def update_from_response(self, usage):
            pass

        def should_compress(self, prompt_tokens=None):
            return False

        def compress(self, messages, current_tokens=None):
            return messages

    engine = ProbeEngine()
    config = {"context": {"engine": "probe"}, "compression": {"enabled": True}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **kw: 204_800)

    def factory(name):
        events.append("factory")
        assert name == "probe"
        return engine

    monkeypatch.setattr("plugins.context_engine.load_context_engine", factory)
    select = agent_init._select_context_engine

    def track_selection(cfg):
        events.append("selection")
        return select(cfg)

    monkeypatch.setattr(agent_init, "_select_context_engine", track_selection)
    job = runner([response(content="No proposal")], Ledger(), [])
    original_complete = job.transport.complete

    def inspect_restricted(request, resolved_agent):
        assert isinstance(resolved_agent.context_compressor, ContextCompressor)
        assert resolved_agent.compression_enabled is False
        return original_complete(request, resolved_agent)

    job.transport.complete = inspect_restricted
    assert job.run().status == "needs_evidence"
    assert events == []

    ordinary = AIAgent(provider="openai", model="test-model", api_mode="chat_completions",
                       enabled_toolsets=["__memory_reasoning_none__"], skip_memory=True,
                       skip_context_files=True, quiet_mode=True)
    try:
        assert ordinary.context_compressor is engine
        assert ordinary.compression_enabled is True
        assert events == ["selection", "factory", "update_model", "bind_session_state"]
    finally:
        ordinary.close()


def test_restricted_agent_never_sends_automatic_compaction_request(fake_route, monkeypatch):
    from agent.context_compressor import ContextCompressor

    config = {"compression": {"enabled": True, "threshold_tokens": 100,
                              "micro_compact": True, "idle_compact_after_seconds": 1,
                              "codex_responses_native": True}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)

    def unexpected_compaction(*args, **kwargs):
        pytest.fail("automatic compaction bypassed the restricted agent budget")

    monkeypatch.setattr(ContextCompressor, "compress", unexpected_compaction)
    monkeypatch.setattr(ContextCompressor, "_micro_compact", unexpected_compaction)
    job = runner([response([call("reason_memory_search", {"query": "large"})]),
                  response(content="Evidence reviewed")], Ledger(), [])

    def search(query, meter):
        meter()
        return Evidence("memory", "memory:large", "sha256:large", "line:1", "now",
                        "complete", "full", {"text": "x" * 30_000})

    job.handlers = HostHandlers(search, job.handlers.exact_read, job.handlers.source_trace,
                                job.handlers.document_read, job.handlers.submit_proposal)
    original_complete = job.transport.complete

    def inspect_restricted(request, resolved_agent):
        assert resolved_agent.compression_enabled is False
        assert resolved_agent.context_compressor._micro_compact_enabled is False
        assert resolved_agent.codex_responses_native_compaction is False
        assert resolved_agent.codex_app_server_auto_compaction == "off"
        return original_complete(request, resolved_agent)

    job.transport.complete = inspect_restricted
    outcome = job.run()
    assert outcome.status == "needs_evidence"
    assert len(job.transport.requests) == 2
    assert outcome.usage["model_requests"] == 2


def test_durable_reservation_blocks_reconstructed_runner(fake_route, tmp_path):
    path = tmp_path / "host-budget.db"
    limits = BudgetLimits(1, 2, 2, 300, 40, 60, 0.02)
    first_ledger = DurableTestLedger(path)
    try:
        first = runner([response(content="First attempt")], first_ledger, [], limits=limits)
        assert first.run().status == "needs_evidence"
    finally:
        first_ledger.close()
    resumed_ledger = DurableTestLedger(path)
    try:
        resumed = runner([response(content="must not send")], resumed_ledger, [], limits=limits)
        result = resumed.run()
        assert result.status == "budget_exhausted"
        assert result.usage["model_requests"] == 1
        assert not resumed.transport.requests
    finally:
        resumed_ledger.close()


def test_unmapped_effort_and_input_bound_stop_before_send(fake_route):
    for change in (
        lambda transport: setattr(transport, "effective_effort", lambda request, route: "medium"),
        lambda transport: setattr(transport, "input_token_upper_bound", lambda request, route: 1000),
    ):
        job = runner([response(content="must not send")], Ledger(), [],
                     limits=BudgetLimits(3, 3, 3, 300, 120, 60, 0.03))
        change(job.transport)
        result = job.run()
        assert result.status in {"restricted_failure", "budget_exhausted"}
        assert not job.transport.requests
        assert result.auto_eligible is False


def test_missing_evidence_never_becomes_noop_or_submission(fake_route):
    events, ledger = [], Ledger()
    job = runner([
        response([call("reason_proposal_submit", {"proposal": {
            "action": "ADD", "evidence_refs": ["message:missing"],
        }})]),
        response(content="I need evidence"),
    ], ledger, events)
    result = job.run()
    assert result.status == "needs_evidence"
    assert result.proposal_receipt is None
    assert not events


def test_duplicate_evidence_stops_for_no_progress(fake_route):
    events, ledger = [], Ledger()
    script = [response([call("reason_memory_search", {"query": "repeat"}, i)]) for i in range(1, 6)]
    job = runner(script, ledger, events,
                 limits=BudgetLimits(8, 8, 8, 1000, 320, 60, 0.1))
    result = job.run()
    assert result.status == "budget_exhausted"
    assert result.usage["model_requests"] < 8
    assert len(job.transport.requests) < 5


def test_related_item_budget_blocks_large_proposal(fake_route):
    events, ledger = [], Ledger()
    job = runner([
        response([call("reason_memory_search", {"query": "project"})]),
        response([call("reason_proposal_submit", {"proposal": {
            "action": "UPDATE", "evidence_refs": ["memory:1"],
            "affected_items": ["one", "two", "three"],
        }})]),
    ], ledger, events,
        limits=BudgetLimits(5, 8, 8, 1000, 200, 60, 0.1, related_items=2))
    result = job.run()
    assert result.status == "budget_exhausted"
    assert result.proposal_receipt is None
    assert [e[0] for e in events] == ["search"]


@pytest.mark.parametrize("field", ["model_requests", "logical_tools", "underlying_reads",
                                   "input_tokens", "output_tokens", "parallelism",
                                   "related_items", "proposal_bytes"])
@pytest.mark.parametrize("bad", [True, 1.5, 0, -1, float("nan"), float("inf"), -float("inf")])
def test_count_limits_require_positive_int(field, bad):
    values = dict(model_requests=5, logical_tools=8, underlying_reads=8,
                  input_tokens=1000, output_tokens=200, elapsed_seconds=60,
                  cost_usd=0.1, parallelism=2, related_items=64, proposal_bytes=65536)
    values[field] = bad
    with pytest.raises(ValueError):
        BudgetLimits(**values)


@pytest.mark.parametrize("field", ["elapsed_seconds", "cost_usd"])
@pytest.mark.parametrize("bad", [True, 0, -1, float("nan"), float("inf"), -float("inf")])
def test_numeric_limits_require_finite_positive_value(field, bad):
    values = dict(model_requests=5, logical_tools=8, underlying_reads=8,
                  input_tokens=1000, output_tokens=200, elapsed_seconds=60,
                  cost_usd=0.1)
    values[field] = bad
    with pytest.raises(ValueError):
        BudgetLimits(**values)


@pytest.mark.parametrize("bad", [True, 1.5, 0, -1, float("nan"), float("inf")])
def test_route_output_limit_is_strict(bad):
    with pytest.raises(ValueError):
        Route("openai", "test-model", "high", max_output_tokens=bad)


@pytest.mark.parametrize("bound", [True, 1.5, 0, -1, float("nan"), float("inf")])
def test_transport_input_bound_rejected_before_send(fake_route, bound):
    job = runner([response(content="unused")], Ledger(), [])
    job.transport.input_token_upper_bound = lambda *_: bound
    assert job.run().status == "restricted_failure"
    assert not job.transport.requests


@pytest.mark.parametrize("cost", [True, 0, -1, float("nan"), float("inf")])
def test_transport_cost_bound_rejected_before_send(fake_route, cost):
    job = runner([response(content="unused")], Ledger(), [])
    job.transport.cost_upper_bound = lambda *_: cost
    assert job.run().status == "restricted_failure"
    assert not job.transport.requests


@pytest.mark.parametrize("kind", ["forbidden", "malformed", "overrun", "partial", "error"])
def test_sent_response_records_known_actual_usage_even_when_rejected(fake_route, kind, tmp_path):
    path = tmp_path / "budget.db"
    ledger = DurableTestLedger(path)
    try:
        item = response(content="done")
        if kind == "forbidden":
            item = response([call("terminal", {"command": "echo nope"})])
        elif kind == "malformed":
            item = response([call("reason_exact_read", {"reference": "x"})])
            item.choices[0].message.tool_calls[0].function.arguments = "{bad"
        elif kind == "overrun":
            item.usage.completion_tokens = 41
        elif kind == "partial":
            item.usage.completion_tokens = None
        job = runner([item], ledger, [])
        if kind == "error":
            job.transport.complete = lambda *_: (_ for _ in ()).throw(RuntimeError("lost response"))
        result = job.run()
        assert result.status in {"restricted_failure", "failed"}
        assert result.usage["model_requests"] == 1
    finally:
        ledger.close()
    rebuilt = DurableTestLedger(path)
    try:
        usage = rebuilt.snapshot("task-1")
        if kind == "error":
            assert usage["actual_input_tokens_unknown"] == 1
        else:
            assert usage["actual_input_tokens"] == 20
            assert usage["actual_cost_usd"] == 0.001
        if kind == "partial":
            assert usage["actual_output_tokens_unknown"] == 1
        elif kind != "error":
            assert usage["actual_output_tokens"] == (41 if kind == "overrun" else 10)
    finally:
        rebuilt.close()


def test_elapsed_parallel_checks_charge_one_wall_interval(monkeypatch):
    job = runner([], Ledger(), [])
    job._last_accounted_time = 10.0
    monkeypatch.setattr("agent.memory_reasoning.runner.time.monotonic", lambda: 11.0)
    threads = [threading.Thread(target=job._check_live) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert job.ledger.snapshot("task-1")["elapsed_seconds"] == 1.0


def test_elapsed_charge_survives_runner_reconstruction(monkeypatch):
    ledger = Ledger()
    limits = BudgetLimits(5, 8, 8, 1000, 200, 1.5, 0.1)
    first = runner([], ledger, [], limits=limits)
    first._last_accounted_time = 10.0
    monkeypatch.setattr("agent.memory_reasoning.runner.time.monotonic", lambda: 11.0)
    first._check_live()
    second = runner([], ledger, [], limits=limits)
    second._last_accounted_time = 11.0
    monkeypatch.setattr("agent.memory_reasoning.runner.time.monotonic", lambda: 12.0)
    with pytest.raises(ToolBudgetExceeded, match="elapsed_seconds"):
        second._check_live()
    assert ledger.snapshot("task-1")["elapsed_seconds"] == 1.0


def test_concurrent_proposals_have_one_host_call_and_frozen_identity():
    job = runner([], Ledger(), [])
    job._last_accounted_time = time.monotonic()
    job.evidence.append(evidence("memory", "memory:1"))
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def submit(submission):
        calls.append(submission)
        entered.set()
        assert release.wait(3)
        return "receipt:one"

    job.handlers = HostHandlers(job.handlers.memory_search, job.handlers.exact_read,
                                job.handlers.source_trace, job.handlers.document_read, submit)
    outcomes = []
    def invoke(value):
        outcomes.append(json.loads(job._handle_tool("reason_proposal_submit", {"proposal": value})))

    first = threading.Thread(target=invoke, args=({"evidence_refs": ["memory:1"], "action": "UPDATE"},))
    second = threading.Thread(target=invoke, args=({"evidence_refs": ["memory:1"], "action": "ADD"},))
    first.start()
    assert entered.wait(3)
    second.start()
    release.set()
    first.join()
    second.join()
    assert len(calls) == 1
    assert {item["status"] for item in outcomes} == {"submitted_untrusted", "proposal_conflict"}
    assert calls[0].proposal_revision == job._proposal_submission.proposal_revision


def test_host_submit_failure_keeps_identity_without_repeat():
    job = runner([], Ledger(), [])
    job._last_accounted_time = time.monotonic()
    job.evidence.append(evidence("memory", "memory:1"))
    calls = []
    def submit(value):
        calls.append(value)
        raise RuntimeError("uncertain host outcome")
    job.handlers = HostHandlers(job.handlers.memory_search, job.handlers.exact_read,
                                job.handlers.source_trace, job.handlers.document_read, submit)
    proposal = {"evidence_refs": ["memory:1"], "action": "UPDATE"}
    with pytest.raises(RuntimeError):
        job._handle_tool("reason_proposal_submit", {"proposal": proposal})
    assert len(calls) == 1
    assert job._proposal_submission is calls[0]
    retry = json.loads(job._handle_tool("reason_proposal_submit", {"proposal": proposal}))
    assert retry == {"status": "submission_uncertain", "proposal_revision": calls[0].proposal_revision}
    assert len(calls) == 1


def test_late_cancel_after_host_receipt_does_not_resubmit():
    flag = {"cancelled": False}
    job = runner([], Ledger(), [], cancelled=lambda: flag["cancelled"])
    job._last_accounted_time = time.monotonic()
    job.evidence.append(evidence("memory", "memory:1"))
    calls = []
    def submit(value):
        calls.append(value)
        flag["cancelled"] = True
        return "receipt:late"
    job.handlers = HostHandlers(job.handlers.memory_search, job.handlers.exact_read,
                                job.handlers.source_trace, job.handlers.document_read, submit)
    proposal = {"evidence_refs": ["memory:1"], "action": "UPDATE"}
    assert json.loads(job._handle_tool("reason_proposal_submit", {"proposal": proposal}))["status"] == "submitted_untrusted"
    assert job.proposal_receipt == "receipt:late" and len(calls) == 1
    with pytest.raises(ToolBudgetExceeded, match="cancelled"):
        job._handle_tool("reason_proposal_submit", {"proposal": proposal})
    assert len(calls) == 1


@pytest.mark.parametrize("status", ["truncated", "missing", "error", "changed_complete"])
def test_latest_evidence_version_controls_submission(status):
    job = runner([], Ledger(), [])
    job._last_accounted_time = time.monotonic()
    job.evidence.append(evidence("memory", "memory:1"))
    newest = Evidence("memory", "memory:1", "sha256:new", "line:1",
                      "2026-09-30T00:00:00Z", "complete" if status == "changed_complete" else status,
                      "new read", {"text": "new"})
    job.evidence.append(newest)
    value = {"evidence_refs": ["memory:1"], "action": "UPDATE"}
    assert json.loads(job._handle_tool("reason_proposal_submit", {"proposal": value}))["status"] == "needs_evidence"
    assert job.proposal_receipt is None
    if status == "changed_complete":
        value["evidence_refs"] = [{"source_kind": "memory", "exact_reference": "memory:1", "digest": "sha256:new"}]
        assert json.loads(job._handle_tool("reason_proposal_submit", {"proposal": value}))["status"] == "submitted_untrusted"
        assert job._proposal_submission.evidence == (newest,)


def test_bare_reference_rejects_ambiguous_source_kinds_before_version_filter():
    job = runner([], Ledger(), [])
    job._last_accounted_time = time.monotonic()
    old = evidence("memory", "shared:1")
    changed = Evidence("memory", "shared:1", "sha256:new", "line:1",
                       "2026-09-30T00:00:00Z", "complete", "new read", {"text": "new"})
    stable = evidence("document", "shared:1")
    job.evidence.extend((old, changed, stable))
    proposal = {"evidence_refs": ["shared:1"], "action": "UPDATE"}
    rejected = json.loads(job._handle_tool("reason_proposal_submit", {"proposal": proposal}))
    assert rejected == {"status": "needs_evidence",
                        "reason": "explicit_source_kind_exact_reference_digest_required"}
    assert job.proposal_receipt is None
    proposal["evidence_refs"] = [{"source_kind": "document", "exact_reference": "shared:1",
                                  "digest": stable.digest}]
    accepted = json.loads(job._handle_tool("reason_proposal_submit", {"proposal": proposal}))
    assert accepted["status"] == "submitted_untrusted"
    assert job._proposal_submission.evidence == (stable,)


def test_evidence_and_submission_nested_content_are_immutable():
    original = {"nested": [{"text": "original"}]}
    item = Evidence("memory", "memory:1", "sha256:1", "line:1", "now", "complete", "full", original)
    original["nested"][0]["text"] = "tampered"
    assert item.content["nested"][0]["text"] == "original"
    with pytest.raises(TypeError):
        item.content["nested"][0]["text"] = "tampered"
    assert json.loads(item.content_json)["nested"][0]["text"] == "original"
    job = runner([], Ledger(), [])
    job._last_accounted_time = time.monotonic()
    job.evidence.append(item)
    job._handle_tool("reason_proposal_submit", {"proposal": {"evidence_refs": ["memory:1"]}})
    submission = job._proposal_submission
    assert hashlib.sha256(submission.evidence_json.encode()).hexdigest() == submission.evidence_digest.removeprefix("sha256:")
    assert submission.evidence[0].content["nested"][0]["text"] == "original"


@pytest.mark.parametrize("exit_kind", ["cancelled", "budget", "malformed", "model_error"])
def test_internal_prompt_never_enters_scratch_or_stdout(fake_route, capsys, exit_kind):
    sentinel = "INTERNAL_PRIVATE_BODY_" + exit_kind.upper()
    outcome = response(content="done")
    if exit_kind == "malformed":
        outcome = response([call("terminal", {"command": "forbidden"})])
    limits = BudgetLimits(1, 8, 8, 1000, 200, 60, 0.1) if exit_kind == "budget" else None
    job = runner([outcome], Ledger(), [], limits=limits,
                 cancelled=(lambda: True) if exit_kind == "cancelled" else None)
    job.task = ReasoningTask("task-leak-" + exit_kind, "owner-1", "rev-1", sentinel)
    if exit_kind == "model_error":
        job.transport.complete = lambda *_: (_ for _ in ()).throw(RuntimeError(sentinel))
    result = job.run()
    assert result.status in {"cancelled", "budget_exhausted", "restricted_failure", "failed", "needs_evidence"}
    assert sentinel not in capsys.readouterr().out
    scratch = Path(os.environ["HERMES_TEST_ISOLATION"])
    for file in scratch.rglob("*"):
        if file.is_file():
            try:
                data = file.read_bytes()
            except PermissionError:
                # Another test's still-open SQLite sidecar may be temporarily locked on Windows.
                continue
            assert sentinel.encode() not in data, file


def test_session_reader_covers_compacted_excludes_rewind_and_pages(tmp_path):
    from hermes_state import SessionDB

    path = tmp_path / "state.db"
    writer = SessionDB(db_path=path)
    try:
        writer.create_session("approved", source="cli")
        for content in ("old compacted", "active fact", "rewound secret", "tail"):
            writer.append_message("approved", "user", content)
        ids = [row["id"] for row in writer.get_messages("approved")]
        writer._write_rowcount("UPDATE messages SET active = 0, compacted = 1 WHERE id = ?", (ids[0],))
        writer._write_rowcount("UPDATE messages SET active = 0, compacted = 0 WHERE id = ?", (ids[2],))
        with ScopedSessionEvidenceReader(db_path=path, owner_key="owner-1",
                                         allowed_session_ids=frozenset({"approved"}),
                                         page_size=1, max_rows=8, max_field_bytes=64) as reader:
            queries = []
            original = reader._db._read_all
            def tracked(sql, params):
                queries.append((sql, params))
                return original(sql, params)
            reader._db._read_all = tracked
            metered = []
            before = reader.read("session:approved", lambda: metered.append(1))
            assert before.status == "complete"
            assert len(metered) == len(queries) >= 4
            assert all("LIMIT" in sql for sql, _ in queries)
            text = " ".join(row["content"] for row in before.content["messages"])
            assert "old compacted" in text and "rewound secret" not in text
            assert reader.search("old compacted").status == "complete"
            assert reader.search("rewound secret").content["hits"] == ()
            writer._write_rowcount("UPDATE messages SET content = ? WHERE id = ?", ("changed compacted", ids[0]))
            after_compaction = reader.read("session:approved").digest
            assert after_compaction != before.digest
            writer._write_rowcount("UPDATE messages SET content = ? WHERE id = ?", ("changed rewind", ids[2]))
            assert reader.read("session:approved").digest == after_compaction
    finally:
        writer.close()


def test_session_reader_bounds_field_and_row_before_materialization(tmp_path):
    from hermes_state import SessionDB

    path = tmp_path / "state.db"
    writer = SessionDB(db_path=path)
    try:
        writer.create_session("approved", source="cli")
        for value in ("x" * 3000, "second", "third"):
            writer.append_message("approved", "user", value)
        with ScopedSessionEvidenceReader(db_path=path, owner_key="owner-1",
                                         allowed_session_ids=frozenset({"approved", "absent"}),
                                         page_size=1, max_rows=2, max_field_bytes=32) as reader:
            queries = []
            original = reader._db._read_all
            reader._db._read_all = lambda sql, params: queries.append((sql, params)) or original(sql, params)
            result = reader.read("session:approved")
            assert result.status == "truncated"
            assert result.content["message_count"] == 2
            assert result.content["messages"][0]["content_truncated"] is True
            assert len(result.content["messages"][0]["content"].encode()) <= 32
            assert all("LIMIT" in sql for sql, _ in queries)
            assert reader.search("missing").status == "missing"
            count = 0
            def budget_meter():
                nonlocal count
                count += 1
                if count == 3:
                    from agent.memory_reasoning import ToolBudgetExceeded
                    raise ToolBudgetExceeded("underlying_reads")
            with pytest.raises(ToolBudgetExceeded):
                reader.read("session:approved", budget_meter)
            assert count == 3
    finally:
        writer.close()


def test_session_reader_utf8_field_and_total_bytes(tmp_path):
    from hermes_state import SessionDB

    path = tmp_path / "state.db"
    writer = SessionDB(db_path=path)
    try:
        writer.create_session("approved", source="cli")
        writer.append_message("approved", "user", "中🙂")
        writer.append_message("approved", "user", "🙂中")
        for field_bytes, total_bytes, expected in (
            (1, 1, ("", "")),
            (3, 3, ("中",)),
            (5, 7, ("中", "🙂")),
        ):
            with ScopedSessionEvidenceReader(db_path=path, owner_key="owner-1",
                                             allowed_session_ids=frozenset({"approved"}),
                                             page_size=1, max_rows=2, max_field_bytes=field_bytes,
                                             max_bytes=total_bytes) as reader:
                result = reader.read("session:approved")
            rows = result.content["messages"]
            assert tuple(row["content"] for row in rows) == expected
            assert all(len(row["content"].encode("utf-8")) <= field_bytes for row in rows)
            assert sum(len(row["content"].encode("utf-8")) for row in rows) <= total_bytes
            assert result.status == "truncated" and result.content["truncated"] is True
            assert "bounded content prefixes" in result.coverage
            assert all(row["content_truncated"] for row in rows)
    finally:
        writer.close()


@pytest.mark.parametrize("bad", [True, 1.5, -1, float("nan"), float("inf"), -float("inf")])
def test_actual_usage_invalid_dimension_is_unknown_not_zero(fake_route, bad):
    item = response(content="unused")
    item.usage.prompt_tokens = bad
    job = runner([item], Ledger(), [])
    outcome = job.run()
    assert outcome.status == "restricted_failure"
    assert outcome.usage["actual_input_tokens_unknown"] == 1
    assert "actual_input_tokens" not in outcome.usage
    assert outcome.usage["actual_output_tokens"] == 10


def test_zero_actual_cost_is_valid_when_measured(fake_route):
    job = runner([response(content="done")], Ledger(), [])
    job.transport.actual_cost = lambda *_: 0.0
    outcome = job.run()
    assert outcome.usage["actual_cost_usd"] == 0.0
    assert "actual_cost_usd_unknown" not in outcome.usage


@pytest.mark.parametrize("bad", [True, -1, float("nan"), float("inf"), -float("inf"), None])
def test_invalid_actual_cost_is_unknown(fake_route, bad):
    job = runner([response(content="done")], Ledger(), [])
    job.transport.actual_cost = lambda *_: bad
    outcome = job.run()
    assert outcome.status == "restricted_failure"
    assert outcome.usage["actual_cost_usd_unknown"] == 1
    assert "actual_cost_usd" not in outcome.usage
    assert outcome.usage["actual_input_tokens"] == 20


@pytest.mark.parametrize("bad", [True, 0, -1, float("nan"), float("inf"), -float("inf")])
def test_deadline_must_be_future_finite_number(bad):
    job = runner([], Ledger(), [])
    with pytest.raises(ValueError):
        RestrictedReasoningRunner(task=job.task, route=job.route, limits=job.limits,
                                  handlers=job.handlers, ledger=job.ledger, transport=job.transport,
                                  cancelled=lambda: False, deadline_monotonic=bad)


def test_tool_result_body_does_not_enter_scratch(fake_route, capsys):
    sentinel = "PRIVATE_EVIDENCE_TOOL_RESULT_CONTENT"
    job = runner([response([call("reason_memory_search", {"query": "find"})]),
                  response(content="done")], Ledger(), [])
    def search(query, meter):
        meter()
        return Evidence("memory", "memory:1", "sha256:tool", "line:1", "now",
                        "complete", "full", {"text": sentinel})
    job.handlers = HostHandlers(search, job.handlers.exact_read, job.handlers.source_trace,
                                job.handlers.document_read, job.handlers.submit_proposal)
    job.run()
    assert sentinel not in capsys.readouterr().out
    scratch = Path(os.environ["HERMES_TEST_ISOLATION"])
    for file in scratch.rglob("*"):
        if file.is_file():
            try:
                assert sentinel.encode() not in file.read_bytes(), file
            except PermissionError:
                continue


@pytest.mark.parametrize("spill_case,exit_kind", [
    ("oversized_result", "model_error"),
    ("aggregate_budget", "budget"),
])
def test_large_evidence_never_spills_from_real_agent(fake_route, capsys, spill_case, exit_kind):
    sentinel = "PRIVATE_EVIDENCE_" + uuid4().hex
    calls = 1 if spill_case == "oversized_result" else 3
    limits = BudgetLimits(3, 3, 8, 1000, 200, 60, 0.1) if exit_kind == "budget" else None
    script = [response([call("reason_memory_search", {"query": str(i)}, i)
                        for i in range(1, calls + 1)])]
    if exit_kind == "budget":
        script.append(response([call("reason_memory_search", {"query": "budget"}, calls + 1)]))
    job = runner(script, Ledger(), [], limits=limits)
    per_result_chars = []
    per_result_thresholds = []
    turn_budget = []

    def search(query, meter):
        from agent.tool_executor import _budget_for_agent

        meter()
        budget = _budget_for_agent(job._agent)
        turn_budget.append(budget.turn_budget)
        size = budget.default_result_size + 10_000 if spill_case == "oversized_result" else int(
            budget.default_result_size * 0.8)
        body = sentinel + "x" * size
        per_result_chars.append(len(body))
        per_result_thresholds.append(budget.default_result_size)
        return Evidence("memory", "memory:" + query, "sha256:tool", "line:1", "now",
                        "complete", "full", {"text": body})

    job.handlers = HostHandlers(search, job.handlers.exact_read, job.handlers.source_trace,
                                job.handlers.document_read, job.handlers.submit_proposal)
    if exit_kind == "model_error":
        original_complete = job.transport.complete

        def fail_after_tools(request, resolved_agent):
            if job.transport.requests:
                raise RuntimeError("synthetic model error")
            return original_complete(request, resolved_agent)

        job.transport.complete = fail_after_tools
    result = job.run()
    assert len(per_result_chars) == calls
    if spill_case == "oversized_result":
        assert per_result_chars[0] > per_result_thresholds[0]
    else:
        assert all(size < threshold for size, threshold in zip(per_result_chars, per_result_thresholds))
        assert sum(per_result_chars) > turn_budget[0]
        tool_contents = [message["content"] for message in job.transport.requests[1]["messages"]
                         if message["role"] == "tool"]
        assert len(tool_contents) == calls
        assert sum(len(content) for content in tool_contents) > turn_budget[0]
    assert result.status == ("budget_exhausted" if exit_kind == "budget" else "restricted_failure")
    captured = capsys.readouterr()
    assert sentinel not in captured.out + captured.err
    scratch = Path(os.environ["HERMES_TEST_ISOLATION"])
    spillover = scratch / "home" / "cache" / "spillover"
    assert not spillover.exists() or not any(path.is_file() for path in spillover.rglob("*"))
    for file in scratch.rglob("*"):
        if file.is_file():
            try:
                assert sentinel.encode("utf-8") not in file.read_bytes(), file
            except PermissionError:
                continue


@pytest.mark.parametrize("failure", [False, True])
def test_owned_agent_client_closes_on_normal_and_error_exit(fake_route, monkeypatch, failure):
    closed = []
    def client_factory(**kwargs):
        client = SimpleNamespace(close=lambda: closed.append(client))
        return client
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", client_factory)
    job = runner([response(content="done")], Ledger(), [])
    if failure:
        job.transport.complete = lambda *_: (_ for _ in ()).throw(RuntimeError("model failure"))
    job.run()
    assert len(closed) == 1


def test_partial_agent_initialization_closes_owned_client(fake_route, monkeypatch):
    from run_agent import AIAgent

    closed = []
    def failing_init(self, *args, **kwargs):
        self.client = SimpleNamespace(close=lambda: closed.append(True))
        self.session_id = "memory-reasoning-partial-init"
        raise RuntimeError("initialization failed")
    monkeypatch.setattr(AIAgent, "__init__", failing_init)
    outcome = runner([], Ledger(), []).run()
    assert outcome.status == "failed"
    assert closed == [True]
