"""Provider-reported spend in its own column, and every spend reader adding the two columns.

An ``actual`` amount (a provider's own billed figure, returned by ``ProviderProfile.get_usage_cost``) is
stored in ``actual_cost_usd`` and an estimate in ``estimated_cost_usd``, for main-loop turns, auxiliary
calls, Codex app-server turns and background-review forks. The two columns hold disjoint calls, so
every in-repo reader adds them. A fake profile returns the ``CostResult`` directly, so these tests don't
depend on how the provider's report reaches ``get_usage_cost``.
"""

import json
import logging
import math
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

HOST = "https://api.billfake.test/v1"


def _cost(amount, status="actual"):
    from agent.usage_pricing import CostResult

    source = "provider_cost_api" if status == "actual" else "official_docs_snapshot"
    return CostResult(amount_usd=Decimal(str(amount)), status=status, source=source, label="x")


@pytest.fixture
def bill_profile():
    """A provider whose ``get_usage_cost`` returns ``profile.next_cost`` (a reported 0.5 by default)."""
    import providers
    from providers.base import ProviderProfile

    class BillFakeProfile(ProviderProfile):
        next_cost = None

        def get_usage_cost(self, model, usage, **_kw):
            return self.next_cost

    providers.list_providers()  # discovery first, so the fake survives it
    profile = BillFakeProfile(name="billfake", base_url=HOST, hostnames=("billfake.test",))
    profile.next_cost = _cost("0.5")
    providers.register_provider(profile)
    yield profile
    import hermes_cli.auth
    providers._REGISTRY.pop("billfake", None)
    providers._SOURCES.pop("billfake", None)
    providers._PROVIDER_LIST_CACHE = None
    hermes_cli.auth.PROVIDER_REGISTRY.pop("billfake", None)


@pytest.fixture
def db(tmp_path):
    from hermes_state import SessionDB

    store = SessionDB(tmp_path / "state.db")
    yield store
    store.close()


def _agent(db, session_id, **kwargs):
    from run_agent import AIAgent

    with patch("agent.process_bootstrap.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        agent = AIAgent(api_key="k", base_url=HOST, provider="billfake", model="fake-model", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, session_id=session_id, platform="cli",
                        save_trajectories=False, session_db=db, **kwargs)
    agent._session_db_created = True
    agent.client = SimpleNamespace()  # a MagicMock client would pass for a MoA client
    return agent


def _chat_usage():
    return SimpleNamespace(prompt_tokens=100, completion_tokens=7, total_tokens=107,
                           prompt_tokens_details=SimpleNamespace(cached_tokens=0), completion_tokens_details=None)


def _record_turn(agent):
    from agent import turn_usage

    turn_usage.record_response_usage(agent, SimpleNamespace(usage=_chat_usage(), id=None, model="fake-model"),
                                     messages=[{"role": "user", "content": "hi"}], api_call_count=1,
                                     api_duration=0.1, compression_attempts=0, max_compression_attempts=3)


def _rows(db, session_id):
    with db._read_ctx() as conn:
        return {r["task"]: dict(r) for r in conn.execute(
            "SELECT task, estimated_cost_usd, actual_cost_usd, cost_status, input_tokens FROM session_model_usage"
            " WHERE session_id = ?", (session_id,))}


def _split(row):
    return (row["estimated_cost_usd"] or 0.0, row["actual_cost_usd"] or 0.0)  # NULL = nothing stored


# ── writers ──────────────────────────────────────────────────────────────────────────────────


def test_turns_split_reported_and_estimated_spend_and_readers_sum_them(bill_profile, db):
    """A mixed session: one reported call, one estimated call. Each lands in its own column (the
    other column is never nulled), and every spend reader adds the two."""
    from hermes_cli.web_routers import analytics

    db.create_session("t", source="cli", model="fake-model")
    agent = _agent(db, "t")
    try:
        _record_turn(agent)
        bill_profile.next_cost = _cost("0.25", status="estimated")
        _record_turn(agent)
        assert agent.session_estimated_cost_usd == pytest.approx(0.75)  # the agent's figure is the whole spend
        assert agent.session_actual_cost_usd == pytest.approx(0.5)
        db.flush_token_counts()
    finally:
        agent.close()
    assert _split(db.get_session("t")) == (pytest.approx(0.25), pytest.approx(0.5))
    assert _split(_rows(db, "t")[""]) == (pytest.approx(0.25), pytest.approx(0.5))
    db.create_session("only-estimated", source="cli")
    db.update_token_counts("only-estimated", input_tokens=1, model="m", billing_provider="p", api_call_count=1,
                           estimated_cost_usd=0.125)
    db._execute_write(lambda conn: conn.execute("UPDATE sessions SET message_count = 1"))
    assert db.usage_totals()["cost_usd"] == pytest.approx(0.875)
    assert [s["id"] for s in db.list_prune_candidates(min_cost=0.7)] == ["t"]  # `hermes sessions prune --min-cost`
    with patch.object(analytics, "_open_session_db_for_profile", return_value=db), patch.object(db, "close"):
        totals = analytics._get_usage_analytics(days=1)["totals"]
    assert totals["total_estimated_cost"] + totals["total_actual_cost"] == pytest.approx(0.875)


def test_moa_advisors_stay_estimated_when_the_aggregator_is_reported(bill_profile, db):
    """MoA advisors are priced by core at each advisor's own rate: an estimate, even when the
    aggregator's provider reports its own call. Only the aggregator's amount is ``actual``."""
    from agent.usage_pricing import CanonicalUsage

    db.create_session("m", source="cli", model="fake-model")
    agent = _agent(db, "m")
    agent.client = SimpleNamespace(consume_reference_usage=lambda: (CanonicalUsage(input_tokens=50), 0.25),
                                   last_aggregator_slot=None)
    try:
        _record_turn(agent)
        assert agent.session_estimated_cost_usd == pytest.approx(0.75)
        assert agent.session_actual_cost_usd == pytest.approx(0.5)
        db.flush_token_counts()
    finally:
        agent.close()
    row = db.get_session("m")
    assert _split(row) == (pytest.approx(0.25), pytest.approx(0.5))
    assert (row["cost_status"], row["input_tokens"], row["api_call_count"]) == ("actual", 150, 1)
    assert _split(_rows(db, "m")[""]) == (pytest.approx(0.25), pytest.approx(0.5))


def test_moa_advisors_and_an_estimated_aggregator_share_one_write(bill_profile, db):
    from agent.usage_pricing import CanonicalUsage

    bill_profile.next_cost = _cost("0.5", status="estimated")
    db.create_session("m", source="cli", model="fake-model")
    agent = _agent(db, "m")
    agent.client = SimpleNamespace(consume_reference_usage=lambda: (CanonicalUsage(input_tokens=50), 0.25),
                                   last_aggregator_slot=None)
    try:
        _record_turn(agent)
        assert getattr(agent, "session_actual_cost_usd", 0.0) == 0.0
        db.flush_token_counts()
    finally:
        agent.close()
    assert _split(db.get_session("m")) == (pytest.approx(0.75), 0.0)


def test_aux_reported_cost_lands_in_actual_cost_usd(bill_profile, db):
    from agent.aux_accounting import record_aux_usage, reset_accounting_context, set_accounting_context

    db.create_session("s", source="cli")
    token = set_accounting_context(db, "s")
    try:
        record_aux_usage(SimpleNamespace(model="fake-model", usage=_chat_usage()), "title_generation",
                         provider="billfake", base_url=HOST)
        bill_profile.next_cost = _cost("0.25", status="estimated")
        record_aux_usage(SimpleNamespace(model="fake-model", usage=_chat_usage()), "title_generation",
                         provider="billfake", base_url=HOST)
    finally:
        reset_accounting_context(token)
    row = _rows(db, "s")["title_generation"]
    assert _split(row) == (pytest.approx(0.25), pytest.approx(0.5))
    by_task = db.auxiliary_usage_by_task("s")["title_generation"]
    # ``estimated_cost_usd`` keeps meaning "the task's spend" for the -z usage report; the reported part is apart.
    assert (by_task["estimated_cost_usd"], by_task["actual_cost_usd"]) == (pytest.approx(0.75), pytest.approx(0.5))


def test_codex_app_server_turn_splits_the_columns(bill_profile, db):
    from agent.codex_runtime import _record_codex_app_server_usage

    db.create_session("c", source="cli", model="fake-model")
    agent = _agent(db, "c")
    turn = SimpleNamespace(token_usage_last={"inputTokens": 100, "cachedInputTokens": 0, "outputTokens": 7,
                                             "reasoningOutputTokens": 0, "totalTokens": 107})
    try:
        result = _record_codex_app_server_usage(agent, turn)
        assert (result["estimated_cost_usd"], result["cost_status"]) == (pytest.approx(0.5), "actual")
        assert agent.session_actual_cost_usd == pytest.approx(0.5)
        db.flush_token_counts()
    finally:
        agent.close()
    assert _split(db.get_session("c")) == (0.0, pytest.approx(0.5))


def test_background_review_fork_spend_is_split(db):
    from agent import background_review

    fork = SimpleNamespace(model="m", provider="billfake", base_url=HOST, session_input_tokens=10,
                           session_output_tokens=1, session_cache_read_tokens=0, session_cache_write_tokens=0,
                           session_reasoning_tokens=0, session_api_calls=2, session_estimated_cost_usd=0.75,
                           session_actual_cost_usd=0.5, session_cost_status="actual",
                           session_cost_source="provider_cost_api")
    db.create_session("p", source="cli")
    background_review._record_review_usage_to_parent(
        SimpleNamespace(_session_db=db, session_id="p"), background_review._snapshot_review_usage(fork))
    assert _split(_rows(db, "p")["background_review"]) == (pytest.approx(0.25), pytest.approx(0.5))


def test_none_cost_keeps_the_stored_value_in_both_writers(db):
    """``estimated_cost_usd=None`` (an actual call) must keep the stored estimate, never null it:
    SQLite's NULL + x is NULL, so both the sessions UPDATE and the per-model upsert must coalesce."""
    db.create_session("s", source="cli")
    db.update_token_counts("s", input_tokens=1, model="m", billing_provider="p", api_call_count=1,
                           estimated_cost_usd=0.25)
    db.update_token_counts("s", input_tokens=1, model="m", billing_provider="p", api_call_count=1,
                           estimated_cost_usd=None, actual_cost_usd=0.5)
    db.update_token_counts("s", input_tokens=1, model="m", billing_provider="p", api_call_count=1,
                           estimated_cost_usd=0.125, actual_cost_usd=None)
    db.record_auxiliary_usage("s", "vision", model="v", actual_cost_usd=0.5)
    db.record_auxiliary_usage("s", "vision", model="v", estimated_cost_usd=0.25)
    assert _split(db.get_session("s")) == (pytest.approx(0.375), pytest.approx(0.5))
    rows = _rows(db, "s")
    assert _split(rows[""]) == (pytest.approx(0.375), pytest.approx(0.5))
    assert _split(rows["vision"]) == (pytest.approx(0.25), pytest.approx(0.5))


def test_worker_store_accepts_the_aux_split(db):
    import hermes_state_runtime as rt

    db.create_session("s", source="cli")
    epoch = rt.begin_runtime_epoch(db, instance_id="owner")
    scope = dict(epoch=epoch, execution_id="w", session_id="s", generation=0)
    rt.register_worker_execution(db, **scope, kind="compute", adoption_secret="private")
    rt.mutate_worker_execution(db, **scope, sequence=1, operation="usage.auxiliary", payload={
        "task": "vision", "model": "v", "input_tokens": 3, "actual_cost_usd": 0.5, "cost_status": "actual",
        "cost_source": "provider_cost_api"})
    assert _rows(db, "s")["vision"]["actual_cost_usd"] == pytest.approx(0.5)


# ── readers ──────────────────────────────────────────────────────────────────────────────────


def test_insights_report_both_buckets_and_count_aux_reported_spend(db):
    from agent.insights import InsightsEngine

    db.create_session("s", source="cli", model="m")
    db.update_token_counts("s", input_tokens=10, model="m", billing_provider="p", api_call_count=1,
                           actual_cost_usd=0.5, cost_status="actual", cost_source="provider_cost_api")
    db.update_token_counts("s", input_tokens=10, model="m", billing_provider="p", api_call_count=1,
                           estimated_cost_usd=0.25, cost_status="estimated", cost_source="official_docs_snapshot")
    db.record_auxiliary_usage("s", "vision", model="v", billing_provider="p", input_tokens=5,
                              actual_cost_usd=0.125, cost_status="actual", cost_source="provider_cost_api")
    engine = InsightsEngine(db)
    report = engine.generate(days=30)
    assert report["overview"]["actual_cost"] == pytest.approx(0.625)
    assert report["overview"]["estimated_cost"] == pytest.approx(0.25)
    text = engine.format_terminal(report)
    assert "Reported:           $0.6" in text and "~$0.6" not in text
    assert "Estimated:          ~$0.25" in text
    assert "reported" in engine.format_gateway(report)


def test_project_tree_total_adds_both_columns():
    from tui_gateway.project_tree import _project_node

    node = _project_node("p", "P", None, [], 2, 0.0, [], sessions=[
        {"id": "a", "actual_cost_usd": 0.5, "estimated_cost_usd": 0.25}, {"id": "b", "estimated_cost_usd": 0.125}])
    assert node["totalCostUsd"] == pytest.approx(0.875)


# ── non-finite costs: dropped at the sinks, read as 0 by the readers ──────────────────────────


@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_sinks_drop_a_non_finite_cost_delta_and_keep_the_tokens(db, caplog, bad):
    """NaN used to fail the per-model row's NOT NULL constraint (losing the whole delta, tokens
    included); inf poisoned every later sum. Both are dropped with a warning; tokens still land."""
    db.create_session("s", source="cli")
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        db.update_token_counts("s", input_tokens=5, model="m", billing_provider="p", api_call_count=1,
                               estimated_cost_usd=bad)
        db.update_token_counts("s", input_tokens=5, model="m", billing_provider="p", api_call_count=1,
                               actual_cost_usd=bad, cost_status="actual")
        db.record_auxiliary_usage("s", "vision", model="v", input_tokens=3, estimated_cost_usd=bad,
                                  actual_cost_usd=bad)
    row = db.get_session("s")
    assert row["input_tokens"] == 10 and _split(row) == (0.0, 0.0)
    rows = _rows(db, "s")
    assert rows[""]["input_tokens"] == 10 and _split(rows[""]) == (0.0, 0.0)
    assert rows["vision"]["input_tokens"] == 3 and _split(rows["vision"]) == (0.0, 0.0)
    assert len([r for r in caplog.records if "non-finite" in r.getMessage()]) == 4


def test_readers_treat_a_stored_inf_as_zero(db):
    """Rows written before the sinks checked can hold inf; readers must neither sum it nor emit it
    (JSON has no inf: the dashboard's response would fail to serialize)."""
    from agent.insights import InsightsEngine
    from hermes_cli.web_routers import analytics
    from tui_gateway.project_tree import _project_node

    db.create_session("s", source="cli", model="m")
    db.update_token_counts("s", input_tokens=10, model="m", billing_provider="p", api_call_count=1,
                           estimated_cost_usd=0.25, cost_status="estimated", cost_source="official_docs_snapshot")
    db.record_auxiliary_usage("s", "vision", model="v", billing_provider="p", input_tokens=5, estimated_cost_usd=0.125)
    inf = float("inf")
    db._execute_write(lambda conn: (
        conn.execute("UPDATE sessions SET actual_cost_usd = ?, message_count = 1, ended_at = started_at", (inf,)),
        conn.execute("UPDATE session_model_usage SET actual_cost_usd = ?", (inf,))))
    assert db.usage_totals()["cost_usd"] == pytest.approx(0.25)
    assert [s["id"] for s in db.list_prune_candidates(min_cost=0.1)] == ["s"]
    assert db.list_prune_candidates(min_cost=1.0) == []
    by_task = db.auxiliary_usage_by_task("s")["vision"]
    assert (by_task["estimated_cost_usd"], by_task["actual_cost_usd"]) == (pytest.approx(0.125), 0.0)
    with patch.object(analytics, "_open_session_db_for_profile", return_value=db), patch.object(db, "close"), \
            patch.object(analytics, "_model_capabilities", return_value={}):
        for payload in (analytics._get_usage_analytics(days=1), analytics._get_models_analytics(days=1)):
            json.dumps(payload, allow_nan=False, default=str)
    report = InsightsEngine(db).generate(days=30)
    json.dumps(report, allow_nan=False, default=str)
    assert math.isfinite(report["overview"]["actual_cost"]) and report["overview"]["actual_cost"] == 0.0
    node = _project_node("p", "P", None, [], 1, 0.0, [], sessions=[dict(db.get_session("s"))])
    assert node["totalCostUsd"] == pytest.approx(0.25)


def test_queued_deltas_drop_a_non_finite_cost_before_coalescing(db, caplog):
    """Adjacent queued deltas merge before the sink runs: one inf/NaN used to turn the whole merged
    cost non-finite, so the sink dropped every finite delta with it. Each is dropped on its own now."""
    db.create_session("s", source="cli")
    route = dict(model="m", billing_provider="p", api_call_count=1)
    batch = [("s", dict(route, input_tokens=1, estimated_cost_usd=cost))
             for cost in (0.25, float("inf"), 0.5, float("nan"), 0.125)]
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        merged = db._coalesce_token_deltas(batch)
        db._apply_token_batch(batch)
    assert len(merged) == 1 and merged[0][1]["estimated_cost_usd"] == pytest.approx(0.875)
    assert str(batch[1][1]["estimated_cost_usd"]) == "inf"  # the caller's delta is not mutated
    row = db.get_session("s")
    assert (row["input_tokens"], row["estimated_cost_usd"]) == (5, pytest.approx(0.875))
    assert _rows(db, "s")[""]["estimated_cost_usd"] == pytest.approx(0.875)
    assert any("non-finite" in r.getMessage() for r in caplog.records)


def test_session_payloads_clamp_a_stored_non_finite_cost():
    """Rows written before the sinks checked can hold inf: the project tree frame and the API server's
    session response must stay valid JSON. An unknown (None) cost stays None."""
    from gateway.platforms.api_server import APIServerAdapter
    from tui_gateway.methods_projects import _project_tree_row

    session = {"id": "s", "actual_cost_usd": float("inf"), "estimated_cost_usd": None}
    for payload in (_project_tree_row(session), APIServerAdapter._session_response(session)):
        assert (payload["actual_cost_usd"], payload["estimated_cost_usd"]) == (0.0, None)
        json.dumps(payload, allow_nan=False, default=str)
    session = {"id": "s", "actual_cost_usd": 0.5, "estimated_cost_usd": float("nan")}
    assert APIServerAdapter._session_response(session)["actual_cost_usd"] == 0.5
    assert _project_tree_row(session)["estimated_cost_usd"] == 0.0
