"""v5 efficiency metrics: per-turn cost, wasted tokens, tool output truncation, tool overhead and
prompt-cache breaks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import lifecycle
from hermes_cli.observability import relay_shared_metrics
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability import shared_metrics_efficiency as eff
from hermes_cli.observability.shared_metrics_model import record_model_friction
from tests.hermes_cli.test_relay_shared_metrics_runtime import (
    _stored_values,
    direct_runtime,
)


def _flush() -> None:
    relay_shared_metrics._get_runtime(retry_failed=True).relay.subscribers.flush()


def _rows(tmp_path, metric):
    _flush()
    return sorted((tuple(sorted(d.items())), v) for d, v in _stored_values(tmp_path, metric))


def _usage(total, cache_read=0):
    return {"input_tokens": total // 2, "output_tokens": total - total // 2, "cache_read_tokens": cache_read,
            "total_tokens": total}


def _turn(session_id, task_id, result, *, calls=(), tools=(), model="model-a", **start):
    """One user turn: ``calls`` are the usages of its primary calls, ``tools`` the tools it ran."""
    base = {"session_id": session_id, "task_id": task_id, "provider": "openrouter", "model": model}
    lifecycle.invoke_hook("pre_llm_call", **base, platform="cli", **start)
    for i, usage in enumerate(calls or [None]):
        request = {**base, "api_request_id": f"{task_id}-r{i}"}
        lifecycle.invoke_hook("pre_api_request", **request)
        lifecycle.invoke_hook("post_api_request", **request, usage=usage)
    for i, (tool, toolset) in enumerate(tools):
        call = {**base, "tool_call_id": f"{task_id}-c{i}", "tool_name": tool, "toolset": toolset}
        lifecycle.invoke_hook("pre_tool_call", **call, args={})
        lifecycle.invoke_hook("post_tool_call", **call, args={}, result="ok", status="ok")
    if result is not None:
        relay_shared_metrics.finish_task_run(session_id=session_id, task_id=task_id, platform="cli", result=result)


def _agent(session_id="s1", model="model-a", tools=None):
    tools = tools if tools is not None else [
        {"type": "function", "function": {"name": name, "description": "d" * 400, "parameters": {"type": "object"}}}
        for name in ("read_file", "terminal", "acme_private_tool")
    ]
    return SimpleNamespace(provider="openrouter", model=model, session_id=session_id, tools=tools,
                           valid_tool_names={t["function"]["name"] for t in tools})


def _dims(**kw):
    return tuple(sorted(kw.items()))


ROUTE = {"model": "model-a", "provider": "openrouter"}


def test_task_cost_is_one_row_per_user_turn_with_its_tokens_and_work(direct_runtime, tmp_path):
    """Tokens sum over the turn's primary calls; a session-close abort, a delegated child and an
    internal fork sharing the session id are not user turns."""
    _turn("s1", "t1", {"completed": True}, calls=[_usage(1500), _usage(900)],
          tools=[("terminal", "terminal")] * 3)
    _turn("s1", "t2", {"interrupted": True, "turn_exit_reason": "interrupted_by_user"}, calls=[_usage(60_000)])
    _turn("s1", "t3", {"failed": True, "turn_exit_reason": "failed"}, calls=[None])
    _turn("s1", "t4", None, calls=[_usage(10)])  # still running at close
    lifecycle.finalize_session(session_id="s1")
    _turn("child", "t1", {"completed": True}, calls=[_usage(10)], parent_session_id="s1")
    lifecycle.finalize_session(session_id="child")

    common = dict(ROUTE)
    assert _rows(tmp_path, "hermes.task_cost.count") == sorted([
        (_dims(**common, api_calls_bucket="2", outcome="completed", tokens_bucket="2k_to_10k",
               tool_calls_bucket="3_to_5"), 1),
        (_dims(**common, api_calls_bucket="1", outcome="interrupted", tokens_bucket="50k_to_200k",
               tool_calls_bucket="0"), 1),
        (_dims(**common, api_calls_bucket="1", outcome="failed", tokens_bucket="unknown",
               tool_calls_bucket="0"), 1),
    ])
    # The interrupt threw its tokens away.
    assert _rows(tmp_path, "hermes.wasted_tokens.count") == [
        (_dims(**common, reason="interrupt", tokens_bucket="50k_to_200k"), 1)]


def test_undo_and_retry_count_the_tokens_of_the_turns_they_discard(direct_runtime, tmp_path):
    """/undo 2 discards two turns; an interrupted turn then undone was wasted once; a session this
    process never saw still counts, tokens unknown."""
    _turn("s1", "t1", {"completed": True}, calls=[_usage(1_000)])
    _turn("s1", "t2", {"completed": True}, calls=[_usage(20_000)])
    _turn("s1", "t3", {"completed": True}, calls=[_usage(300_000)])
    record_model_friction("undo", session_id="s1", provider="openrouter", model="model-a", turns=2)
    _turn("s1", "t4", {"interrupted": True, "turn_exit_reason": "interrupted_by_user"}, calls=[_usage(5_000)])
    record_model_friction("undo", session_id="s1", provider="openrouter", model="model-a")
    record_model_friction("retry", session_id="s1", provider="openrouter", model="model-a")
    record_model_friction("retry", session_id="gone", provider="openrouter", model="model-a")

    assert _rows(tmp_path, "hermes.wasted_tokens.count") == sorted([
        (_dims(**ROUTE, reason="undo", tokens_bucket="200k_to_1m"), 1),
        (_dims(**ROUTE, reason="undo", tokens_bucket="10k_to_50k"), 1),
        (_dims(**ROUTE, reason="interrupt", tokens_bucket="2k_to_10k"), 1),
        (_dims(**ROUTE, reason="retry", tokens_bucket="lt_2k"), 1),
        (_dims(**ROUTE, reason="retry", tokens_bucket="unknown"), 1),
    ])


def test_tool_output_truncation_is_judged_after_the_turn_budget(direct_runtime, tmp_path):
    from tools.tool_result_storage import enforce_turn_budget, maybe_persist_tool_result
    from tools.budget_config import BudgetConfig

    agent = _agent()
    budget = BudgetConfig(default_result_size=5_000, turn_budget=12_000, preview_size=100)
    from tools.tool_output_truncate import truncate_head_tail

    cut = json.dumps({"output": truncate_head_tail("e" * 300_000, 3_000), "exit_code": 0})
    results = [("read_file", "a" * 200), ("terminal", "b" * 80_000), ("mcp_acme_query", "c" * 9_000),
               ("read_file", "d" * 4_900), ("execute_code", cut)]
    batch = []
    for i, (name, raw) in enumerate(results):
        kept = maybe_persist_tool_result(raw, name, f"c{i}", env=None, config=budget)
        eff.note_tool_result(agent, name, f"c{i}", raw, kept)
        batch.append({"role": "tool", "tool_call_id": f"c{i}", "content": kept})
    before = [m["content"] for m in batch]
    enforce_turn_budget(batch, env=None, config=budget)
    eff.record_tool_batch(agent, batch, before)
    assert not getattr(agent, "_shared_metrics_tool_outputs", None)

    rows = _rows(tmp_path, "hermes.tool_output_truncation.count")
    assert "acme" not in json.dumps(rows)
    got = {(dict(d)["tool"], dict(d)["original_size_bucket"], dict(d)["truncated"]) for d, _ in rows}
    assert ("read_file", "lt_1k", "no") in got
    assert ("terminal", "50k_to_100k", "yes") in got
    assert ("mcp", "1k_to_10k", "yes") in got
    assert ("execute_code", "100k_to_500k", "yes") in got  # cut inside the tool: its original size
    assert sum(v for _, v in rows) == 5


def test_tool_overhead_and_unused_toolsets_report_once_per_conversation(direct_runtime, tmp_path, monkeypatch):
    toolsets = {"read_file": "file", "terminal": "terminal", "acme_private_tool": "acme-private"}
    monkeypatch.setattr("model_tools.get_toolset_for_tool", toolsets.get)
    agent = _agent()
    _turn("s1", "t1", None, calls=[_usage(100)])
    eff.observe_request_tools(agent, agent.tools)
    eff.observe_request_tools(agent, agent.tools)  # same array: latched, no recount
    _turn("s1", "t1", {"completed": True}, tools=[("terminal", "terminal")])
    lifecycle.finalize_session(session_id="s1")

    overhead = _rows(tmp_path, "hermes.tool_overhead.count")
    assert [dict(d) for d, _ in overhead] == [{
        "enabled_tool_count_bucket": "3_to_5", "execution_surface": "cli", "tool_schema_tokens_bucket": "lt_2k"}]
    assert _rows(tmp_path, "hermes.tool_enabled_unused.count") == sorted([
        (_dims(toolset="custom", used="no"), 1), (_dims(toolset="file", used="no"), 1),
        (_dims(toolset="terminal", used="yes"), 1)])
    assert "acme" not in json.dumps(_stored_values(tmp_path, "hermes.tool_enabled_unused.count"))


def test_several_known_causes_before_one_cold_read_are_one_break(direct_runtime, tmp_path):
    agent = _agent()
    _turn("s1", "t1", {"completed": True}, calls=[_usage(9_000, cache_read=8_000)])
    eff.record_cache_break(agent, "compression")
    eff.record_cache_break(agent, "compression")
    eff.record_cache_break(agent, "model_switch")
    _turn("s1", "t2", {"completed": True}, calls=[_usage(9_000, cache_read=0)])
    assert _rows(tmp_path, "hermes.cache_break.count") == [(_dims(**ROUTE, cause="compression"), 1)]


def test_cache_breaks_count_known_causes_once_and_unannounced_cold_reads_as_misses(direct_runtime, tmp_path,
                                                                                  monkeypatch):
    agent = _agent()
    _turn("s1", "t1", {"completed": True}, calls=[_usage(9_000, cache_read=0), _usage(9_000, cache_read=8_000)])
    eff.record_cache_break(agent, "compression")
    _turn("s1", "t2", {"completed": True}, calls=[_usage(9_000, cache_read=0)])  # expected cold read
    _turn("s1", "t3", {"completed": True}, calls=[_usage(9_000, cache_read=8_000), _usage(9_000, cache_read=0)])
    # A different route's first read is cold by nature, not a break.
    _turn("s1", "t4", {"completed": True}, calls=[_usage(9_000, cache_read=0)], model="model-b")
    # Idle past the TTL: expiry.
    _turn("s1", "t5", {"completed": True}, calls=[_usage(9_000, cache_read=8_000)])
    rt = relay_shared_metrics._get_runtime(retry_failed=True)
    rt._sessions["s1"].efficiency.cache_ended_ns -= int(eff.CACHE_TTL_S * 1e9) + 1
    _turn("s1", "t6", {"completed": True}, calls=[_usage(9_000, cache_read=0)])
    # The tool array changing mid-conversation.
    monkeypatch.setattr("model_tools.get_toolset_for_tool", lambda name: None)
    eff.observe_request_tools(agent, agent.tools)
    eff.observe_request_tools(agent, agent.tools[:1])
    # Internal forks and persistence-disabled side agents never count.
    eff.record_cache_break(SimpleNamespace(**vars(_agent()), _memory_write_origin="background_review"), "compression")
    eff.record_cache_break(SimpleNamespace(**vars(_agent()), _persist_disabled=True), "compression")

    assert _rows(tmp_path, "hermes.cache_break.count") == sorted([
        (_dims(**ROUTE, cause="compression"), 1), (_dims(**ROUTE, cause="provider_reported_miss"), 1),
        (_dims(**ROUTE, cause="cache_expired"), 1), (_dims(**ROUTE, cause="toolset_change"), 1)])


def test_prompt_rebuild_names_a_model_switch_apart_from_other_rebuilds(monkeypatch):
    causes = []
    monkeypatch.setattr(eff, "record_cache_break", lambda agent, cause: causes.append(cause))
    agent = _agent(model="model-b")
    stored = "You are Hermes.\nModel: model-a\nProvider: openrouter\n"
    eff.record_prompt_rebuild(agent, stored, "stale_runtime", stored.replace("model-a", "model-b"))
    eff.record_prompt_rebuild(_agent(model="model-a"), stored, "stale_runtime", stored + "Host: x\n")
    eff.record_prompt_rebuild(agent, None, "null", "rebuilt")
    eff.record_prompt_rebuild(agent, stored, "stale_runtime", stored)  # replayed bytes: no break
    eff.record_prompt_rebuild(agent, None, "missing", "fresh")  # a brand-new conversation
    assert causes == ["model_switch", "system_prompt_rebuild", "system_prompt_rebuild"]


def test_nothing_is_recorded_while_disabled(direct_runtime, tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.read_raw_config_readonly", dict)
    agent = _agent()
    eff.note_tool_result(agent, "terminal", "c1", "x" * 10, "x" * 10)
    assert not getattr(agent, "_shared_metrics_tool_outputs", None)
    eff.record_cache_break(agent, "compression")
    eff.observe_request_tools(agent, agent.tools)
    assert not (tmp_path / "hermes-home" / "telemetry").exists()


@pytest.mark.parametrize("count,bucket", [(0, "0"), (2, "2"), (5, "3_to_5"), (25, "11_to_25"), (50, "26_to_50"),
                                          (100, "51_to_100"), (101, "gte_101")])
def test_turn_activity_buckets_reach_long_agentic_loops(count, bucket):
    assert eff.activity_bucket(count) == bucket
    assert bucket in contract.TURN_ACTIVITY_BUCKETS


def test_v3_schema_accepts_exactly_the_contract_values():
    from hermes_cli import observability

    schema = json.loads((Path(observability.__file__).parent / "schemas/hermes.shared_metrics.v4.schema.json").read_text())
    by_name = {d["properties"]["name"]["const"]: d for d in schema["$defs"].values() if "properties" in d}
    refs = {item["$ref"].rsplit("/", 1)[1] for item in schema["properties"]["metrics"]["items"]["oneOf"]}
    for metric in (contract.TASK_COST_METRIC, contract.WASTED_TOKENS_METRIC, contract.TOOL_OUTPUT_TRUNCATION_METRIC,
                   contract.TOOL_OVERHEAD_METRIC, contract.TOOL_ENABLED_UNUSED_METRIC, contract.CACHE_BREAK_METRIC):
        assert any(schema["$defs"][r]["properties"]["name"]["const"] == metric for r in refs if r.endswith("_counter"))
        dims = by_name[metric]["properties"]["dimensions"]["properties"]
        expected = contract._COUNTER_DIMENSION_VALUES[metric]
        for field_name, spec in dims.items():
            if "enum" in spec:
                assert set(spec["enum"]) == set(expected[field_name]), (metric, field_name)
        assert set(dims) == set(expected) | ({"model", "provider"} & set(dims)), metric
