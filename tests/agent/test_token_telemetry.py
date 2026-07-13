import json

from agent.token_telemetry import (
    TokenEfficiencyStore,
    build_token_efficiency_record,
    finalize_token_efficiency_record,
    finalize_token_efficiency_no_usage,
    finalize_token_efficiency_error,
    build_compression_efficiency_event,
    build_token_efficiency_optimizer_preview,
    render_token_efficiency_report,
    summarize_token_efficiency_records,
)


def test_build_record_attributes_prompt_blocks_without_raw_prompt_text():
    messages = [
        {"role": "system", "content": "SYSTEM SECRET PREFIX with stable guidance"},
        {"role": "user", "content": "Please inspect the project"},
        {"role": "assistant", "content": "I will read files"},
        {"role": "tool", "content": "VERY LARGE TOOL OUTPUT with sensitive local text"},
        {"role": "assistant", "content": "[CONTEXT COMPACTION] ## Progress\nOld context summary"},
        {"role": "user", "content": [{"type": "text", "text": "see image"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Execute shell commands",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        }
    ]

    record = build_token_efficiency_record(
        session_id="s1",
        turn_id="t1",
        api_request_id="r1",
        platform="cli",
        provider="openrouter",
        model="test/model",
        api_mode="chat_completions",
        messages=messages,
        tools=tools,
        rough_request_tokens=1234,
    )

    blocks = {block["kind"]: block for block in record["blocks"]}
    assert {"system", "tools_schema", "history_recent", "tool_results", "summary", "attachments"} <= set(blocks)
    assert blocks["tools_schema"]["count"] == 1
    assert blocks["attachments"]["image_count"] == 1
    assert blocks["system"]["cacheable"] is True
    assert blocks["system"]["stable_hash"]
    assert record["rough_request_tokens"] == 1234

    serialized = json.dumps(record)
    assert "SYSTEM SECRET PREFIX" not in serialized
    assert "VERY LARGE TOOL OUTPUT" not in serialized
    assert "Please inspect the project" not in serialized
    assert "Execute shell commands" not in serialized


def test_finalize_record_adds_usage_cache_metrics_and_avoidable_diagnostics():
    record = build_token_efficiency_record(
        session_id="s1",
        turn_id="t1",
        api_request_id="r1",
        platform="cli",
        provider="anthropic",
        model="claude",
        api_mode="chat_completions",
        messages=[
            {"role": "system", "content": "x" * 8000},
            {"role": "tool", "content": "y" * 12000},
            {"role": "user", "content": "question"},
        ],
        tools=[{"name": f"tool_{i}", "description": "z" * 2000} for i in range(8)],
        rough_request_tokens=12000,
    )

    finalized = finalize_token_efficiency_record(
        record,
        usage={
            "input_tokens": 9000,
            "output_tokens": 1000,
            "cache_read_tokens": 3000,
            "cache_write_tokens": 1000,
            "reasoning_tokens": 250,
        },
    )

    assert finalized["actual"]["input_tokens"] == 9000
    assert finalized["actual"]["cache_read_tokens"] == 3000
    assert finalized["cache"]["provider_reported"] is True
    assert finalized["cache"]["hit_ratio"] == 3000 / 13000
    assert finalized["diagnostics"]
    assert any(item["kind"] in {"tool_schema_overhead", "tool_result_bloat"} for item in finalized["diagnostics"])


def test_store_appends_jsonl_without_raw_prompts_and_summarizes(tmp_path):
    path = tmp_path / "token_efficiency.jsonl"
    store = TokenEfficiencyStore(path)
    record = build_token_efficiency_record(
        session_id="s1",
        turn_id="t1",
        api_request_id="r1",
        platform="cli",
        provider="openrouter",
        model="model-a",
        api_mode="chat_completions",
        messages=[{"role": "system", "content": "raw secret system prompt"}],
        tools=[],
        rough_request_tokens=100,
    )
    store.append(finalize_token_efficiency_record(record, {"input_tokens": 80, "output_tokens": 20}))

    text = path.read_text()
    assert "raw secret system prompt" not in text
    rows = store.read_recent(limit=5)
    assert len(rows) == 1
    assert rows[0]["api_request_id"] == "r1"

    summary = summarize_token_efficiency_records(rows)
    assert summary["records"] == 1
    assert summary["total_input_tokens"] == 80
    assert summary["total_output_tokens"] == 20
    assert summary["top_blocks"][0]["rough_tokens"] >= 0


def test_render_token_efficiency_report_prioritizes_reduction_recommendations(tmp_path):
    path = tmp_path / "token_efficiency.jsonl"
    store = TokenEfficiencyStore(path)
    bloated = {
        "schema": "hermes.token_efficiency.v1",
        "session_id": "s1",
        "api_request_id": "r2",
        "model": "model-a",
        "provider": "openrouter",
        "rough_request_tokens": 10000,
        "blocks": [
            {"kind": "tools_schema", "rough_tokens": 4200, "count": 40},
            {"kind": "tool_results", "rough_tokens": 2800},
            {"kind": "history_recent", "rough_tokens": 2000},
            {"kind": "system", "rough_tokens": 1000, "stable_hash": "abc"},
        ],
        "actual": {"input_tokens": 9000, "output_tokens": 800, "cache_read_tokens": 300, "cache_write_tokens": 700},
        "cache": {"provider_reported": True, "hit_ratio": 0.03},
        "diagnostics": [
            {"kind": "tool_schema_overhead", "severity": "warn", "ratio": 0.42},
            {"kind": "tool_result_bloat", "severity": "warn", "ratio": 0.28},
            {"kind": "low_cache_hit_ratio", "severity": "info", "ratio": 0.03},
        ],
    }
    store.append(bloated)

    report = render_token_efficiency_report(store=store, limit=10)

    assert "Token Efficiency Report" in report
    assert "observe-only" in report
    assert "tools_schema" in report
    assert "tool_results" in report
    assert "Cache hit ratio" in report
    assert "narrower toolsets" in report
    assert "prune or summarize old tool outputs" in report
    assert "preserve stable prefixes" in report
    assert "raw" not in report.lower()


def test_render_token_efficiency_report_handles_empty_store(tmp_path):
    store = TokenEfficiencyStore(tmp_path / "missing.jsonl")

    report = render_token_efficiency_report(store=store)

    assert "Token Efficiency Report" in report
    assert "No token efficiency records yet" in report
    assert "Run a model call" in report


def test_no_usage_record_preserves_attribution_and_reports_blind_spot():
    record = build_token_efficiency_record(
        session_id="s1",
        turn_id="t1",
        api_request_id="r-no-usage",
        platform="cli",
        provider="custom",
        model="no-usage-model",
        api_mode="chat_completions",
        messages=[{"role": "system", "content": "secret prefix"}, {"role": "user", "content": "hello"}],
        tools=[],
        rough_request_tokens=321,
    )

    finalized = finalize_token_efficiency_no_usage(record, reason="provider_missing_usage")

    assert finalized["status"] == "no_usage"
    assert finalized["reason"] == "provider_missing_usage"
    assert finalized["actual"]["input_tokens"] == 0
    assert any(item["kind"] == "missing_usage" for item in finalized["diagnostics"])
    assert "secret prefix" not in json.dumps(finalized)


def test_error_record_sanitizes_exception_and_tracks_retry_without_prompt_text():
    record = build_token_efficiency_record(
        session_id="s1",
        turn_id="t1",
        api_request_id="r-error",
        platform="cli",
        provider="openrouter",
        model="model-a",
        api_mode="chat_completions",
        messages=[{"role": "user", "content": "sensitive user text"}],
        tools=[],
        rough_request_tokens=111,
    )

    finalized = finalize_token_efficiency_error(
        record,
        error=RuntimeError("HTTP 429: secret request payload should not be stored"),
        retry_count=2,
        will_retry=True,
    )

    assert finalized["status"] == "error"
    assert finalized["error"]["type"] == "RuntimeError"
    assert finalized["error"]["retry_count"] == 2
    assert finalized["error"]["will_retry"] is True
    assert any(item["kind"] == "retry_overhead" for item in finalized["diagnostics"])
    serialized = json.dumps(finalized)
    assert "sensitive user text" not in serialized
    assert "secret request payload" not in serialized


def test_compression_efficiency_event_reports_roi_without_summary_text():
    event = build_compression_efficiency_event(
        session_id="s1",
        turn_id="t-compress",
        before_tokens=100_000,
        after_tokens=24_000,
        trigger="auto_threshold",
        summary_text="compressed secret summary should not be stored",
    )

    assert event["schema"] == "hermes.token_efficiency.compression.v1"
    assert event["status"] == "compression"
    assert event["before_tokens"] == 100_000
    assert event["after_tokens"] == 24_000
    assert event["saved_tokens"] == 76_000
    assert event["roi_ratio"] == 0.76
    assert event["summary_hash"]
    assert "compressed secret summary" not in json.dumps(event)


def test_report_surfaces_no_usage_error_and_compression_blind_spots(tmp_path):
    store = TokenEfficiencyStore(tmp_path / "telemetry.jsonl")
    base = build_token_efficiency_record(
        session_id="s1",
        turn_id="t1",
        api_request_id="r1",
        platform="cli",
        provider="custom",
        model="model-a",
        api_mode="chat_completions",
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
        rough_request_tokens=1000,
    )
    store.append(finalize_token_efficiency_no_usage(base, reason="provider_missing_usage"))
    store.append(finalize_token_efficiency_error(base, error=TimeoutError("timeout with private payload"), retry_count=1, will_retry=False))
    store.append(build_compression_efficiency_event(session_id="s1", turn_id="tc", before_tokens=50_000, after_tokens=20_000))

    report = render_token_efficiency_report(store=store, session_id="s1")

    assert "No-usage records:     1" in report
    assert "Error records:        1" in report
    assert "Compression events:   1" in report
    assert "provider missing usage" in report
    assert "retry/fallback overhead" in report
    assert "compression saved" in report
    assert "private payload" not in report


def test_optimizer_preview_proposes_safe_actions_without_mutation_or_raw_prompt_text():
    records = [
        {
            "schema": "hermes.token_efficiency.v1",
            "session_id": "s1",
            "api_request_id": "r-hotspot",
            "rough_request_tokens": 120_000,
            "blocks": [
                {"kind": "tools_schema", "rough_tokens": 50_000, "count": 80},
                {"kind": "tool_results", "rough_tokens": 32_000},
                {"kind": "history_recent", "rough_tokens": 36_000},
                {"kind": "system", "rough_tokens": 2_000, "stable_hash": "abc"},
            ],
            "actual": {"input_tokens": 100_000, "output_tokens": 2_000, "cache_read_tokens": 1_000, "cache_write_tokens": 11_000},
            "cache": {"hit_ratio": 0.08, "provider_reported": True},
            "diagnostics": [
                {"kind": "tool_schema_overhead", "severity": "warn", "ratio": 0.42},
                {"kind": "tool_result_bloat", "severity": "warn", "ratio": 0.27},
                {"kind": "low_cache_hit_ratio", "severity": "info", "ratio": 0.08},
            ],
        },
        {
            "schema": "hermes.token_efficiency.compression.v1",
            "status": "compression",
            "before_tokens": 90_000,
            "after_tokens": 30_000,
            "saved_tokens": 60_000,
            "roi_ratio": 0.66,
        },
    ]

    preview = build_token_efficiency_optimizer_preview(records)

    assert preview["ok"] is True
    assert preview["mode"] == "preview_only"
    assert preview["mutation_available"] is False
    assert preview["requires_approval"] is True
    assert preview["auto_applied"] is False
    assert preview["guardrail"]["no_request_mutation"] is True
    proposal_keys = {proposal["key"] for proposal in preview["proposals"]}
    assert {"narrow_toolsets", "compress_now", "prune_tool_results", "stabilize_prefix", "lazy_retrieve_history"} <= proposal_keys
    for proposal in preview["proposals"]:
        assert proposal["mutation_available"] is False
        assert proposal["requires_approval"] is True
        assert proposal["auto_applied"] is False
        assert proposal["estimated_savings_tokens"] >= 0
        assert proposal["confidence"] in {"low", "medium", "high"}
    serialized = json.dumps(preview)
    assert "raw" not in serialized.lower()
    assert "secret" not in serialized.lower()


def test_optimizer_preview_empty_records_is_safe_and_inert():
    preview = build_token_efficiency_optimizer_preview([])

    assert preview["ok"] is True
    assert preview["mode"] == "preview_only"
    assert preview["proposals"] == []
    assert preview["mutation_available"] is False
    assert preview["guardrail"]["no_auto_apply"] is True


def test_optimizer_preview_action_specs_are_function_first_and_inspectable():
    records = [
        {
            "schema": "hermes.token_efficiency.v1",
            "session_id": "s1",
            "api_request_id": "r-hotspot",
            "rough_request_tokens": 140_000,
            "blocks": [
                {"kind": "tools_schema", "rough_tokens": 56_000, "count": 90},
                {"kind": "tool_results", "rough_tokens": 30_000},
                {"kind": "history_recent", "rough_tokens": 44_000},
                {"kind": "system", "rough_tokens": 3_000, "stable_hash": "abc"},
            ],
            "actual": {"input_tokens": 120_000, "output_tokens": 3_000, "cache_read_tokens": 1_000, "cache_write_tokens": 19_000},
            "cache": {"hit_ratio": 0.05, "provider_reported": True},
            "diagnostics": [
                {"kind": "tool_schema_overhead", "severity": "warn", "ratio": 0.40},
                {"kind": "tool_result_bloat", "severity": "warn", "ratio": 0.21},
                {"kind": "low_cache_hit_ratio", "severity": "info", "ratio": 0.05},
            ],
        },
    ]

    preview = build_token_efficiency_optimizer_preview(records)
    proposals = {proposal["key"]: proposal for proposal in preview["proposals"]}

    assert preview["guardrail"]["function_first"] is True
    assert preview["guardrail"]["no_capability_degradation_without_review"] is True
    assert preview["guardrail"]["avoid_over_engineering"] is True
    assert set(proposals) >= {"narrow_toolsets", "compress_now", "prune_tool_results", "stabilize_prefix", "lazy_retrieve_history"}
    for key, proposal in proposals.items():
        spec = proposal["action_spec"]
        assert spec["key"] == key
        assert spec["kind"] in {"toolset_scope", "compression", "tool_result_policy", "cache_prefix", "lazy_retrieval"}
        assert spec["would_change"]
        assert spec["would_not_change"]
        assert spec["capability_risk"] in {"low", "medium", "high"}
        assert spec["requires_human_review"] is True
        assert spec["apply_endpoint"] is None
        assert spec["auto_apply_allowed"] is False
        assert spec["fallback"]
    narrow = proposals["narrow_toolsets"]["action_spec"]
    assert "lost_capabilities_to_review" in narrow
    assert "terminal" in narrow["preserve_capabilities"]
    compress = proposals["compress_now"]["action_spec"]
    assert compress["preserve_recent_decisions"] is True
    lazy = proposals["lazy_retrieve_history"]["action_spec"]
    assert lazy["retrieval_fallback"] == "session_search_or_full_context"


def test_optimizer_preview_visibility_collapses_when_no_hotspots():
    preview = build_token_efficiency_optimizer_preview([])

    visibility = preview["visibility"]
    assert visibility["needs_attention"] is False
    assert visibility["recommended_surface"] == "diagnostics_collapsed"
    assert visibility["cockpit_card"] is None
    assert visibility["reason"] == "no_actionable_token_hotspots"
    assert preview["guardrail"]["function_first"] is True


def test_optimizer_preview_visibility_surfaces_only_actionable_hotspots():
    records = [
        {
            "schema": "hermes.token_efficiency.v1",
            "session_id": "s1",
            "api_request_id": "r-hotspot",
            "rough_request_tokens": 180_000,
            "blocks": [
                {"kind": "tools_schema", "rough_tokens": 75_000, "count": 100},
                {"kind": "tool_results", "rough_tokens": 40_000},
                {"kind": "history_recent", "rough_tokens": 52_000},
            ],
            "actual": {"input_tokens": 150_000, "output_tokens": 4_000, "cache_read_tokens": 1_000, "cache_write_tokens": 25_000},
            "cache": {"hit_ratio": 0.03, "provider_reported": True},
            "diagnostics": [
                {"kind": "tool_schema_overhead", "severity": "warn", "ratio": 0.42},
                {"kind": "tool_result_bloat", "severity": "warn", "ratio": 0.22},
                {"kind": "low_cache_hit_ratio", "severity": "info", "ratio": 0.03},
            ],
        }
    ]

    preview = build_token_efficiency_optimizer_preview(records)
    visibility = preview["visibility"]

    assert visibility["needs_attention"] is True
    assert visibility["recommended_surface"] == "mission_control_review"
    assert visibility["reason"] == "actionable_token_hotspots"
    card = visibility["cockpit_card"]
    assert card["title"] == "Token efficiency hotspot"
    assert card["autonomy_gate"] == "review_only_no_auto_apply"
    assert card["recommended_action"] == "Review optimizer preview; do not apply automatically."
    assert card["proposal_count"] >= 3
    assert card["estimated_savings_tokens"] > 0
    assert card["function_first"] is True
