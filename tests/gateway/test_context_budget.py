from __future__ import annotations

from gateway.context_budget import (
    ContextBudget,
    build_budget_proof,
    build_policy_static_prefix,
    compact_tool_response,
    compile_context_budget,
    make_context_item,
)


def test_conversation_direct_can_prove_zero_tool_schema_budget() -> None:
    proof = build_budget_proof(
        profile_name="conversation_direct",
        tool_schema_tokens_est=0,
        answerability={"state": "answerable", "max_claim_strength": "memory_truth"},
        answer_evidence=[{"id": "profile:debug_marker", "value": "1231231X"}],
        forbidden_claims=["external_verification"],
        current_user_message="Mi volt a debug markerem?",
    )

    assert proof["tool_schema_budget_zero"] is True
    assert proof["protected_preserved"] is True
    assert proof["assembly"]["provider_cache_observable"] is False


def test_overflow_drops_supporting_context_before_answer_evidence() -> None:
    budget = ContextBudget(
        total_input_budget_tokens=200,
        static_system_budget_tokens=20,
        tool_schema_budget_tokens=0,
        memory_packet_budget_tokens=50,
        recent_history_budget_tokens=50,
        safety_margin_tokens=20,
    )
    static_prefix = "stable"
    items = [
        make_context_item("answerability", "answerability", '{"state":"answerable"}', tokens_est=10),
        make_context_item("answer_evidence", "answer_evidence", "debug marker 1231231X", tokens_est=10),
        make_context_item("forbidden", "forbidden_claims", "do not guess", tokens_est=10),
        make_context_item("user", "current_user_message", "Mi volt?", tokens_est=10),
        make_context_item("support", "supporting_context", "noise " * 1000, tokens_est=400),
        make_context_item("history", "recent_history", "old chat " * 1000, tokens_est=400),
    ]

    assembly = compile_context_budget(
        profile_name="conversation_direct",
        budget=budget,
        static_prefix=static_prefix,
        tool_schema_tokens_est=0,
        items=items,
    )
    selected = {item.item_id for item in assembly.selected_items}
    dropped = {item.item_id for item in assembly.dropped_items}

    assert {"answerability", "answer_evidence", "forbidden", "user"}.issubset(selected)
    assert {"support", "history"}.issubset(dropped)
    assert assembly.overflow is True


def test_minimum_viable_context_preserves_protected_when_budget_tiny() -> None:
    budget = ContextBudget(
        total_input_budget_tokens=20,
        static_system_budget_tokens=10,
        tool_schema_budget_tokens=0,
        memory_packet_budget_tokens=5,
        recent_history_budget_tokens=0,
        safety_margin_tokens=5,
    )
    items = [
        make_context_item("answerability", "answerability", '{"state":"answerable"}', tokens_est=10),
        make_context_item("answer_evidence", "answer_evidence", "1231231X", tokens_est=10),
        make_context_item("support", "supporting_context", "drop me", tokens_est=10),
    ]

    assembly = compile_context_budget(
        profile_name="conversation_direct",
        budget=budget,
        static_prefix="stable static prefix",
        tool_schema_tokens_est=0,
        items=items,
    )

    assert assembly.minimum_viable_context_used is True
    assert {item.item_id for item in assembly.selected_items} == {"answerability", "answer_evidence"}
    assert {item.item_id for item in assembly.dropped_items} == {"support"}


def test_static_prefix_hash_stable_when_dynamic_suffix_changes() -> None:
    prefix = build_policy_static_prefix(
        profile_name="conversation_direct",
        tool_profile="conversation_direct",
        max_claim_strength="memory_truth",
        latency_mode="conversation_warm",
        forbidden_claims=["external_verification"],
    )
    budget = ContextBudget(1000, 100, 0, 100, 100, 50)

    first = compile_context_budget(
        profile_name="conversation_direct",
        budget=budget,
        static_prefix=prefix,
        tool_schema_tokens_est=0,
        items=[make_context_item("user", "current_user_message", "first")],
    )
    second = compile_context_budget(
        profile_name="conversation_direct",
        budget=budget,
        static_prefix=prefix,
        tool_schema_tokens_est=0,
        items=[make_context_item("user", "current_user_message", "second")],
        previous_static_prefix_hash=first.static_prefix_hash,
    )
    changed = compile_context_budget(
        profile_name="conversation_direct",
        budget=budget,
        static_prefix=prefix + " changed",
        tool_schema_tokens_est=0,
        items=[make_context_item("user", "current_user_message", "second")],
        previous_static_prefix_hash=first.static_prefix_hash,
    )

    assert first.static_prefix_hash == second.static_prefix_hash
    assert first.dynamic_suffix_hash != second.dynamic_suffix_hash
    assert second.cache_break_reason is None
    assert changed.cache_break_reason == "STATIC_PREFIX_HASH_CHANGED_REQUIRES_PROFILE_VERSION_BUMP"


def test_tool_response_model_facing_cap_preserves_answerability_and_inspect_ref() -> None:
    payload = {
        "memory_answerability": {"state": "answerable", "max_claim_strength": "memory_truth"},
        "answer_evidence": [{"id": "profile:debug_marker", "value": "1231231X"}],
        "supporting_context": ["noise " * 1000],
        "forbidden_claims": ["external_verification"],
        "diagnostics": ["full candidate trace should not be model-facing"],
    }

    envelope = compact_tool_response(
        "brainstack_recall",
        payload,
        cap_tokens=80,
        inspect_ref="inspect://trace-1",
        continuation_token="cont-1",
    )

    assert envelope.truncated is True
    assert envelope.model_facing["answerability"]["state"] == "answerable"
    assert envelope.model_facing["answer_evidence"][0]["value"] == "1231231X"
    assert envelope.model_facing["supporting_context"] == []
    assert envelope.model_facing["inspect_ref"] == "inspect://trace-1"


def test_secret_and_private_path_are_redacted_from_model_facing_items() -> None:
    item = make_context_item(
        "diagnostic",
        "diagnostics",
        "secret sk-abcdef1234567890 path /home/lauratom/secret/file.txt",
    )

    assert "sk-abcdef" not in item.text
    assert "/home/lauratom" not in item.text
    assert "[REDACTED_SECRET_SHAPED]" in item.text
    assert "[REDACTED_PRIVATE_PATH]" in item.text
