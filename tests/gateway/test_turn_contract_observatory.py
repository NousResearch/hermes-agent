from gateway.turn_contract import (
    StructuralSignals,
    build_hidden_tool_prompt_audit,
    compile_turn_contract,
    sample_size_latency_summary,
)


def test_turn_contract_is_observe_only_and_profile_local():
    contract = compile_turn_contract(
        platform="discord",
        prompt="Mi volt a markerem?",
        tool_profile="memory-only",
        model_profile="kimi",
        brainstack_sufficiency={
            "state": "answerable",
            "max_claim_strength": "memory_truth",
            "answer_type": "explicit_user_fact",
            "answer_evidence_count": 1,
            "requires_external_tools_for_memory_answer": False,
        },
        idempotency_parts=("discord", "guild", "channel", "message", "user"),
        causal_index=3,
    )

    data = contract.to_dict()

    assert data["schema"] == "hermes.turn_contract.v1"
    assert data["observe_only"] is True
    assert data["allowed_tool_profile"] == "conversation_tools"
    assert data["reason_code"] == "MEMORY_SUFFICIENT_NO_EXTERNAL_TOOLS"
    assert data["causal_index"] == 3
    assert data["idempotency_key_hash"].startswith("sha256:")


def test_structural_heavy_signal_not_keyword_router():
    signals = StructuralSignals.from_prompt("/heavy web https://example.com")
    contract = compile_turn_contract(
        platform="discord",
        prompt="/heavy web https://example.com",
        tool_profile="full",
        structural_signals=signals,
    )

    assert contract.reason_code == "HEAVY_COMMAND"
    assert contract.turn_class == "external.heavy_requested"
    assert contract.structural_signals["url_count"] == 1


def test_no_tools_hidden_prompt_audit_detects_residue():
    audit = build_hidden_tool_prompt_audit(
        tool_profile="no-tools",
        tool_names=["web_search"],
        system_prompt="You can use tools.",
    )

    assert audit.hidden_tool_prompt_detected is True
    assert audit.tools_array_empty is False
    assert audit.tool_use_system_prompt_present is True


def test_small_sample_summary_never_claims_p95():
    summary = sample_size_latency_summary([1.0, 2.0, 3.0])

    assert summary["sample_size"] == 3
    assert summary["p95_seconds"] is None
    assert summary["p95_claim_allowed"] is False
    assert "30" in summary["small_sample_rule"]


def test_large_sample_summary_can_claim_p95():
    summary = sample_size_latency_summary([float(i) for i in range(30)])

    assert summary["sample_size"] == 30
    assert summary["p95_seconds"] is not None
    assert summary["p95_claim_allowed"] is True
