"""Adapter-specific classification: metadata, nested controls, opaque/media."""
import pytest
from agent.final_wire_admission import (
    COVERED_MAIN, FinalAttemptIdentity, ProviderBoundUnsupportedAccounting,
    admit_final_json, project_final_body,
)


def test_feedback_structured_constraints_accept_schema_type_object():
    from agent.final_wire_admission import estimate_final_body_pressure
    base = {"messages": [{"role": "user", "content": "hi"}]}
    assert estimate_final_body_pressure({**base, "response_format": {"type": "json_schema", "json_schema": {"name": "inert", "schema": {"type": "object", "properties": {"value": {"type": "string", "description": "x" * 4400}}}}}}) > estimate_final_body_pressure(base) + 1000


def identity(family):
    return FinalAttemptIdentity(COVERED_MAIN, family, "inert", "https://inert.invalid", 100000, "classification")


@pytest.mark.parametrize("family,body", [
    ("chat_completions", {"messages": [{"role": "user", "content": "hi"}], "anthropic_version": "context?"}),
    ("gemini_native", {"contents": [{"parts": [{"text": "hi"}]}], "generationConfig": {"maxOutputTokens": 1, "newContext": "x" * 4400}}),
    ("chat_completions", {"messages": "unclassified shape"}),
    ("anthropic_messages", {"messages": {}, "max_tokens": 1}),
])
def test_valid_json_unknown_family_or_nested_context_shapes_refuse(family, body):
    with pytest.raises(ProviderBoundUnsupportedAccounting):
        admit_final_json(body, identity(family))


@pytest.mark.parametrize("family,base,metadata", [
    ("chat_completions", {"messages": [{"role": "user", "content": "hi"}]}, {"metadata": {"trace": "x" * 4400}, "prompt_cache_key": "cache", "temperature": .2, "stream": True}),
    ("codex_responses", {"input": "hi"}, {"include": ["reasoning.encrypted_content"], "store": False, "service_tier": "default"}),
    ("anthropic_bedrock", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}, {"anthropic_version": "bedrock-2023-05-31", "anthropic_beta": ["inert"], "stream": True}),
    ("bedrock_converse", {"messages": [{"role": "user", "content": [{"text": "hi"}]}]}, {"guardrailConfig": {"guardrailIdentifier": "x" * 4400}, "performanceConfig": {"latency": "standard"}, "requestMetadata": {"trace": "inert"}}),
    ("gemini_native", {"contents": [{"parts": [{"text": "hi"}]}], "generationConfig": {"maxOutputTokens": 1}}, {"safetySettings": [{"category": "inert", "threshold": "inert"}]}),
])
def test_exact_family_metadata_excluded_with_reasons(family, base, metadata):
    first = project_final_body(base, identity(family))
    second = project_final_body({**base, **metadata}, identity(family))
    assert second.coverage == first.coverage == "COMPLETE_LOCAL"
    assert second.estimated_input == first.estimated_input
    assert second.excluded_metadata


@pytest.mark.parametrize("family,body", [
    ("codex_responses", {"input": "hi", "previous_response_id": "opaque-id"}),
    ("gemini_native", {"contents": [{"parts": [{"text": "hi"}]}], "cachedContent": "cached/opaque", "generationConfig": {"maxOutputTokens": 1}}),
])
def test_provider_held_reference_visible_cost_and_opaque_not_complete_local(family, body):
    snap = project_final_body(body, identity(family))
    assert snap.coverage == "OPAQUE"
    assert snap.coverage_reason
    assert snap.estimated_input > project_final_body({k: v for k, v in body.items() if k not in {"previous_response_id", "cachedContent"}}, identity(family)).estimated_input


@pytest.mark.parametrize("family,body", [
    ("chat_completions", {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}),
    ("anthropic_messages", {"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}], "max_tokens": 1}),
    ("bedrock_converse", {"messages": [{"role": "user", "content": [{"image": {"format": "png", "source": {"bytes": "AAAA"}}}]}]}),
    ("gemini_native", {"contents": [{"parts": [{"inlineData": {"mimeType": "image/png", "data": "AAAA"}}]}], "generationConfig": {"maxOutputTokens": 1}}),
])
def test_recognized_native_image_keeps_heuristic_and_estimated_coverage(family, body):
    snap = project_final_body(body, identity(family))
    assert snap.coverage == "ESTIMATED_MULTIMODAL"
    assert snap.estimated_input >= 1500
