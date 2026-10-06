"""Acceptance contracts for opt-in, direct xAI Responses compaction.

Exercise the real routing, conversion and compression entrypoints; only the
provider client and local summary generation are replaced with hermetic fakes.
"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import native_compaction
from agent.chat_completion_helpers import _build_codex_kwargs
from agent.codex_responses_adapter import (
    _native_responses_replay_items,
    classify_responses_route,
    estimate_native_responses_preflight_tokens,
    has_replayable_native_compaction_checkpoint,
)
from agent.context_compressor import ContextCompressor
from agent.conversation_compression import CompressionCommitFence, compress_context
from agent.transports.codex import ResponsesApiTransport


def _agent(**overrides):
    transport = ResponsesApiTransport()
    agent = SimpleNamespace(
        provider="xai", base_url="https://api.x.ai/v1", model="grok-4.7",
        api_mode="codex_responses", runtime_capabilities={"native_compaction": True},
        capabilities={}, codex_responses_native_compaction=True,
        codex_responses_compact_threshold=40_000, compression_enabled=True,
        compression_checkpoint_required=False, session_id="xai-compaction-test",
        max_tokens=1024, request_overrides={},
        context_compressor=SimpleNamespace(threshold_tokens=50_000),
        _get_transport=lambda: transport,
        _prepare_messages_for_non_vision_model=lambda messages: messages,
        _resolved_api_call_timeout=lambda: 37.0,
    )
    vars(agent).update(overrides)
    return agent


def _request(agent, messages):
    return _build_codex_kwargs(
        agent, messages, [], None, agent.request_overrides, agent.session_id,
    )


def _checkpoint(blob="old-checkpoint"):
    return {
        "type": "compaction", "id": "cmp_old", "encrypted_content": blob,
        "_issuer_kind": "xai_responses", "_issuer_model": "grok-4.7",
    }


def _transcript():
    return [
        {"role": "user", "content": "Remember the launch code: cedar."},
        {"role": "assistant", "content": "Old explanation. " * 2000},
        {"role": "user", "content": "Keep that fact for later."},
        {"role": "assistant", "content": "Checkpoint carrier.",
         "codex_reasoning_items": [_checkpoint()]},
        {"role": "user", "content": "What is the launch code?"},
        {"role": "assistant", "content": "The code is cedar."},
        {"role": "user", "content": "This new turn must stay outside compact input."},
    ]


@pytest.mark.parametrize("base_url", ["https://api.x.ai/v1", "", None])
@pytest.mark.parametrize("model", ["grok-4.7", "xai/GROK_4.7", "grok-4.7-fast"])
def test_direct_xai_grok_47_family_advertises_native_compaction(base_url, model):
    assert native_compaction.resolve_native_compaction_capabilities(
        provider="xai", base_url=base_url, model=model,
    )["native_compaction"] is True


@pytest.mark.parametrize("provider,base_url,model", [
    ("xai-oauth", "https://api.x.ai/v1", "grok-4.7"),
    ("xai", "https://relay.example/v1", "grok-4.7"),
    ("xai", "https://api.x.ai.evil.example/v1", "grok-4.7"),
    ("xai", "https://sub.api.x.ai/v1", "grok-4.7"),
    ("openrouter", "https://openrouter.ai/api/v1", "xai/grok-4.7"),
    ("xai", "https://api.x.ai/v1", "grok-4.6"),
    ("xai", "https://api.x.ai/v1", "grok-4.70"),
])
def test_other_routes_and_models_cannot_advertise_xai_compaction(provider, base_url, model):
    assert native_compaction.resolve_native_compaction_capabilities(
        provider=provider, base_url=base_url, model=model,
    )["native_compaction"] is False


@pytest.mark.parametrize("overrides,expected", [
    ({}, True),
    ({"codex_responses_native_compaction": False}, False),
    ({"compression_enabled": False}, False),
    ({"compression_checkpoint_required": True}, False),
    ({"runtime_capabilities": {"native_compaction": False}}, False),
    ({"model": "grok-4.6"}, False),
])
def test_xai_eligibility_obeys_existing_switches_without_context_management(overrides, expected):
    agent = _agent(**overrides)
    route = classify_responses_route(agent)._asdict()
    assert native_compaction.native_compaction_eligible(agent, **route) is expected
    assert native_compaction.native_compaction_context_management(agent, **route) is None
    assert "context_management" not in _request(agent, _transcript())


@pytest.mark.parametrize("entrypoint", ["request", "build_kwargs", "convert_messages"])
def test_xai_replay_prunes_history_and_strips_checkpoint_provenance(entrypoint):
    agent = _agent()
    messages = _transcript()
    before = deepcopy(messages)
    params = dict(
        model=agent.model, **classify_responses_route(agent)._asdict(),
        base_url=agent.base_url, native_compaction_eligible=True,
    )
    transport = agent._get_transport()
    if entrypoint == "request":
        wire = _request(agent, messages)
        assert "context_management" not in wire
        items = wire["input"]
    elif entrypoint == "build_kwargs":
        wire = transport.build_kwargs(messages=messages, **params)
        assert "context_management" not in wire
        items = wire["input"]
    else:
        items = transport.convert_messages(messages, **params)

    assert items[0] == {"type": "compaction", "encrypted_content": "old-checkpoint"}
    assert not any(item.get("content") == messages[1]["content"] for item in items)
    # Existing pruning retains plaintext user asks after the checkpoint.
    assert [item["content"] for item in items if item.get("role") == "user"] == [
        message["content"] for message in messages if message["role"] == "user"
    ]
    assert any(item.get("content") == messages[5]["content"] for item in items)
    assert messages == before

    openai = _agent(provider="openai", base_url="https://api.openai.com/v1", model="gpt-5.6")
    foreign_items = _request(openai, messages)["input"]
    assert not any(item.get("type") == "compaction" for item in foreign_items)
    assert any(item.get("content") == messages[1]["content"] for item in foreign_items)
    assert not has_replayable_native_compaction_checkpoint(openai, messages)


def test_xai_preflight_counts_the_same_pruned_payload_as_the_request():
    from agent.model_metadata import estimate_request_tokens_rough

    agent = _agent()
    messages = _transcript()
    assert has_replayable_native_compaction_checkpoint(agent, messages)
    items = _native_responses_replay_items(agent, messages)
    assert items == _request(agent, messages)["input"]
    estimate = estimate_native_responses_preflight_tokens(agent, messages, system_prompt="Stable prompt")
    assert estimate == estimate_request_tokens_rough(items, system_prompt="Stable prompt", tools=None)
    without_checkpoint = deepcopy(messages)
    del without_checkpoint[3]["codex_reasoning_items"]
    assert estimate < estimate_native_responses_preflight_tokens(
        agent, without_checkpoint, system_prompt="Stable prompt",
    )


@pytest.fixture
def compression_agent(monkeypatch):
    compressor = ContextCompressor(
        model="grok-4.7", provider="xai", api_mode="codex_responses",
        config_context_length=100_000, quiet_mode=True,
    )
    # A no-op local engine is sufficient to observe dispatch without doing I/O.
    monkeypatch.setattr(compressor, "compress", Mock(side_effect=lambda messages, **kw: deepcopy(messages)))
    monkeypatch.setattr(compressor, "note_native_compaction_checkpoint", Mock(
        wraps=compressor.note_native_compaction_checkpoint,
    ))
    response = SimpleNamespace(
        object="response.compaction",
        output=[SimpleNamespace(type="compaction", id="cmp_new", encrypted_content="new-checkpoint")],
        usage=SimpleNamespace(input_tokens=9000, output_tokens=500, dropped_message_count=4),
    )
    client = SimpleNamespace(responses=SimpleNamespace(compact=Mock(return_value=response)))
    return _agent(
        context_compressor=compressor, client=client,
        _create_request_openai_client=Mock(return_value=client),
        _cached_system_prompt="Stable system prompt\nwith exact whitespace.\n",
        _build_system_prompt=Mock(side_effect=AssertionError("native compaction must keep the cached prompt")),
        _compression_feasibility_checked=True, _current_task_id="xai-task",
        _session_db=None, _emit_status=Mock(), _emit_warning=Mock(),
        _usage_anchor={"stale": True}, _turn_base_usage_anchor={"stale": True},
    )


@pytest.mark.parametrize("blocked_local_engine", [False, True])
def test_successful_compact_keeps_transcript_and_commits_checkpoint_side_effects(
    compression_agent, monkeypatch, blocked_local_engine,
):
    from tools import file_tools_read_tracking

    agent = compression_agent
    cc = agent.context_compressor
    if blocked_local_engine:
        cc._record_structural_no_op("local summary cannot shorten this transcript")
    gate = Mock(wraps=cc._automatic_compression_blocked)
    monkeypatch.setattr(cc, "_automatic_compression_blocked", gate)
    no_progress = Mock(wraps=cc._record_structural_no_op)
    monkeypatch.setattr(cc, "_record_structural_no_op", no_progress)
    reads = {
        "xai-task": {"dedup_generation_reads": {"file"}, "dedup_hits": {"file": 2}},
        "other-task": {"dedup_generation_reads": {"other"}, "dedup_hits": {"other": 1}},
    }
    monkeypatch.setattr(file_tools_read_tracking, "_read_tracker", reads)
    messages = _transcript()
    messages[5]["codex_reasoning_items"] = [{
        "type": "reasoning", "id": "rs_latest", "encrypted_content": "latest-reasoning",
        "summary": [], "_issuer_kind": "xai_responses", "_issuer_model": "grok-4.7",
    }]
    before = deepcopy(messages)
    previous_sidecar = messages[5]["codex_reasoning_items"]
    # Use the real request builder as the compact-input contract, before mutation.
    expected_input = _request(agent, messages[:6])["input"]
    fence = CompressionCommitFence()
    observed_fence = []
    compact = agent.client.responses.compact
    response = compact.return_value

    def respond(**kwargs):
        observed_fence.append(fence.commit_in_flight)
        return response

    compact.side_effect = respond
    returned, prompt = compress_context(
        agent, messages, "uncached fallback prompt", approx_tokens=80_000,
        task_id="xai-task", commit_fence=fence,
    )

    compact.assert_called_once()
    call = compact.call_args.kwargs
    assert call["model"] == "grok-4.7"
    assert call["input"] == expected_input
    assert call["instructions"] == agent._cached_system_prompt
    assert call["timeout"] == agent._resolved_api_call_timeout()
    assert observed_fence == [True]
    assert not fence.commit_in_flight
    assert returned is messages and len(returned) == len(before)
    assert prompt == agent._cached_system_prompt
    sidecar = messages[5]["codex_reasoning_items"]
    assert sidecar is not previous_sidecar
    assert sidecar[:-1] == before[5]["codex_reasoning_items"] == previous_sidecar
    assert sidecar[-1] == {
        "type": "compaction", "encrypted_content": "new-checkpoint",
        "_issuer_kind": "xai_responses", "_issuer_model": "grok-4.7",
    }
    expected_transcript = deepcopy(before)
    expected_transcript[5]["codex_reasoning_items"] = sidecar
    assert returned == expected_transcript  # no injected turn or truncated history
    cc.note_native_compaction_checkpoint.assert_called_once_with()
    assert cc.awaiting_real_usage_after_compression
    assert agent._usage_anchor is None and agent._turn_base_usage_anchor is None
    assert reads["xai-task"]["dedup_generation_reads"] == set()
    assert reads["xai-task"]["dedup_hits"] == {}
    assert reads["other-task"]["dedup_generation_reads"] == {"other"}
    cc.compress.assert_not_called()
    gate.assert_not_called()
    no_progress.assert_not_called()
    assert _request(agent, messages)["input"][0]["encrypted_content"] == "new-checkpoint"


@pytest.mark.parametrize("failure", ["http_error", "missing_item", "missing_client", "no_assistant"])
def test_failed_native_compaction_falls_back_to_local_compression(compression_agent, failure):
    import httpx
    from openai import InternalServerError

    agent = compression_agent
    compact = agent.client.responses.compact
    messages = _transcript()
    if failure == "http_error":
        compact.side_effect = InternalServerError(
            "compact failed", body=None,
            response=httpx.Response(503, request=httpx.Request("POST", "https://api.x.ai/v1/responses/compact")),
        )
    elif failure == "missing_item":
        compact.return_value.output = [SimpleNamespace(type="message", content=[])]
    elif failure == "missing_client":
        agent.client = None
        agent._create_request_openai_client.return_value = None
    else:
        messages = [{"role": "user", "content": "No assistant boundary yet."}]
    before = deepcopy(messages)
    fence = CompressionCommitFence()

    returned, prompt = compress_context(
        agent, messages, agent._cached_system_prompt, approx_tokens=80_000, commit_fence=fence,
    )

    if failure in {"http_error", "missing_item"}:
        compact.assert_called_once()
    else:
        compact.assert_not_called()
    agent.context_compressor.compress.assert_called_once()
    assert agent.context_compressor.compress.call_args.args[0] == before
    agent.context_compressor.note_native_compaction_checkpoint.assert_not_called()
    assert messages == before and returned == before
    assert prompt == agent._cached_system_prompt
    assert not fence.commit_in_flight


def test_required_memory_checkpoint_uses_local_path_without_calling_xai(compression_agent):
    agent = compression_agent
    agent.compression_checkpoint_required = True
    agent._memory_manager = SimpleNamespace(
        supports_pre_compress_checkpoint=Mock(return_value=True),
        on_pre_compress=Mock(return_value="Durably saved evidence"),
    )
    messages = _transcript()
    before = deepcopy(messages)

    compress_context(agent, messages, agent._cached_system_prompt, approx_tokens=80_000)

    agent.client.responses.compact.assert_not_called()
    agent.context_compressor.compress.assert_called_once()
    assert agent._memory_manager.on_pre_compress.call_args.kwargs["require_checkpoint"] is True
    assert messages == before
