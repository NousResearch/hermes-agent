from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest

from agent.llm_egress_firewall import EgressBlocked, SanitizedTextRejected
from agent.llm_egress_runtime import (
    _restore_source_provenance_sidecar,
    _typed_payload_violation_locations,
    authorize_agent_sdk_kwargs,
    dispatch_authorized_agent_request,
)
from agent.source_provenance import SourceProvenanceRegistry


def _agent(tmp_path: Path, registry: SourceProvenanceRegistry | None = None):
    return SimpleNamespace(
        provider="custom",
        model="test-model",
        base_url="https://llm.example.test/v1",
        api_mode="chat_completions",
        session_id="session-1",
        _current_turn_id="turn-1",
        _current_api_request_id="req-1",
        _llm_egress_policy_digest=sha256(b"policy-1").hexdigest(),
        _llm_egress_state_dir=tmp_path,
        _source_provenance_registry=registry or SourceProvenanceRegistry(),
    )


def _grant(tmp_path: Path, registry: SourceProvenanceRegistry):
    path = tmp_path / "source.py"
    content = b"verified source\n"
    path.write_bytes(content)
    return registry.issue_file_slice(
        path=path,
        line_start=1,
        line_end=1,
        content=content,
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest=sha256(b"policy-1").hexdigest(),
    )


def test_typed_payload_violation_locations_are_content_free():
    """Live egress diagnostics identify the typed boundary without text leaks."""

    from agent.llm_egress_firewall import SanitizedSegment

    locations = _typed_payload_violation_locations(
        {"input": [SanitizedSegment("c2VjcmV0LXBheWxvYWQ=")]}
    )

    assert locations == (
        ("$.map[0].value.sequence[0]", "SanitizedSegment", 20, ("base64_payload",)),
    )


def test_runtime_authorizes_mixed_exact_source_and_bounded_sanitized_text(tmp_path):
    registry = SourceProvenanceRegistry()
    _grant(tmp_path, registry)
    agent = _agent(tmp_path, registry)
    kwargs = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "Fix CI now."},
            {"role": "user", "content": "CI:\nverified source\nDo fix."},
        ],
        "temperature": 0,
    }

    authorized, receipt = authorize_agent_sdk_kwargs(agent, kwargs)

    assert authorized == kwargs
    assert receipt.decision.source_grant_count == 1
    assert receipt.decision.source_segment_count == 1
    wire = json.loads(receipt.payload_bytes)
    assert wire == kwargs
    assert "session_id" not in wire
    assert "turn_id" not in wire
    assert "request_id" not in wire
    assert "policy_digest" not in wire


def test_runtime_granted_caps_default_to_the_configured_request_caps(tmp_path):
    registry = SourceProvenanceRegistry()
    path = tmp_path / "large-source.txt"
    content = b"plain source sentence\n" * 12
    path.write_bytes(content)
    registry.issue_file_slice(
        path=path,
        line_start=1,
        line_end=12,
        content=content,
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest=sha256(b"policy-1").hexdigest(),
    )
    agent = _agent(tmp_path, registry)
    agent._llm_egress_max_serialized_bytes = 128

    with pytest.raises(EgressBlocked) as exc_info:
        authorize_agent_sdk_kwargs(
            agent,
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": content.decode("utf-8")}],
            },
        )

    assert "serialized_bytes_exceeded" in exc_info.value.decision.reason_codes


def test_runtime_keeps_sdk_controls_out_of_authorized_body(tmp_path):
    agent = _agent(tmp_path)
    timeout = object()
    kwargs = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Fix CI now."}],
        "timeout": timeout,
    }
    authorized, receipt = authorize_agent_sdk_kwargs(agent, kwargs)
    assert authorized["timeout"] is timeout
    assert "timeout" not in json.loads(receipt.payload_bytes)


def test_runtime_scans_extra_headers_and_query_as_request_content(tmp_path):
    agent = _agent(tmp_path)
    calls = []
    with pytest.raises((EgressBlocked, SanitizedTextRejected)):
        dispatch_authorized_agent_request(
            agent,
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Fix CI now."}],
                "extra_headers": {
                    "Authorization": "Bearer sk-test-credential-1234567890"
                },
                "extra_query": {"trace": "safe"},
            },
            lambda request: calls.append(request),
        )
    assert calls == []


def test_runtime_verifies_authorized_payload_at_provider_boundary(
    tmp_path, monkeypatch
):
    agent = _agent(tmp_path)
    calls = []
    original = __import__(
        "agent.llm_egress_firewall", fromlist=["AuthorizedEgress"]
    ).AuthorizedEgress.verify_payload
    verified = []

    def _verify(self, candidate):
        verified.append(candidate)
        return original(self, candidate)

    monkeypatch.setattr(
        "agent.llm_egress_firewall.AuthorizedEgress.verify_payload", _verify
    )
    dispatch_authorized_agent_request(
        agent,
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Fix CI now."}],
        },
        lambda request: calls.append(request),
    )
    assert calls
    assert len(verified) == 1


@pytest.mark.parametrize(
    "text",
    [
        "SECRET_TOKEN=super-secret-value",
        "Read /Users/private/repository/file.py",
        "ZW5jb2RlZCBwcml2YXRlIGRldGFpbA==",
    ],
)
def test_runtime_denies_unsafe_text_before_provider_callback(tmp_path, text):
    agent = _agent(tmp_path)
    calls = []
    with pytest.raises((EgressBlocked, SanitizedTextRejected)):
        dispatch_authorized_agent_request(
            agent,
            {"model": "test-model", "messages": [{"role": "user", "content": text}]},
            lambda request: calls.append(request),
        )
    assert calls == []


def test_codex_generated_context_is_redacted_without_using_untrusted_budget(tmp_path):
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    context = (
        "Hermes generated instructions.\n" * 3000
        + "Workspace: /Users/private/project/file.py\n"
        + "Protocol sample: c2VjcmV0LXBheWxvYWQ=\n"
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "gpt-5.6-terra",
            "input": [{"role": "system", "content": context}],
            "tools": [{"type": "function", "description": context}],
        },
    )

    wire = json.loads(receipt.payload_bytes)
    rendered = json.dumps(wire)
    assert receipt.allowed
    assert "<private-path>" in rendered
    assert "<redacted-base64>" in rendered
    assert "/Users/private/project/file.py" not in rendered
    assert "c2VjcmV0LXBheWxvYWQ=" not in rendered
    assert len(receipt.payload_bytes) > 32_768


def test_codex_generated_context_redaction_honors_mapping_routes(tmp_path):
    agent = _agent(tmp_path)
    route = {
        "provider": "openai-codex",
        "model": "gpt-5.6-terra",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
    }
    generated = "Workspace: /Users/private/project/file.py\n" + (
        "Protocol sample: c2VjcmV0LXBheWxvYWQ=\n"
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "gpt-5.6-terra",
            "instructions": generated,
            "tools": [{
                "type": "function",
                "description": generated,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "c2VjcmV0LXBheWxvYWQ=": {"type": "string"},
                    },
                },
            }],
        },
        route=route,
    )

    rendered = json.dumps(json.loads(receipt.payload_bytes))
    assert receipt.allowed
    assert authorized["instructions"] != generated
    assert authorized["instructions"] == (
        "Workspace: <private-path>\n"
        "Protocol sample: <redacted-base64>\n"
    )
    assert "<private-path>" in rendered
    assert "<redacted-base64>" in rendered
    assert "/Users/private/project/file.py" not in rendered
    assert rendered.count("c2VjcmV0LXBheWxvYWQ=") == 1


def test_codex_generated_tool_schema_preserves_encoded_property_names(tmp_path):
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    property_name = "c2VjcmV0LXBheWxvYWQ="

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "gpt-5.6-terra",
            "tools": [{
                "type": "function",
                "parameters": {
                    "type": "object",
                    "properties": {property_name: {"type": "string"}},
                },
            }],
        },
    )

    assert receipt.allowed
    assert authorized["tools"][0]["parameters"]["properties"] == {
        property_name: {"type": "string"}
    }


def test_protected_remote_nested_tool_schema_name_mapping_is_not_a_tool_name(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "nous"
    agent.base_url = "https://inference-api.nousresearch.com/v1"
    agent.api_mode = "chat_completions"
    tools = [{
        "type": "function",
        "function": {
            "name": "tool_catalog",
            "description": "Invoke a named tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            },
        },
    }]

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {"model": "fallback-model", "messages": [], "tools": tools},
    )

    assert receipt.allowed
    assert authorized["tools"] == tools


def test_codex_generated_context_still_hard_blocks_secrets(tmp_path):
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"

    with pytest.raises(EgressBlocked) as exc_info:
        authorize_agent_sdk_kwargs(
            agent,
            {
                "model": "gpt-5.6-terra",
                "input": [
                    {"role": "system", "content": "SECRET_TOKEN=super-secret-value"}
                ],
            },
        )

    assert "secret_detected" in exc_info.value.decision.reason_codes


def test_codex_reasoning_replay_encrypted_content_is_not_treated_as_a_credential(tmp_path):
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    encrypted_replay = "gAAAAABm3gZrX7xP0Yk3V8nQ2aBcD4eFgH5jKlMnOpQrStUvWxYz0123456789"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "gpt-5.6-terra",
            "input": [
                {
                    "type": "reasoning",
                    "encrypted_content": encrypted_replay,
                    "summary": [],
                }
            ],
        },
    )

    assert receipt.allowed
    assert authorized["input"][0]["encrypted_content"] == encrypted_replay


def test_codex_reasoning_replay_uses_body_shape_when_api_mode_missing(tmp_path):
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = None
    encrypted_replay = "gAAAAABm3gZrX7xP0Yk3V8nQ2aBcD4eFgH5jKlMnOpQrStUvWxYz0123456789"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "gpt-5.6-terra",
            "input": [
                {
                    "type": "reasoning",
                    "encrypted_content": encrypted_replay,
                    "summary": [{"type": "summary_text", "text": "checked PR"}],
                }
            ],
        },
    )

    assert receipt.allowed
    assert authorized["input"][0]["encrypted_content"] == encrypted_replay
    assert authorized["input"][0]["summary"][0]["text"] == "checked PR"


def test_codex_user_supplied_encrypted_token_still_blocks(tmp_path):
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"

    with pytest.raises(EgressBlocked) as exc_info:
        authorize_agent_sdk_kwargs(
            agent,
            {
                "model": "gpt-5.6-terra",
                "input": [
                    {
                        "role": "user",
                        "content": "gAAAAABm3gZrX7xP0Yk3V8nQ2aBcD4eFgH5jKlMnOpQrStUvWxYz0123456789",
                    }
                ],
            },
        )

    assert "secret_detected" in exc_info.value.decision.reason_codes


def test_codex_user_content_private_path_is_not_silently_redacted(tmp_path):
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"

    with pytest.raises(EgressBlocked) as exc_info:
        authorize_agent_sdk_kwargs(
            agent,
            {
                "model": "gpt-5.6-terra",
                "input": [{"role": "user", "content": "Read /Users/private/file.py"}],
            },
        )

    assert "private_absolute_path" in exc_info.value.decision.reason_codes


def test_protected_codex_elides_bound_kanban_show_result(tmp_path, monkeypatch):
    """Board data remains local instead of causing a remote fallback loop."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "nous"
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_kanban_show_123"
    board_text = (
        '{"task":{"body":"untrusted c2VjcmV0LXBheWxvYWQ= '
        'token=super-secret-value /Users/private/source.py"}}'
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "gpt-5.6-terra",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "kanban_show", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_name": "kanban_show",
                    "tool_call_id": call_id,
                    "content": board_text,
                },
            ],
        },
    )

    rendered = authorized["messages"][1]["content"]
    assert receipt.allowed
    assert rendered.startswith("kanban_show completed locally.")
    assert "c2VjcmV0LXBheWxvYWQ=" not in rendered
    assert "super-secret-value" not in rendered
    assert "/Users/private/source.py" not in rendered


def test_protected_codex_kanban_projection_drops_raw_assignment_text(
    tmp_path, monkeypatch
):
    """A valid board record cannot replay arbitrary title/body text remotely."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_kanban_show_projection_boundary"
    board_text = json.dumps(
        {
            "task": {
                "title": "[Audit] Paper-safety posture 20260903",
                "body": "private assignment c2VjcmV0LXBheWxvYWQ=",
                "status": "blocked",
                "workspace_access": "assigned",
            },
            "parents": [{"body": "private dependency detail"}],
            "children": [{"body": "private child detail"}],
        }
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": [
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "kanban_show",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": board_text,
                },
            ],
        },
    )

    rendered = authorized["input"][1]["output"]
    assert receipt.allowed
    assert "20260903" not in rendered
    assert "private assignment" not in rendered
    assert "private dependency" not in rendered
    assert "c2VjcmV0LXBheWxvYWQ=" not in rendered
    assert '"status": "blocked"' in rendered


def test_protected_codex_kanban_projection_replays_only_versioned_task_spec(
    tmp_path, monkeypatch
):
    """The trusted producer contract restores the assignment without history."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_kanban_show_task_spec"
    board_text = json.dumps(
        {
            "task": {"status": "running", "workspace_access": "assigned"},
            "protected_task_spec": {
                "version": "v1",
                "title": "Repair the current PR feedback worker",
                "body": (
                    "Inspect the current checkout and report the result. "
                    "PAPER_SAFETY_SENTINEL_OK "
                    "token=super-secret-value c2VjcmV0LXBheWxvYWQ= "
                    "/Users/private/source.py"
                ),
            },
            "comments": [{"body": "obsolete attempt"}],
        }
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": [
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "kanban_show",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": board_text,
                },
            ],
        },
    )

    rendered = authorized["input"][1]["output"]
    assert receipt.allowed
    assert "Repair the current PR feedback worker" in rendered
    assert "Inspect the current checkout and report the result." in rendered
    assert "PAPER_SAFETY_SENTINEL_OK" in rendered
    assert "obsolete attempt" not in rendered
    assert "super-secret-value" not in rendered
    assert "c2VjcmV0LXBheWxvYWQ=" not in rendered
    assert "/Users/private/source.py" not in rendered


def test_protected_codex_elides_responses_kanban_show_output(tmp_path, monkeypatch):
    """Responses API function output follows the same no-egress boundary."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_kanban_show_responses"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "gpt-5.6-terra",
            "input": [
                {
                    "id": call_id,
                    "call_id": call_id,
                    "type": "function_call",
                    "function": {"name": "kanban_show", "arguments": "{}"},
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": "c2VjcmV0LXBheWxvYWQ= token=super-secret-value",
                },
            ],
        },
    )

    rendered = authorized["input"][1]["output"]
    assert receipt.allowed
    assert rendered.startswith("kanban_show completed locally.")
    assert "c2VjcmV0LXBheWxvYWQ=" not in rendered
    assert "super-secret-value" not in rendered


def test_protected_codex_elides_bound_kanban_attachments_output(tmp_path, monkeypatch):
    """Read-only attachment results cannot trigger an egress retry."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_kanban_attachments_123"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "gpt-5.6-terra",
            "input": [
                {
                    "id": call_id,
                    "call_id": call_id,
                    "type": "function_call",
                    "function": {"name": "kanban_attachments", "arguments": "{}"},
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": "c2VjcmV0LXBheWxvYWQ= token=super-secret-value",
                },
            ],
        },
    )

    rendered = authorized["input"][1]["output"]
    assert receipt.allowed
    assert rendered.startswith("kanban_attachments completed locally;")
    assert "c2VjcmV0LXBheWxvYWQ=" not in rendered
    assert "super-secret-value" not in rendered


def test_protected_codex_allows_benign_bound_kanban_attachments_output(
    tmp_path, monkeypatch
):
    """A benign bound attachment result must not become untrusted provenance."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_kanban_attachments_benign"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "gpt-5.6-terra",
            "input": [
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "kanban_attachments",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": "No attachments found.",
                },
            ],
        },
    )

    assert receipt.allowed
    assert authorized["input"][1]["output"].startswith(
        "kanban_attachments completed locally;"
    )


def test_protected_codex_projects_bound_kanban_lifecycle_result(
    tmp_path, monkeypatch
):
    """Lifecycle results replay only a fixed outcome, never raw control-plane text."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_kanban_comment_result"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "gpt-5.6-terra",
            "input": [
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "kanban_comment",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(
                        {
                            "ok": True,
                            "comment": "raw private path /Users/private/source.py",
                        }
                    ),
                },
            ],
        },
    )

    assert receipt.allowed
    assert authorized["input"][1]["output"] == (
        "Kanban lifecycle action completed locally; its raw control-plane result "
        "was omitted from remote replay."
    )


def test_protected_nous_elides_bound_kanban_show_result(tmp_path, monkeypatch):
    """The same safe projection covers protected free-provider workers."""

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_safe_projection")
    agent = _agent(tmp_path)
    agent.provider = "nous"
    agent.base_url = "https://inference-api.nousresearch.com/v1"
    call_id = "call_kanban_show_nous"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "poolside/laguna-xs-2.1:free",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "kanban_show", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "c2VjcmV0LXBheWxvYWQ= token=super-secret-value",
                },
            ],
        },
    )

    assert receipt.allowed
    assert authorized["messages"][1]["content"].startswith(
        "kanban_show completed locally."
    )


def test_protected_nous_elides_canonicalized_kanban_show_result(tmp_path, monkeypatch):
    """Bridge-id normalization cannot turn a real board result untrusted."""

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_safe_projection")
    agent = _agent(tmp_path)
    agent.provider = "nous"
    agent.base_url = "https://inference-api.nousresearch.com/v1"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "poolside/laguna-xs-2.1:free",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_kanban_show|fc_response_item",
                            "type": "function",
                            "function": {"name": "kanban_show", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_kanban_show",
                    "content": "c2VjcmV0LXBheWxvYWQ=",
                },
            ],
        },
    )

    assert receipt.allowed
    assert authorized["messages"][1]["content"].startswith(
        "kanban_show completed locally."
    )


def test_protected_nous_keeps_unbound_kanban_output_blocked(tmp_path, monkeypatch):
    """Provider broadening never treats an asserted call ID as trusted."""

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_safe_projection")
    agent = _agent(tmp_path)
    agent.provider = "nous"
    agent.base_url = "https://inference-api.nousresearch.com/v1"

    with pytest.raises(EgressBlocked) as exc_info:
        authorize_agent_sdk_kwargs(
            agent,
            {
                "model": "poolside/laguna-xs-2.1:free",
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "call_unbound_kanban_show",
                        "content": "c2VjcmV0LXBheWxvYWQ=",
                    }
                ],
            },
        )

    assert "base64_payload" in exc_info.value.decision.reason_codes


def test_protected_nous_redacts_generated_cloud_system_context(tmp_path):
    """Generated cloud framing may be redacted, unlike user/source content."""

    agent = _agent(tmp_path)
    agent.provider = "nous"
    agent.base_url = "https://inference-api.nousresearch.com/v1"
    generated = "Hermes framing.\n" * 3_000 + (
        "Workspace: /Users/private/worktree\n"
        "Protocol sample: c2VjcmV0LXBheWxvYWQ="
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "poolside/laguna-xs-2.1:free",
            "messages": [
                {
                    "role": "system",
                    "content": generated,
                }
            ],
        },
    )

    assert receipt.allowed
    assert "<private-path>" in authorized["messages"][0]["content"]
    assert "<redacted-base64>" in authorized["messages"][0]["content"]
    assert "/Users/private/worktree" not in authorized["messages"][0]["content"]
    assert "c2VjcmV0LXBheWxvYWQ=" not in authorized["messages"][0]["content"]
    assert len(receipt.payload_bytes) > 32_768


def test_protected_nous_generated_context_allows_numeric_output_cap(tmp_path):
    """A numeric JSON control must not be misclassified as base64 text."""

    agent = _agent(tmp_path)
    agent.provider = "nous"
    agent.base_url = "https://inference-api.nousresearch.com/v1"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "poolside/laguna-xs-2.1:free",
            "messages": [
                {
                    "role": "system",
                    "content": "Generated schema example: c2VjcmV0LXBheWxvYWQ=",
                }
            ],
            "max_tokens": 4096,
        },
    )

    assert receipt.allowed
    assert authorized["max_tokens"] == 4096
    assert "<redacted-base64>" in authorized["messages"][0]["content"]


def test_protected_remote_generated_context_preserves_worker_identifiers(tmp_path):
    agent = _agent(tmp_path)
    agent.provider = "nous"
    agent.base_url = "https://inference-api.nousresearch.com/v1"
    generated = (
        "profile ci-hygiene-fixer follows non-gate-weakening rules; "
        "reproduction_command is recorded for HTTP HYGIENE checks."
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "poolside/laguna-xs-2.1:free",
            "messages": [{"role": "system", "content": generated}],
        },
    )

    assert receipt.allowed
    assert authorized["messages"][0]["content"] == generated


def test_protected_nous_keeps_generated_cloud_secrets_blocked(tmp_path):
    """Redaction never converts secrets into remote-safe text."""

    agent = _agent(tmp_path)
    agent.provider = "nous"
    agent.base_url = "https://inference-api.nousresearch.com/v1"

    with pytest.raises(EgressBlocked) as exc_info:
        authorize_agent_sdk_kwargs(
            agent,
            {
                "model": "poolside/laguna-xs-2.1:free",
                "messages": [
                    {"role": "system", "content": "SECRET_TOKEN=super-secret-value"}
                ],
            },
        )

    assert "secret_detected" in exc_info.value.decision.reason_codes


def test_protected_nous_keeps_user_kanban_content_blocked(tmp_path, monkeypatch):
    """The cloud framing allowance never promotes task/source input."""

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_safe_projection")
    agent = _agent(tmp_path)
    agent.provider = "nous"
    agent.base_url = "https://inference-api.nousresearch.com/v1"

    with pytest.raises(EgressBlocked) as exc_info:
        authorize_agent_sdk_kwargs(
            agent,
            {
                "model": "poolside/laguna-xs-2.1:free",
                "messages": [
                    {
                        "role": "user",
                        "content": "c2VjcmV0LXBheWxvYWQ=",
                    }
                ],
            },
        )

    assert "base64_payload" in exc_info.value.decision.reason_codes


def test_protected_codex_does_not_elide_unbound_kanban_show_result(
    tmp_path, monkeypatch
):
    """Only an actual prior Kanban tool call may discard its output."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"

    with pytest.raises(EgressBlocked) as exc_info:
        authorize_agent_sdk_kwargs(
            agent,
            {
                "model": "gpt-5.6-terra",
                "messages": [
                    {
                        "role": "tool",
                        "tool_name": "kanban_show",
                        "tool_call_id": "call_unbound_kanban_show",
                        "content": "c2VjcmV0LXBheWxvYWQ=",
                    }
                ],
            },
        )

    assert "base64_payload" in exc_info.value.decision.reason_codes


def test_protected_codex_does_not_elide_unbound_responses_kanban_output(
    tmp_path, monkeypatch
):
    """Responses output without the actual prior call remains fail-closed."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"

    with pytest.raises(EgressBlocked) as exc_info:
        authorize_agent_sdk_kwargs(
            agent,
            {
                "model": "gpt-5.6-terra",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_unbound_kanban_show",
                        "output": "c2VjcmV0LXBheWxvYWQ=",
                    }
                ],
            },
        )

    assert "base64_payload" in exc_info.value.decision.reason_codes


def test_protected_codex_projects_bound_github_list_terminal_metadata(
    tmp_path, monkeypatch
):
    """A bounded GitHub list retains review fields, not opaque API identifiers."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_github_snapshot_1234"
    command = (
        "gh issue list --repo NousResearch/hermes-agent --state open "
        "--limit 20 --json number,title,url,labels"
    )
    raw_rows = [
        {
            "id": "c2VjcmV0LXBheWxvYWQ=",
            "number": 123,
            "title": "Desktop resume reliability",
            "url": "https://github.com/NousResearch/hermes-agent/issues/123",
            "labels": [{"id": "c2VjcmV0LWxhYmVs", "name": "desktop"}],
        }
    ]
    kwargs = {
        "model": agent.model,
        "input": [
            {
                "type": "function_call",
                "name": "terminal",
                "call_id": call_id,
                "arguments": json.dumps({"command": command}),
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"exit_code": 0, "output": json.dumps(raw_rows)}),
            },
        ],
    }

    authorized, _ = authorize_agent_sdk_kwargs(agent, kwargs)

    projected = json.loads(json.loads(authorized["input"][1]["output"])["output"])
    assert projected["items"] == [
        {
            "labels": ["desktop"],
            "number": 123,
            "title": "Desktop resume reliability",
            "url": "https://github.com/NousResearch/hermes-agent/issues/123",
        }
    ]
    assert "c2VjcmV0LXBheWxvYWQ=" not in authorized["input"][1]["output"]


def test_protected_codex_redacts_base64_like_pr_view_ref(tmp_path, monkeypatch):
    """A safe projected ref cannot deadlock replay because its spelling is opaque."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_github_pr_view_ref"
    row = {
        "number": 90820,
        "baseRefName": "main",
        "headRefName": "final-v2-landing",
        "title": "Kanban worker controls",
        "url": "https://github.com/NousResearch/hermes-agent/pull/90820",
        "state": "CLOSED",
    }

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": [
                {
                    "type": "function_call",
                    "name": "terminal",
                    "call_id": call_id,
                    "arguments": json.dumps(
                        {
                            "command": (
                                "gh pr view 90820 --repo NousResearch/hermes-agent "
                                "--json number,baseRefName,headRefName,title,url,state"
                            )
                        }
                    ),
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(
                        {"exit_code": 0, "output": json.dumps(row)}
                    ),
                },
            ],
        },
    )

    assert receipt.allowed
    projected = json.loads(json.loads(authorized["input"][1]["output"])["output"])
    assert projected["items"][0]["number"] == 90820
    assert projected["items"][0]["baseRefName"] == "main"
    assert "c2VjcmV0LXBheWxvYWQ=" not in authorized["input"][1]["output"]


@pytest.mark.parametrize("tool_name", ("write_file", "patch"))
def test_protected_codex_elides_bound_file_mutation_result(
    tmp_path, monkeypatch, tool_name
):
    """A successful local edit never replays source or diff bytes remotely."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = f"call_{tool_name}_result"
    unsafe_result = (
        "updated /Users/private/repository/file.py\n"
        "+ encoded = 'c2VjcmV0LXBheWxvYWQ='\n"
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": [
                {
                    "type": "function_call",
                    "name": tool_name,
                    "call_id": call_id,
                    "arguments": json.dumps({"path": "file.py"}),
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": unsafe_result,
                },
            ],
        },
    )

    assert receipt.allowed
    assert unsafe_result not in json.dumps(authorized)
    assert "Inspect git diff" in authorized["input"][1]["output"]


def test_protected_codex_projects_bound_github_api_extract_metadata(
    tmp_path, monkeypatch
):
    """A verified GitHub REST list fetch never replays opaque API fields."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_github_api_extract_1234"
    api_url = (
        "https://api.github.com/repos/NousResearch/hermes-agent/issues"
        "?state=open&per_page=20"
    )
    raw_rows = [
        {
            "id": "c2VjcmV0LXBheWxvYWQ=",
            "node_id": "c2VjcmV0LW5vZGUtYnl0ZXM=",
            "number": 123,
            "title": "Desktop resume reliability",
            "state": "open",
            "html_url": "https://github.com/NousResearch/hermes-agent/issues/123",
            "labels": [{"id": "c2VjcmV0LWxhYmVs", "name": "desktop"}],
            "user": {"id": 42, "login": "contributor"},
        }
    ]
    kwargs = {
        "model": agent.model,
        "input": [
            {
                "type": "function_call",
                "name": "web_extract",
                "call_id": call_id,
                "arguments": json.dumps({"urls": [api_url]}),
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(
                    {
                        "success": True,
                        "results": [
                            {
                                "url": api_url,
                                "content": json.dumps(raw_rows),
                            }
                        ],
                    }
                ),
            },
        ],
    }

    authorized, _ = authorize_agent_sdk_kwargs(agent, kwargs)

    projected = json.loads(authorized["input"][1]["output"])
    assert projected["github_api_extract_projection"] == "v1"
    assert projected["results"] == [
        {
            "url": api_url,
            "items": [
                {
                    "labels": ["desktop"],
                    "number": 123,
                    "state": "open",
                    "title": "Desktop resume reliability",
                }
            ],
        }
    ]
    assert "c2VjcmV0LXBheWxvYWQ=" not in authorized["input"][1]["output"]


def test_protected_codex_elides_bound_github_api_curl_arguments(tmp_path, monkeypatch):
    """A bounded REST list ``curl`` does not replay its opaque URL path."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_github_api_curl_1234"
    command = (
        "curl -fsSL "
        "'https://api.github.com/repos/NousResearch/hermes-agent/issues?state=open&per_page=20'"
    )
    kwargs = {
        "model": agent.model,
        "input": [
            {
                "type": "function_call",
                "name": "terminal",
                "call_id": call_id,
                "arguments": json.dumps({"command": command}),
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"exit_code": 0, "output": "20 issues listed"}),
            },
        ],
    }

    authorized, _ = authorize_agent_sdk_kwargs(agent, kwargs)

    assert authorized["input"][0]["arguments"] == json.dumps(
        {"command": "curl GitHub REST list (details omitted)"}, separators=(",", ":")
    )
    assert "NousResearch" not in authorized["input"][0]["arguments"]


def test_protected_codex_elides_bound_plain_github_list_output(tmp_path, monkeypatch):
    """Unstructured GitHub titles cannot make a protected worker leak content."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_github_plain_list_1234"
    kwargs = {
        "model": agent.model,
        "input": [
            {
                "type": "function_call",
                "name": "terminal",
                "call_id": call_id,
                "arguments": json.dumps(
                    {
                        "command": (
                            "gh issue list --repo NousResearch/hermes-agent "
                            "--state open --limit 20"
                        )
                    }
                ),
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(
                    {"exit_code": 0, "output": "123  c2VjcmV0LXBheWxvYWQ="}
                ),
            },
        ],
    }

    authorized, _ = authorize_agent_sdk_kwargs(agent, kwargs)

    assert (
        authorized["input"][1]["output"]
        == "GitHub list output omitted; use --json for bounded fields."
    )


def test_protected_codex_projects_combined_github_list_diagnostic(tmp_path, monkeypatch):
    """A bounded workspace/GitHub list chain retains no raw API identifiers."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_github_chain_1234"
    command = (
        "pwd && git status --short && "
        "gh issue list --repo NousResearch/hermes-agent --state open --limit 20 "
        "--json number,title,labels,assignees,updatedAt,createdAt,url && "
        "gh pr list --repo NousResearch/hermes-agent --state open --limit 100 "
        "--json number,title,headRefName,baseRefName,updatedAt,createdAt,url"
    )
    rows = [
        {"id": "c2VjcmV0LXBheWxvYWQ=", "number": 123, "title": "Issue", "state": "open"},
        {"node_id": "c2VjcmV0LW5vZGUtYnl0ZXM=", "number": 124, "title": "PR", "state": "open"},
    ]
    kwargs = {
        "model": agent.model,
        "input": [
            {
                "type": "function_call",
                "name": "terminal",
                "call_id": call_id,
                "arguments": json.dumps({"command": command}),
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(
                    {"exit_code": 0, "output": "/workspace\n M file.py\n" + json.dumps(rows)}
                ),
            },
        ],
    }

    authorized, _ = authorize_agent_sdk_kwargs(agent, kwargs)

    replay = authorized["input"][1]["output"]
    assert "c2VjcmV0LXBheWxvYWQ=" not in replay
    assert json.loads(json.loads(replay)["output"])["items"] == [
        {"number": 123, "state": "open", "title": "Issue"},
        {"number": 124, "state": "open", "title": "PR"},
    ]


def test_protected_codex_projects_combined_github_issue_views(tmp_path, monkeypatch):
    """A bounded chain of issue views excludes bodies and comments."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_github_views_1234"
    command = " && ".join(
        "gh issue view " + number + " --repo NousResearch/hermes-agent "
        "--json number,title,body,labels,assignees,state,url,comments,createdAt,updatedAt"
        for number in ("98168", "98160")
    )
    raw = "\n".join(
        json.dumps(
            {"id": f"c2VjcmV0LXBheWxvYWQ={number}", "number": int(number), "title": "Issue", "body": "c2VjcmV0LXBheWxvYWQ=", "comments": [{"body": "c2VjcmV0LXBheWxvYWQ="}]}
        )
        for number in ("98168", "98160")
    )
    kwargs = {"model": agent.model, "input": [
        {"type": "function_call", "name": "terminal", "call_id": call_id, "arguments": json.dumps({"command": command})},
        {"type": "function_call_output", "call_id": call_id, "output": json.dumps({"exit_code": 0, "output": raw})},
    ]}

    authorized, _ = authorize_agent_sdk_kwargs(agent, kwargs)

    replay = authorized["input"][1]["output"]
    assert "c2VjcmV0LXBheWxvYWQ=" not in replay
    assert [item["number"] for item in json.loads(json.loads(replay)["output"])["items"]] == [98168, 98160]


@pytest.mark.parametrize("output_field", ["stderr", "output"])
def test_protected_codex_replays_bounded_pr_feedback_failure_excerpt(
    tmp_path, monkeypatch, output_field
):
    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_pr_feedback_failure"
    kwargs = {
        "model": agent.model,
        "input": [
            {
                "type": "function_call",
                "name": "terminal",
                "call_id": call_id,
                "arguments": json.dumps(
                    {
                        "command": (
                            "env HERMES_HOME=$HERMES_CONTROL_HOME python3 -m "
                            "hermes_cli.main github-pr-feedback inspect-pr "
                            "--repository acme/widgets --pr-number 17"
                        )
                    }
                ),
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(
                    {
                        "exit_code": 1,
                        output_field: (
                            "Traceback: ModuleNotFoundError: No module named "
                            "dotenv at /Users/operator/private.py"
                        ),
                    }
                ),
            },
        ],
    }

    authorized, receipt = authorize_agent_sdk_kwargs(agent, kwargs)

    rendered = json.loads(authorized["input"][1]["output"])
    assert receipt.allowed
    assert rendered["terminal_result"] == "github_pr_feedback"
    assert rendered["exit_code"] == 1
    assert "ModuleNotFoundError" in rendered["error_excerpt"]
    assert "/Users/operator" not in rendered["error_excerpt"]
    assert "raw_output" in rendered


def test_protected_codex_preserves_audit_process_and_receipt_handoff(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_pr_feedback_failure"
    kwargs = {
        "model": agent.model,
        "input": [
            {
                "type": "function_call",
                "name": "terminal",
                "call_id": call_id,
                "arguments": json.dumps(
                    {
                        "command": (
                            "env HERMES_HOME=$HERMES_CONTROL_HOME python3 -m "
                            "hermes_cli.main github-pr-feedback inspect-pr "
                            "--repository acme/widgets --pr-number 17"
                        )
                    }
                ),
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(
                    {
                        "exit_code": 0,
                        "session_id": "proc_162741633896",
                        "pid": 88779,
                        "output": json.dumps({"status": "audit_handoff_retryable", "receipt_id": "a" * 64, "handoff_reason": "base_refresh_required", "retryable": True}),
                    }
                ),
            },
        ],
    }

    authorized, receipt = authorize_agent_sdk_kwargs(agent, kwargs)

    rendered = json.loads(authorized["input"][1]["output"])
    assert receipt.allowed
    assert rendered["terminal_result"] == "github_pr_feedback"
    assert rendered["session_id"] == "proc_162741633896"
    assert rendered["pid"] == 88779
    assert rendered["json"]["receipt_id"] == "a" * 64
    assert rendered["json"]["handoff_reason"] == "base_refresh_required"
    assert rendered["json"]["retryable"] is True
    assert "raw_output" in rendered


def test_protected_codex_omits_rejected_terminal_command_replay(
    tmp_path, monkeypatch
):
    """A command rejected before execution need not be replayed remotely."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_rejected_terminal_1234"
    dangerous_command = "python - <<'PY'\nprint('c2VjcmV0LXBheWxvYWQ=')\nPY"
    kwargs = {
        "model": agent.model,
        "input": [
            {
                "type": "function_call",
                "name": "terminal",
                "call_id": call_id,
                "arguments": json.dumps({"command": dangerous_command}),
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(
                    {
                        "exit_code": -1,
                        "error": "BLOCKED: Command flagged as dangerous before execution",
                    }
                ),
            },
        ],
    }

    authorized, _ = authorize_agent_sdk_kwargs(agent, kwargs)

    assert json.loads(authorized["input"][0]["arguments"]) == {
        "command": "<rejected terminal command omitted>"
    }
    assert "c2VjcmV0LXBheWxvYWQ=" not in authorized["input"][0]["arguments"]


def test_protected_codex_projects_bound_github_issue_view_metadata(
    tmp_path, monkeypatch
):
    """A bounded issue view never replays a body containing credential-shaped text."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_github_issue_view_1234"
    kwargs = {
        "model": agent.model,
        "input": [
            {
                "type": "function_call",
                "name": "terminal",
                "call_id": call_id,
                "arguments": json.dumps(
                    {
                        "command": (
                            "gh issue view 123 --repo NousResearch/hermes-agent "
                            "--json number,title,url,labels,body"
                        )
                    }
                ),
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(
                    {
                        "exit_code": 0,
                        "output": json.dumps(
                            {
                                "id": "c2VjcmV0LXBheWxvYWQ=",
                                "number": 123,
                                "title": "Desktop resume reliability",
                                "url": "https://github.com/NousResearch/hermes-agent/issues/123",
                                "body": "token=live-secret-value",
                            }
                        ),
                    }
                ),
            },
        ],
    }

    authorized, _ = authorize_agent_sdk_kwargs(agent, kwargs)

    projected = json.loads(json.loads(authorized["input"][1]["output"])["output"])
    assert projected["items"] == [
        {
            "number": 123,
            "title": "Desktop resume reliability",
            "url": "https://github.com/NousResearch/hermes-agent/issues/123",
        }
    ]
    assert "live-secret-value" not in authorized["input"][1]["output"]


def test_protected_codex_projects_bound_github_pr_view_metadata(
    tmp_path, monkeypatch
):
    """A pull-request view has the same opaque-id/body replay boundary."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_github_pr_view_1234"
    kwargs = {
        "model": agent.model,
        "input": [
            {
                "type": "function_call",
                "name": "terminal",
                "call_id": call_id,
                "arguments": json.dumps(
                    {
                        "command": (
                            "gh pr view 123 --repo NousResearch/hermes-agent "
                            "--json number,title,url,labels,body"
                        )
                    }
                ),
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(
                    {
                        "exit_code": 0,
                        "output": json.dumps(
                            {
                                "id": "c2VjcmV0LXBheWxvYWQ=",
                                "number": 123,
                                "title": "Desktop resume reliability",
                                "url": "https://github.com/NousResearch/hermes-agent/pull/123",
                                "body": "token=live-secret-value",
                            }
                        ),
                    }
                ),
            },
        ],
    }

    authorized, _ = authorize_agent_sdk_kwargs(agent, kwargs)

    projected = json.loads(json.loads(authorized["input"][1]["output"])["output"])
    assert projected["items"] == [
        {
            "number": 123,
            "title": "Desktop resume reliability",
            "url": "https://github.com/NousResearch/hermes-agent/pull/123",
        }
    ]
    assert "live-secret-value" not in authorized["input"][1]["output"]


def test_protected_codex_preserves_exact_github_pr_identity_fields(
    tmp_path, monkeypatch
):
    """A governed PR preflight retains only its bounded identity tuple."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_github_pr_identity_1234"
    command = (
        "gh pr view 719 --repo mrkillbob/luna-bot "
        "--json baseRefName,baseRefOid,headRefName,headRefOid,headRepository"
    )
    raw_identity = {
        "baseRefName": "stable",
        "baseRefOid": "b21d4ae47ebb596929357145a8ca7b819bbc3d8a",
        "headRefName": "codex/q446-execution-margin-auto-flatten-owner",
        "headRefOid": "987abfb478cdcf9c3d1725286c9ccf779e915832",
        "headRepository": {
            "name": "luna-bot",
            "nameWithOwner": "mrkillbob/luna-bot",
        },
        "id": "c2VjcmV0LXBheWxvYWQ=",
    }
    kwargs = {
        "model": agent.model,
        "input": [
            {
                "type": "function_call",
                "name": "terminal",
                "call_id": call_id,
                "arguments": json.dumps({"command": command}),
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(
                    {"exit_code": 0, "output": json.dumps(raw_identity)}
                ),
            },
        ],
    }

    authorized, _ = authorize_agent_sdk_kwargs(agent, kwargs)

    projected = json.loads(json.loads(authorized["input"][1]["output"])["output"])
    assert projected["items"] == [
        {
            "baseRefName": raw_identity["baseRefName"],
            "baseRefOid": raw_identity["baseRefOid"],
            "headRefName": raw_identity["headRefName"],
            "headRefOid": raw_identity["headRefOid"],
            "headRepository": raw_identity["headRepository"],
        }
    ]
    assert "c2VjcmV0LXBheWxvYWQ=" not in authorized["input"][1]["output"]


def test_protected_codex_projects_structured_github_pr_identity_fields(
    tmp_path, monkeypatch
):
    """Responses arrays retain the same bounded governed PR identity tuple."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_structured_github_pr_identity"
    raw_identity = {
        "baseRefName": "main",
        "baseRefOid": "b21d4ae47ebb596929357145a8ca7b819bbc3d8a",
        "headRefName": "codex/small-reviewable-fix",
        "headRefOid": "987abfb478cdcf9c3d1725286c9ccf779e915832",
        "headRepository": {"nameWithOwner": "mrkillbob/hermes-agent"},
        "id": "c2VjcmV0LXBheWxvYWQ=",
    }
    terminal_result = json.dumps(
        {"exit_code": 0, "output": json.dumps(raw_identity)}
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": [
                {
                    "type": "function_call",
                    "name": "terminal",
                    "call_id": call_id,
                    "arguments": json.dumps(
                        {
                            "command": (
                                "gh pr view 719 --repo mrkillbob/hermes-agent "
                                "--json baseRefName,baseRefOid,headRefName,"
                                "headRefOid,headRepository"
                            )
                        }
                    ),
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": [{"type": "input_text", "text": terminal_result}],
                },
            ],
        },
    )

    assert receipt.allowed
    projected = json.loads(json.loads(authorized["input"][1]["output"])["output"])
    assert projected["items"] == [
        {
            "baseRefName": raw_identity["baseRefName"],
            "baseRefOid": raw_identity["baseRefOid"],
            "headRefName": raw_identity["headRefName"],
            "headRefOid": raw_identity["headRefOid"],
            "headRepository": raw_identity["headRepository"],
        }
    ]
    assert raw_identity["id"] not in json.dumps(authorized)


@pytest.mark.parametrize(
    "payload",
    (
        "c2VjcmV0LXBheWxvYWQ=",
        "API_TOKEN=not-a-real-secret-token-123456789",
    ),
)
def test_structured_terminal_replay_remains_outcome_only(
    tmp_path, monkeypatch, payload
):
    """Responses API output arrays receive the same terminal replay policy."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_structured_terminal_output"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": [
                {
                    "type": "function_call",
                    "name": "terminal",
                    "call_id": call_id,
                    "arguments": json.dumps(
                        {"command": "printf test-output"}
                    ),
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": [{"type": "input_text", "text": payload}],
                },
            ],
        },
    )

    assert receipt.allowed
    assert payload not in json.dumps(authorized)
    assert json.loads(authorized["input"][1]["output"]) == {
        "terminal_result": "completed",
        "exit_code": None,
        "raw_output": "omitted_from_remote_replay",
    }


def test_protected_codex_projects_bound_pytest_failure_diagnostics(
    tmp_path, monkeypatch
):
    """A bound pytest call retains safe failure facts without replaying source."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_pytest_failure"
    raw_output = (
        "============================= test session starts =============================\n"
        "ERROR collecting tests/test_state.py\n"
        "E   ImportError: cannot import name 'ConversationWorktreeRecord'\n"
        "=========================== short test summary info ============================\n"
        "ERROR tests/test_state.py - ImportError: missing state API\n"
        "!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!\n"
    )
    terminal_result = json.dumps({"exit_code": 1, "output": raw_output})

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": [
                {
                    "type": "function_call",
                    "name": "terminal",
                    "call_id": call_id,
                    "arguments": json.dumps(
                        {"command": "python3 -m pytest -q tests"}
                    ),
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": [{"type": "input_text", "text": terminal_result}],
                },
            ],
        },
    )

    assert receipt.allowed
    rendered = json.loads(authorized["input"][1]["output"])
    assert rendered["terminal_result"] == "pytest"
    assert rendered["exit_code"] == 1
    assert "ConversationWorktreeRecord" in "\n".join(rendered["diagnostics"])
    assert raw_output not in json.dumps(authorized)


def test_protected_codex_elides_bound_browser_exec_output(
    tmp_path, monkeypatch
):
    """Browser Use results follow the exact-call local-action replay boundary."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_browser_exec_output"
    unsafe_output = json.dumps(
        {"status": "ok", "text": "c2VjcmV0LXBheWxvYWQ="}
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": [
                {
                    "type": "function_call",
                    "name": "browser_exec",
                    "call_id": call_id,
                    "arguments": json.dumps(
                        {"code": "print('c2VjcmV0LXBheWxvYWQ=')"}
                    ),
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": unsafe_output,
                },
            ],
        },
    )

    assert receipt.allowed
    assert unsafe_output not in json.dumps(authorized)
    assert "<redacted-base64>" in authorized["input"][0]["arguments"]
    assert json.loads(authorized["input"][1]["output"]) == {
        "terminal_result": "completed",
        "exit_code": None,
        "raw_output": "omitted_from_remote_replay",
    }


def test_protected_codex_projects_bound_paginated_github_api_output(
    tmp_path, monkeypatch
):
    """The White-Knight ``gh api --paginate`` list path stays bounded."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_github_api_paginate_output"
    command = (
        "gh api --paginate "
        "'/repos/NousResearch/hermes-agent/issues?state=open'"
    )
    raw_output = json.dumps(
        [
            {
                "id": "c2VjcmV0LXBheWxvYWQ=",
                "number": 102658,
                "title": "Encoded-looking title is still bounded",
                "state": "open",
                "user": {"id": 42, "login": "contributor"},
            }
        ]
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": [
                {
                    "type": "function_call",
                    "name": "terminal",
                    "call_id": call_id,
                    "arguments": json.dumps({"command": command}),
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps({"exit_code": 0, "output": raw_output}),
                },
            ],
        },
    )

    assert receipt.allowed
    assert "c2VjcmV0LXBheWxvYWQ=" not in json.dumps(authorized)
    assert "NousResearch" not in authorized["input"][0]["arguments"]
    projected = json.loads(authorized["input"][1]["output"])
    assert json.loads(projected["output"])["items"] == [
        {
            "number": 102658,
            "state": "open",
            "title": "Encoded-looking title is still bounded",
        }
    ]


def test_protected_codex_elides_structured_rg_output(tmp_path, monkeypatch):
    """Structured search output never falls through to recursive text replay."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_structured_rg_output"
    payload = "README.md:42:c2VjcmV0LXBheWxvYWQ="

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": [
                {
                    "type": "function_call",
                    "name": "terminal",
                    "call_id": call_id,
                    "arguments": json.dumps(
                        {"command": "rg -n needle README.md"}
                    ),
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": [{"type": "input_text", "text": payload}],
                },
            ],
        },
    )

    assert receipt.allowed
    assert payload not in json.dumps(authorized)
    assert (
        authorized["input"][1]["output"]
        == "search completed locally; structured output omitted from remote replay."
    )


def test_protected_codex_rejects_unknown_structured_tool_output(tmp_path, monkeypatch):
    """Structured output is elided only when bound to a recognized local call."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"

    with pytest.raises(EgressBlocked, match="base64_payload"):
        authorize_agent_sdk_kwargs(
            agent,
            {
                "model": agent.model,
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_unknown_structured_output",
                        "output": [
                            {
                                "type": "input_text",
                                "text": "c2VjcmV0LXBheWxvYWQ=",
                            }
                        ],
                    }
                ],
            },
        )


@pytest.mark.parametrize(
    "unsafe_text",
    (
        "c2VjcmV0LXBheWxvYWQ=",
        "API_TOKEN=not-a-real-secret-token-123456789",
    ),
)
def test_protected_codex_redacts_bound_web_search_replay(
    tmp_path, monkeypatch, unsafe_text
):
    """Public search evidence remains usable without replaying unsafe atoms."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_web_search_replay"
    output = (
        '<external_tool_result source="web_search">\n'
        + json.dumps(
            {
                "search": {
                    "web": [
                        {
                            "url": "https://example.com/result",
                            "title": "Public result",
                            "description": (
                                f"Untrusted public snippet {unsafe_text} "
                                "from /Users/private/repository/file.py"
                            ),
                        }
                    ]
                }
            }
        )
        + "\n</external_tool_result>"
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": [
                {
                    "type": "function_call",
                    "name": "web_search",
                    "call_id": call_id,
                    "arguments": json.dumps({"query": "public source"}),
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                },
            ],
        },
    )

    rendered = authorized["input"][1]["output"]
    assert receipt.allowed
    assert "https://example.com/result" in rendered
    assert "Public result" in rendered
    assert unsafe_text not in rendered
    assert "/Users/private/repository/file.py" not in rendered


def test_protected_codex_projects_git_diff_name_only_output(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_git_diff_name_only"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": [
                {
                    "type": "function_call",
                    "name": "terminal",
                    "call_id": call_id,
                    "arguments": json.dumps({"command": "git diff --name-only"}),
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(
                        {
                            "exit_code": 0,
                            "output": "agent/llm_egress_runtime.py\n../secret.txt\n",
                        }
                    ),
                },
            ],
        },
    )

    rendered = json.loads(authorized["input"][1]["output"])
    projection = json.loads(rendered["output"])
    assert receipt.allowed
    assert projection["files"] == ["agent/llm_egress_runtime.py"]
    assert projection["omitted_files"] == 1


def test_protected_codex_projects_bound_file_mutation_outcome(tmp_path, monkeypatch):
    """A patch receipt keeps outcome facts while dropping its unified diff."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_patch_outcome"
    patch_result = json.dumps(
        {
            "success": True,
            "diff": "@@ -1 +1 @@\n-secret\n+changed\n",
            "files_modified": ["/Users/private/worktree/hermes_state.py"],
        }
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": [
                {
                    "type": "function_call",
                    "name": "patch",
                    "call_id": call_id,
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": patch_result,
                },
            ],
        },
    )

    rendered = json.loads(authorized["input"][1]["output"])
    assert receipt.allowed
    assert rendered == {
        "file_mutation": "completed",
        "success": True,
        "files_modified_count": 1,
    }
    assert "secret" not in receipt.payload_bytes.decode("utf-8")


def test_protected_codex_projects_bound_file_mutation_error(tmp_path, monkeypatch):
    """A failed patch keeps a bounded safe error for the next model turn."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_patch_error"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": [
                {
                    "type": "function_call",
                    "name": "patch",
                    "call_id": call_id,
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(
                        {
                            "success": False,
                            "error": "Patch validation failed (no files were modified): missing symbol",
                        }
                    ),
                },
            ],
        },
    )

    rendered = json.loads(authorized["input"][1]["output"])
    assert receipt.allowed
    assert rendered["file_mutation"] == "completed"
    assert rendered["success"] is False
    assert "missing symbol" in rendered["error"]


def test_protected_codex_projects_github_pr_feedback_output(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_pr_feedback"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": [
                {
                    "type": "function_call",
                    "name": "terminal",
                    "call_id": call_id,
                    "arguments": json.dumps(
                        {
                            "command": (
                                "env HERMES_HOME=/Users/mikedemott/.hermes "
                                "/Users/mikedemott/.local/bin/hermes "
                                "github-pr-feedback inspect-pr "
                                "--repository acme/widgets --pr-number 17 --feedback-id 120"
                            )
                        }
                    ),
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(
                        {
                            "exit_code": 0,
                            "output": json.dumps(
                                {
                                    "repository": "acme/widgets",
                                    "number": 17,
                                    "head_sha": "a" * 40,
                                    "feedback_id": "120",
                                    "feedback_body_excerpt": "bounded review finding",
                                    "opaque_node_id": "c2VjcmV0LXBheWxvYWQ=",
                                }
                            ),
                        }
                    ),
                },
            ],
        },
    )

    rendered = json.loads(authorized["input"][1]["output"])
    assert receipt.allowed
    assert rendered["terminal_result"] == "github_pr_feedback"
    assert rendered["json"]["feedback_body_excerpt"] == "bounded review finding"
    assert "opaque_node_id" not in rendered["json"]


def test_protected_codex_rejects_unbound_web_search_shaped_replay(
    tmp_path, monkeypatch
):
    """A web-shaped result receives no exemption without an exact call binding."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"

    with pytest.raises(EgressBlocked, match="base64_payload"):
        authorize_agent_sdk_kwargs(
            agent,
            {
                "model": agent.model,
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_unbound_web_result",
                        "output": "c2VjcmV0LXBheWxvYWQ=",
                    }
                ],
            },
        )


def test_protected_codex_elides_bound_web_extract_source_replay(
    tmp_path, monkeypatch
):
    """Raw web-extract source cannot trip the cloud egress scanner."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_web_extract_replay"
    output = (
        "source=public\n"
        "c2VjcmV0LXBheWxvYWQ=\n"
        "/Users/private/repository/data_loader.py\n"
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": [
                {
                    "type": "function_call",
                    "name": "web_extract",
                    "call_id": call_id,
                    "arguments": json.dumps(
                        {
                            "urls": [
                                "https://raw.githubusercontent.com/acme/widget/main/app.py"
                            ]
                        }
                    ),
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                },
            ],
        },
    )

    rendered = authorized["input"][1]["output"]
    assert receipt.allowed
    assert "c2VjcmV0LXBheWxvYWQ=" not in rendered
    assert "/Users/private/repository/data_loader.py" not in rendered
    assert "raw excerpts omitted" in rendered


@pytest.mark.parametrize("structured", (False, True))
def test_protected_codex_projects_exact_kanban_assignee_roster(
    tmp_path, monkeypatch, structured
):
    """A worker can recover bounded on-disk profile names without raw replay."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    call_id = "call_kanban_assignees_roster"
    terminal_result = json.dumps(
        {
            "exit_code": 0,
            "output": json.dumps(
                [
                    {
                        "name": "task-orchestrator",
                        "on_disk": True,
                        "counts": {"blocked": 2},
                    },
                    {
                        "name": "stale-board-owner",
                        "on_disk": False,
                        "counts": {"blocked": 1},
                    },
                    {
                        "name": "API_TOKEN=not-a-profile-secret",
                        "on_disk": True,
                        "counts": {},
                    },
                ]
            ),
        }
    )
    output = (
        [{"type": "input_text", "text": terminal_result}]
        if structured
        else terminal_result
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": [
                {
                    "type": "function_call",
                    "name": "terminal",
                    "call_id": call_id,
                    "arguments": json.dumps(
                        {"command": "hermes kanban assignees --json"}
                    ),
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                },
            ],
        },
    )

    assert receipt.allowed
    projected = json.loads(authorized["input"][1]["output"])
    assert projected == {
        "exit_code": 0,
        "output": json.dumps(
            {
                "kanban_assignees_projection": "v1",
                "assignees": ["task-orchestrator"],
                "omitted_entries": 2,
            },
            separators=(",", ":"),
        ),
    }
    assert "API_TOKEN" not in json.dumps(authorized)


def test_runtime_does_not_manufacture_boundaries_for_oversized_sanitized_text(
    tmp_path,
):
    agent = _agent(tmp_path)
    agent._llm_egress_max_sanitized_bytes = 128_000
    text = "ordinary bounded repair context. " * 2_000

    with pytest.raises(ValueError, match="sanitized segment exceeds byte cap"):
        authorize_agent_sdk_kwargs(
            agent,
            {"model": "test-model", "messages": [{"role": "system", "content": text}]},
        )


def test_protected_kanban_splits_large_line_bounded_context_without_changing_wire_text(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent._llm_egress_max_sanitized_bytes = 128_000
    text = "\n".join(
        f"source=kanban-task-context line={index} ordinary repair evidence."
        for index in range(900)
    )
    assert len(text.encode("utf-8")) > 32_768

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {"model": "test-model", "messages": [{"role": "system", "content": text}]},
    )

    assert authorized["messages"][0]["content"] == text
    assert json.loads(receipt.payload_bytes)["messages"][0]["content"] == text


def test_protected_kanban_splits_single_oversized_line_without_relaxing_cap(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent._llm_egress_max_sanitized_bytes = 128_000
    text = "ordinary bounded repair context " * 2_000
    assert "\n" not in text
    assert len(text.encode("utf-8")) > 32_768

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {"model": "test-model", "messages": [{"role": "system", "content": text}]},
    )

    assert authorized["messages"][0]["content"] == text
    assert json.loads(receipt.payload_bytes)["messages"][0]["content"] == text


def test_protected_provider_route_splits_without_dispatcher_marker(
    tmp_path, monkeypatch
):
    """Route protection must survive provider/fallback agent reconstruction."""
    monkeypatch.delenv("HERMES_KANBAN_PROTECTED_REMOTE", raising=False)
    agent = _agent(tmp_path)
    agent.provider = "nous"
    agent.model = "tencent/hy3:free"
    agent.base_url = "https://inference-api.nousresearch.com/v1"
    agent._llm_egress_max_sanitized_bytes = 128_000

    text = "bounded protected repair context. " * 2_000
    assert len(text.encode("utf-8")) > 32_768

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {"model": agent.model, "messages": [{"role": "system", "content": text}]},
    )

    assert authorized["messages"][0]["content"] == text
    assert json.loads(receipt.payload_bytes)["messages"][0]["content"] == text


def test_reconstructed_kanban_worker_redacts_paths_without_marker(
    tmp_path, monkeypatch
):
    """Task identity must restore protected redaction after fallback rebuild."""
    monkeypatch.delenv("HERMES_KANBAN_PROTECTED_REMOTE", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_01234567")
    agent = _agent(tmp_path)
    agent.provider = "nous"
    agent.model = "tencent/hy3:free"
    agent.base_url = "https://inference-api.nousresearch.com/v1"
    path = "/Users/private/hermes/worktree/kanban.db"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "messages": [{"role": "system", "content": f"Inspect {path}"}],
        },
    )

    wire = json.loads(receipt.payload_bytes)
    assert path not in authorized["messages"][0]["content"]
    assert "<private-path>" in authorized["messages"][0]["content"]
    assert wire["messages"][0]["content"] == authorized["messages"][0]["content"]


@pytest.mark.parametrize(
    "identifier",
    [
        "github-pr-repair:v2",
        "ci_receipt_not_passing",
        "data-authority-patch-steward",
        "timestamp_coercion_guard",
        "t_498d6a2a",
        "84057c81a75d3ef064ca20e037662dc9b1962904",
    ],
)
def test_protected_kanban_admits_validated_application_identifiers(
    tmp_path, monkeypatch, identifier
):
    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)

    authorized, _ = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "messages": [{"role": "system", "content": f"routing {identifier}"}],
        },
    )

    assert authorized["messages"][0]["content"] == f"routing {identifier}"


def test_protected_kanban_admits_exact_pr_receipt_decomposer_structure(
    tmp_path, monkeypatch
):
    from hermes_cli.kanban_decompose import _SYSTEM_PROMPT, _USER_TEMPLATE

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    monkeypatch.setenv("HERMES_CONTROL_HOME", "/Users/operator/.hermes")
    agent = _agent(tmp_path)
    lower_sha = "8ea9309f1c38ac8da8064e16acae05da86ba2df4"
    upper_sha = "D41A011C51B41FE599440426624C8EE49D256C14"
    receipt_sha = (
        "0123456789ABCDEF0123456789ABCDEF"
        "0123456789ABCDEF0123456789ABCDEF"
    )
    body = (
        "Run `git status --short --branch`, then `git rev-parse --verify HEAD`. "
        "Fetch --no-recurse-submodules from https://github.com/acme/widget.git. "
        f"Require base {lower_sha}, head {upper_sha}, and receipt {receipt_sha}. "
        "Acknowledge with /Users/operator/.hermes/hermes-agent/venv/bin/python."
    )
    roster = (
        "  - pr-repair-steward: compare before/after evidence for "
        "equities/options and unit/static checks"
    )
    user_prompt = _USER_TEMPLATE.format(
        task_id="t_ff23ef8a",
        title="PR repair: acme/widget#103",
        body=body,
        handoffs="(no root handoff comments were recorded)",
        roster=roster,
        default_assignee="pr-repair-steward",
    )

    authorized, _ = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        },
    )

    assert authorized["messages"][0] == {
        "role": "system",
        "content": _SYSTEM_PROMPT,
    }
    authorized_user = authorized["messages"][1]["content"]
    assert "/Users/operator" not in authorized_user
    assert "$HERMES_CONTROL_HOME/hermes-agent/venv/bin/python" in authorized_user
    assert lower_sha in authorized_user
    assert upper_sha in authorized_user
    assert receipt_sha in authorized_user


@pytest.mark.parametrize(
    ("unsafe_text", "reason"),
    [
        ("c2VjcmV0LXBheWxvYWQ=", "base64_payload"),
        ("SECRET_TOKEN=super-secret-value", "secret_detected"),
        ("AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPP", "base64_payload"),
        (
            "raw review source: def _approved_sanitized_segments(value): "
            "return provider/runtime",
            "base64_payload",
        ),
    ],
)
def test_protected_kanban_pr_receipt_lexical_exceptions_remain_fail_closed(
    tmp_path, monkeypatch, unsafe_text, reason
):
    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)

    with pytest.raises(EgressBlocked) as exc_info:
        authorize_agent_sdk_kwargs(
            agent,
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": unsafe_text}],
            },
        )

    assert reason in exc_info.value.decision.reason_codes


def test_runtime_dispatches_exactly_once_with_authorized_bytes(tmp_path):
    agent = _agent(tmp_path)
    calls = []
    result = dispatch_authorized_agent_request(
        agent,
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Fix CI now."}],
        },
        lambda request: calls.append(request) or "ok",
    )
    assert result == "ok"
    assert calls == [
        {
            "messages": [{"content": "Fix CI now.", "role": "user"}],
            "model": "test-model",
        }
    ]


def test_provider_callback_cannot_mutate_authorized_request(tmp_path):
    agent = _agent(tmp_path)

    def mutate(request):
        request["messages"] = [{"role": "user", "content": "replacement"}]

    with pytest.raises(TypeError):
        dispatch_authorized_agent_request(
            agent,
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Fix CI now."}],
            },
            mutate,
        )


def test_protected_kanban_runtime_sanitizes_tool_paths_before_egress(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "managed" / "t_12345678"
    profile_home = tmp_path / "profiles" / "worker"
    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    agent = _agent(tmp_path / "egress")

    authorized, _ = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "tool",
                    "content": (
                        f"pwd={workspace} home={profile_home} "
                        "other=/Users/private/repository/file.py"
                    ),
                }
            ],
        },
    )

    content = authorized["messages"][0]["content"]
    assert str(tmp_path) not in content
    assert "pwd=." in content
    assert "$HERMES_PROFILE_HOME" in content
    assert "<private-path>" in content


def test_protected_kanban_runtime_does_not_hide_encoded_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)

    with pytest.raises((EgressBlocked, SanitizedTextRejected)):
        authorize_agent_sdk_kwargs(
            agent,
            {
                "model": "test-model",
                "messages": [{"role": "tool", "content": "c2VjcmV0LXBheWxvYWQ="}],
            },
        )


@pytest.mark.parametrize(
    "output",
    [
        "https://github.com/acme/widget.git",
        "refs/heads/codex/fix-135",
        "a" * 40,
        "b" * 64,
    ],
)
def test_protected_kanban_admits_bounded_generic_terminal_stdout(
    tmp_path, monkeypatch, output
):
    """Generic terminal output cannot deadlock a protected cloud worker."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    kwargs = {
        "model": "test-model",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_terminal123",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_terminal123",
                "content": output,
            },
        ],
    }

    authorized, receipt = authorize_agent_sdk_kwargs(agent, kwargs)

    assert receipt.allowed
    assert json.loads(authorized["messages"][1]["content"]) == {
        "terminal_result": "completed",
        "exit_code": None,
        "raw_output": "omitted_from_remote_replay",
    }
    assert output not in authorized["messages"][1]["content"]
    assert receipt.decision.source_segment_count == 0


def test_protected_kanban_search_result_projects_source_content_to_locations(
    tmp_path, monkeypatch
):
    """Remote workers get bounded search locations, then exact reads.

    A repository search can validly match base64-like fixture/code text.  That
    must not be replayed as untrusted provider input, but the file and line
    location remain enough for the worker to request a provenance-bound read.
    """

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "nous"
    result = json.dumps({
        "total_count": 1,
        "matches": [{
            "path": "tests/fixture.py",
            "line": 42,
            "content": "encoded = 'c2VjcmV0LXBheWxvYWQ='",
        }],
    })

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_search123",
                            "type": "function",
                            "function": {
                                "name": "search_files",
                                "arguments": '{"pattern":"encoded"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_search123",
                    "content": result,
                },
            ],
        },
    )

    rendered = json.loads(authorized["messages"][1]["content"])
    assert receipt.allowed
    assert rendered == {
        "search_files_projection": "locations-v1",
        "total_count": 1,
        "matches": [{"path": "tests/fixture.py", "line": 42}],
    }


def test_protected_kanban_tool_search_result_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "nous"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_tool_search_projection",
                        "type": "function",
                        "function": {
                            "name": "tool_search",
                            "arguments": '{"query":"github feedback"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_tool_search_projection",
                    "content": '{"total_available":1,"results":[]}',
                },
            ],
        },
    )

    assert receipt.allowed
    assert authorized["messages"][1]["content"].startswith(
        "tool_search completed locally."
    )


def test_protected_kanban_tool_describe_result_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "nous"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_tool_describe_projection",
                        "type": "function",
                        "function": {
                            "name": "tool_describe",
                            "arguments": '{"names":["github_pr_feedback"]}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_tool_describe_projection",
                    "content": (
                        '{"tools":{"github_pr_feedback":'
                        '{"description":"post a governed review",'
                        '"parameters":{"type":"object"}}}}'
                    ),
                },
            ],
        },
    )

    assert receipt.allowed
    assert authorized["messages"][1]["content"].startswith(
        "tool_search completed locally."
    )


def test_protected_kanban_search_file_listing_projects_safe_relative_paths(
    tmp_path, monkeypatch
):
    """File discovery must return paths that can be provenance-read next."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "nous"
    result = json.dumps({
        "total_count": 3,
        "files": [
            "./pyproject.toml",
            "tests/test_runner.py",
            ".github/workflows/ci.yml",
            ".git/FETCH_HEAD",
        ],
    })

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_search_files_listing",
                        "type": "function",
                        "function": {
                            "name": "search_files",
                            "arguments": '{"pattern":"*.toml","output_mode":"files"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_search_files_listing",
                    "content": result,
                },
            ],
        },
    )

    assert receipt.allowed
    assert json.loads(authorized["messages"][1]["content"]) == {
        "search_files_projection": "locations-v1",
        "total_count": 3,
        "files": [
            "pyproject.toml",
            "tests/test_runner.py",
            ".github/workflows/ci.yml",
        ],
    }


def test_protected_kanban_keeps_opaque_tool_protocol_identifiers(tmp_path, monkeypatch):
    """Opaque provider call IDs are protocol linkage, not remote text."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "nous"
    opaque_id = "A" * 64
    result = json.dumps({"total_count": 1, "files": [".git/FETCH_HEAD"]})

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": opaque_id,
                            "type": "function",
                            "function": {
                                "name": "search_files",
                                "arguments": '{"pattern":"needle"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": opaque_id,
                    "content": result,
                },
            ],
        },
    )

    assert receipt.allowed
    assert authorized["messages"][0]["tool_calls"][0]["id"] == opaque_id
    assert authorized["messages"][1]["tool_call_id"] == opaque_id
    assert json.loads(authorized["messages"][1]["content"]) == {
        "search_files_projection": "locations-v1",
        "total_count": 1,
    }


def test_protected_kanban_redacts_nested_terminal_argument_replay(
    tmp_path, monkeypatch
):
    """Nested chat-completions terminal arguments cannot carry opaque text."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "nous"
    opaque_command = "git show c2VjcmV0LXBheWxvYWQ="

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_terminal_nested",
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": json.dumps({"command": opaque_command}),
                            },
                        }
                    ],
                }
            ],
        },
    )

    assert receipt.allowed
    replay = authorized["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert "git show" in replay
    assert "<redacted-base64>" in replay
    assert opaque_command not in replay


def test_protected_kanban_elides_scratch_read_without_source_metadata(
    tmp_path, monkeypatch
):
    """Worker-created scratch files cannot abort a remote task on provenance."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "nous"
    call_id = "call_scratch_read"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"/tmp/kanban_list.txt"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_name": "read_file",
                    "tool_call_id": call_id,
                    "content": "untrusted c2VjcmV0LXBheWxvYWQ=",
                    "_source_provenance": {"presentation_kind": "forged"},
                },
            ],
        },
    )

    assert receipt.allowed
    assert authorized["messages"][1]["content"] == (
        "read_file completed locally, but its raw content cannot be replayed on "
        "this protected route. Request only the needed narrow range again."
    )


def test_protected_kanban_redacts_readonly_search_argument_replay(
    tmp_path, monkeypatch
):
    """A base64-shaped search token cannot deadlock its already-run tool call."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "nous"
    arguments = json.dumps({"pattern": "DISABLE", "path": "src"})

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "input": [
                {
                    "type": "function_call",
                    "name": "search_files",
                    "call_id": "call_search_argument123",
                    "arguments": arguments,
                },
            ],
        },
    )

    assert receipt.allowed
    assert "DISABLE" not in authorized["input"][0]["arguments"]
    assert "<redacted-base64>" in authorized["input"][0]["arguments"]


def test_protected_kanban_redacts_nested_readonly_search_argument_replay(
    tmp_path, monkeypatch
):
    """Chat-completions function arguments take the same replay-only path."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "nous"
    arguments = json.dumps({"pattern": "DISABLE", "path": "src"})

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_search_argument123",
                            "type": "function",
                            "function": {
                                "name": "search_files",
                                "arguments": arguments,
                            },
                        }
                    ],
                }
            ],
        },
    )

    replayed = authorized["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert receipt.allowed
    assert "DISABLE" not in replayed
    assert "<redacted-base64>" in replayed


def test_protected_kanban_projects_readonly_git_grep_result_to_locations(
    tmp_path, monkeypatch
):
    """A bounded line-number search must not replay matching source text."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "nous"
    arguments = json.dumps(
        {"command": "git grep -n -E 'paper|safety' -- README.md live_runner.py"}
    )
    output = json.dumps(
        {
            "output": "README.md:42:c2VjcmV0LXBheWxvYWQ=\nlive_runner.py:8:PAPER_MODE = True\n",
            "exit_code": 0,
        }
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "input": [
                {
                    "type": "function_call",
                    "name": "terminal",
                    "call_id": "call_git_grep_123",
                    "arguments": arguments,
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_git_grep_123",
                    "output": output,
                },
            ],
        },
    )

    assert receipt.allowed
    rendered = json.loads(authorized["input"][1]["output"])
    assert rendered == {
        "exit_code": 0,
        "output": json.dumps(
            {
                "git_grep_locations": "locations-v1",
                "matches": [
                    {"path": "README.md", "line": 42},
                    {"path": "live_runner.py", "line": 8},
                ],
            },
            separators=(",", ":"),
        ),
    }


def test_protected_kanban_projects_readonly_rg_result_to_locations(
    tmp_path, monkeypatch
):
    """A no-context line-number search must not replay matching source text."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "nous"
    arguments = json.dumps({"command": "rg -n 'paper|safety' README.md live_runner.py"})
    output = json.dumps(
        {
            "output": "README.md:42:c2VjcmV0LXBheWxvYWQ=\nlive_runner.py:8:PAPER_MODE = True\n",
            "exit_code": 0,
        }
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "input": [
                {
                    "type": "function_call",
                    "name": "terminal",
                    "call_id": "call_rg_123",
                    "arguments": arguments,
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_rg_123",
                    "output": output,
                },
            ],
        },
    )

    assert receipt.allowed
    rendered = json.loads(authorized["input"][1]["output"])
    assert "c2VjcmV0LXBheWxvYWQ=" not in receipt.payload_bytes.decode("utf-8")
    assert rendered["output"] == json.dumps(
        {
            "git_grep_locations": "locations-v1",
            "matches": [
                {"path": "README.md", "line": 42},
                {"path": "live_runner.py", "line": 8},
            ],
        },
        separators=(",", ":"),
    )


def test_protected_kanban_projects_context_rg_result_to_locations(
    tmp_path, monkeypatch
):
    """Context search output must not block a protected worker or leak source."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "nous"
    arguments = json.dumps({"command": "rg -n -C 2 'paper|safety' README.md"})
    output = json.dumps(
        {
            "output": (
                "README.md-40-context c2VjcmV0LXBheWxvYWQ=\n"
                "README.md:42:matched c2VjcmV0LXBheWxvYWQ=\n"
                "README.md-44-more context c2VjcmV0LXBheWxvYWQ=\n"
            ),
            "exit_code": 0,
        }
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "input": [
                {
                    "type": "function_call",
                    "name": "terminal",
                    "call_id": "call_context_rg_123",
                    "arguments": arguments,
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_context_rg_123",
                    "output": output,
                },
            ],
        },
    )

    assert receipt.allowed
    rendered = json.loads(authorized["input"][1]["output"])
    assert rendered["output"] == json.dumps(
        {
            "git_grep_locations": "locations-v1",
            "matches": [{"path": "README.md", "line": 42}],
            "omitted_matches": 2,
        },
        separators=(",", ":"),
    )


def test_protected_kanban_projects_exact_git_workspace_diagnostic(
    tmp_path, monkeypatch
):
    """Commit subjects are not replayed as trusted cloud context."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "nous"
    output = "/private/worktree\nfix: c2VjcmV0LXBheWxvYWQ=\n"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_git_workspace123",
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": json.dumps({
                                    "command": (
                                        "git rev-parse --show-toplevel && "
                                        "git branch --show-current && "
                                        "git log --oneline -5"
                                    )
                                }),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_git_workspace123",
                    "content": output,
                },
            ],
        },
    )

    assert receipt.allowed
    assert authorized["messages"][1]["content"] == (
        "git workspace diagnostic completed locally; raw paths and commit "
        "subjects were omitted from remote replay."
    )


def test_protected_terminal_file_bytes_are_bounded_non_source_context(
    tmp_path, monkeypatch
):
    """Normal terminal reads must reach the approved cloud worker.

    Without a read_file grant this remains non-source context, so it cannot
    silently acquire source authority during serialization.
    """

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    innocent_source = "def calculate_total(items):\n    return sum(items)\n"

    calls = []
    dispatch_authorized_agent_request(
        agent,
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_terminal_file_read",
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": '{"command":"cat internal_source.py"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_terminal_file_read",
                    "content": innocent_source,
                },
            ],
        },
        lambda request: calls.append(request),
    )

    assert calls == [{
        "model": "test-model",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_terminal_file_read",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command":"cat internal_source.py"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_terminal_file_read",
                "content": innocent_source,
            },
        ],
    }]


def test_exact_applied_secret_is_denied_at_final_provider_boundary(
    tmp_path, monkeypatch
):
    from hermes_cli import env_loader

    home = tmp_path / "profile-home"
    home.mkdir()
    secret = "purple-lantern-river-cobalt"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(home.resolve()),
        {"EXTERNAL_VALUE": secret},
    )
    agent = _agent(tmp_path / "egress")
    calls = []

    with pytest.raises(EgressBlocked) as exc_info:
        dispatch_authorized_agent_request(
            agent,
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": f"result: {secret}"}],
            },
            lambda request: calls.append(request),
        )

    assert "exact_secret_detected" in exc_info.value.decision.reason_codes
    assert calls == []


def test_tool_syntax_without_recognized_terminal_call_remains_blocked(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)

    with pytest.raises((EgressBlocked, SanitizedTextRejected)):
        authorize_agent_sdk_kwargs(
            agent,
            {
                "model": "test-model",
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "call_unbound123",
                        "content": (
                            "https://github.com/acme/widget.git "
                            "token=opaqueValue123456789"
                        ),
                    }
                ],
            },
        )


def test_recognized_terminal_syntax_does_not_exempt_adjacent_base64(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)

    with pytest.raises((EgressBlocked, SanitizedTextRejected)):
        authorize_agent_sdk_kwargs(
            agent,
            {
                "model": "test-model",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_terminal123",
                                "type": "function",
                                "function": {"name": "terminal", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_terminal123",
                        "content": "--branch c2VjcmV0LXBheWxvYWQ=",
                    },
                ],
            },
        )


def test_recognized_terminal_diagnostic_atoms_do_not_trigger_base64_blocks(
    tmp_path, monkeypatch
):
    """Verified local terminal output may retain non-encoded code evidence."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    output = "PASS _SCHWAB_PARENT_SEED_ASSEMBLER line 5243"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_terminal123",
                            "type": "function",
                            "function": {"name": "terminal", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_terminal123",
                    "content": output,
                },
            ],
        },
    )

    assert receipt.allowed
    assert authorized["messages"][1]["content"] == output


def test_protected_kanban_admits_bounded_codex_function_output(
    tmp_path, monkeypatch
):
    """Responses API tool output follows the same usable cloud path."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    kwargs = {
        "model": "test-model",
        "input": [
            {
                "id": "call_terminal123",
                "call_id": "call_terminal123",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            },
            {
                "type": "function_call_output",
                "call_id": "call_terminal123",
                "output": "https://github.com/acme/widget.git\nworking tree clean",
            },
        ],
    }

    authorized, receipt = authorize_agent_sdk_kwargs(agent, kwargs)

    assert receipt.allowed
    assert authorized["input"][1]["output"] == kwargs["input"][1]["output"]
    assert receipt.decision.source_segment_count == 0


def test_codex_responses_restores_source_metadata_by_call_id_and_hash():
    """Responses results get only their exact sidecar provenance copy."""

    call_id = "call_read_file_123"
    result = '{"path":"source.py","content":"1|safe = True"}'
    sidecar = [
        {
            "message_index": 0,
            "tool_call_id": call_id,
            "content_sha256": sha256(result.encode("utf-8")).hexdigest(),
            "request_id": "turn-1:api:1",
            "source_grant_digests": ["grant-digest"],
            "presentation_kind": "read_file_json_v1",
        }
    ]
    output = {
        "type": "function_call_output",
        "call_id": call_id,
        "output": result,
    }
    body = {"input": [{"type": "function_call", "call_id": call_id}, output]}
    original_input = body["input"]
    sidecar_before = [dict(entry) for entry in sidecar]

    restored = _restore_source_provenance_sidecar(body, sidecar)

    assert restored["input"] is not original_input
    assert restored["input"][1] is not output
    assert restored["input"][1]["_source_provenance"] == {
        "content_sha256": sha256(result.encode("utf-8")).hexdigest(),
        "request_id": "turn-1:api:1",
        "source_grant_digests": ["grant-digest"],
        "presentation_kind": "read_file_json_v1",
    }
    assert "_source_provenance" not in output
    assert body["input"] is original_input
    assert sidecar == sidecar_before


@pytest.mark.parametrize("mutation", ("wrong_call", "wrong_hash"))
def test_codex_responses_does_not_restore_ambiguous_or_mismatched_metadata(mutation):
    """A mismatched sidecar never grants a Responses result provenance."""

    result = '{"path":"source.py","content":"1|safe = True"}'
    call_id = "call_read_file_123"
    sidecar = [
        {
            "message_index": 0,
            "tool_call_id": "call_other" if mutation == "wrong_call" else call_id,
            "content_sha256": (
                "0" * 64
                if mutation == "wrong_hash"
                else sha256(result.encode("utf-8")).hexdigest()
            ),
            "request_id": "turn-1:api:1",
            "source_grant_digests": ["grant-digest"],
            "presentation_kind": "read_file_json_v1",
        }
    ]
    body = {
        "input": [
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            }
        ]
    }

    restored = _restore_source_provenance_sidecar(body, sidecar)

    assert "_source_provenance" not in restored["input"][0]


def test_codex_responses_does_not_restore_ambiguous_duplicate_result():
    """One sidecar entry cannot bind more than one identical Responses result."""

    call_id = "call_read_file_123"
    result = '{"path":"source.py","content":"1|safe = True"}'
    sidecar = [
        {
            "message_index": 0,
            "tool_call_id": call_id,
            "content_sha256": sha256(result.encode("utf-8")).hexdigest(),
            "request_id": "turn-1:api:1",
            "source_grant_digests": ["grant-digest"],
            "presentation_kind": "read_file_json_v1",
        }
    ]
    body = {
        "input": [
            {"type": "function_call_output", "call_id": call_id, "output": result},
            {"type": "function_call_output", "call_id": call_id, "output": result},
        ]
    }

    restored = _restore_source_provenance_sidecar(body, sidecar)

    assert all("_source_provenance" not in item for item in restored["input"])


def test_codex_responses_restores_source_metadata_from_one_input_text_result():
    """The canonical one-item Responses text array binds to its sidecar."""

    call_id = "call_read_file_123"
    result = '{"path":"source.py","content":"1|safe = True"}'
    body = {
        "input": [
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": [{"type": "input_text", "text": result}],
            }
        ]
    }
    sidecar = [
        {
            "message_index": 0,
            "tool_call_id": call_id,
            "content_sha256": sha256(result.encode("utf-8")).hexdigest(),
            "request_id": "turn-1:api:1",
            "source_grant_digests": ["grant-digest"],
            "presentation_kind": "read_file_json_v1",
        }
    ]

    restored = _restore_source_provenance_sidecar(body, sidecar)

    assert restored["input"][0]["_source_provenance"]["content_sha256"] == sidecar[0][
        "content_sha256"
    ]


def test_codex_responses_consumes_sidecar_once_across_input_carriers():
    """One source record cannot authorize both normalized input carriers."""

    call_id = "call_read_file_123"
    result = '{"path":"source.py","content":"1|safe = True"}'
    output = {"type": "function_call_output", "call_id": call_id, "output": result}
    sidecar = [
        {
            "message_index": 0,
            "tool_call_id": call_id,
            "content_sha256": sha256(result.encode("utf-8")).hexdigest(),
            "request_id": "turn-1:api:1",
            "source_grant_digests": ["grant-digest"],
            "presentation_kind": "read_file_json_v1",
        }
    ]
    body = {"input": [dict(output)], "extra_body": {"input": [dict(output)]}}

    restored = _restore_source_provenance_sidecar(body, sidecar)

    assert "_source_provenance" in restored["input"][0]
    assert "_source_provenance" not in restored["extra_body"]["input"][0]


@pytest.mark.parametrize(
    "output",
    (
        [],
        [
            {"type": "input_text", "text": '{"content":"safe"}'},
            {"type": "input_text", "text": '{"content":"also safe"}'},
        ],
        [{"type": "input_image", "image_url": "https://example.test/image"}],
        {"type": "input_text", "text": '{"content":"safe"}'},
        [{"type": "input_text"}],
        [{"type": "input_text", "text": 1}],
        [{"type": "output_text", "text": '{"content":"safe"}'}],
    ),
)
def test_codex_responses_does_not_restore_ambiguous_or_malformed_output(output):
    """Only one exact input_text item is eligible for structured restoration."""

    call_id = "call_read_file_123"
    result = '{"content":"safe"}'
    sidecar = [
        {
            "message_index": 0,
            "tool_call_id": call_id,
            "content_sha256": sha256(result.encode("utf-8")).hexdigest(),
            "request_id": "turn-1:api:1",
            "source_grant_digests": ["grant-digest"],
            "presentation_kind": "read_file_json_v1",
        }
    ]
    body = {
        "input": [
            {"type": "function_call_output", "call_id": call_id, "output": output}
        ]
    }

    restored = _restore_source_provenance_sidecar(body, sidecar)

    assert "_source_provenance" not in restored["input"][0]


def test_source_provenance_sidecar_keeps_chat_restoration_compatible():
    """Existing Chat results retain their exact content-bound restoration."""

    content = '{"path":"source.py","content":"1|safe = True"}'
    body = {
        "messages": [
            {"role": "tool", "tool_call_id": "call_read_file_123", "content": content}
        ]
    }
    sidecar = [
        {
            "message_index": 0,
            "tool_call_id": "call_read_file_123",
            "content_sha256": sha256(content.encode("utf-8")).hexdigest(),
            "request_id": "turn-1:api:1",
            "source_grant_digests": ["grant-digest"],
            "presentation_kind": "read_file_json_v1",
        }
    ]

    restored = _restore_source_provenance_sidecar(body, sidecar)

    assert restored["messages"][0]["_source_provenance"]["content_sha256"] == sidecar[0][
        "content_sha256"
    ]
    assert "_source_provenance" not in body["messages"][0]


@pytest.fixture
def non_scratch_source_root():
    """Provide a real source root that never matches the scratch-read policy."""

    checkout_root = Path.cwd().resolve()
    with tempfile.TemporaryDirectory(
        prefix="hermes-codex-responses-", dir=checkout_root
    ) as root:
        source_root = Path(root)
        assert source_root.is_relative_to(checkout_root)
        yield source_root


@pytest.fixture
def scratch_source_root():
    """Provide a real source root that the scratch-read policy must elide."""

    for parent in (Path("/private/tmp"), Path("/tmp")):
        try:
            with tempfile.TemporaryDirectory(
                prefix="hermes-codex-responses-", dir=parent
            ) as root:
                root_path = Path(root)
                if str(root_path).startswith(("/tmp/", "/private/tmp/")):
                    yield root_path
                    return
        except OSError:
            continue
    raise RuntimeError("no writable scratch temporary root")


def _real_codex_read_file_replay(
    tmp_path,
    monkeypatch,
    *,
    source_root: Path,
    call_id: str,
    structured_output: bool = False,
):
    """Build one live-provenance read replay through the real converter."""

    from agent.codex_responses_adapter import _chat_messages_to_responses_input
    from agent.source_provenance_tools import (
        attach_trusted_source_provenance_metadata,
        build_source_provenance_sidecar,
        source_provenance_activation,
    )
    from agent.tool_dispatch_helpers import make_tool_result_message
    from tools.file_tools import read_file_tool

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    source = source_root / "source.py"
    source.write_text("safe = True\n", encoding="utf-8")
    try:
        source_argument = str(source.relative_to(Path.cwd()))
    except ValueError:
        source_argument = str(source)
    if source_root.is_relative_to(Path.cwd()):
        assert not source_argument.startswith(("/tmp/", "/private/tmp/"))
    agent = _agent(tmp_path / "egress")
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    agent._current_api_request_id = "turn-1:api:1"
    with source_provenance_activation(agent, "read_file"):
        result = read_file_tool(source_argument, task_id="codex-responses-read")
    metadata = attach_trusted_source_provenance_metadata(
        agent, "read_file", content=result
    )
    tool_result = make_tool_result_message(
        "read_file", result, call_id, source_provenance=metadata
    )
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": source_argument}),
                    },
                }
            ],
        },
        tool_result,
    ]
    sidecar = build_source_provenance_sidecar(messages)
    assert sidecar, result
    if structured_output:
        messages = [
            messages[0],
            {
                **tool_result,
                "content": [{"type": "input_text", "text": result}],
            },
        ]
    input_items = _chat_messages_to_responses_input(messages)
    agent._current_api_request_id = "turn-1:api:2"
    return agent, result, input_items, sidecar


@pytest.mark.parametrize("mutation", ("stale_request", "forged_grant"))
def test_protected_codex_responses_rejects_stale_sidecar_before_source_replay(
    tmp_path, monkeypatch, non_scratch_source_root, mutation
):
    """A matching hash cannot bypass source validation with a stale envelope."""

    agent, _result, input_items, sidecar = _real_codex_read_file_replay(
        tmp_path,
        monkeypatch,
        source_root=non_scratch_source_root,
        call_id="call_read_file_123",
    )
    if mutation == "stale_request":
        sidecar = [{**sidecar[0], "request_id": "stale-request"}]
    else:
        sidecar = [{**sidecar[0], "source_grant_digests": ["forged-grant"]}]

    with pytest.raises(EgressBlocked) as exc_info:
        authorize_agent_sdk_kwargs(
            agent,
            {
                "model": agent.model,
                "input": input_items,
                "_hermes_source_provenance": sidecar,
            },
        )

    assert "untrusted_provenance" in exc_info.value.decision.reason_codes


def test_real_read_file_responses_replays_one_input_text_with_source_segment(
    tmp_path, monkeypatch, non_scratch_source_root
):
    """One canonical input_text item retains shape after validated replay."""

    agent, result, input_items, sidecar = _real_codex_read_file_replay(
        tmp_path,
        monkeypatch,
        source_root=non_scratch_source_root,
        call_id="call_read_file_123",
        structured_output=True,
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": input_items,
            "_hermes_source_provenance": sidecar,
        },
    )

    assert authorized["input"][1]["output"] == [
        {"type": "input_text", "text": result}
    ]
    assert "_source_provenance" not in json.dumps(authorized)
    assert "_hermes_source_provenance" not in authorized
    assert receipt.decision.source_grant_count == 1
    assert receipt.decision.source_segment_count == 1


@pytest.mark.parametrize(
    "call_id",
    (
        "codex_mcp__hermes-tools__read_file_exec-" + "0" * 64,
        "call_read_file_123|fc_responses_item_123",
    ),
)
def test_real_read_file_responses_replays_real_converted_effective_call_id(
    tmp_path, monkeypatch, non_scratch_source_root, call_id
):
    """Sidecar identity follows the actual clamped/split Responses call ID."""

    agent, result, input_items, sidecar = _real_codex_read_file_replay(
        tmp_path,
        monkeypatch,
        source_root=non_scratch_source_root,
        call_id=call_id,
    )

    assert input_items[0]["call_id"] == input_items[1]["call_id"]
    assert input_items[1]["call_id"] != call_id
    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": input_items,
            "_hermes_source_provenance": sidecar,
        },
    )

    assert authorized["input"][1]["output"] == result
    assert "_source_provenance" not in json.dumps(authorized)
    assert receipt.decision.source_grant_count == 1
    assert receipt.decision.source_segment_count == 1


def test_protected_codex_responses_elides_duplicate_sidecar_records(
    tmp_path, monkeypatch, non_scratch_source_root
):
    """Ambiguous sidecar records are ignored and leave the read outcome-only."""

    agent, result, input_items, sidecar = _real_codex_read_file_replay(
        tmp_path,
        monkeypatch,
        source_root=non_scratch_source_root,
        call_id="call_read_file_123",
    )
    sidecar = [sidecar[0], {**sidecar[0], "request_id": "stale-request"}]

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": input_items,
            "_hermes_source_provenance": sidecar,
        },
    )

    assert receipt.allowed
    assert authorized["input"][1]["output"] == (
        "read_file completed locally, but its raw content cannot be replayed on "
        "this protected route. Request only the needed narrow range again."
    )
    assert result not in json.dumps(authorized)
    assert receipt.decision.source_segment_count == 0


def test_responses_read_file_fails_closed_on_policy_mismatch(
    tmp_path, monkeypatch, non_scratch_source_root
):
    """An exact prior-policy read cannot cross a later egress policy boundary."""

    agent, _result, input_items, sidecar = _real_codex_read_file_replay(
        tmp_path,
        monkeypatch,
        source_root=non_scratch_source_root,
        call_id="call_read_file_123",
    )
    agent._llm_egress_policy_digest = sha256(b"policy-2").hexdigest()

    with pytest.raises(EgressBlocked) as exc_info:
        authorize_agent_sdk_kwargs(
            agent,
            {
                "model": agent.model,
                "input": input_items,
                "_hermes_source_provenance": sidecar,
            },
        )

    assert "grant_binding_mismatch" in exc_info.value.decision.reason_codes
    assert "untrusted_provenance" in exc_info.value.decision.reason_codes


def test_real_scratch_read_file_responses_is_elided(
    tmp_path, monkeypatch, scratch_source_root
):
    """A genuine scratch read stays elided after real provenance/conversion flow."""

    agent, result, input_items, sidecar = _real_codex_read_file_replay(
        tmp_path,
        monkeypatch,
        source_root=scratch_source_root,
        call_id="call_read_file_123",
    )
    assert str(scratch_source_root).startswith(("/tmp/", "/private/tmp/"))

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": input_items,
            "_hermes_source_provenance": sidecar,
        },
    )

    assert receipt.allowed
    assert authorized["input"][1]["output"] == (
        "read_file completed locally, but its raw content cannot be replayed on "
        "this protected route. Request only the needed narrow range again."
    )
    assert result not in json.dumps(authorized)
    assert receipt.decision.source_segment_count == 0


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_sidecar",
        "wrong_call",
        "wrong_hash",
        "duplicate_result",
    ),
)
def test_responses_read_file_elides_when_sidecar_is_unattached(
    tmp_path, monkeypatch, non_scratch_source_root, mutation
):
    """Ignored Responses sidecars cannot turn an otherwise elidable read into a block."""

    agent, result, input_items, sidecar = _real_codex_read_file_replay(
        tmp_path,
        monkeypatch,
        source_root=non_scratch_source_root,
        call_id="call_read_file_123",
    )
    if mutation == "missing_sidecar":
        sidecar = []
    elif mutation == "wrong_call":
        sidecar = [{**sidecar[0], "tool_call_id": "call_other"}]
    elif mutation == "wrong_hash":
        sidecar = [{**sidecar[0], "content_sha256": "0" * 64}]
    else:
        input_items.append(dict(input_items[1]))

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": input_items,
            "_hermes_source_provenance": sidecar,
        },
    )

    assert receipt.allowed
    assert all(
        item.get("output")
        == "read_file completed locally, but its raw content cannot be replayed on "
        "this protected route. Request only the needed narrow range again."
        for item in authorized["input"]
        if item.get("type") == "function_call_output"
    )
    assert result not in json.dumps(authorized)
    assert receipt.decision.source_segment_count == 0


@pytest.mark.parametrize(
    "mutation",
    (
        "changed_content",
        "changed_line_numbering",
        "truncated_json",
    ),
)
def test_responses_read_file_fails_closed_for_unbound_or_changed_presentations(
    tmp_path, monkeypatch, non_scratch_source_root, mutation
):
    """Attached provenance still blocks changed Responses presentations."""

    agent, result, input_items, sidecar = _real_codex_read_file_replay(
        tmp_path,
        monkeypatch,
        source_root=non_scratch_source_root,
        call_id="call_read_file_123",
    )
    changed = json.loads(result)
    if mutation == "changed_content":
        changed["content"] = "1|changed = True\n2|"
        output = json.dumps(changed)
    elif mutation == "changed_line_numbering":
        changed["content"] = "2|safe = True\n3|"
        output = json.dumps(changed)
    else:
        output = result[:-1]
    input_items[1] = {**input_items[1], "output": output}
    sidecar = [
        {
            **sidecar[0],
            "content_sha256": sha256(output.encode("utf-8")).hexdigest(),
        }
    ]

    with pytest.raises(EgressBlocked) as exc_info:
        authorize_agent_sdk_kwargs(
            agent,
            {
                "model": agent.model,
                "input": input_items,
                "_hermes_source_provenance": sidecar,
            },
        )

    assert "untrusted_provenance" in exc_info.value.decision.reason_codes


def test_responses_read_file_elides_after_sidecar_is_consumed(
    tmp_path, monkeypatch, non_scratch_source_root
):
    """A consumed sidecar leaves the second Responses result outcome-only."""

    agent, result, input_items, sidecar = _real_codex_read_file_replay(
        tmp_path,
        monkeypatch,
        source_root=non_scratch_source_root,
        call_id="call_read_file_123",
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "messages": [
                {"role": "assistant", "content": ""},
                {
                    "role": "tool",
                    "tool_call_id": sidecar[0]["tool_call_id"],
                    "content": result,
                }
            ],
            "input": input_items,
            "_hermes_source_provenance": sidecar,
        },
    )

    assert receipt.allowed
    assert authorized["input"][1]["output"] == (
        "read_file completed locally, but its raw content cannot be replayed on "
        "this protected route. Request only the needed narrow range again."
    )
    assert result not in json.dumps(authorized)
    assert receipt.decision.source_segment_count == 0


@pytest.mark.parametrize(
    "output",
    (
        lambda result: [
            {"type": "input_text", "text": result},
            {"type": "input_text", "text": "second item"},
        ],
        lambda result: [
            {"type": "input_text", "text": result},
            {"type": "input_image", "image_url": "https://example.test/image"},
        ],
        lambda _result: [{"type": "input_text"}],
    ),
)
def test_responses_read_file_fails_closed_for_malformed_or_media_output(
    tmp_path, monkeypatch, non_scratch_source_root, output
):
    """Composite, media-bearing, and malformed Responses output stays elided."""

    agent, result, input_items, sidecar = _real_codex_read_file_replay(
        tmp_path,
        monkeypatch,
        source_root=non_scratch_source_root,
        call_id="call_read_file_123",
    )
    input_items[1] = {**input_items[1], "output": output(result)}

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": input_items,
            "_hermes_source_provenance": sidecar,
        },
    )

    assert receipt.allowed
    assert authorized["input"][1]["output"] == (
        "read_file completed locally, but its raw content cannot be replayed on "
        "this protected route. Request only the needed narrow range again."
    )
    assert result not in json.dumps(authorized)
    assert receipt.decision.source_segment_count == 0


def test_responses_source_metadata_is_internal_and_audit_is_content_free(
    tmp_path, monkeypatch, non_scratch_source_root
):
    """Exact source reaches only the provider payload, never private metadata or audit."""

    agent, result, input_items, sidecar = _real_codex_read_file_replay(
        tmp_path,
        monkeypatch,
        source_root=non_scratch_source_root,
        call_id="call_read_file_123",
    )

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": agent.model,
            "input": input_items,
            "_hermes_source_provenance": sidecar,
        },
    )

    assert authorized["input"][1]["output"] == result
    assert "_source_provenance" not in json.dumps(authorized)
    assert "_hermes_source_provenance" not in authorized
    audit = (agent._llm_egress_state_dir / "llm-egress-receipts.jsonl").read_text(
        encoding="utf-8"
    )
    assert result not in audit
    assert "safe = True" not in audit


def test_real_read_file_wire_result_keeps_exact_source_provenance(
    tmp_path, monkeypatch
):
    from agent.source_provenance_tools import (
        attach_trusted_source_provenance_metadata,
        source_provenance_activation,
    )
    from agent.tool_dispatch_helpers import make_tool_result_message
    from tools.file_tools import read_file_tool

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    source = tmp_path / "source.py"
    source.write_text("first = 1\nsecond = 2\n", encoding="utf-8")
    agent = _agent(tmp_path / "egress")
    agent._current_api_request_id = "turn-1:api:1"

    with source_provenance_activation(agent, "read_file"):
        result = read_file_tool(str(source), task_id="egress-real-read")
    metadata = attach_trusted_source_provenance_metadata(
        agent, "read_file", content=result
    )
    message = make_tool_result_message(
        "read_file",
        result,
        "call_read_1",
        source_provenance=metadata,
    )
    agent._current_api_request_id = "turn-1:api:2"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {"model": "test-model", "messages": [message]},
    )

    assert authorized["messages"][0]["content"] == result
    assert "_source_provenance" not in authorized["messages"][0]
    assert receipt.decision.source_grant_count == 1
    assert receipt.decision.source_segment_count == 1


def test_read_alias_can_attach_trusted_source_provenance(tmp_path, monkeypatch):
    from agent.source_provenance_tools import (
        attach_trusted_source_provenance_metadata,
        source_provenance_activation,
    )
    from tools.file_tools import read_file_tool

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    source = tmp_path / "source.py"
    source.write_text("first = 1\n", encoding="utf-8")
    agent = _agent(tmp_path / "egress")
    agent._current_api_request_id = "turn-1:api:1"

    with source_provenance_activation(agent, "read"):
        result = read_file_tool(str(source), task_id="egress-read-alias")
    metadata = attach_trusted_source_provenance_metadata(
        agent, "read", content=result
    )

    assert metadata is not None
    assert metadata["source_grant_digests"]


def test_bound_read_file_error_without_provenance_is_elided(tmp_path, monkeypatch):
    from agent.tool_dispatch_helpers import make_tool_result_message

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path / "egress")
    agent.provider = "nous"
    agent.base_url = "https://inference-api.nousresearch.com/v1"
    call_id = "call_read_error_1"
    error_result = '{"error":"File is outside the permitted read root"}'
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path":"outside.txt"}',
                },
            }],
        },
        make_tool_result_message("read_file", error_result, call_id),
    ]

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {"model": "test-model", "messages": messages},
    )

    assert receipt.allowed
    assert authorized["messages"][1]["content"] != error_result
    assert receipt.decision.source_grant_count == 0
    assert receipt.decision.source_segment_count == 0


def test_real_read_file_codex_responses_result_keeps_exact_source_provenance(
    tmp_path, monkeypatch
):
    from agent.codex_responses_adapter import _chat_messages_to_responses_input
    from agent.source_provenance_tools import (
        attach_trusted_source_provenance_metadata,
        build_source_provenance_sidecar,
        source_provenance_activation,
    )
    from agent.tool_dispatch_helpers import make_tool_result_message
    from tools.file_tools import read_file_tool

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    source = tmp_path / "source.py"
    source.write_text("first = 1\nsecond = 2\n", encoding="utf-8")
    agent = _agent(tmp_path / "egress")
    agent.provider = "openai-codex"
    agent.api_mode = "codex_responses"
    agent._current_api_request_id = "turn-1:api:1"

    with source_provenance_activation(agent, "read_file"):
        result = read_file_tool(str(source), task_id="egress-real-codex-read")
    metadata = attach_trusted_source_provenance_metadata(
        agent, "read_file", content=result
    )
    message = make_tool_result_message(
        "read_file", result, "call_read_1", source_provenance=metadata
    )
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_read_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        },
        message,
    ]
    sidecar = build_source_provenance_sidecar(messages)
    input_items = _chat_messages_to_responses_input(messages)
    agent._current_api_request_id = "turn-1:api:2"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "input": input_items,
            "_hermes_source_provenance": sidecar,
        },
    )

    assert authorized["input"][1]["output"] == result
    assert "_source_provenance" not in authorized["input"][1]
    assert receipt.decision.source_grant_count == 1
    assert receipt.decision.source_segment_count == 1


def test_codex_sdk_bypass_input_keeps_exact_source_provenance(
    tmp_path, monkeypatch
):
    from agent.codex_responses_adapter import _chat_messages_to_responses_input
    from agent.source_provenance_tools import (
        attach_trusted_source_provenance_metadata,
        build_source_provenance_sidecar,
        source_provenance_activation,
    )
    from agent.tool_dispatch_helpers import make_tool_result_message
    from tools.file_tools import read_file_tool

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    source = tmp_path / "source.py"
    source.write_text("first = 1\nsecond = 2\n", encoding="utf-8")
    agent = _agent(tmp_path / "egress")
    agent.provider = "openai-codex"
    agent.api_mode = "codex_responses"
    agent._current_api_request_id = "turn-1:api:1"

    with source_provenance_activation(agent, "read_file"):
        result = read_file_tool(str(source), task_id="egress-codex-bypass-read")
    metadata = attach_trusted_source_provenance_metadata(
        agent, "read_file", content=result
    )
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_read_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        },
        make_tool_result_message(
            "read_file", result, "call_read_1", source_provenance=metadata
        ),
    ]
    sidecar = build_source_provenance_sidecar(messages)
    input_items = _chat_messages_to_responses_input(messages)
    agent._current_api_request_id = "turn-1:api:2"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "extra_body": {"input": input_items},
            "_hermes_source_provenance": sidecar,
        },
    )

    assert authorized["extra_body"]["input"][1]["output"] == result
    assert receipt.decision.source_grant_count == 1
    assert receipt.decision.source_segment_count == 1


def test_parallel_read_file_codex_results_keep_distinct_source_provenance(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from agent.codex_responses_adapter import _chat_messages_to_responses_input
    import agent.source_provenance as provenance
    from agent.source_provenance_tools import (
        attach_trusted_source_provenance_metadata,
        build_source_provenance_sidecar,
        source_provenance_activation,
    )
    from agent.tool_dispatch_helpers import make_tool_result_message
    from tools.file_tools import read_file_tool

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    first_source = tmp_path / "pyproject.toml"
    second_source = tmp_path / "pytest.ini"
    first_source.write_text("[project]\nname = 'sample'\n", encoding="utf-8")
    second_source.write_text("[pytest]\naddopts = -q\n", encoding="utf-8")
    agent = _agent(tmp_path / "egress")
    agent.provider = "openai-codex"
    agent.api_mode = "codex_responses"
    agent._current_api_request_id = "turn-1:api:1"

    constructor_barrier = Barrier(2)

    class RacingRegistry(provenance.SourceProvenanceRegistry):
        def __init__(self):
            super().__init__()
            constructor_barrier.wait(timeout=5)

    monkeypatch.setattr(provenance, "SourceProvenanceRegistry", RacingRegistry)

    def read(source):
        with source_provenance_activation(agent, "read_file"):
            return read_file_tool(
                str(source), task_id="egress-parallel-codex-read"
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(read, (first_source, second_source)))

    messages = [{
        "role": "assistant",
        "tool_calls": [
            {
                "id": f"call_read_{index}",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }
            for index in range(2)
        ],
    }]
    for index, result in enumerate(results):
        metadata = attach_trusted_source_provenance_metadata(
            agent, "read_file", content=result
        )
        messages.append(
            make_tool_result_message(
                "read_file",
                result,
                f"call_read_{index}",
                source_provenance=metadata,
            )
        )
    sidecar = build_source_provenance_sidecar(messages)
    input_items = _chat_messages_to_responses_input(messages)
    agent._current_api_request_id = "turn-1:api:2"

    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        {
            "model": "test-model",
            "input": input_items,
            "_hermes_source_provenance": sidecar,
        },
    )

    assert [
        item["output"]
        for item in authorized["input"]
        if item.get("type") == "function_call_output"
    ] == results
    assert receipt.decision.source_grant_count == 2
    assert receipt.decision.source_segment_count == 2


@pytest.mark.parametrize("mutation", ["missing", "stale", "forged", "ambiguous"])
def test_read_file_codex_responses_result_fails_closed_without_exact_metadata(
    tmp_path, monkeypatch, mutation
):
    from agent.codex_responses_adapter import _chat_messages_to_responses_input
    from agent.source_provenance_tools import (
        attach_trusted_source_provenance_metadata,
        build_source_provenance_sidecar,
        source_provenance_activation,
    )
    from agent.tool_dispatch_helpers import make_tool_result_message
    from tools.file_tools import read_file_tool

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    source = tmp_path / "source.py"
    source.write_text("safe = True\n", encoding="utf-8")
    agent = _agent(tmp_path / "egress")
    agent.provider = "openai-codex"
    agent.api_mode = "codex_responses"
    agent._current_api_request_id = "turn-1:api:1"
    with source_provenance_activation(agent, "read_file"):
        result = read_file_tool(str(source), task_id=f"egress-codex-{mutation}")
    metadata = attach_trusted_source_provenance_metadata(
        agent, "read_file", content=result
    )
    message = make_tool_result_message(
        "read_file", result, "call_read_1", source_provenance=metadata
    )
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_read_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        },
        message,
    ]
    sidecar = build_source_provenance_sidecar(messages)
    input_items = _chat_messages_to_responses_input(messages)
    if mutation == "missing":
        sidecar = []
    elif mutation == "stale":
        sidecar[0]["request_id"] = "turn-1:api:1"
    elif mutation == "forged":
        sidecar[0]["content_sha256"] = "0" * 64
    else:
        input_items.append(dict(input_items[1]))
    agent._current_api_request_id = "turn-1:api:2"

    try:
        authorized, receipt = authorize_agent_sdk_kwargs(
            agent,
            {
                "model": "test-model",
                "input": input_items,
                "_hermes_source_provenance": sidecar,
            },
        )
    except EgressBlocked as exc:
        assert "untrusted_provenance" in exc.decision.reason_codes
    else:
        assert authorized["input"][1]["output"] != result
        assert receipt.decision.source_grant_count == 0
        assert receipt.decision.source_segment_count == 0


@pytest.mark.parametrize("mutation", ["missing", "stale", "forged"])
def test_read_file_wire_result_fails_closed_without_exact_metadata(
    tmp_path, monkeypatch, mutation
):
    from agent.source_provenance_tools import (
        attach_trusted_source_provenance_metadata,
        source_provenance_activation,
    )
    from agent.tool_dispatch_helpers import make_tool_result_message
    from tools.file_tools import read_file_tool

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    source = tmp_path / "source.py"
    source.write_text("safe = True\n", encoding="utf-8")
    agent = _agent(tmp_path / "egress")
    agent._current_api_request_id = "turn-1:api:1"
    with source_provenance_activation(agent, "read_file"):
        result = read_file_tool(str(source), task_id=f"egress-{mutation}")
    metadata = attach_trusted_source_provenance_metadata(
        agent, "read_file", content=result
    )
    if mutation == "missing":
        metadata = None
    elif mutation == "stale":
        metadata = {**metadata, "request_id": "turn-1:api:1"}
    else:
        metadata = {**metadata, "content_sha256": "0" * 64}
    message = make_tool_result_message(
        "read_file", result, "call_read_1", source_provenance=metadata
    )
    agent._current_api_request_id = "turn-1:api:2"

    with pytest.raises(EgressBlocked) as exc_info:
        authorize_agent_sdk_kwargs(
            agent,
            {"model": "test-model", "messages": [message]},
        )

    assert "untrusted_provenance" in exc_info.value.decision.reason_codes


@pytest.mark.parametrize("action,payload", [
    ("inspect-pr", {"state": "CLOSED", "head_sha": "a" * 40}),
    ("retire-feedback", {"status": "retired", "pr_state": "CLOSED", "task_id": "t_12345678"}),
])
def test_protected_feedback_replay_preserves_retirement_state(tmp_path, monkeypatch, action, payload):
    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    kwargs = {"model": agent.model, "input": [
        {"type": "function_call", "name": "terminal", "call_id": "call_retire",
         "arguments": json.dumps({"command": f"hermes github-pr-feedback {action} --repository acme/widgets --pr-number 17"})},
        {"type": "function_call_output", "call_id": "call_retire",
         "output": json.dumps({"exit_code": 0, "output": json.dumps(payload)})},
    ]}
    authorized, receipt = authorize_agent_sdk_kwargs(agent, kwargs)
    assert receipt.allowed
    rendered = json.loads(authorized["input"][1]["output"])
    assert all(rendered["json"].get(key) == value for key, value in payload.items())
