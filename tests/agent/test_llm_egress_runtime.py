from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.llm_egress_firewall import EgressBlocked, SanitizedTextRejected
from agent.llm_egress_runtime import (
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
                "extra_headers": {"Authorization": "token=secret-value"},
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
        "token=super-secret-value",
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
                "input": [{"role": "system", "content": "token=super-secret-value"}],
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
    assert "untrusted" in rendered
    assert rendered.startswith("kanban_show completed locally.")
    assert "c2VjcmV0LXBheWxvYWQ=" not in rendered
    assert "super-secret-value" not in rendered
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
                    {"role": "system", "content": "token=super-secret-value"}
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
        ("token=super-secret-value", "secret_detected"),
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
    """Ordinary terminal evidence must not deadlock a protected cloud worker.

    The result is still typed as bounded non-source text; it is not promoted
    to a source grant merely because its shape resembles a URL, ref, or hash.
    """

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
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
    assert authorized["messages"][1]["content"] == output
    assert receipt.decision.source_segment_count == 0


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
                        "content": "https://github.com/acme/widget.git run_id=1129",
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
