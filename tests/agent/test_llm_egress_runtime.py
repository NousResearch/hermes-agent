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
def test_protected_kanban_never_promotes_generic_terminal_stdout_by_shape(
    tmp_path, monkeypatch, output
):
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

    with pytest.raises(EgressBlocked) as exc_info:
        authorize_agent_sdk_kwargs(agent, kwargs)

    assert "untrusted_provenance" in exc_info.value.decision.reason_codes


def test_protected_terminal_file_bytes_keep_untrusted_provenance(
    tmp_path, monkeypatch
):
    """Ungrantable terminal reads must never become sanitized by omission."""

    monkeypatch.setenv("HERMES_KANBAN_PROTECTED_REMOTE", "1")
    agent = _agent(tmp_path)
    calls = []
    innocent_source = "def calculate_total(items):\n    return sum(items)\n"

    with pytest.raises(EgressBlocked) as exc_info:
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

    assert "untrusted_provenance" in exc_info.value.decision.reason_codes
    assert calls == []


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


def test_protected_kanban_rejects_generic_codex_function_output(
    tmp_path, monkeypatch
):
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
                "output": "https://github.com/acme/widget.git run_id=1129",
            },
        ],
    }

    with pytest.raises(EgressBlocked) as exc_info:
        authorize_agent_sdk_kwargs(agent, kwargs)

    assert "untrusted_provenance" in exc_info.value.decision.reason_codes


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
