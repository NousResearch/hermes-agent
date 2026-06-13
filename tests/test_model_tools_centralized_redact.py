"""Tests for centralized secret redaction in model_tools.handle_function_call.

Every tool result returned by ``handle_function_call`` must be redacted
before it reaches the LLM context. Individual tools already redact their
own output (terminal, code_exec, file, browser), but this centralized
gate catches every tool path: MCP, web, custom plugins, etc.
"""

import json

import pytest

import model_tools


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    """Ensure redaction is enabled before each test.

    Individual tests that want to test the disabled path must override
    this explicitly inside their own body AFTER this fixture runs.
    """
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


def _run_handle_function_call(monkeypatch, *, dispatch_result):
    """Drive ``handle_function_call`` with a mocked registry dispatch."""
    from tools.registry import registry

    monkeypatch.setattr(
        registry, "dispatch",
        lambda name, args, **kw: dispatch_result,
    )
    # Skip unrelated side effects (read-loop tracker, approval, etc.).
    monkeypatch.setattr(model_tools, "_READ_SEARCH_TOOLS", frozenset())

    # Clear any stale plugin state so has_hook() returns False for
    # transform_tool_result, keeping the simplest code path.
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)

    return model_tools.handle_function_call(
        "dummy_tool",
        {},
        task_id="t1",
        session_id="s1",
        tool_call_id="tc1",
        skip_pre_tool_call_hook=True,
    )


# ── Positive tests: redaction works ────────────────────────────────────

def test_redacts_openai_sk_key_in_tool_result(monkeypatch):
    result = _run_handle_function_call(
        monkeypatch,
        dispatch_result=json.dumps({
            "output": "Using key sk-proj-abc123def456ghi789jkl012 for API calls"
        }),
    )
    assert "abc123def456" not in result
    assert "sk-pro" in result  # prefix preserved
    assert "..." in result     # masking indicator


def test_redacts_github_token_in_tool_result(monkeypatch):
    result = _run_handle_function_call(
        monkeypatch,
        dispatch_result=json.dumps({
            "output": "Authenticated with ghp_abc123def456ghi789jkl"
        }),
    )
    assert "abc123def456" not in result


def test_redacts_bearer_token_in_tool_result(monkeypatch):
    result = _run_handle_function_call(
        monkeypatch,
        dispatch_result=json.dumps({
            "output": "Authorization: Bearer ya29.abcdefghijklmnopqrstuvwxyz123456"
        }),
    )
    # "Authorization" prefix preserved, token masked
    assert "ya29." not in result or "..." in result


def test_redacts_jwt_token_in_tool_result(monkeypatch):
    result = _run_handle_function_call(
        monkeypatch,
        dispatch_result=json.dumps({
            "output": "token=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNqyP44"
        }),
    )
    # eyJhbGciOi... is a JWT — should be masked
    assert "eyJhbGci" not in result or "..." in result


def test_redacts_env_assignment_in_tool_result(monkeypatch):
    result = _run_handle_function_call(
        monkeypatch,
        dispatch_result=json.dumps({
            "output": "OPENAI_API_KEY=sk-abc123def456ghi789jkl012"
        }),
    )
    assert "abc123def456" not in result


def test_redacts_private_key_block_in_tool_result(monkeypatch):
    result = _run_handle_function_call(
        monkeypatch,
        dispatch_result=json.dumps({
            "output": (
                "-----BEGIN RSA PRIVATE KEY-----\n"
                "MIIEpAIBAAKCAQEAsomethingprivate\n"
                "-----END RSA PRIVATE KEY-----"
            )
        }),
    )
    # The full private key content is replaced with a redacted marker
    assert "MIIEpAIBAAKCAQEA" not in result
    assert "somethingprivate" not in result
    assert "REDACTED" in result


def test_redacts_db_connection_string_password(monkeypatch):
    result = _run_handle_function_call(
        monkeypatch,
        dispatch_result=json.dumps({
            "output": "postgresql://user:hunter2@localhost:5432/db"
        }),
    )
    assert "hunter2" not in result


# ── Negative tests: non-secret outputs pass through ─────────────────────

def test_non_secret_output_passes_through(monkeypatch):
    result = _run_handle_function_call(
        monkeypatch,
        dispatch_result=json.dumps({
            "output": "Hello, world! The answer is 42."
        }),
    )
    parsed = json.loads(result)
    assert parsed["output"] == "Hello, world! The answer is 42."


def test_empty_result_handled_gracefully(monkeypatch):
    result = _run_handle_function_call(
        monkeypatch,
        dispatch_result=json.dumps({"output": ""}),
    )
    parsed = json.loads(result)
    assert parsed["output"] == ""


def test_json_structure_preserved(monkeypatch):
    """Redaction must not corrupt JSON structure."""
    result = _run_handle_function_call(
        monkeypatch,
        dispatch_result=json.dumps({
            "exit_code": 0,
            "files": ["a.txt", "b.txt"],
            "nested": {"key": "value"},
        }),
    )
    parsed = json.loads(result)
    assert parsed["exit_code"] == 0
    assert parsed["files"] == ["a.txt", "b.txt"]
    assert parsed["nested"] == {"key": "value"}


# ── Config flag tests ──────────────────────────────────────────────────

def test_respects_redact_secrets_disabled(monkeypatch):
    """When security.redact_secrets is false, results are NOT redacted."""
    # Override the autouse fixture which enables redaction.
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "false")
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)

    result = _run_handle_function_call(
        monkeypatch,
        dispatch_result=json.dumps({
            "output": "Using key sk-proj-abc123def456ghi789jkl012"
        }),
    )
    assert "abc123def456" in result  # NOT redacted
    assert "sk-proj-abc123def456" in result


def test_redact_secrets_flag_restored_after_disabled(monkeypatch):
    """Re-enabling the flag should resume redaction."""
    # Disable first — override autouse fixture
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "false")
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)

    result = _run_handle_function_call(
        monkeypatch,
        dispatch_result=json.dumps({"output": "sk-proj-abc123def456"}),
    )
    assert "abc123def456" in result  # confirmed disabled

    # Re-enable — monkeypatch.setattr overrides the disabled value set above
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)

    result = _run_handle_function_call(
        monkeypatch,
        dispatch_result=json.dumps({"output": "sk-proj-abc123def456"}),
    )
    assert "abc123def456" not in result  # redacted again


# ── Defense-in-depth: double redaction is harmless ────────────────────

def test_pre_redacted_output_is_harmless(monkeypatch):
    """Tools that already redact (terminal, etc.) should pass through safely."""
    # Simulate a tool that already redacted its own output
    result = _run_handle_function_call(
        monkeypatch,
        dispatch_result=json.dumps({
            "output": "Using key sk-proj-...abc123def  (already redacted)"
        }),
    )
    # The centralized redaction should not corrupt already-redacted content
    parsed = json.loads(result)
    assert "output" in parsed
    assert len(parsed["output"]) > 0
