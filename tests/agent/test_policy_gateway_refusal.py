"""Structured governance refusals must not invalidate a working credential (#125058)."""
import httpx
import openai
import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_failure_copy import nonretryable_copy


DENIED = {
    "type": "wardryx_denied",
    "reason": 'estimated cost $0.02 exceeds policy "deny-beta" hard ceiling; no approval can authorize this',
    "retryable": False,
}
APPROVAL = {
    "type": "approval_pending",
    "approval_token_required": True,
    "detail": "resubmit this request with header x-fuse-approval-token after approval",
    "reason": "estimated cost $0.02 exceeds policy threshold; human approval required",
    "retryable": False,
}


def classify(body, status=403):
    response = httpx.Response(status, request=httpx.Request("POST", "https://policy.example/v1/chat/completions"))
    error = openai.APIStatusError("Forbidden", response=response, body=body)
    return classify_api_error(error, provider="custom", model="test", base_url="https://policy.example/v1")


@pytest.mark.parametrize("error", [DENIED, APPROVAL])
@pytest.mark.parametrize("wrapped", [True, False])
def test_governance_refusal_preserves_key_and_explains_policy(error, wrapped):
    result = classify({"error": error} if wrapped else error)
    assert result.reason == FailoverReason.provider_policy_blocked
    assert not result.is_auth
    assert not result.should_rotate_credential
    assert not result.retryable
    # Do not route around an explicit governance refusal with a different provider.
    assert not result.should_fallback
    assert result.message == error["reason"]
    copy = nonretryable_copy(result, provider="custom", model="test", summary="Forbidden")
    assert error["reason"] in copy
    assert "rejected your API key" not in copy
    assert "switch models" not in copy


@pytest.mark.parametrize("body", [
    {}, {"error": {}}, {"error": "forbidden"},
    {"error": {"type": "unknown_error", "reason": "unexpected failure", "retryable": False}},
    {"error": {"type": "unknown_error", "reason": "policy service failed", "retryable": False}},
    {"error": {"type": "wardryx_denied", "reason": "policy denied", "retryable": True}},
    {"error": {"type": "wardryx_denied", "reason": {"policy": "denied"}, "retryable": False}},
    *[{"error": {**DENIED, "type": code}} for code in
      ("authentication_error", "invalid_api_key", "permission_denied", "unauthorized")],
    {"error": {**APPROVAL, "code": "invalid_api_key"}},
    {"error": {**DENIED, "retryable": "false"}},
])
def test_ambiguous_or_auth_errors_keep_existing_auth_fallback(body):
    result = classify(body)
    assert result.reason == FailoverReason.auth
    assert result.should_fallback


def test_401_remains_auth_even_with_policy_envelope():
    assert classify({"error": DENIED}, status=401).is_auth


@pytest.mark.parametrize("message, expected", [
    ("insufficient credits", FailoverReason.billing),
    ("Enable JavaScript and cookies to continue", FailoverReason.upstream_blocked),
])
def test_existing_decisive_403_handlers_keep_precedence(message, expected):
    assert classify({"error": {**DENIED, "message": message}}).reason == expected


def test_policy_verdict_does_not_bench_persisted_credentials(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock
    from agent.credential_pool import load_pool
    from agent.agent_runtime_helpers import recover_with_credential_pool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"credential_pool": {"openai-codex": [
        {"id": "fixture", "auth_type": "oauth", "source": "manual", "priority": 0,
         "access_token": "fixture-only", "refresh_token": "fixture-refresh", "expires_at_ms": 4_000_000_000_000}
    ]}}))
    pool = load_pool("openai-codex")
    assert pool.select(model="test").id == "fixture"
    before = auth.read_bytes()
    agent = SimpleNamespace(
        provider="openai-codex", model="test", base_url="https://chatgpt.com/backend-api/codex",
        api_key="fixture-only", _credential_pool=pool, _credential_pool_entry_id="fixture",
        _swap_credential=Mock(),
    )
    verdict = classify({"error": DENIED})
    assert recover_with_credential_pool(
        agent, status_code=403, has_retried_429=False, classified_reason=verdict.reason,
    ) == (False, False)
    agent._swap_credential.assert_not_called()
    assert auth.read_bytes() == before
    reloaded = load_pool("openai-codex")
    assert reloaded.select(model="test").id == "fixture"
    assert reloaded.select(model="other").id == "fixture"


def test_terminal_result_carries_policy_reason_not_key_advice():
    from agent.turn_recovery import nonretryable_client_error_result

    class Agent:
        log_prefix = ""
        verbose = False
        verbose_logging = False

        def _summarize_api_error(self, error):
            return str(error)

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    verdict = classify({"error": APPROVAL})
    agent = Agent()
    from unittest.mock import Mock
    agent._vprint = Mock()
    result = nonretryable_client_error_result(
        agent, Exception("Forbidden"), verdict, status_code=403, api_kwargs=None,
        api_messages=[], messages=[], conversation_history=None, api_call_count=1,
        approx_tokens=10, provider="custom", base_url="https://policy.example/v1", model="test",
    )
    assert result["failure_reason"] == "provider_policy_blocked"
    assert result["failure_retryable"] is False
    assert APPROVAL["reason"] in result["final_response"]
    assert "rejected your API key" not in result["final_response"]
    diagnostics = "\n".join(call.args[0] for call in agent._vprint.call_args_list)
    assert "gateway administrator" in diagnostics
    assert "pick another model" not in diagnostics
