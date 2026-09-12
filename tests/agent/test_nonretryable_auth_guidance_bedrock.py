"""Terminal auth guidance for the Bedrock provider.

``agent.turn_recovery._print_nonretryable_auth_guidance`` is what the user sees
when a turn ends on a non-retryable ``auth`` verdict. Bedrock has no API key in
the usual sense — the credential is an AWS session (SSO, ``credential_process``,
an instance/task role) — so the generic "Your API key was rejected ... Run:
hermes setup" remedy is wrong for it. The Bedrock branch must name the AWS
remedy instead; every other provider keeps the generic text.
"""

from types import SimpleNamespace

from agent.error_classifier import FailoverReason
from agent.turn_recovery import _print_nonretryable_auth_guidance


def _agent_capturing_lines():
    lines = []
    return SimpleNamespace(
        log_prefix="", _vprint=lambda line, force=False: lines.append(line)
    ), lines


def _auth_verdict():
    return SimpleNamespace(
        reason=FailoverReason.auth, is_auth=True, billing_unverified=False, retryable=False
    )


def test_bedrock_auth_guidance_names_the_aws_session_not_an_api_key():
    agent, lines = _agent_capturing_lines()
    _print_nonretryable_auth_guidance(
        agent, _auth_verdict(), status_code=None, provider="bedrock",
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com", model="amazon.nova-pro-v1:0",
    )
    text = "\n".join(lines)
    assert "aws sso login" in text
    assert "AWS" in text
    assert "API key was rejected" not in text
    assert "hermes setup" not in text


def test_non_bedrock_provider_keeps_generic_api_key_guidance():
    agent, lines = _agent_capturing_lines()
    _print_nonretryable_auth_guidance(
        agent, _auth_verdict(), status_code=None, provider="openai",
        base_url="https://api.openai.com/v1", model="gpt-5",
    )
    text = "\n".join(lines)
    assert "API key was rejected" in text
    assert "aws sso login" not in text
