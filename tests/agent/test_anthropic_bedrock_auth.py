"""Native Bedrock authentication uses one explicit method (#146546), without network I/O."""

from contextlib import contextmanager

import httpx
import pytest

anthropic = pytest.importorskip("anthropic")
pytest.importorskip("boto3")

from agent import anthropic_adapter, secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def requests_sent(tmp_path, monkeypatch):
    monkeypatch.setattr(anthropic_adapter, "_anthropic_sdk", anthropic)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "launch-bearer")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    credentials = tmp_path / "credentials"
    credentials.write_text("[target]\naws_access_key_id=target-key\naws_secret_access_key=target-secret\n", encoding="utf-8")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    sent = []

    def send(_client, request, **kwargs):
        sent.append(request)
        return httpx.Response(200, request=request, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-test",
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    monkeypatch.setattr(httpx.Client, "send", send)
    return sent


@contextmanager
def _profile(home, env):
    home.mkdir(exist_ok=True)
    (home / ".env").write_text(env, encoding="utf-8")
    home_token = set_hermes_home_override(str(home))
    secret_token = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home))
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def _request():
    with anthropic_adapter.build_anthropic_bedrock_client("us-east-1", timeout=42) as client:
        assert client.max_retries == 0
        assert client.timeout.read == 42
        response = client.messages.create(model="claude-test", max_tokens=8, messages=[{"role": "user", "content": "hi"}])
        assert response.content[0].text == "ok"


@pytest.mark.parametrize("iam", ["", "AWS_PROFILE=target\n", "AWS_ACCESS_KEY_ID=target-key\nAWS_SECRET_ACCESS_KEY=target-secret\n"], ids=["bearer-only", "named-profile", "key-pair"])
def test_native_bearer_takes_precedence_in_each_profile(tmp_path, requests_sent, iam):
    for name in ("first", "second", "first"):
        with _profile(tmp_path / name, f"AWS_BEARER_TOKEN_BEDROCK={name}-bearer\n{iam}"):
            _request()
        assert requests_sent[-1].headers["authorization"] == f"Bearer {name}-bearer"


@pytest.mark.parametrize("iam", ["AWS_PROFILE=target\n", "AWS_ACCESS_KEY_ID=target-key\nAWS_SECRET_ACCESS_KEY=target-secret\nAWS_SESSION_TOKEN=target-session\n"], ids=["named-profile", "key-pair"])
def test_native_iam_can_sign_with_an_ambient_bearer(tmp_path, requests_sent, iam):
    with _profile(tmp_path / "iam", iam):
        _request()
    assert requests_sent[-1].headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=target-key/")
    if "AWS_SESSION_TOKEN" in iam:
        assert requests_sent[-1].headers["x-amz-security-token"] == "target-session"


@pytest.mark.parametrize("bearer", [True, False], ids=["bearer", "default-chain"])
def test_standalone_native_auth_keeps_environment_credentials(requests_sent, monkeypatch, bearer):
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setenv("AWS_PROFILE", "target")
    if not bearer:
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK")
    _request()
    authorization = requests_sent[-1].headers["authorization"]
    if bearer:
        assert authorization == "Bearer launch-bearer"
    else:
        assert authorization.startswith("AWS4-HMAC-SHA256 Credential=target-key/")
