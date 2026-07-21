import io
import json
from urllib.error import HTTPError

import pytest

from hermes_cli import model_role_resolution as roles


def test_resolve_logical_role_posts_authenticated_request_and_returns_atomic_tuple(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "role": "bob-main",
                    "model": "gpt-5.6-terra",
                    "provider": "openai-codex",
                    "api_mode": "codex_responses",
                    "reason": "first_healthy_candidate",
                    "policy_version": "2026-07-20",
                    "candidate_index": 0,
                    "evidence": [],
                }
            ).encode()

    def fake_urlopen(request, *, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode())
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(roles.urllib_request, "urlopen", fake_urlopen)

    resolved = roles.resolve_logical_role(
        "bob-main",
        proxy_base_url="http://127.0.0.1:18989",
        api_key="test-key",
    )

    assert resolved == {
        "model": "gpt-5.6-terra",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
    }
    assert captured == {
        "url": "http://127.0.0.1:18989/api/model-route/resolve",
        "body": {"role": "bob-main"},
        "authorization": "Bearer test-key",
        "timeout": roles.MODEL_ROLE_RESOLUTION_TIMEOUT_SECONDS,
    }


def test_resolve_logical_role_fails_closed_with_proxy_error(monkeypatch):
    error_body = io.BytesIO(b'{"error":{"code":"unknown_model_role","message":"Unknown model role"}}')

    def fake_urlopen(request, *, timeout):
        raise HTTPError(request.full_url, 404, "Not Found", hdrs=None, fp=error_body)

    monkeypatch.setattr(roles.urllib_request, "urlopen", fake_urlopen)

    with pytest.raises(roles.LogicalModelRoleResolutionError, match="Unknown model role"):
        roles.resolve_logical_role("missing", api_key="test-key")


def test_resolve_logical_model_role_leaves_physical_model_unchanged():
    assert roles.resolve_logical_model_role("gpt-5.6-terra") is None


def test_resolve_logical_model_runtime_replaces_the_full_runtime_tuple(monkeypatch):
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(
        "hermes_cli.model_role_resolution.resolve_logical_model_role",
        lambda model: {
            "model": "gpt-5.6-terra",
            "provider": "openai-codex",
            "api_mode": "codex_responses",
        },
    )
    monkeypatch.setattr(
        rp,
        "resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "stale-provider",
            "api_mode": "stale-mode",
            "api_key": "provider-key",
            "base_url": "http://provider.example/v1",
        },
    )

    model, runtime = rp.resolve_logical_model_runtime("role:bob-main")

    assert model == "gpt-5.6-terra"
    assert runtime["provider"] == "openai-codex"
    assert runtime["api_mode"] == "codex_responses"
