import pytest

from plugins.platforms.teams.preflight import preflight_teams_config


VALID = {"client_id": "11111111-1111-1111-1111-111111111111", "client_secret": "super-secret", "tenant_id": "22222222-2222-2222-2222-222222222222"}


class Response:
    def __init__(self, status, payload):
        self.status = status
        self.payload = payload

    async def json(self):
        return self.payload


@pytest.mark.anyio
async def test_success_returns_sanitized_checklist_and_never_secret():
    calls = []

    async def post(url, data, timeout):
        calls.append((url, data, timeout))
        return Response(200, {"access_token": "token-value", "expires_in": 3600})

    result = await preflight_teams_config(VALID, http_post=post)
    assert result["ok"] is True
    assert result["category"] == "success"
    assert result["checklist"]["capabilities"]
    assert result["checklist"]["permissions"][0]["status"] == "not_verifiable"
    assert result["checklist"]["permissions"][0]["next_step"]
    assert "super-secret" not in str(result)
    assert "token-value" not in str(result)
    assert calls[0][1]["client_secret"] == "super-secret"


@pytest.mark.anyio
@pytest.mark.parametrize("status,category", [(401, "invalid_credentials"), (403, "permission_denied")])
async def test_token_http_errors_are_distinguished_without_body(status, category):
    async def post(url, data, timeout):
        return Response(status, {"error": "secret-bearing raw body"})

    result = await preflight_teams_config(VALID, http_post=post)
    assert result["ok"] is False
    assert result["category"] == category
    assert "secret-bearing" not in str(result)


@pytest.mark.anyio
async def test_timeout_is_sanitized():
    async def post(url, data, timeout):
        raise TimeoutError("secret-timeout-detail")

    result = await preflight_teams_config(VALID, http_post=post)
    assert result == {"ok": False, "category": "timeout", "message": "Token acquisition timed out."}


@pytest.mark.anyio
async def test_malformed_config_does_not_call_http():
    called = False

    async def post(*args, **kwargs):
        nonlocal called
        called = True

    result = await preflight_teams_config({**VALID, "tenant_id": "bad tenant"}, http_post=post)
    assert result["ok"] is False
    assert result["category"] == "malformed_configuration"
    assert called is False
    assert "client_secret" not in str(result)


@pytest.mark.anyio
async def test_network_error_is_sanitized():
    async def post(url, data, timeout):
        raise OSError("secret raw network response")

    result = await preflight_teams_config(VALID, http_post=post)
    assert result["category"] == "network_error"
    assert "secret raw" not in str(result)
