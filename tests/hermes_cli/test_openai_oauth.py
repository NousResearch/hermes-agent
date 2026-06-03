import base64
import json
import time

import pytest

from hermes_cli.auth import (
    AuthError,
    DEFAULT_OPENAI_OAUTH_BASE_URL,
    PROVIDER_REGISTRY,
    _import_openai_oauth_external_tokens,
    get_auth_status,
    get_openai_oauth_auth_status,
    resolve_openai_oauth_runtime_credentials,
)


def _jwt_with_exp(exp_epoch: int) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256", "typ": "JWT"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "exp": exp_epoch,
                "https://api.openai.com/auth": {"chatgpt_account_id": "acct-test-123"},
            }
        ).encode()
    ).decode().rstrip("=")
    return f"{header}.{payload}.sig"


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def test_provider_registry_contains_openai_oauth():
    assert "openai-oauth" in PROVIDER_REGISTRY
    cfg = PROVIDER_REGISTRY["openai-oauth"]
    assert cfg.auth_type == "oauth_external"
    assert cfg.inference_base_url == DEFAULT_OPENAI_OAUTH_BASE_URL


def test_resolve_openai_oauth_runtime_credentials_reads_auth_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "providers": {
                "openai-oauth": {
                    "tokens": {
                        "access_token": _jwt_with_exp(int(time.time()) + 3600),
                        "refresh_token": "refresh-token",
                    },
                    "account_id": "acct-file-456",
                    "last_refresh": "2026-05-20T00:00:00Z",
                    "auth_mode": "chatgpt",
                }
            },
        },
    )

    creds = resolve_openai_oauth_runtime_credentials(refresh_if_expiring=False)

    assert creds["provider"] == "openai-oauth"
    assert creds["base_url"] == DEFAULT_OPENAI_OAUTH_BASE_URL
    assert creds["source"] == "hermes-auth-store"
    assert creds["account_id"] == "acct-file-456"


def test_resolve_openai_oauth_runtime_credentials_refreshes_expiring_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "providers": {
                "openai-oauth": {
                    "tokens": {
                        "access_token": _jwt_with_exp(int(time.time()) - 10),
                        "refresh_token": "refresh-token",
                    },
                    "account_id": "acct-file-456",
                }
            },
        },
    )

    monkeypatch.setattr(
        "hermes_cli.auth.refresh_codex_oauth_pure",
        lambda *args, **kwargs: {
            "access_token": _jwt_with_exp(int(time.time()) + 3600),
            "refresh_token": "refresh-new",
            "last_refresh": "2026-05-20T00:00:00Z",
        },
    )

    creds = resolve_openai_oauth_runtime_credentials()
    assert creds["api_key"]


def test_import_openai_oauth_external_tokens(tmp_path, monkeypatch):
    auth_path = tmp_path / "external-auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "openai": {
                    "type": "oauth",
                    "access": _jwt_with_exp(int(time.time()) + 3600),
                    "refresh": "refresh-token",
                    "accountId": "acct-file-456",
                }
            }
        )
    )
    monkeypatch.setenv("HERMES_OPENAI_OAUTH_AUTH_PATH", str(auth_path))

    imported = _import_openai_oauth_external_tokens()
    assert imported is not None
    assert imported["tokens"]["refresh_token"] == "refresh-token"
    assert imported["account_id"] == "acct-file-456"


def test_get_openai_oauth_auth_status_dispatches(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "providers": {
                "openai-oauth": {
                    "tokens": {
                        "access_token": _jwt_with_exp(int(time.time()) + 3600),
                        "refresh_token": "refresh-token",
                    },
                    "account_id": "acct-file-456",
                }
            },
        },
    )

    status = get_openai_oauth_auth_status()
    dispatched = get_auth_status("openai-oauth")

    assert status["logged_in"] is True
    assert status["provider"] == "openai-oauth"
    assert dispatched["logged_in"] is True


def test_openai_oauth_missing_store_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})

    with pytest.raises(AuthError) as exc_info:
        resolve_openai_oauth_runtime_credentials(refresh_if_expiring=False)
    assert exc_info.value.code == "openai_oauth_auth_missing"


# ---------------------------------------------------------------------------
# Codepath coverage: openai-oauth in model normalization, vision, delegation
# ---------------------------------------------------------------------------


def test_openai_oauth_model_normalize_strips_prefix():
    from hermes_cli.model_normalize import normalize_model_for_provider

    assert normalize_model_for_provider("openai/gpt-5.4", "openai-oauth") == "gpt-5.4"


def test_openai_oauth_model_normalize_bare_name_unchanged():
    from hermes_cli.model_normalize import normalize_model_for_provider

    assert normalize_model_for_provider("gpt-5.4", "openai-oauth") == "gpt-5.4"


def test_openai_oauth_supports_vision_tool_results():
    from tools.vision_tools import _supports_media_in_tool_results

    assert _supports_media_in_tool_results("openai-oauth", "gpt-5.4") is True


def test_openai_oauth_in_strip_vendor_providers():
    from hermes_cli.model_normalize import _STRIP_VENDOR_ONLY_PROVIDERS

    assert "openai-oauth" in _STRIP_VENDOR_ONLY_PROVIDERS


def test_delegation_preserves_openai_oauth_provider():
    from tools.delegate_tool import _resolve_delegation_credentials

    class FakeParent:
        provider = "openai-oauth"
        base_url = "https://chatgpt.com/backend-api/codex"
        api_key = "tok_test"
        api_mode = "codex_responses"

    cfg = {
        "base_url": "https://chatgpt.com/backend-api/codex",
        "provider": "openai-oauth",
    }
    result = _resolve_delegation_credentials(cfg, FakeParent())
    assert result["provider"] == "openai-oauth"
    assert result["api_mode"] == "codex_responses"
