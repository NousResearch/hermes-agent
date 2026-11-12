"""Synthetic tests for the Codex subscription-proxy adapter."""

from __future__ import annotations

import base64
import json
from unittest.mock import patch

import pytest

from hermes_cli.proxy.adapters import ADAPTERS, get_adapter
from hermes_cli.proxy.adapters.base import UpstreamCredential
from hermes_cli.proxy.adapters.codex import CodexAdapter
from hermes_cli.proxy.server import _forwarded_request_headers


CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"


def _synthetic_jwt() -> str:
    payload = {
        "https://api.openai.com/auth": {
            "chatgpt_account_id": "synthetic-account",
            "chatgpt_data_residency": "eu",
        }
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def test_codex_adapter_is_registered_and_only_forwards_responses():
    assert "codex" in ADAPTERS
    adapter = get_adapter("codex")
    assert isinstance(adapter, CodexAdapter)
    assert adapter.allowed_paths == frozenset({"/responses"})


def test_codex_adapter_resolves_only_hermes_auth_read_only():
    token = _synthetic_jwt()
    result = {
        "api_key": token,
        "base_url": CODEX_BASE_URL,
        "last_refresh": "2026-01-01T00:00:00Z",
    }
    with patch(
        "hermes_cli.proxy.adapters.codex.resolve_codex_runtime_credentials",
        return_value=result,
    ) as resolve:
        credential = CodexAdapter().get_credential()

    resolve.assert_called_once_with(read_only=True)
    assert credential.bearer == token
    assert credential.base_url == CODEX_BASE_URL
    assert credential.headers["originator"] == "hermes-agent"
    assert credential.headers["ChatGPT-Account-ID"] == "synthetic-account"
    assert credential.headers["x-openai-internal-codex-residency"] == "eu"
    assert token not in repr(credential)


def test_codex_adapter_refuses_nonofficial_endpoint_without_forwarding_token():
    token = "synthetic-token-not-for-network"
    with patch(
        "hermes_cli.proxy.adapters.codex.resolve_codex_runtime_credentials",
        return_value={"api_key": token, "base_url": "https://attacker.invalid/codex"},
    ) as resolve:
        with pytest.raises(RuntimeError, match="official Codex endpoint") as exc:
            CodexAdapter().get_credential()

    resolve.assert_called_once_with(read_only=True)
    assert token not in str(exc.value)


def test_proxy_replaces_client_identity_headers_with_upstream_identity():
    credential = UpstreamCredential(
        bearer="synthetic-upstream-token",
        base_url=CODEX_BASE_URL,
        headers={
            "User-Agent": "HermesAgent/test",
            "originator": "hermes-agent",
            "ChatGPT-Account-ID": "synthetic-account",
        },
    )
    forwarded = _forwarded_request_headers(
        {
            "Authorization": "Bearer client-placeholder",
            "user-agent": "untrusted-client",
            "ORIGINATOR": "untrusted-client",
            "chatgpt-account-id": "untrusted-account",
            "Accept": "application/json",
        },
        credential,
    )

    assert forwarded["Authorization"] == "Bearer synthetic-upstream-token"
    assert forwarded["User-Agent"] == "HermesAgent/test"
    assert forwarded["originator"] == "hermes-agent"
    assert forwarded["ChatGPT-Account-ID"] == "synthetic-account"
    assert forwarded["Accept"] == "application/json"
    assert all("client-placeholder" not in value for value in forwarded.values())
