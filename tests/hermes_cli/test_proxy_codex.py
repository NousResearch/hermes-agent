from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_cli.proxy.adapters import ADAPTERS, get_adapter
from hermes_cli.proxy.adapters.codex import OpenAICodexAdapter
from hermes_cli.proxy.cli import cmd_proxy_start
from hermes_cli.proxy.server import is_loopback_host, run_server


def _jwt_with_account(account_id: str = "acct-123") -> str:
    payload = {
        "exp": 4_102_444_800,
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def _entry(account_id: str = "acct-123"):
    return SimpleNamespace(
        runtime_api_key=_jwt_with_account(account_id),
        runtime_base_url="https://chatgpt.com/backend-api/codex",
        base_url="https://chatgpt.com/backend-api/codex",
        expires_at="2099-01-01T00:00:00Z",
    )


def test_registry_exposes_one_canonical_codex_adapter_and_alias():
    assert ADAPTERS["openai-codex"] is OpenAICodexAdapter
    assert "codex" not in ADAPTERS
    assert isinstance(get_adapter("openai-codex"), OpenAICodexAdapter)
    assert isinstance(get_adapter("codex"), OpenAICodexAdapter)


def test_codex_loopback_host_validation_and_cli_rejection(capsys):
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert is_loopback_host("localhost")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("192.168.1.5")
    assert not is_loopback_host("example.com")

    adapter = OpenAICodexAdapter()
    args = SimpleNamespace(provider="codex", host="0.0.0.0", port=8645)
    with (
        patch("hermes_cli.proxy.cli.get_adapter", return_value=adapter),
        patch.object(adapter, "is_authenticated", return_value=True),
        patch("hermes_cli.proxy.cli.run_server") as run,
    ):
        assert cmd_proxy_start(args) == 2
    run.assert_not_called()
    assert "loopback-only" in capsys.readouterr().err


def test_codex_server_rejects_non_loopback_programmatic_bind():
    adapter = OpenAICodexAdapter()
    try:
        asyncio.run(run_server(adapter, host="0.0.0.0", port=0))
    except RuntimeError as exc:
        assert "loopback-only" in str(exc)
    else:
        raise AssertionError("programmatic non-loopback Codex bind was accepted")


def test_codex_adapter_authentication_is_pool_backed():
    pool = MagicMock()
    pool.has_available.return_value = True
    with patch("hermes_cli.proxy.adapters.codex.load_pool", return_value=pool):
        assert OpenAICodexAdapter().is_authenticated() is True


def test_codex_adapter_selects_responses_credential_and_required_headers():
    pool = MagicMock()
    pool.has_credentials.return_value = True
    pool.select.return_value = _entry()
    adapter = OpenAICodexAdapter()
    with patch("hermes_cli.proxy.adapters.codex.load_pool", return_value=pool):
        credential = adapter.get_credential()

    assert credential.bearer == _jwt_with_account()
    assert credential.base_url == "https://chatgpt.com/backend-api/codex"
    assert adapter.allowed_paths == frozenset({"/responses", "/models"})
    assert adapter.get_upstream_headers(credential) == {
        "User-Agent": "codex_cli_rs/0.0.0 (Hermes Agent)",
        "originator": "codex_cli_rs",
        "ChatGPT-Account-ID": "acct-123",
    }


def test_codex_adapter_rejects_untrusted_upstream_before_returning_bearer():
    pool = MagicMock()
    pool.has_credentials.return_value = True
    bad = _entry()
    bad.runtime_base_url = "http://127.0.0.1:9999/steal"
    bad.base_url = "http://127.0.0.1:9999/steal"
    pool.select.return_value = bad
    adapter = OpenAICodexAdapter()
    with patch("hermes_cli.proxy.adapters.codex.load_pool", return_value=pool):
        try:
            adapter.get_credential()
        except RuntimeError as exc:
            assert "untrusted" in str(exc).lower()
        else:
            raise AssertionError("untrusted Codex upstream was accepted")


def test_codex_owned_account_header_survives_missing_jwt_claim_as_owned_only():
    adapter = OpenAICodexAdapter()
    credential = SimpleNamespace(bearer="malformed-token")
    headers = adapter.get_upstream_headers(credential)
    assert "ChatGPT-Account-ID" not in headers
    assert "ChatGPT-Account-ID" in adapter.get_owned_upstream_header_names()


def test_codex_adapter_401_refreshes_matching_credential():
    pool = MagicMock()
    pool.has_credentials.return_value = True
    pool.select.return_value = _entry("first")
    pool.try_refresh_matching.return_value = _entry("refreshed")
    adapter = OpenAICodexAdapter()
    with patch("hermes_cli.proxy.adapters.codex.load_pool", return_value=pool):
        failed = adapter.get_credential()
        retry = adapter.get_retry_credential(
            failed_credential=failed,
            status_code=401,
        )

    assert retry is not None
    assert retry.bearer == _jwt_with_account("refreshed")
    pool.try_refresh_matching.assert_called_once_with(api_key_hint=failed.bearer)
    pool.mark_exhausted_and_rotate.assert_not_called()


def test_codex_adapter_reuses_concurrently_refreshed_current_credential():
    pool = MagicMock()
    refreshed = _entry("new-token")
    pool.try_refresh_matching.side_effect = [refreshed, None]
    pool.select.return_value = refreshed
    adapter = OpenAICodexAdapter()
    adapter._pool = pool
    failed = SimpleNamespace(bearer=_jwt_with_account("old-token"))

    first = adapter.get_retry_credential(
        failed_credential=failed,
        status_code=401,
    )
    second = adapter.get_retry_credential(
        failed_credential=failed,
        status_code=401,
    )

    assert first is not None and first.bearer == refreshed.runtime_api_key
    assert second is not None and second.bearer == refreshed.runtime_api_key
    pool.mark_exhausted_and_rotate.assert_not_called()


def test_codex_adapter_concurrent_401_refresh_is_serialized():
    pool = MagicMock()
    in_flight = threading.Event()
    overlap = threading.Event()
    counter = {"n": 0}

    def refresh(*, api_key_hint):
        _ = api_key_hint
        if in_flight.is_set():
            overlap.set()
        in_flight.set()
        try:
            time.sleep(0.03)
            counter["n"] += 1
            return _entry(f"refreshed-{counter['n']}")
        finally:
            in_flight.clear()

    pool.try_refresh_matching.side_effect = refresh
    adapter = OpenAICodexAdapter()
    adapter._pool = pool
    failed = SimpleNamespace(bearer=_jwt_with_account("failed"))
    results = []

    def worker():
        results.append(
            adapter.get_retry_credential(
                failed_credential=failed,
                status_code=401,
            )
        )

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 3
    assert not overlap.is_set()


def test_codex_adapter_429_marks_failed_credential_and_rotates():
    pool = MagicMock()
    pool.has_credentials.return_value = True
    pool.select.return_value = _entry("first")
    pool.mark_exhausted_and_rotate.return_value = _entry("second")
    adapter = OpenAICodexAdapter()
    with patch("hermes_cli.proxy.adapters.codex.load_pool", return_value=pool):
        failed = adapter.get_credential()
        retry = adapter.get_retry_credential(
            failed_credential=failed,
            status_code=429,
        )

    assert retry is not None
    assert retry.bearer == _jwt_with_account("second")
    pool.mark_exhausted_and_rotate.assert_called_once_with(
        status_code=429,
        api_key_hint=failed.bearer,
    )
    pool.try_refresh_matching.assert_not_called()
