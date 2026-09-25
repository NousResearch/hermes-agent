"""Auxiliary credential recovery must not disable the owning main route.

Regression for #119533.
"""

from __future__ import annotations

import json

import pytest


class _ProviderError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"provider failed with HTTP {status_code}")
        self.status_code = status_code


def _write_openrouter_pool(tmp_path, monkeypatch, keys: list[str]) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    entries = [
        {
            "id": f"key-{index}",
            "label": f"key-{index}",
            "auth_type": "api_key",
            "priority": index,
            "source": "manual",
            "access_token": key,
        }
        for index, key in enumerate(keys)
    ]
    (hermes_home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {"openrouter": entries}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.mark.parametrize("status_code", [402, 429])
def test_aux_failure_preserves_single_key_for_main_route(tmp_path, monkeypatch, status_code):
    from agent.auxiliary_client import _recover_provider_pool
    from agent.credential_pool import load_pool

    _write_openrouter_pool(tmp_path, monkeypatch, ["shared-key"])

    assert _recover_provider_pool(
        "openrouter", _ProviderError(status_code), failed_api_key="shared-key"
    ) is False

    main_entry = load_pool("openrouter").select()
    assert main_entry is not None
    assert main_entry.runtime_api_key == "shared-key"
    assert main_entry.last_status is None


def test_aux_auth_failure_exhausts_single_rejected_key(tmp_path, monkeypatch):
    from agent.auxiliary_client import _recover_provider_pool
    from agent.credential_pool import STATUS_EXHAUSTED, load_pool

    _write_openrouter_pool(tmp_path, monkeypatch, ["rejected-key"])

    assert _recover_provider_pool(
        "openrouter", _ProviderError(401), failed_api_key="rejected-key"
    ) is False

    pool = load_pool("openrouter")
    assert pool.entries()[0].last_status == STATUS_EXHAUSTED
    assert pool.select() is None


def test_aux_failure_still_rotates_to_distinct_key(tmp_path, monkeypatch):
    from agent.auxiliary_client import _recover_provider_pool
    from agent.credential_pool import STATUS_EXHAUSTED, load_pool

    _write_openrouter_pool(tmp_path, monkeypatch, ["failed-key", "healthy-key"])

    assert _recover_provider_pool(
        "openrouter", _ProviderError(429), failed_api_key="failed-key"
    ) is True

    pool = load_pool("openrouter")
    entries = {entry.runtime_api_key: entry for entry in pool.entries()}
    assert entries["failed-key"].last_status == STATUS_EXHAUSTED
    assert pool.select().runtime_api_key == "healthy-key"