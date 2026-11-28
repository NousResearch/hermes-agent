"""A remembered Codex model-entitlement failure blocks a dead /model switch (#140736)."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from agent.credential_pool_model_cooldowns import MODEL_ENTITLEMENT_BENCH_SECONDS
from hermes_cli.auth import write_credential_pool
from hermes_cli.model_switch import switch_model

_PROVIDER = "openai-codex"
_BASE_URL = "https://chatgpt.com/backend-api/codex"
_MODEL = "gpt-5.4-mini"
_ACCEPTED = {"accepted": True, "persist": True, "recognized": True, "message": None}


def _entry(name: str, *, cooldown: float | None) -> dict:
    row = {
        "id": name,
        "label": name,
        "auth_type": "oauth",
        "priority": 1,
        "source": "manual",
        "access_token": f"token-{name}",
        "refresh_token": f"refresh-{name}",
    }
    if cooldown is not None:
        row["model_cooldowns"] = {_MODEL: cooldown}
    return row


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _switch(model: str = _MODEL):
    with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
         patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
             "api_key": "token-a", "base_url": _BASE_URL, "api_mode": "responses",
         }), \
         patch("hermes_cli.models_validate.validate_requested_model", return_value=_ACCEPTED) as validate, \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch("hermes_cli.models.detect_provider_for_model", return_value=None):
        result = switch_model(
            raw_input=model,
            current_provider=_PROVIDER,
            current_model="gpt-5.3-codex",
            current_base_url=_BASE_URL,
            current_api_key="token-a",
        )
    return result, validate


def test_all_live_credentials_benched_refuses_before_remote_validation(isolated_home):
    future = time.time() + MODEL_ENTITLEMENT_BENCH_SECONDS
    write_credential_pool(_PROVIDER, [_entry("a", cooldown=future), _entry("b", cooldown=future)])

    result, validate = _switch()

    assert result.success is False
    assert "unsupported by every saved" in result.error_message
    assert "hermes auth reset openai-codex" in result.error_message
    assert validate.call_count == 0


def test_one_eligible_credential_without_the_bench_allows_the_switch(isolated_home):
    future = time.time() + MODEL_ENTITLEMENT_BENCH_SECONDS
    write_credential_pool(_PROVIDER, [_entry("a", cooldown=future), _entry("b", cooldown=None)])

    result, validate = _switch()

    assert result.success is True
    assert validate.call_count == 1


def test_short_model_cooldowns_do_not_become_entitlement_refusals(isolated_home):
    short = time.time() + 300
    write_credential_pool(_PROVIDER, [_entry("a", cooldown=short), _entry("b", cooldown=short)])

    result, validate = _switch()

    assert result.success is True
    assert validate.call_count == 1


@pytest.mark.parametrize("stored,requested", [
    ("gpt-5.4", "gpt-5.4-900k"),
    ("gpt-5.4-900k", "gpt-5.4"),
])
def test_codex_context_alias_matches_the_same_entitlement_bench(isolated_home, stored, requested):
    future = time.time() + MODEL_ENTITLEMENT_BENCH_SECONDS
    row = _entry("a", cooldown=None)
    row["model_cooldowns"] = {stored: future}
    write_credential_pool(_PROVIDER, [row])

    result, validate = _switch(requested)

    assert result.success is False
    assert validate.call_count == 0
