"""Regression for #154794: a bare custom route to OpenAI needs a usable key before dispatch."""

import json

import pytest

from hermes_cli.auth import AuthError
from hermes_cli.runtime_provider import resolve_runtime_provider

OPENAI_URL = "https://api.openai.com/v1"
LOCAL_URL = "http://127.0.0.1:11434/v1"


def _configure(tmp_path, monkeypatch, *, provider="custom", base_url=OPENAI_URL):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(json.dumps({
        "model": {"default": "gpt-4o", "provider": provider, "base_url": base_url},
    }), encoding="utf-8")
    (home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENROUTER_API_KEY",
                 "CUSTOM_BASE_URL", "HERMES_MODEL", "HERMES_PROVIDER"):
        monkeypatch.delenv(name, raising=False)


def test_bare_custom_openai_without_key_refused_before_request(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch)
    with pytest.raises(AuthError, match="OPENAI_API_KEY") as exc:
        resolve_runtime_provider(requested="custom", explicit_base_url=OPENAI_URL, target_model="gpt-4o")
    assert exc.value.code == "missing_api_key"
    assert "request was not sent" in str(exc.value)

    # Legal keyless local servers must stay usable.
    local = resolve_runtime_provider(requested="custom", explicit_base_url=LOCAL_URL)
    assert local["api_key"] == "no-key-required"
    assert local["base_url"] == LOCAL_URL

    # A real key in the supported process environment must also stay usable.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-valid-0123456789")
    keyed = resolve_runtime_provider(requested="custom", explicit_base_url=OPENAI_URL)
    assert keyed["api_key"] == "sk-test-valid-0123456789"


def test_inherited_openai_default_blocked_by_cron_preflight(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, provider="openai")
    from cron.scheduler_preflight import _preflight_check_provider_key

    config = {"model": {"default": "gpt-4o", "provider": "openai", "base_url": OPENAI_URL}}
    reason = _preflight_check_provider_key({"id": "daily", "model": None, "provider": None}, config)
    assert reason is not None
    assert "provider credential missing" in reason
    assert "OPENAI_API_KEY" in reason
