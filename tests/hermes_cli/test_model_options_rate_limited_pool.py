"""A rate-limited provider stays in the GUI model picker (#103829, #124510).

Claude Pro/Max and ChatGPT subscriptions hit usage windows constantly. The 429 puts the pooled
credential in cooldown (credential-wide ``exhausted``, or one model's ``model_cooldowns``), and
the Desktop / dashboard / TUI ``model.options`` payload treated that as signed out: the whole
provider vanished from the menu until the window reset, then came back. These tests drive the
real payload builder against a real ``auth.json`` pool in a temp home.
"""

import json
import time

import pytest

from hermes_cli.inventory import build_model_options_payload, load_picker_context


def _pool_entry(provider: str, cooldown: str) -> dict:
    now = time.time()
    oauth = provider == "anthropic"
    entry = {
        "id": "e1", "label": "subscription", "priority": 0, "last_status": "ok",
        "auth_type": "oauth" if oauth else "api_key",
        "source": "manual:hermes_pkce" if oauth else "manual",
        "access_token": "sk-ant-oat01-test" if oauth else "test-key-123",
        "refresh_token": "refresh" if oauth else None,
        "expires_at_ms": int((now + 6 * 3600) * 1000),
    }
    if cooldown == "exhausted":
        entry.update(last_status="exhausted", last_status_at=now, last_error_code=429,
                     last_error_reset_at=now + 3 * 3600)
    else:
        entry["model_cooldowns"] = {"some-model": now + 3 * 3600}
    return entry


@pytest.fixture
def pooled_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Only the pooled credential under test may authenticate anything.
    (home / "config.yaml").write_text(
        "model:\n  provider: nous\n  default: test-model\nauth:\n  adopt_external_logins: false\n")

    def write(provider: str, cooldown: str) -> None:
        (home / "auth.json").write_text(json.dumps({
            "version": 1, "providers": {},
            "credential_pool": {provider: [_pool_entry(provider, cooldown)]}}))

    return write


# anthropic reaches the picker through the overlay section, gemini through the models.dev one.
@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
@pytest.mark.parametrize("cooldown", ["exhausted", "model_cooldown"])
@pytest.mark.parametrize("explicit_only", [True, False])
def test_rate_limited_pool_keeps_its_provider_row(pooled_home, provider, cooldown, explicit_only):
    pooled_home(provider, cooldown)

    rows = build_model_options_payload(load_picker_context(), explicit_only=explicit_only)["providers"]

    row = next((r for r in rows if r["slug"] == provider), None)
    assert row is not None, f"a {cooldown} {provider} pool dropped the provider from model.options"
    assert row["models"], "the kept row must still offer the provider's models"


def test_picker_visibility_keeps_the_full_custom_probe_budget(monkeypatch):
    """Keeping cooldown rows visible must not shorten the current custom endpoint's live probe."""
    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", lambda *_a, **_kw: [])
    monkeypatch.setattr("hermes_cli.models.provider_model_ids", lambda *_a, **_kw: [])
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *_a, **_kw: None)
    monkeypatch.setattr("hermes_cli.models_local.fetch_ollama_local_models", lambda *_a, **_kw: None)

    def answers_after_1_5s(_api_key, _api_url, _native, _preserve, headers=None, timeout=5.0,
                           api_mode=None, **_kw):
        return ["slow-discovered-model"] if timeout >= 5.0 else None

    monkeypatch.setattr("hermes_cli.model_switch_providers._fetch_picker_live_models", answers_after_1_5s)
    ctx = load_picker_context().with_overrides(
        current_provider="custom", current_base_url="http://127.0.0.1:9999/v1", current_model="kept-model")

    rows = build_model_options_payload(ctx)["providers"]

    custom = next(r for r in rows if r["slug"] == "custom")
    assert "slow-discovered-model" in custom["models"]
