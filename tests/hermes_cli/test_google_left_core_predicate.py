"""Tests for the prepared Google (Gemini + Solstice) left-core migration predicate."""

from __future__ import annotations

import pytest

from hermes_cli.left_core_migration import _GOOGLE_PROVIDERS, google_in_use


@pytest.mark.parametrize("alias", _GOOGLE_PROVIDERS)
def test_explicit_google_or_solstice_provider_selection_counts_as_in_use(tmp_path, alias):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(f"model:\n  provider: {alias}\n", encoding="utf-8")
    assert google_in_use(home) is True


def test_custom_provider_shadowing_google_alias_does_not_trigger(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  provider: google\ncustom_providers:\n  - name: google\n    base_url: https://proxy.example/v1\n",
        encoding="utf-8",
    )
    assert google_in_use(home) is False


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://generativelanguage.googleapis.com/v1beta", True),
        ("https://generativelanguage.googleapis.com/v1beta/openai", False),
        ("https://aiplatform.googleapis.com/v1beta1", True),
        ("https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi", False),
    ],
)
def test_url_matching_distinguishes_native_and_express_from_openai_compat_and_vertex_oauth(
    tmp_path, base_url, expected,
):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"model:\n  provider: custom\n  base_url: {base_url}\n",
        encoding="utf-8",
    )
    assert google_in_use(home) is expected


def test_env_key_autodetect_counts_when_gemini_would_win_auto(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("GEMINI_API_KEY=AQ.real-test-key-value\n", encoding="utf-8")
    assert google_in_use(home) is True


def test_env_key_used_only_for_tts_with_another_selected_provider_does_not_trigger(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("GEMINI_API_KEY=AQ.real-test-key-value\n", encoding="utf-8")
    (home / "config.yaml").write_text("model:\n  provider: anthropic\ntts:\n  provider: gemini\n", encoding="utf-8")
    assert google_in_use(home) is False


@pytest.mark.parametrize("higher_var", ["OPENROUTER_API_KEY", "OPENAI_API_KEY"])
def test_env_key_outranked_by_higher_priority_provider_does_not_trigger(tmp_path, higher_var):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text(
        f"GEMINI_API_KEY=AQ.real-test-key-value\n{higher_var}=sk-higher-priority-key\n",
        encoding="utf-8",
    )
    assert google_in_use(home) is False


def test_lm_studio_key_does_not_outrank_gemini_autodetect(tmp_path):
    """LM Studio is excluded from env auto-detect, so GEMINI_API_KEY still wins auto."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("GEMINI_API_KEY=AQ.real-test-key-value\nLM_API_KEY=lm-local-token\n", encoding="utf-8")
    assert google_in_use(home) is True


def test_openrouter_pool_credential_outranks_gemini_env_key(tmp_path):
    import json

    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("GEMINI_API_KEY=AQ.real-test-key-value\n", encoding="utf-8")
    (home / "auth.json").write_text(
        json.dumps({"credential_pool": {"openrouter": [{"source": "manual", "api_key": "sk-or-v1-real"}]}}),
        encoding="utf-8",
    )
    assert google_in_use(home) is False


def test_custom_loopback_base_url_prevents_gemini_env_key_trigger(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("GEMINI_API_KEY=AQ.real-test-key-value\n", encoding="utf-8")
    (home / "config.yaml").write_text("model:\n  base_url: http://127.0.0.1:8000/v1\n", encoding="utf-8")
    assert google_in_use(home) is False
