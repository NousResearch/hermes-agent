"""Regression tests for section-3 (``providers:``) same-endpoint grouping in
``list_authenticated_providers`` and for ``format_model_for_display``.

Salvaged with PR #36998 (@antydizajn): section 3 folds ``providers:`` entries
that share (api_url, credential, api_mode, extra_headers) into one picker row,
mirroring section 4's grouping for ``custom_providers:``. These are invariant
tests — grouping identity, header-routed separation, list-of-dict model
declarations, and display-only RID stripping.

Section-4 builtin-shadow dedup tests (added for credential-aware dedup fix):
a ``custom_providers`` entry is now only hidden when it duplicates a built-in's
endpoint **AND** credential — not just the URL. We exercise the dedup
decision in isolation to keep the tests offline and fast.
"""

import os

import hermes_cli.providers as providers_mod
from hermes_cli.model_switch import (
    format_model_for_display,
    list_authenticated_providers,
)


def _custom(monkeypatch, custom_providers):
    """Run the picker with a minimal stub for network probes."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *a, **k: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *a, **k: [])
    return list_authenticated_providers(
        user_providers={},
        custom_providers=custom_providers,
        max_models=50,
    )


def test_same_key_same_url_shadows_builtin(monkeypatch):
    """A custom_providers entry with the same endpoint and same API key as the
    built-in alibaba-coding-plan row is hidden (#16970 invariant).

    We let the full built-in path run (alibaba-coding-plan is in the real
    PROVIDER_REGISTRY) and set the matching env var so the credential
    identity lands in ``_builtin_endpoints``.
    """
    same_key = "sk-test-shared-abc"
    monkeypatch.setenv("ALIBABA_CODING_PLAN_API_KEY", same_key)
    rows = _custom(monkeypatch, [
        {
            "name": "My Dashscope",
            "base_url": "https://coding-intl.dashscope.aliyuncs.com/v1",
            "api_key": same_key,
            "model": "qwen3.7-max",
        },
    ])
    user_rows = [p for p in rows if p.get("source") == "user-config"]
    assert not user_rows, "Same-key shadow must be hidden"


def test_distinct_key_same_url_keeps_own_row(monkeypatch):
    """A custom_providers entry with the same endpoint but a different API key
    (or a different auth mode such as OAuth vs key) keeps its own picker row.

    Covers the reporter's case: topup_nous_api (API key) beside the built-in
    Nous Portal (OAuth device-code) on inference-api.nousresearch.com.

    Nous is an OAuth-login built-in — its credential identity in
    ``_builtin_endpoints`` is the sentinel "oauth". No key-based custom entry
    can match that, so our topup row must survive the section-4 dedup.
    """
    topup_key = "sk-nous-topup-123"
    monkeypatch.setenv(
        "HERMES_CUSTOM_INFERENCE_API_NOUSRESEARCH_COM_API_KEY",
        topup_key,
    )
    rows = _custom(monkeypatch, [
        {
            "name": "topup_nous_api",
            "base_url": "https://inference-api.nousresearch.com/v1",
            "key_env": "HERMES_CUSTOM_INFERENCE_API_NOUSRESEARCH_COM_API_KEY",
            "model": "deepseek/deepseek-v4-flash-0731",
            "models": ["deepseek/deepseek-v4-flash-0731", "sakana/sakana-namazu"],
        },
    ])
    user_rows = [p for p in rows if p.get("source") == "user-config"]
    assert len(user_rows) == 1
    row = user_rows[0]
    assert row["slug"] == "custom:topup_nous_api"
    assert "sakana/sakana-namazu" in row["models"]
    assert "deepseek/deepseek-v4-flash-0731" in row["models"]


def test_distinct_key_same_url_dashscope_keeps_own_row(monkeypatch):
    """Mirror of the Nous case using an API-key built-in: the custom row uses
    a different API key value than the built-in and must NOT be hidden.

    We set ALIBABA_CODING_PLAN_API_KEY to a value the custom row does NOT
    carry; the dedup then sees the built-in credential ≠ custom credential
    and surfaces the row.
    """
    builtin_key = "sk-alibaba-456"
    other_key = "sk-other-distinct-key"
    monkeypatch.setenv("ALIBABA_CODING_PLAN_API_KEY", builtin_key)
    rows = _custom(monkeypatch, [
        {
            "name": "my-dashscope-other-key",
            "base_url": "https://coding-intl.dashscope.aliyuncs.com/v1",
            "api_key": other_key,
            "model": "qwen3.7-max",
        },
    ])
    user_rows = [p for p in rows if p.get("source") == "user-config"]
    assert len(user_rows) == 1
    assert "my-dashscope-other-key" in user_rows[0]["slug"]


def test_no_credentials_recorded_keeps_custom_row(monkeypatch):
    """When the built-in's credentials are not captured (e.g. env var unset
    at probe time), the dedup conservatively keeps the custom row rather than
    hiding it on a missing-cred signal.
    """
    # Unset the env var so alibaba-coding-plan has no captured credential
    monkeypatch.delenv("ALIBABA_CODING_PLAN_API_KEY", raising=False)
    rows = _custom(monkeypatch, [
        {
            "name": "my-dashscope-unset-key",
            "base_url": "https://coding-intl.dashscope.aliyuncs.com/v1",
            "api_key": "sk-any-key",
            "model": "qwen3.7-max",
        },
    ])
    user_rows = [p for p in rows if p.get("source") == "user-config"]
    assert len(user_rows) == 1
    assert "my-dashscope-unset-key" in user_rows[0]["slug"]


class TestFormatModelForDisplay:
    def test_palantir_rid_stripped_to_trailing_slug(self):
        rid = "ri.language-model-service..language-model.anthropic-claude-4-7-opus"
        assert format_model_for_display(rid) == "anthropic-claude-4-7-opus"
