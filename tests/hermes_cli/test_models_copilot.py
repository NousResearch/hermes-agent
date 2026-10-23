"""Copilot endpoint metadata selects the wire for current and unknown model families."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    ("model_id", "catalog_id", "endpoints", "expected"),
    [
        ("grok-4.5", None, None, "codex_responses"),
        ("grok-4.6", None, None, "codex_responses"),
        ("grok-4.7", None, None, "codex_responses"),
        ("xai/grok-4.7", None, None, "codex_responses"),
        ("github-copilot/grok-4.7", None, None, "codex_responses"),
        ("owner/sub/grok-4.7", None, None, "chat_completions"),
        ("gpt-6.1-sol", None, None, "codex_responses"),
        ("gpt-5-mini", None, None, "chat_completions"),
        ("new-model", None, None, "chat_completions"),
        ("new-model", "other-model", ["/responses"], "chat_completions"),
        ("new-model", "new-model", None, "chat_completions"),
        ("grok-4.7", "grok-4.7", [], "codex_responses"),
        ("new-model", "new-model", ["/responses"], "codex_responses"),
        ("new-model", "new-model", ["/chat/completions"], "chat_completions"),
        ("gpt-6.1-sol", "gpt-6.1-sol", ["/chat/completions"], "chat_completions"),
        ("gpt-5-mini", "gpt-5-mini", ["/responses"], "codex_responses"),
        ("new-model", "new-model", ["/chat/completions", "/responses"], "chat_completions"),
        ("gpt-6.1-sol", "gpt-6.1-sol", ["/chat/completions", "/responses"], "codex_responses"),
        ("gpt-5-mini", "gpt-5-mini", ["/chat/completions", "/responses"], "chat_completions"),
        ("xai/grok-4.7", "grok-4.7", ["/responses"], "codex_responses"),
        ("claude-opus-4.8", "claude-opus-4.8", ["/v1/messages"], "chat_completions"),
        ("new-model", "new-model", ["ws:/responses"], "chat_completions"),
    ],
)
def test_copilot_selects_advertised_http_wire_or_compatible_fallback(
    monkeypatch, model_id, catalog_id, endpoints, expected,
):
    from hermes_cli import models

    catalog = [{"id": catalog_id, "supported_endpoints": endpoints}] if catalog_id else None
    fetch = Mock(return_value=catalog)
    monkeypatch.setattr(models, "fetch_github_model_catalog", fetch)
    assert models.copilot_model_api_mode(model_id, catalog=catalog, api_key="test-token") == expected
    assert fetch.call_count == (1 if catalog is None else 0)


@pytest.mark.parametrize("provider", ["copilot", "github-copilot", None])
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize(
    ("model_id", "endpoints", "configured_mode", "expected"),
    [
        ("grok-4.7", None, None, "codex_responses"),
        ("grok-4.7", None, "codex_responses", "codex_responses"),
        ("grok-4.7", None, "chat_completions", "codex_responses"),
        ("grok-4.7", ["/responses"], "chat_completions", "codex_responses"),
        ("new-model", ["/responses"], None, "codex_responses"),
        ("new-model", ["/responses"], "chat_completions", "codex_responses"),
        ("new-model", ["/chat/completions"], "codex_responses", "chat_completions"),
        ("new-model", ["/chat/completions", "/responses"], "codex_responses", "codex_responses"),
        ("new-model", ["/chat/completions", "/responses"], "chat_completions", "chat_completions"),
        ("gpt-5-mini", ["/chat/completions", "/responses"], "codex_responses", "chat_completions"),
        ("claude-opus-4.8", ["/v1/messages"], "codex_responses", "chat_completions"),
    ],
)
def test_runtime_respects_endpoint_metadata_for_default_and_override(
    tmp_path, monkeypatch, provider, override, model_id, endpoints, configured_mode, expected,
):
    from hermes_cli import models
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    model = {"default": "gpt-6.1-sol" if override else model_id}
    if provider:
        model["provider"] = provider
    if configured_mode:
        model["api_mode"] = configured_mode
    (tmp_path / "config.yaml").write_text(json.dumps({"model": model}), encoding="utf-8")
    catalog = [{"id": model_id, "supported_endpoints": endpoints}] if endpoints is not None else None
    monkeypatch.setattr(models, "fetch_github_model_catalog", lambda **_kwargs: catalog)

    request = {} if provider else {"requested": "copilot"}
    runtime = resolve_runtime_provider(
        explicit_api_key="test-token", target_model=model_id if override else None, **request,
    )

    assert runtime["provider"] == "copilot"
    assert runtime["api_mode"] == expected
