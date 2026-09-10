"""ACP ``session/set_model``: an explicit ``provider:model`` prefix is a selection, never re-detected.

Regression for #59089: ``anthropic:claude-sonnet-5`` while already on anthropic resolved to the same
provider, so the bare-name fallback ``detect_provider_for_model`` ran and could hand the session to
OpenRouter because the bare name appears in its catalog.
"""

from __future__ import annotations

from acp_adapter.server import HermesACPAgent


def test_explicit_provider_prefix_skips_detection(monkeypatch):
    calls: list[tuple[str, str]] = []

    def hijack(model, current):
        calls.append((model, current))
        return ("openrouter", f"anthropic/{model}")

    monkeypatch.setattr("hermes_cli.models.detect_provider_for_model", hijack)
    assert HermesACPAgent._resolve_model_selection("anthropic:claude-sonnet-5", "anthropic") == (
        "anthropic", "claude-sonnet-5")
    assert calls == []


def test_bare_name_still_uses_detection(monkeypatch):
    monkeypatch.setattr("hermes_cli.models.detect_provider_for_model", lambda m, c: ("deepseek", m))
    assert HermesACPAgent._resolve_model_selection("deepseek-flash", "anthropic") == ("deepseek", "deepseek-flash")
