"""claude-code-cli provider resolution — every alias must reach the runtime,
never fall through to OpenRouter/custom (#25267)."""

from __future__ import annotations

import pytest

from hermes_cli.runtime_provider import (
    CLAUDE_CODE_CLI_PROVIDER_NAMES,
    resolve_runtime_provider,
)


@pytest.mark.parametrize("name", sorted(CLAUDE_CODE_CLI_PROVIDER_NAMES))
def test_every_alias_resolves_to_the_claude_code_runtime(name, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    runtime = resolve_runtime_provider(requested=name)
    assert runtime["provider"] == "claude-code-cli"
    assert runtime["api_mode"] == "claude_code"
    assert runtime["base_url"] == "claude-code://local"
    assert runtime["requested_provider"] == name


def test_aliases_match_the_provider_profile():
    from providers import get_provider_profile

    profile = get_provider_profile("claude-code-cli")
    assert {profile.name, *profile.aliases} == set(CLAUDE_CODE_CLI_PROVIDER_NAMES)


def test_claude_code_alias_still_means_anthropic():
    from hermes_cli.auth import resolve_provider

    assert resolve_provider("claude-code") == "anthropic"
