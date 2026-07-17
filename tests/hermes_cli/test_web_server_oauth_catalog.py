import asyncio

import hermes_cli.web_server as web_server


def test_oauth_catalog_exposes_provider_description(monkeypatch):
    monkeypatch.setattr(
        web_server,
        "_resolve_provider_status",
        lambda _provider_id, _status_fn: {"logged_in": False},
    )

    result = asyncio.run(web_server.list_oauth_providers())
    claude_code = next(
        provider for provider in result["providers"] if provider["id"] == "claude-code"
    )

    assert claude_code["name"] == "Anthropic OAuth (Claude Code)"
    assert claude_code["description"] == (
        "Requires extra usage credits on top of a Claude Max plan."
    )
