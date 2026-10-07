"""Web setup reports the selected backend without interrupting config writes."""

import pytest


@pytest.mark.parametrize("backend", ["firecrawl", "perplexity", "exa"])
def test_web_selection_reports_the_backend_it_writes(monkeypatch, backend):
    import rabbit_cli.tools_config_providers as providers

    messages = []
    monkeypatch.setattr(providers, "_print_success", messages.append)
    row = {"name": "Test provider", "web_backend": backend, "env_vars": []}
    config = {}
    providers._configure_provider(row, config)
    selected = config["web"]["backend"]
    assert selected == backend
    assert f"  Web backend set to: {selected}" in messages
