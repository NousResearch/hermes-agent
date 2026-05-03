"""Tests for hermes memory status output."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from hermes_cli import memory_setup


class _UnavailableProvider:
    def is_available(self):
        return False

    def get_config_schema(self):
        return [
            {"key": "api_key", "env_var": "HINDSIGHT_API_KEY", "url": "https://example.invalid"},
            {"key": "api_key", "env_var": "HINDSIGHT_API_KEY"},
            {"key": "llm_api_key", "env_var": "HINDSIGHT_LLM_API_KEY"},
        ]


def test_cmd_status_deduplicates_missing_env_vars(monkeypatch):
    monkeypatch.setattr(
        memory_setup,
        "_get_available_providers",
        lambda: [("hindsight", "API key / local", _UnavailableProvider())],
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"provider": "hindsight"}},
    )
    monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)
    monkeypatch.delenv("HINDSIGHT_LLM_API_KEY", raising=False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        memory_setup.cmd_status(None)

    out = buf.getvalue()
    assert out.count("HINDSIGHT_API_KEY") == 1
    assert "HINDSIGHT_API_KEY  → https://example.invalid" in out
    assert out.count("HINDSIGHT_LLM_API_KEY") == 1
