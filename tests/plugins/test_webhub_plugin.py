"""Tests for the bundled observability/webhub plugin."""

from __future__ import annotations

import importlib


def _reload_webhub(monkeypatch, tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: home)
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: home / "config.yaml")
    monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: home / ".env")
    mod = importlib.import_module("plugins.observability.webhub")
    return importlib.reload(mod), home


def test_webhub_records_openrouter_success_and_writes_briefing(monkeypatch, tmp_path):
    webhub, home = _reload_webhub(monkeypatch, tmp_path)

    webhub.on_post_api_request(
        session_id="s1",
        turn_id="t1",
        api_request_id="r1",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4.5",
        api_mode="chat_completions",
        api_duration=1.25,
        usage={
            "input_tokens": 120,
            "output_tokens": 30,
            "cache_read_tokens": 10,
            "cache_write_tokens": 0,
            "reasoning_tokens": 5,
            "request_count": 1,
            "total_tokens": 160,
        },
        finish_reason="stop",
    )

    summary = webhub.get_dashboard_summary(window_hours=24)
    assert summary["requests"]["total"] == 1
    assert summary["requests"]["ok"] == 1
    assert summary["tokens"]["total"] == 160
    assert "OpenRouter requests: 1 total" in summary["briefing_markdown"]
    assert (home / "observability" / "webhub" / "latest_briefing.md").exists()


def test_webhub_records_errors_and_renders_prometheus(monkeypatch, tmp_path):
    webhub, _home = _reload_webhub(monkeypatch, tmp_path)

    webhub.on_api_request_error(
        session_id="s1",
        turn_id="t2",
        api_request_id="r2",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4.1",
        api_mode="chat_completions",
        api_duration=0.5,
        error={"type": "rate_limit", "message": "Too many requests"},
    )

    summary = webhub.get_dashboard_summary(window_hours=24)
    assert summary["requests"]["error"] == 1
    assert summary["recent_errors"][0]["error_type"] == "rate_limit"

    metrics = webhub.render_prometheus_metrics()
    assert "hermes_webhub_requests_total 1" in metrics
    assert 'hermes_webhub_requests_status_total{status="error"} 1' in metrics
