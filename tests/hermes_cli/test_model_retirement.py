"""Primary-model retirement detection and model-independent recovery guidance."""
from __future__ import annotations

import pytest

from hermes_cli.model_retirement import (
    GPT55_CODEX_RETIREMENT,
    find_retired_primary_model,
    format_retirement_notice,
)


def _config(provider: str, model: str) -> dict:
    return {"model": {"provider": provider, "default": model}}


@pytest.mark.parametrize("provider", ["openai-codex", "chatgpt", "chatgpt-codex"])
@pytest.mark.parametrize("model", ["gpt-5.5", "openai/gpt-5.5"])
def test_finds_retired_configured_primary_codex_model(provider, model):
    issue = find_retired_primary_model(_config(provider, model))

    assert issue is not None
    assert issue.config_path == "model.default"
    assert issue.provider == "openai-codex"
    assert issue.configured_model == model
    assert issue.retirement is GPT55_CODEX_RETIREMENT


@pytest.mark.parametrize(
    "provider",
    ["openai-api", "openrouter", "github-copilot", "custom", "auto", ""],
)
def test_same_slug_on_other_routes_is_not_flagged(provider):
    assert find_retired_primary_model(_config(provider, "gpt-5.5")) is None


def test_detector_does_not_scan_non_primary_slots():
    cfg = {
        "model": {"provider": "openai-codex", "default": "gpt-6.1-sol"},
        "fallback_providers": [
            {"provider": "openai-codex", "model": "gpt-5.5"},
        ],
        "auxiliary": {
            "compression": {"provider": "openai-codex", "model": "gpt-5.5"},
        },
        "delegation": {"provider": "openai-codex", "model": "gpt-5.5"},
    }

    assert find_retired_primary_model(cfg) is None


def test_retirement_notice_is_actionable_without_selecting_replacement():
    issue = find_retired_primary_model(_config("openai-codex", "gpt-5.5"))
    assert issue is not None

    notice = format_retirement_notice(issue)

    assert "openai-codex/gpt-5.5" in notice
    assert "October 14, 2026" in notice
    assert "hermes model --refresh" in notice
    assert "available to your account" in notice
    assert "No configuration was changed" in notice
    assert "gpt-6" not in notice


def test_startup_warning_is_actionable_and_non_mutating(monkeypatch, capsys):
    import hermes_cli.main as cli_main

    cfg = _config("openai-codex", "gpt-5.5")
    before = {"model": dict(cfg["model"])}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)

    cli_main._warn_retired_xai_models()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "openai-codex/gpt-5.5" in captured.err
    assert "hermes model --refresh" in captured.err
    assert "No configuration was changed" in captured.err
    assert cfg == before


def test_gateway_startup_logs_primary_retirement_warning(monkeypatch, caplog):
    import logging
    from hermes_cli.startup_model_retirement import log_retired_primary_model

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: _config("openai-codex", "gpt-5.5"),
    )

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        log_retired_primary_model(logging.getLogger("gateway.run"))

    assert "openai-codex/gpt-5.5" in caplog.text
    assert "hermes model --refresh" in caplog.text


def test_doctor_reports_retired_primary_as_manual_even_with_fix(monkeypatch):
    from hermes_cli.doctor_config import _check_primary_model_retirement

    cfg = _config("openai-codex", "gpt-5.5")
    before = {"model": dict(cfg["model"])}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)

    finding = _check_primary_model_retirement(True)

    assert finding.fixed == 0
    assert finding.issues == []
    assert len(finding.manual_issues) == 1
    assert "hermes model --refresh" in finding.manual_issues[0]
    assert cfg == before
