"""English overlays customize presentation without rewriting machine-facing status or failure text."""

from types import SimpleNamespace

import hermes_yaml as yaml
import pytest

from agent import i18n
from agent.error_classifier import FailoverReason
from agent.turn_failure_copy import exhausted_copy, nonretryable_copy
from gateway.config import Platform
from gateway.run import _prepare_gateway_status_message


@pytest.fixture
def english_overlay(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "locales").mkdir(parents=True)
    values = {
        "display.status.turn.retry_wait": "Custom retry {wait} {attempt}/{max}",
        "explainer.failure.exhausted_lead.first_rate_limit": "Custom lead {label}",
        "explainer.failure.exhausted_situation_reset": "Custom reset {window}",
        "explainer.failure.exhausted": "Custom failure {lead} {situation} {summary}",
        "explainer.failure.nonretryable.data_policy_denied": "Custom privacy {model} {label}",
    }
    (home / "locales" / "en.yaml").write_text(yaml.safe_dump(values), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    i18n.reset_language_cache()
    yield
    i18n.reset_language_cache()


def test_english_overlay_cannot_turn_a_noisy_retry_into_a_delivered_status(english_overlay):
    line = i18n.tl("display.status.turn.retry_wait", wait="5.0", attempt=1, max=3)
    assert i18n.render_localized(line) == "Custom retry 5.0 1/3"
    assert _prepare_gateway_status_message(Platform.TELEGRAM, "info", line) is None


def test_failure_storage_keeps_source_english_while_the_reply_uses_the_overlay(english_overlay):
    fields = {"label": "Nous", "attempts": 1, "summary": "HTTP 429", "reset_seconds": 75}
    stored = exhausted_copy("rate_limit", **fields)
    shown = exhausted_copy("rate_limit", lang=None, **fields)
    assert "Custom" not in stored
    assert "Custom lead Nous" in shown and "Custom reset" in shown
    assert "resets in" in stored and "/retry" in stored

    verdict = SimpleNamespace(reason=FailoverReason.data_policy_denied)
    fields = {"provider": "nous", "model": "free-model", "summary": "policy"}
    stored = nonretryable_copy(verdict, **fields)
    shown = nonretryable_copy(verdict, lang=None, **fields)
    assert "Custom" not in stored and "/compress" in stored
    assert shown.startswith("Custom privacy free-model ")
