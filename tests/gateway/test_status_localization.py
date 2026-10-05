"""Agent status lines are matched in English but shown to chat users in the active language.

Real temp HERMES_HOME with user overlays for ``en`` and ``de``; no i18n mocks."""

from __future__ import annotations

import hermes_yaml as yaml
import pytest

from agent import i18n, i18n_layers
from gateway.config import Platform
from gateway.run import _prepare_gateway_status_message
from gateway.warning_notifications import DiagnosticText

_CATALOGS = {
    "en": {"probe": {"noisy": "⚠ Auxiliary compression failed: {detail}", "plain": "Working on {what}"}},
    "de": {"probe": {"noisy": "⚠ Hilfskomprimierung fehlgeschlagen: {detail}", "plain": "Arbeite an {what}"}},
}


@pytest.fixture
def german_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "locales").mkdir(parents=True)
    for lang, data in _CATALOGS.items():
        (home / "locales" / f"{lang}.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({"display": {"language": "de"}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    i18n_layers._reset_registry_for_tests()
    i18n.reset_language_cache()
    yield home
    i18n_layers._reset_registry_for_tests()
    i18n.reset_language_cache()


def test_status_line_is_filtered_in_english_and_delivered_localized(german_home):
    plain = i18n.tl("probe.plain", what="x")
    assert plain == "Working on x"  # matchers, logs and transcripts see English
    assert _prepare_gateway_status_message(Platform.TELEGRAM, "lifecycle", plain) == "Arbeite an x"
    # A diagnostic wrapper keeps the key, so the sink still localizes it.
    assert _prepare_gateway_status_message(Platform.TELEGRAM, "lifecycle", DiagnosticText(plain)) == "Arbeite an x"
    # The noise filter matches the English value; the German text alone would slip through it.
    assert _prepare_gateway_status_message(Platform.TELEGRAM, "lifecycle", i18n.tl("probe.noisy", detail="y")) is None
