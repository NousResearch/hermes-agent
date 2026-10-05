"""Compression status lines: matchers read the English constants, chat shows the active language.

The emit sites only call ``<TEMPLATE>.format(...)``; the value stays the constant's English rendering (noise filter,
progress gate, TUI tagging) and the gateway sink renders the catalog line. Real temp HERMES_HOME overlay, no i18n mocks.
"""

import pytest

import gateway.run as gateway_run
from agent import i18n
from agent.conversation_compression import (
    CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE,
    PREFLIGHT_COMPRESSION_STATUS_TEMPLATE,
    ROUTINE_COMPRESSION_STATUS_SAMPLES,
)
from gateway.config import Platform
from gateway.run import _prepare_gateway_status_message
from hermes_constants import get_hermes_home

_DE_PREFLIGHT = "📦 Vorab-Komprimierung: ~{tokens} Tokens >= Schwelle {threshold}. Das kann einen Moment dauern."


@pytest.fixture
def german_overlay(monkeypatch):
    overlay = get_hermes_home() / "locales"
    overlay.mkdir(parents=True, exist_ok=True)
    (overlay / "de.yaml").write_text(f'"display.status.compress.preflight": "{_DE_PREFLIGHT}"\n', encoding="utf-8")
    monkeypatch.setenv("HERMES_LANGUAGE", "de")
    i18n.reset_language_cache()
    yield
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    i18n.reset_language_cache()


def test_routine_compression_status_is_filtered_in_english_and_opt_in_shows_it_localized(german_overlay, monkeypatch):
    line = PREFLIGHT_COMPRESSION_STATUS_TEMPLATE.format(tokens=120000, threshold=100000)
    assert line == "📦 Preflight compression: ~120,000 tokens >= 100,000 threshold. This may take a moment."

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    assert _prepare_gateway_status_message(Platform.TELEGRAM, "lifecycle", line) is None

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"compression": {"progress_notices": True}})
    assert _prepare_gateway_status_message(Platform.TELEGRAM, "lifecycle", line) == _DE_PREFLIGHT.format(
        tokens="120,000", threshold="100,000")


@pytest.mark.parametrize("line", [
    *ROUTINE_COMPRESSION_STATUS_SAMPLES,
    CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(tokens=250000, threshold=200000, reason="cooldown: 120s remaining"),
], ids=lambda line: line.i18n_key)
def test_english_chat_rendering_is_the_matched_constant(line):
    """An English chat must read exactly what the matchers matched: the catalog cannot drift from the constants."""
    assert i18n.t(line.i18n_key, lang="en", **line.i18n_kwargs) == line
