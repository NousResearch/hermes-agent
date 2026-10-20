"""Localisation of lifecycle/warning status lines (``agent.status_i18n``).

Covers the two properties that matter: English output is byte-for-byte unchanged, and the
English-matching classifications downstream (gateway noise filter, compaction-progress tagging)
still work when the user sees a translated line.
"""

from __future__ import annotations

import re

import pytest

from agent.status_i18n import (
    GENERIC_TEXTS,
    STATUS_TEXTS,
    normalize_status,
    translate_status,
)

SAMPLES = {
    "tokens": "120,000", "threshold": "100,000", "idle_seconds": "900", "attempt": "1",
    "cap": "3", "before": "310", "after": "48", "new_ctx": "70,000", "old_ctx": "150,000",
    "count": "4", "reason": "cooldown: 60s", "preflight_tokens": "900,000",
    "context_length": "1,048,576", "provider": "custom", "model": "gniu-flash",
    "error": "ReadTimeout", "previous_model": "ds-flash", "previous_provider": "litellm",
    "summary": "402 Payment Required", "n": "3", "max_retries": "5", "used": "60",
    "max": "60", "idle": "300", "elapsed": "45", "glyph": "🧠", "cost": "0.42",
    "budget": "2", "wait": "4.0", "note": "", "tool": "bash", "code": "destructive_command",
    "label": "DeepSeek",
}


def _fill(template: str) -> str:
    return re.sub(
        r"\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}",
        lambda m: SAMPLES.get(m.group("name"), "X"),
        template,
    )


@pytest.fixture
def zh(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    yield


@pytest.fixture
def en(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    yield


ALL_TEXTS = {**STATUS_TEXTS, **GENERIC_TEXTS}


def test_english_is_short_circuited(en):
    """English users must see the exact constants — no catalog lookup, no drift."""
    for en_text, _zh in ALL_TEXTS.values():
        raw = _fill(en_text)
        assert translate_status(raw) == raw


def test_every_catalog_entry_translates(zh):
    """Each mapped line renders in Chinese and differs from the English original."""
    for key, (en_text, _zh) in ALL_TEXTS.items():
        raw = _fill(en_text)
        assert translate_status(raw) != raw, f"{key} was not translated"


def test_placeholders_survive_translation(zh):
    """Dynamic values (counts, model names, errors) must reach the translated line."""
    out = translate_status(_fill("⚠️  Session compressed {count} times — accuracy may degrade. Consider /new to start fresh."))
    assert SAMPLES["count"] in out
    out = translate_status(_fill("✅ Primary model restored: {model} via {provider}; fallback {previous_model} via {previous_provider} is no longer active."))
    assert SAMPLES["model"] in out and SAMPLES["previous_provider"] in out


def test_unknown_lines_pass_through(zh):
    assert translate_status("just a normal sentence") == "just a normal sentence"
    assert normalize_status("🗜️ 正在压缩上下文 — 正在摘要之前的对话，以便继续…").startswith("🗜️ Compacting context")


def test_normalize_round_trip(zh):
    """Translated -> English round-trip powers every downstream English match."""
    for key, (en_text, _zh) in ALL_TEXTS.items():
        localized = translate_status(_fill(en_text))
        if "{" in localized:  # sample didn't cover every placeholder
            continue
        assert normalize_status(localized) != localized, f"{key} could not be normalised back"


def test_compaction_progress_recognised_when_localized(zh):
    from agent.conversation_compression import is_compaction_progress_status as is_progress

    running = translate_status("🗜️ Compacting context — summarizing earlier conversation so I can continue...")
    preflight = translate_status("📦 Preflight compression: ~120,000 tokens >= 100,000 threshold. This may take a moment.")
    done = translate_status("✓ Context compaction complete — continuing turn...")
    assert is_progress(running) is True
    assert is_progress(preflight) is True
    assert is_progress(done) is False


def test_gateway_noise_filter_still_suppresses_localized_chatter(zh):
    from gateway.run import _TELEGRAM_NOISY_STATUS_RE

    localized = translate_status("🗜️ Compacting context — summarizing earlier conversation so I can continue...")
    assert _TELEGRAM_NOISY_STATUS_RE.search(normalize_status(localized))
