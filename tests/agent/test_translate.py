"""Tests for agent.translate -- on-demand translation layer for tool output,
errors and fetched content (#123591).

Safety contract under test:
- code spans, file paths and flags are never translated (placeholder round-trip)
- the original is always kept alongside the translation
- uncertainty is flagged, never presented confidently
- target language follows the UI locale with an independent override
- automatic translation is OFF by default
"""

from __future__ import annotations

import pytest

from agent import translate


def _echo_backend(prompt: str, target: str) -> str:
    """Stub backend: emulates a perfectly faithful model reply (protected source only)."""
    return prompt.split("\n\n", 1)[1] if "\n\n" in prompt else prompt


def test_resolve_target_defaults_to_ui_locale(monkeypatch):
    monkeypatch.setattr(translate.i18n, "get_language", lambda: "ru")
    assert translate.resolve_target(None) == "ru"


def test_resolve_target_override_normalizes_alias():
    assert translate.resolve_target("russian") == "ru"
    assert translate.resolve_target("  DE  ") == "de"


def test_resolve_target_unknown_falls_back_to_english():
    assert translate.resolve_target("klingon") == "en"


def test_empty_source_needs_no_backend():
    result = translate.translate_text("", target="ru", backend=_echo_backend)
    assert result.translation == ""
    assert result.source == ""


def test_protect_round_trip_keeps_code_paths_and_flags_verbatim():
    source = (
        "Build failed with 2 errors.\n"
        "```\nTraceback (most recent call last):\n  File `app.py` line 1\n```\n"
        "See `src/main.py` and run with --verbose. Output went to /tmp/build log.txt."
    )
    result = translate.translate_text(source, target="ru", backend=_echo_backend)
    assert result.translation == source  # echo backend: faithful MT restores everything
    assert "src/main.py" in result.translation
    assert "--verbose" in result.translation
    assert "/tmp/build" in result.translation


def test_protected_source_hides_machine_exact_text_from_backend():
    seen = {}

    def spy_backend(prompt: str, target: str) -> str:
        seen["prompt"] = prompt
        return prompt.split("\n\n", 1)[1] if "\n\n" in prompt else prompt

    source = "Error in `deploy.sh --force`: file not found"
    translate.translate_text(source, target="ru", backend=spy_backend)
    assert "deploy.sh --force" not in seen["prompt"]
    assert "KEEP" in seen["prompt"]


def test_dropped_placeholder_marks_uncertain():
    def lossy_backend(prompt: str, target: str) -> str:
        return "перевод без плейсхолдеров"

    source = "Run `make build` now"
    result = translate.translate_text(source, target="ru", backend=lossy_backend)
    assert result.uncertain is True


def test_rough_marker_marks_uncertain_and_is_stripped():
    def rough_backend(prompt: str, target: str) -> str:
        return "перевод текста\nROUGH: idiom rendered loosely"

    result = translate.translate_text("plain prose", target="ru", backend=rough_backend)
    assert result.uncertain is True
    assert "ROUGH" not in result.translation


def test_confident_reply_is_not_flagged():
    result = translate.translate_text("plain prose", target="ru", backend=_echo_backend)
    assert result.uncertain is False


def test_render_keeps_original_alongside_translation():
    rendered = translate.render_translation(
        translate.TranslationResult(
            source="Build failed", translation="Сборка не удалась",
            target="ru", uncertain=False,
        )
    )
    assert "Build failed" in rendered
    assert "Сборка не удалась" in rendered


def test_render_flags_uncertain_translation():
    rendered = translate.render_translation(
        translate.TranslationResult(
            source="Build failed", translation="Сборка не удалась",
            target="ru", uncertain=True,
        )
    )
    assert "Build failed" in rendered  # original still reachable
    assert "rough" in rendered.lower() or "uncertain" in rendered.lower()


def test_auto_translate_is_off_by_default():
    assert translate.auto_enabled("some-session") is False
    assert translate.should_auto_translate("some-session", "Некоторая ошибка") is False


def test_auto_mode_translates_foreign_script():
    translate.set_auto("sess-auto", True)
    try:
        assert translate.should_auto_translate("sess-auto", "Некоторая ошибка", target="en") is True
        assert translate.should_auto_translate("sess-auto", "plain english", target="en") is False
    finally:
        translate.set_auto("sess-auto", False)


def test_session_target_override_round_trip():
    translate.set_target("sess-1", "de")
    try:
        assert translate.effective_target("sess-1", None) == "de"
        assert translate.effective_target("sess-1", "fr") == "fr"  # explicit arg wins
    finally:
        translate.set_target("sess-1", None)
