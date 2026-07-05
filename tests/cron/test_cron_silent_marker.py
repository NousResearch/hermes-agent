"""Tests for the [SILENT] marker classification in cron scheduler.

Regression: previously the guard used exact whole-text equality
(``deliver_content.strip().upper() == SILENT_MARKER``), which failed to
suppress delivery when the agent prepended reasoning before the marker
despite the prompt instructing "respond with exactly [SILENT]".

The fix moved to a last-non-empty-line check.
"""

from cron.scheduler import SILENT_MARKER, _classify_silent_marker


def test_silent_clean_exact_marker_with_whitespace():
    """Response is exactly the marker (with surrounding whitespace) → suppress."""
    assert _classify_silent_marker("  [SILENT]  ") == "silent_clean"
    assert _classify_silent_marker("[SILENT]") == "silent_clean"
    assert _classify_silent_marker("\n\n[SILENT]\n") == "silent_clean"


def test_silent_dirty_reasoning_then_marker():
    """Prefix reasoning + trailing [SILENT] on its own line → suppress (with warn)."""
    text = (
        "Проверил кроны за последние сутки, ошибок не обнаружено, все прошло штатно.\n"
        "\n"
        "[SILENT]"
    )
    assert _classify_silent_marker(text) == "silent_dirty"


def test_deliver_when_marker_not_last_line():
    """[SILENT] in the middle of a real report must NOT suppress delivery."""
    text = (
        "Отчёт по кронам:\n"
        "- job A: OK\n"
        "- job B вернул [SILENT] — норм, подавлено\n"
        "- job C: OK\n"
        "\n"
        "Итого 3 задачи, все успешно."
    )
    assert _classify_silent_marker(text) == "deliver"


def test_deliver_empty_response():
    """Empty/whitespace response is handled by a separate guard → 'deliver'."""
    assert _classify_silent_marker("") == "deliver"
    assert _classify_silent_marker("   \n\n  ") == "deliver"


def test_silent_marker_case_insensitive():
    """Marker matching is case-insensitive (upper() in classifier)."""
    assert _classify_silent_marker("[silent]") == "silent_clean"
    assert _classify_silent_marker("reasoning\n\n[Silent]") == "silent_dirty"


def test_silent_marker_constant():
    """Sanity: the marker constant is what we expect."""
    assert SILENT_MARKER == "[SILENT]"
