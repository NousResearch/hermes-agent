"""Tests for the multilingual stop-phrase matcher.

These guard against accidental drift in the conservative phrase set —
false-positives in any language would erode trust ("why did the bot
interrupt itself?") so the bar is high.
"""

from gateway.stop_phrases import (
    MAX_STOP_LEN,
    SLASH_STOP_COMMANDS,
    STOP_PHRASES,
    matches_stop_phrase,
    normalize,
)


# --- Empty / whitespace / slash universals ----------------------------------


def test_empty_string_matches_universal():
    assert matches_stop_phrase("") == "universal"


def test_none_matches_universal():
    assert matches_stop_phrase(None) == "universal"


def test_whitespace_only_matches_universal():
    assert matches_stop_phrase("   \t\n  ") == "universal"


def test_lone_slash_matches_slash():
    assert matches_stop_phrase("/") == "slash"


def test_slash_stop_commands_match_slash():
    for cmd in ["/stop", "/cancel", "/halt", "/abort"]:
        assert matches_stop_phrase(cmd) == "slash", cmd


# --- English exact matches --------------------------------------------------


def test_english_basic_words_match():
    for w in ["stop", "wait", "halt", "cancel", "abort", "pause"]:
        assert matches_stop_phrase(w) == "en", w


def test_english_case_insensitive():
    for w in ["STOP", "Stop", "sToP", "WAIT", "Halt"]:
        assert matches_stop_phrase(w) == "en", w


def test_english_trailing_punctuation_tolerated():
    for w in ["stop!", "wait.", "halt!.", "cancel!", "abort."]:
        assert matches_stop_phrase(w) == "en", w


def test_english_please_phrases():
    assert matches_stop_phrase("please stop") == "en"
    assert matches_stop_phrase("please wait") == "en"


# --- Other languages --------------------------------------------------------


def test_spanish_phrases_match():
    """Spanish 'para' deliberately excluded (also means preposition 'for')."""
    for w in ["alto", "espera", "cancela", "pausa"]:
        assert matches_stop_phrase(w) == "es", w


def test_spanish_para_alone_does_not_match():
    """Confirms 'para' (the preposition 'for') doesn't trigger a stop."""
    assert matches_stop_phrase("para") is None


def test_french_phrases_match():
    """Words unique to French resolve to 'fr'.

    Phrases shared with other languages (e.g. 'pause' in fr/de, 'stop' in
    en/fr/nl/pl) resolve to whichever language is checked first in dict
    iteration order. Either way the action is 'stop' — what matters.
    """
    for w in ["arrête", "arrete", "attends", "annule"]:
        result = matches_stop_phrase(w)
        assert result == "fr", f"{w} got {result}"


def test_german_phrases_match():
    for w in ["stopp", "warte", "abbrechen"]:
        assert matches_stop_phrase(w) == "de", w


def test_japanese_phrases_match():
    for w in ["止まれ", "止まって", "待って", "ストップ"]:
        assert matches_stop_phrase(w) == "ja", w


def test_japanese_full_stop_punctuation():
    assert matches_stop_phrase("止まって。") == "ja"


def test_mandarin_simplified_phrases_match():
    # 停 and 等等 are shared with traditional. Lookup hits Hans first by dict order.
    result = matches_stop_phrase("暂停")
    assert result == "zh-Hans"


def test_mandarin_traditional_specific():
    result = matches_stop_phrase("暫停")
    assert result == "zh-Hant"


def test_korean_phrases_match():
    """'잠깐' deliberately excluded (often a filler 'just a moment')."""
    for w in ["멈춰", "정지", "일시정지", "중단"]:
        assert matches_stop_phrase(w) == "ko", w


def test_russian_phrases_match():
    for w in ["стой", "стоп", "пауза", "отмена"]:
        assert matches_stop_phrase(w) == "ru", w


def test_arabic_phrases_match():
    for w in ["قف", "انتظر", "توقف"]:
        assert matches_stop_phrase(w) == "ar", w


def test_hindi_phrases_match():
    for w in ["रुको", "ठहरो", "बंद करो"]:
        assert matches_stop_phrase(w) == "hi", w


# --- Negative cases (the bar for these is HIGH) -----------------------------


def test_long_message_with_stop_buried_does_not_match():
    """The conservative length cap is the main false-positive defense."""
    msg = "we should stop including Bob in the email today"
    assert len(msg) > MAX_STOP_LEN
    assert matches_stop_phrase(msg) is None


def test_word_with_stop_prefix_does_not_match():
    """'stopover' is a word, 'stop' is the phrase. Exact-word match required."""
    assert matches_stop_phrase("stopover") is None
    assert matches_stop_phrase("stopgap") is None
    assert matches_stop_phrase("stoppage") is None


def test_normal_conversation_does_not_match():
    benign = [
        "hello there",
        "thanks for the help",
        "ok cool",
        "what about X?",
        "by the way also include the docs",
        "actually let me think about this",
        "could you also check Y",
        "instead of Z try W",  # contains "instead" but is a long sentence
        "different task perhaps",
    ]
    for msg in benign:
        result = matches_stop_phrase(msg)
        assert result is None, f"false-positive on: {msg!r} → {result}"


def test_unknown_short_word_does_not_match():
    for w in ["hi", "hey", "ok", "yes", "no", "sure", "ah"]:
        assert matches_stop_phrase(w) is None, w


def test_emoji_only_does_not_match():
    assert matches_stop_phrase("👍") is None
    assert matches_stop_phrase("🤔") is None


# --- normalize() helper -----------------------------------------------------


def test_normalize_lowercases():
    assert normalize("STOP") == "stop"
    assert normalize("Stop") == "stop"


def test_normalize_strips_whitespace():
    assert normalize("  stop  ") == "stop"


def test_normalize_strips_trailing_punct():
    assert normalize("stop!") == "stop"
    assert normalize("stop.") == "stop"
    assert normalize("stop!.") == "stop"
    assert normalize("止まって。") == "止まって"


def test_normalize_handles_none_and_empty():
    assert normalize(None) == ""
    assert normalize("") == ""


# --- Coverage sanity --------------------------------------------------------


def test_all_phrases_under_length_cap():
    """Every entry in STOP_PHRASES must be <= MAX_STOP_LEN normalized."""
    for lang, phrases in STOP_PHRASES.items():
        for p in phrases:
            n = normalize(p)
            assert len(n) <= MAX_STOP_LEN, f"{lang}: {p!r} normalized to {n!r} exceeds {MAX_STOP_LEN}"


def test_slash_commands_set_is_finite():
    assert isinstance(SLASH_STOP_COMMANDS, frozenset)
    assert "/" in SLASH_STOP_COMMANDS
    assert "/stop" in SLASH_STOP_COMMANDS


def test_lang_codes_are_canonical():
    """Language codes follow ISO 639 / IETF BCP-47 conventions."""
    valid = {
        "en", "es", "fr", "de", "pt", "it", "nl",
        "ja", "ko", "ru", "ar", "hi", "tr", "pl",
        "zh-Hans", "zh-Hant",
    }
    for lang in STOP_PHRASES.keys():
        assert lang in valid, f"unexpected language code: {lang}"
