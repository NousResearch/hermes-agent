"""Tests for the post-STT keyword gate on Discord voice channels.

The gate lives in ``DiscordAdapter._matches_voice_keyword``,
``_fold_case_accents`` and ``_keyword_block_coverage`` in
plugins/platforms/discord/adapter.py.  The matching logic is pure (no
discord.py, no I/O), so it is tested via the standard
``object.__new__(DiscordAdapter)`` helper used elsewhere in the voice suite.
"""

from unittest.mock import patch

import pytest

from plugins.platforms.discord.adapter import DiscordAdapter


def _adapter(keywords, similarity=0.5):
    """Minimal DiscordAdapter whose gate is armed with *keywords*.

    ``similarity`` is the fuzzy-match tolerance; 0 disables fuzzy matching.
    """
    adapter = object.__new__(DiscordAdapter)
    adapter._voice_keywords = [kw.lower() for kw in keywords]
    adapter._voice_keyword_similarity = similarity
    return adapter


class TestMatchesVoiceKeyword:
    def test_disabled_gate_passthrough(self):
        # Empty keyword list = gate disabled: every utterance is kept as-is.
        adapter = _adapter([])
        assert adapter._matches_voice_keyword("bonjour tout le monde") == \
            "bonjour tout le monde"
        assert adapter._matches_voice_keyword("") == ""

    def test_matching_prefix_is_stripped(self):
        adapter = _adapter(["hey hermes", "jarvis"])
        assert adapter._matches_voice_keyword(
            "hey hermes, quelle heure est-il ?") == "quelle heure est-il ?"
        assert adapter._matches_voice_keyword(
            "Hey Hermes donne la météo") == "donne la météo"
        assert adapter._matches_voice_keyword(
            "jarvis allume la lumière") == "allume la lumière"

    def test_case_insensitive(self):
        adapter = _adapter(["jarvis"])
        assert adapter._matches_voice_keyword("JARVIS la lumière") == \
            "la lumière"
        assert adapter._matches_voice_keyword("jArViS la lumière") == \
            "la lumière"

    def test_accent_insensitive(self):
        adapter = _adapter(["hey hermes"])
        assert adapter._matches_voice_keyword("Hé Hermès, l'heure ?") == \
            "l'heure ?"
        assert adapter._matches_voice_keyword("hey hermès, la météo") == \
            "la météo"

    def test_bare_content_word_without_prefix_is_filtered(self):
        # Just "hermes" (without "hey ") is NOT a full trigger: the word
        # boundary after stripping the 10-char keyword lands mid-word, so the
        # utterance is safely filtered out rather than mis-stripped.
        adapter = _adapter(["hey hermes"])
        assert adapter._matches_voice_keyword("hermès la météo") is None

    def test_non_keyword_utterance_filtered_out(self):
        adapter = _adapter(["hey hermes"])
        assert adapter._matches_voice_keyword("bonjour, tu es là ?") is None
        assert adapter._matches_voice_keyword("salut") is None

    def test_requires_word_boundary(self):
        # "hey hermesphone" must NOT trigger even though it fuzzy-matches:
        # after stripping the keyword the remainder starts with a letter.
        adapter = _adapter(["hey hermes"])
        assert adapter._matches_voice_keyword("hey hermesphone teste") is None

    def test_keyword_only_returns_empty(self):
        adapter = _adapter(["hey hermes"])
        assert adapter._matches_voice_keyword("hey hermes") == ""
        assert adapter._matches_voice_keyword("  HEY HERMES  ") == ""

    def test_separator_after_keyword(self):
        adapter = _adapter(["hey hermes"])
        assert adapter._matches_voice_keyword("hey hermes!coucou") == "coucou"
        assert adapter._matches_voice_keyword("hey hermes: vas-y") == "vas-y"


class TestFuzzyMatching:
    def test_stt_mangled_keyword_is_accepted(self):
        # Whisper hears "hey hermes" as "l'hermesse" — the contiguous "hermes"
        # run covers 60% of the keyword, above the 0.5 default threshold.
        adapter = _adapter(["hey hermes"])
        assert adapter._matches_voice_keyword(
            "l'hermesse, comment vas-tu ?") == "comment vas-tu ?"

    def test_scattered_letters_do_not_false_trigger(self):
        # "quelle heure est-il" scatters the keyword's letters across separate
        # words; no contiguous run reaches the threshold.
        adapter = _adapter(["hey hermes"])
        assert adapter._matches_voice_keyword("quelle heure est-il") is None

    def test_fuzzy_disabled_rejects_mangled_keyword(self):
        adapter = _adapter(["hey hermes"], similarity=0.0)
        assert adapter._matches_voice_keyword(
            "l'hermesse, comment vas-tu ?") is None
        # exact (case/accent-insensitive) matching still works with fuzzy off
        assert adapter._matches_voice_keyword("hey hermes, vas-y") == "vas-y"

    def test_accent_fold_works_with_fuzzy_disabled_on_exact_spelling(self):
        adapter = _adapter(["hermes"], similarity=0.0)
        assert adapter._matches_voice_keyword("Hermès, la météo") == \
            "la météo"

    def test_block_coverage(self):
        # longest contiguous run of "hey hermes" inside "l'hermesse, " is
        # "hermes" (6 of 10 chars) => 0.6
        assert abs(
            DiscordAdapter._keyword_block_coverage("hey hermes", "l'hermesse, ")
            - 0.6
        ) < 1e-6
        # short scattered match => low coverage
        assert DiscordAdapter._keyword_block_coverage(
            "hey hermes", "quelle heure est") < 0.5
        # empty keyword => 0.0
        assert DiscordAdapter._keyword_block_coverage("", "abc") == 0.0


class TestLoadVoiceKeywords:
    def test_accepts_json_list_string(self):
        # ``hermes config set`` may persist the value as a JSON-ish string.
        with patch(
            "hermes_cli.config.read_raw_config",
            return_value={"discord": {"voice_keywords": '["hey hermes"]'}},
        ):
            adapter = object.__new__(DiscordAdapter)
            assert adapter._load_voice_keywords() == ["hey hermes"]

    def test_accepts_comma_string(self):
        with patch(
            "hermes_cli.config.read_raw_config",
            return_value={"discord": {"voice_keywords": "hey hermes,jarvis, "}},
        ):
            adapter = object.__new__(DiscordAdapter)
            assert adapter._load_voice_keywords() == ["hey hermes", "jarvis"]

    def test_accepts_yaml_list(self):
        with patch(
            "hermes_cli.config.read_raw_config",
            return_value={"discord": {"voice_keywords": ["Hey Hermes", "Jarvis"]}},
        ):
            adapter = object.__new__(DiscordAdapter)
            assert adapter._load_voice_keywords() == ["hey hermes", "jarvis"]

    def test_empty_is_disabled(self):
        for raw in (None, "", [], "[]"):
            with patch(
                "hermes_cli.config.read_raw_config",
                return_value={"discord": {"voice_keywords": raw}},
            ):
                adapter = object.__new__(DiscordAdapter)
                assert adapter._load_voice_keywords() == []


class TestLoadVoiceKeywordSimilarity:
    def test_default_is_half(self):
        with patch(
            "hermes_cli.config.read_raw_config",
            return_value={"discord": {}},
        ):
            adapter = object.__new__(DiscordAdapter)
            assert adapter._load_voice_keyword_similarity() == 0.5

    def test_reads_and_clamps(self):
        with patch(
            "hermes_cli.config.read_raw_config",
            return_value={"discord": {"voice_keyword_similarity": 1.9}},
        ):
            adapter = object.__new__(DiscordAdapter)
            assert adapter._load_voice_keyword_similarity() == 1.0
        with patch(
            "hermes_cli.config.read_raw_config",
            return_value={"discord": {"voice_keyword_similarity": -1}},
        ):
            adapter = object.__new__(DiscordAdapter)
            assert adapter._load_voice_keyword_similarity() == 0.0
        with patch(
            "hermes_cli.config.read_raw_config",
            return_value={"discord": {"voice_keyword_similarity": 0.7}},
        ):
            adapter = object.__new__(DiscordAdapter)
            assert adapter._load_voice_keyword_similarity() == 0.7
