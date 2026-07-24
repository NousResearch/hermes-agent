"""Tests for AIAgent._has_natural_response_ending heuristic."""

import pytest
from run_agent import AIAgent


cls = AIAgent


class TestNaturalResponseEnding:
    """Cover punctuation, emoji, caret, slang, and edge cases."""

    # --- Punctuation endings (existing behavior) ---

    def test_period(self):
        assert cls._has_natural_response_ending("Hello world.")

    def test_exclamation(self):
        assert cls._has_natural_response_ending("Cool!")

    def test_question_mark(self):
        assert cls._has_natural_response_ending("Really?")

    def test_colon(self):
        assert cls._has_natural_response_ending("Here:")

    def test_closing_paren(self):
        assert cls._has_natural_response_ending("(done)")

    def test_closing_bracket(self):
        assert cls._has_natural_response_ending("[ok]")

    def test_cjk_punctuation(self):
        assert cls._has_natural_response_ending("こんにちは。")

    # --- Code block endings ---

    def test_triple_backtick(self):
        assert cls._has_natural_response_ending("```\ncode\n```")

    # --- Caret endings (PR #28168) ---

    def test_caret_single(self):
        assert cls._has_natural_response_ending("Voila ^")

    def test_caret_double(self):
        assert cls._has_natural_response_ending("C'est cool ^^")

    # --- Emoji / symbol endings ---

    def test_emoji_emoticons(self):
        assert cls._has_natural_response_ending("Génial 😊")

    def test_symbol_misc(self):
        """⚡ (U+26A1) is in Misc Symbols, below U+1F300."""
        assert cls._has_natural_response_ending("C'est cool ⚡")

    def test_symbol_dingbats(self):
        """❤ (U+2764) is in Dingbats."""
        assert cls._has_natural_response_ending("Merci ❤")

    def test_symbol_check_mark(self):
        """✅ (U+2705) is in Dingbats."""
        assert cls._has_natural_response_ending("Done ✅")

    # --- Slang / laughter endings ---

    class TestFrenchSlang:
        def test_mdr(self):
            assert cls._has_natural_response_ending("Trop drôle mdr")

        def test_ptdr(self):
            assert cls._has_natural_response_ending("N'importe quoi ptdr")

        def test_xptdr(self):
            assert cls._has_natural_response_ending("C'est ouf xptdr")

        def test_lol(self):
            assert cls._has_natural_response_ending("ok lol")

    class TestEnglishSlang:
        def test_lmao(self):
            assert cls._has_natural_response_ending("That's hilarious lmao")

        def test_rofl(self):
            assert cls._has_natural_response_ending("Classic rofl")

        def test_kek(self):
            assert cls._has_natural_response_ending("kek")

        def test_brb(self):
            assert cls._has_natural_response_ending("brb")

        def test_smh(self):
            assert cls._has_natural_response_ending("smh")

    class TestSpanishPortugueseSlang:
        def test_jaja(self):
            assert cls._has_natural_response_ending("jaja")

        def test_rsrs(self):
            assert cls._has_natural_response_ending("rsrs")

        def test_kkk(self):
            assert cls._has_natural_response_ending("kkk")

    class TestJapaneseSlang:
        def test_w(self):
            assert cls._has_natural_response_ending("w")

        def test_www(self):
            assert cls._has_natural_response_ending("www")

        def test_kusa(self):
            assert cls._has_natural_response_ending("kusa")

    class TestChineseSlang:
        def test_233(self):
            assert cls._has_natural_response_ending("233")

        def test_666(self):
            assert cls._has_natural_response_ending("666")

    class TestGamingGeneralSlang:
        def test_gg(self):
            assert cls._has_natural_response_ending("gg")

        def test_gg_wp(self):
            assert cls._has_natural_response_ending("gg wp")

        def test_thx(self):
            assert cls._has_natural_response_ending("thx")

        def test_tldr(self):
            assert cls._has_natural_response_ending("tldr")

    # --- Slang with trailing punctuation/emoji ---

    def test_slang_with_emoji(self):
        """'mdr ^^' — slang followed by caret (already handled by caret)."""
        assert cls._has_natural_response_ending("mdr ^^")

    def test_slang_with_exclamation(self):
        """'lmao!!!' — slang with trailing punctuation."""
        assert cls._has_natural_response_ending("lmao!!!")

    def test_slang_case_insensitive(self):
        """Slang matching is case-insensitive."""
        assert cls._has_natural_response_ending("LMAO")

    # --- Negative cases (should NOT be detected as natural ending) ---

    def test_mid_sentence(self):
        """'mdr' in the middle of a sentence is NOT an ending."""
        assert not cls._has_natural_response_ending("Le mot mdr existe dans le dico")

    def test_empty_string(self):
        assert not cls._has_natural_response_ending("")

    def test_whitespace_only(self):
        assert not cls._has_natural_response_ending("   ")

    def test_no_ending(self):
        """Bare text ending with a letter (no punctuation/emoji/slang)."""
        assert not cls._has_natural_response_ending("Je pense donc")

    def test_number_not_slang(self):
        """A random number is not slang."""
        assert not cls._has_natural_response_ending("The answer is 42")

    def test_regular_word_ending(self):
        """A regular word ending in 'r' is not slang."""
        assert not cls._has_natural_response_ending("C'est un ordinateur")