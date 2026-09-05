"""Conversational approval matching (gateway/plaintext_approval.py).

The security contract, in order of importance:

1. A refusal must NEVER map to an approval. Substring matching would approve
   "no, don't go ahead" because it contains "go ahead"; whole-message matching
   makes that impossible. These are the tests that matter.
2. Anything ambiguous returns None so the approval stays pending (fail-safe).
3. Real conversational affirmatives and refusals resolve, so a person can answer
   a text message the way they'd answer a person.
"""

import pytest

from gateway.plaintext_approval import match_conversational_approval as m


class TestRefusalsNeverApprove:
    """The whole reason matching is whole-message: a refusal containing an
    affirmative phrase must resolve as deny, never as approve."""

    @pytest.mark.parametrize("reply", [
        "no, don't go ahead",
        "no do not go ahead",
        "don't do it",
        "do not do that",
        "no, don't run it",
        "nope",
        "cancel that",
        "never mind",
        "hold off",
        "no thanks",
    ])
    def test_refusal_resolves_as_deny(self, reply):
        assert m(reply) == ("deny", "")

    @pytest.mark.parametrize("reply", [
        "no, don't go ahead",
        "don't do it",
        "do not do that",
        "no dont approve",
    ])
    def test_refusal_is_never_an_approval(self, reply):
        verb, _ = m(reply)
        assert verb == "deny"


class TestAffirmatives:
    @pytest.mark.parametrize("reply", [
        "yes", "yep", "yeah", "sure", "ok", "okay", "confirm",
        "sure, go ahead", "go ahead", "yes please", "please do", "do it",
        "just do it", "sounds good", "that's fine", "proceed", "send it",
        "Sure!", "  yes  ", "OK.", "Yes, go ahead.", "go for it",
    ])
    def test_affirmative_approves_once(self, reply):
        assert m(reply) == ("approve", "")

    @pytest.mark.parametrize("reply", [
        "for this session", "approve for this session", "session",
        "yes for this session",
    ])
    def test_session_scope(self, reply):
        assert m(reply) == ("approve", "session")

    @pytest.mark.parametrize("reply", [
        "always", "always approve", "don't ask again", "do not ask again",
        "stop asking", "yes always",
    ])
    def test_permanent_scope(self, reply):
        assert m(reply) == ("approve", "always")


class TestAmbiguousFallsThrough:
    """None = leave the approval pending. A conditional, a question, or a real
    sentence is not an answer and must not resolve anything."""

    @pytest.mark.parametrize("reply", [
        "",
        "   ",
        "are you sure?",                      # question, even though it says "sure"
        "is that ok?",                        # question containing "ok"
        "yes but not the second one",         # conditional
        "ok so what does that actually do",   # a real question opening with "ok"
        "go ahead and also delete the backups",  # extra instruction, not a bare answer
        "what happens if I say yes",
        "sure thing, but check with me first",
        "maybe",
        "hmm",
        "do it after you check the logs first please",
    ])
    def test_returns_none(self, reply):
        assert m(reply) is None


class TestSmartPunctuationAndTapbacks:
    def test_ios_curly_apostrophe_folds(self):
        # iOS autocorrects don't -> don\u2019t; both must reach the same phrase.
        assert m("don\u2019t do it") == ("deny", "")
        assert m("don't do it") == ("deny", "")

    def test_thats_fine_with_curly_apostrophe(self):
        assert m("that\u2019s fine") == ("approve", "")

    @pytest.mark.parametrize("emoji,expected", [
        ("\U0001f44d", ("approve", "")),
        ("\u2705", ("approve", "")),
        ("\U0001f44e", ("deny", "")),
        ("\u274c", ("deny", "")),
    ])
    def test_tapback_emoji(self, emoji, expected):
        assert m(emoji) == expected


class TestVerbGrammarMatchesSlashHandlers:
    """The returned (verb, args) is synthesized into "/approve <args>", so args
    must be tokens the real handlers parse."""

    def test_args_are_handler_tokens(self):
        from gateway.slash_commands import _APPROVE_CHOICE_BY_ARG

        for reply in ("always", "for this session"):
            verb, args = m(reply)
            assert verb == "approve"
            assert args in _APPROVE_CHOICE_BY_ARG

    def test_once_passes_empty_args(self):
        assert m("yes") == ("approve", "")
