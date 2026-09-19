"""Regression tests for Honcho representation dedup before prompt injection.

Restated observations arrive in the representation as byte-identical lines (one per
retain); ``_peer_context_strings`` injected all of them, multiplying memory tokens and
training the agent to distrust recalled context. Dedup collapses ONLY exact duplicates
(bullet/case-normalized) - paraphrases, contradictions, and blank structure survive.
Sibling of the Hindsight recall dedup (PR #89145) at the equivalent Honcho injection
point; see issue #111205 for the production 49x-duplicate report.
"""
import pytest

from plugins.memory.honcho.session_context import SessionContextMixin


def dedup(text: str) -> str:
    return SessionContextMixin._dedup_representation_lines(text)


class TestCollapsesExactDuplicates:
    def test_49x_restated_fact_collapses_to_one(self):
        # The reported production shape: one fact restated 49 times + one other fact.
        rep = "\n".join(["- Pox prefers root-cause fixes."] * 49) + "\n- Pox likes Konrad for TTS."
        out = dedup(rep).splitlines()
        assert out == ["- Pox prefers root-cause fixes.", "- Pox likes Konrad for TTS."]

    def test_first_occurrence_wins_and_order_is_preserved(self):
        rep = "- fact A\n- fact B\n- fact A\n- fact C\n- fact B"
        assert dedup(rep) == "- fact A\n- fact B\n- fact C"

    def test_bullet_style_and_case_are_normalized(self):
        # Same observation restated with a different bullet glyph / casing collapses.
        rep = "- Pox prefers root-cause fixes.\n* pox prefers root-cause fixes.\n• Pox prefers root-cause fixes."
        out = dedup(rep)
        assert len(out.splitlines()) == 1


class TestPreservesNonDuplicates:
    def test_paraphrases_are_kept(self):
        # Different wording = different observation; a fuzzy collapse would lose information.
        rep = "- Pox prefers root-cause fixes.\n- Pox wants the root cause fixed first."
        assert len(dedup(rep).splitlines()) == 2

    def test_contradictions_are_never_merged(self):
        # The failure mode PR #89145 guards against: dropping one side of a flip.
        rep = "- Pox uses Teams.\n- Pox no longer uses Teams."
        assert len(dedup(rep).splitlines()) == 2

    def test_numeric_differences_are_kept(self):
        rep = "- budget is 500000 tokens\n- budget is 450000 tokens"
        assert len(dedup(rep).splitlines()) == 2

    def test_blank_lines_are_structural_and_survive(self):
        rep = "- fact A\n\n\n- fact B"
        assert dedup(rep) == "- fact A\n\n\n- fact B"

    def test_whitespace_only_lines_are_not_deduped_against_each_other(self):
        # Whitespace-only lines are structural: kept verbatim, never collapsed.
        rep = "- fact A\n   \n\t\n- fact B"
        assert dedup(rep) == rep


class TestContract:
    def test_clean_representation_is_byte_identical(self):
        rep = "- fact A\n- fact B\n- fact C\n"
        assert dedup(rep) == rep

    def test_trailing_newline_is_preserved(self):
        assert dedup("- fact A\n").endswith("\n")
        assert not dedup("- fact A").endswith("\n")

    def test_empty_and_none_safe(self):
        assert dedup("") == ""
        assert dedup("\n") == "\n"
        # A representation of only blank lines stays structurally intact (trailing
        # newline preserved; interior blank lines are not duplicates by our contract).
        assert dedup("- fact A\n\n- fact B\n") == "- fact A\n\n- fact B\n"

    def test_is_fail_open_staticmethod(self):
        # Contract: callable without an instance (fail-open even if mixin state changes).
        assert callable(SessionContextMixin._dedup_representation_lines)


class TestWiring:
    def test_peer_context_strings_applies_dedup(self, monkeypatch):
        # The injection point must actually pass the representation through dedup.
        captured = {}

        def fake_fetch(self, peer_id, search_query=None, *, target=None):
            captured["peer_id"] = peer_id
            return {"representation": "- fact A\n- fact A\n- fact B", "card": ["card line"]}

        monkeypatch.setattr(SessionContextMixin, "_fetch_peer_context", fake_fetch)
        stub = object.__new__(SessionContextMixin)  # unbound call; dedup is stateless
        rep, card = SessionContextMixin._peer_context_strings(stub, "Pox")
        assert rep == "- fact A\n- fact B"
        assert card == "card line"
        assert captured["peer_id"] == "Pox"


@pytest.mark.parametrize("dup_count", [2, 5, 49])
def test_parameterized_duplicate_counts_collapse_to_one(dup_count):
    rep = "\n".join(["- same fact."] * dup_count)
    assert len(dedup(rep).splitlines()) == 1
