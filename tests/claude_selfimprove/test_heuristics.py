from __future__ import annotations

from claude_selfimprove.heuristics import mine_candidates
from claude_selfimprove.scanner import TranscriptTurn


def _turn(role, text, file_path="f1", session_id="s1", source="claude", project="p"):
    return TranscriptTurn(
        source=source,
        project=project,
        session_id=session_id,
        file_path=file_path,
        line_no=0,
        role=role,
        text=text,
        timestamp="2026-08-01T00:00:00Z",
    )


def test_detects_explicit_instruction_never():
    turns = [_turn("user", "Never use git push --force on main, that's a hard rule.")]
    cands = mine_candidates(turns)
    assert len(cands) == 1
    assert cands[0].category == "explicit_instruction"


def test_detects_explicit_instruction_from_now_on():
    turns = [_turn("user", "From now on, always run the linter before committing.")]
    cands = mine_candidates(turns)
    assert cands[0].category == "explicit_instruction"


def test_detects_correction_style_instruction():
    turns = [_turn("user", "No, don't mock the database in integration tests.")]
    cands = mine_candidates(turns)
    assert cands[0].category == "explicit_instruction"


def test_detects_repeated_fix_with_preceding_assistant_context():
    turns = [
        _turn("assistant", "The bug was a missing await in the async handler, I fixed it."),
        _turn("user", "That fixed it, works now."),
    ]
    cands = mine_candidates(turns)
    assert len(cands) == 1
    assert cands[0].category == "repeated_fix"
    assert "missing await" in cands[0].context_text


def test_detects_repeated_procedure_confirmation():
    turns = [
        _turn("assistant", "I split the refactor into three small PRs instead of one big one."),
        _turn("user", "Yes exactly, that's the right approach for this repo."),
    ]
    cands = mine_candidates(turns)
    assert len(cands) == 1
    assert cands[0].category == "repeated_procedure"


def test_ignores_ordinary_conversation():
    turns = [
        _turn("user", "Can you show me the current file listing?"),
        _turn("assistant", "Sure, here's the listing."),
        _turn("user", "Thanks, that looks right."),
    ]
    cands = mine_candidates(turns)
    assert cands == []


def test_instruction_match_short_circuits_confirmation_check():
    # A message that both reads like an instruction AND contains "exactly"
    # should be classified once, as the higher-priority instruction category.
    turns = [_turn("user", "Never do that again, exactly the opposite of what I asked.")]
    cands = mine_candidates(turns)
    assert len(cands) == 1
    assert cands[0].category == "explicit_instruction"


def test_candidate_text_is_redacted():
    turns = [_turn("user", "Never commit a file with api_key: sk_live_abcdef123456 in it.")]
    cands = mine_candidates(turns)
    assert "sk_live_abcdef123456" not in cands[0].text


def test_groups_by_file_and_preserves_order_across_interleaved_batch():
    # Turns from two different files interleaved in the input list; context
    # look-back must not leak across files.
    turns = [
        _turn("assistant", "Fixed the retry loop bug.", file_path="fileA"),
        _turn("assistant", "Unrelated normal reply.", file_path="fileB"),
        _turn("user", "That fixed it, no longer fails.", file_path="fileA"),
        _turn("user", "That fixed it, no longer fails.", file_path="fileB"),
    ]
    cands = mine_candidates(turns)
    by_file = {c.file_path: c for c in cands}
    assert "retry loop" in by_file["fileA"].context_text
    assert "retry loop" not in by_file["fileB"].context_text
