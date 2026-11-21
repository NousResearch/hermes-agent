"""A transcript that merely *mentions* a lean section heading must not suppress it.

Found on a real session (2026-11-05): the compacted turns contained the literal
text "`## Anchor Index`/`## Context Recovery`/`## Pruned Skills`" because the
conversation was about the compression port itself. A full-text heading search
then treated both deterministic sections as already present, so the handoff
shipped with no anchor ledger and no recovery pointer — silently, with every
other check green.

Presence is decided structurally: a real section puts its heading at column 0,
while prose quotes it after a backtick, a bullet marker, or a table pipe. The
test does not depend on guessing a string the prose will never contain.

Known residual gap, now closed for the recovery footer: honest pasted content —
e.g. resubmitting a previous handoff verbatim — also puts the headings at column
0. A heading match alone would swallow the fresh sections and ship the stale
recovery pointer naming the OLD session id. The footer's body carries the
current session id, so its presence test additionally requires that name after
the line-start heading; a pasted footer names the previous session and no
longer counts. The anchor and verbatim-user-messages sections have no such
signature and deliberately stay structure-only — suppressing them on a paste
still loses fresh anchors, but inventing a payload test for them would re-gamble
on wording the transcript will never contain.
"""

from unittest.mock import patch

from agent.context_compressor import (
    ContextCompressor,
    _LEAN_ANCHOR_HEADING,
    _LEAN_RECOVERY_HEADING,
    _LEAN_USER_MESSAGES_HEADING,
    _build_recovery_footer,
    _lean_section_present,
)

SESSION = "sess-impostor"

# Every shape prose can take when it names a heading, none of which opens a line.
PROSE_MENTIONS = {
    "backticked": "55. WRITE notes — symbols (`## Anchor Index`/`## Context Recovery`/`## Pruned Skills`) are absent",
    "bulleted": "  - ## Anchor Index is mentioned in the plan",
    "blockquote": "> ## Context Recovery appears in the docstring",
    "table cell": "| ## Anchor Index | yes |",
    "mid sentence": "we should add ## Anchor Index here",
}

TURNS = [
    {
        "role": "user",
        "content": (
            "fix agent/context_compressor.py line 42, see PR #1234, "
            "commit " + "a" * 40
        ),
    },
]


def _compressor(tail_mode="lean"):
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        c = ContextCompressor(model="test/model", quiet_mode=True, tail_mode=tail_mode)
    c._session_id = SESSION
    return c


def test_line_start_predicate_rejects_every_prose_shape():
    for shape, text in PROSE_MENTIONS.items():
        assert not _lean_section_present(text, _LEAN_ANCHOR_HEADING), shape
        assert not _lean_section_present(text, _LEAN_RECOVERY_HEADING), shape


def test_line_start_predicate_accepts_a_real_section():
    assert _lean_section_present(f"{_LEAN_ANCHOR_HEADING}\nbody", _LEAN_ANCHOR_HEADING)
    assert _lean_section_present(f"## Goal\n\n{_LEAN_ANCHOR_HEADING}\nx", _LEAN_ANCHOR_HEADING)
    assert _lean_section_present(f"intro\n{_LEAN_RECOVERY_HEADING}\nbody", _LEAN_RECOVERY_HEADING)


def test_mentioning_a_heading_does_not_suppress_the_real_section():
    c = _compressor()
    out = c._augment_summary_lean(PROSE_MENTIONS["backticked"], TURNS)

    assert f"\n{_LEAN_ANCHOR_HEADING}" in out, "anchor ledger skipped because prose named its heading"
    assert f"\n{_LEAN_RECOVERY_HEADING}" in out, "recovery footer skipped because prose named its heading"
    assert f"session_id='{SESSION}'" in out


def test_user_messages_section_survives_a_heading_mention():
    c = _compressor()
    out = c._augment_summary_lean(PROSE_MENTIONS["blockquote"], TURNS)
    assert f"\n{_LEAN_USER_MESSAGES_HEADING}" in out


def test_real_sections_are_not_duplicated_on_a_second_pass():
    c = _compressor()
    once = c._augment_summary_lean(PROSE_MENTIONS["backticked"], TURNS)
    twice = c._augment_summary_lean(once, TURNS)
    assert twice == once


def test_a_clean_summary_still_gets_every_section():
    c = _compressor()
    out = c._augment_summary_lean("## Goal\nship it", TURNS)
    assert f"\n{_LEAN_ANCHOR_HEADING}" in out
    assert f"\n{_LEAN_USER_MESSAGES_HEADING}" in out
    assert f"\n{_LEAN_RECOVERY_HEADING}" in out


def test_legacy_mode_still_appends_nothing():
    c = _compressor(tail_mode="legacy")
    assert c._augment_summary_lean("## Goal\nbody", TURNS) == "## Goal\nbody"


# A full handoff from a previous session, pasted back verbatim by the user.
# Every heading genuinely opens a line, session id and all — nothing about it
# is adversarial, which is why wording/structure guessing alone cannot tell it
# from a section this compressor appended itself.
PASTED_OLD_HANDOFF = (
    "## Goal\ncontinuing work from the previous session's handoff.\n"
    + _LEAN_ANCHOR_HEADING + "\n"
    "- commits: a1b2c3d4e5f6\n"
    "\n" + _LEAN_USER_MESSAGES_HEADING + "\n"
    "1. restart the service on :8080\n"
    + _build_recovery_footer(SESSION, 42)
)

NEW_SESSION = "sess-impostor-retry"


def test_a_pasted_old_handoff_must_not_steal_the_fresh_recovery_pointer():
    c = _compressor()
    c._session_id = NEW_SESSION
    out = c._augment_summary_lean(PASTED_OLD_HANDOFF, TURNS)

    # The pasted footer names the OLD session; the fresh one names the CURRENT
    # session and must be appended alongside it, not swallowed by the
    # line-start predicate.
    assert out.count(_LEAN_RECOVERY_HEADING) >= 2
    assert f"session_id='{NEW_SESSION}'" in out
