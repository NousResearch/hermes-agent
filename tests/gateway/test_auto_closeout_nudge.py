"""A5-B: post-restart auto-closeout NUDGE rides the resume note, cache-safe.

The nudge must fire on the two interrupted-build resume branches (so a resumed
session runs closeout before reporting a build "done"), NOT on a clean idle
restore, and — the NON-NEGOTIABLE — it must NOT mutate the cached conversation
prefix. The nudge is appended to the already-injected resume note (the current
turn's inbound message), never to `agent_history`, so the per-conversation
prompt cache is preserved (AGENTS.md: "per-conversation prompt caching is
sacred").

Worktree-bytes discipline: `test_module_under_test_is_worktree_bytes` asserts
gateway.run resolves to the A5 worktree before any behavioral green is trusted.
"""

from __future__ import annotations

import copy

import gateway.run as gr
from gateway.run import _build_resume_pending_message

_WORKTREE_MARKER = "/.worktrees/sendoff-a5-e2e-closeout/"

_CLOSEOUT_TELL = "run the prd-closeout gate"


def test_module_under_test_is_worktree_bytes() -> None:
    """Prove gateway.run is WORKTREE bytes, not a deployed/editable-install copy.

    PEP660 editable install + sys.meta_path finder means PYTHONPATH alone does
    not win; the harness that runs this file must have evicted the __editable__
    finder so this import resolves to the worktree. Assert it before trusting
    the behavioral assertions below.
    """
    assert gr.__file__ and _WORKTREE_MARKER in gr.__file__, (
        f"gateway.run is not worktree bytes: {gr.__file__}"
    )


def _flagged_interrupt_history():
    return [
        {"role": "user", "content": "build the feature"},
        {
            "role": "assistant",
            "content": "Operation interrupted.",
            "_interrupt_close": True,
        },
    ]


# ── The nudge fires on the interrupted-build branches ──────────────────────────

def test_nudge_present_on_surface_and_ask_branch() -> None:
    """Empty-message resume after an interrupted build → nudge in the note."""
    message, surface_and_ask = _build_resume_pending_message(
        agent_history=_flagged_interrupt_history(),
        message="",
        reason_phrase="a gateway restart",
    )
    assert surface_and_ask is True
    assert _CLOSEOUT_TELL in message, message
    assert "real captured evidence" in message, message


def test_nudge_present_when_new_message_after_interrupted_build() -> None:
    """New user message after an interrupted build → nudge still rides along."""
    message, surface_and_ask = _build_resume_pending_message(
        agent_history=_flagged_interrupt_history(),
        message="what happened?",
        reason_phrase="a gateway restart",
    )
    assert surface_and_ask is False
    assert _CLOSEOUT_TELL in message, message
    # The new user message is still appended last (role-alternation preserved).
    assert message.endswith("\n\nwhat happened?"), message


# ── The nudge does NOT fire on a clean, non-interrupted restore ────────────────

def test_no_nudge_on_clean_idle_restore() -> None:
    """A plain assistant tail (no interrupt-close) is a clean restore — the
    build nudge must NOT fire (negative case; avoids nagging every restart).
    """
    message, surface_and_ask = _build_resume_pending_message(
        agent_history=[{"role": "assistant", "content": "done"}],
        message="",
        reason_phrase="a gateway restart",
    )
    assert surface_and_ask is False
    assert _CLOSEOUT_TELL not in message, message


def test_no_nudge_on_interrupted_text_without_flag() -> None:
    """Assistant text that merely says 'interrupted' but carries no
    interrupt-close flag is NOT treated as an interrupted build → no nudge.
    """
    message, _ = _build_resume_pending_message(
        agent_history=[{"role": "assistant", "content": "Operation interrupted."}],
        message="",
        reason_phrase="a gateway restart",
    )
    assert _CLOSEOUT_TELL not in message, message


# ── MANDATORY cache-stability assertion ────────────────────────────────────────

def test_nudge_does_not_mutate_cached_history_prefix() -> None:
    """CACHE-SAFETY (non-negotiable): building the resume note with the nudge
    must NOT mutate `agent_history` — the cached conversation prefix. The nudge
    rides the CURRENT turn's inbound message only. If the builder touched the
    history bytes, the per-conversation prompt cache would be invalidated and
    the user's cost would multiply (AGENTS.md invariant).
    """
    history = _flagged_interrupt_history()
    before = copy.deepcopy(history)

    message, _ = _build_resume_pending_message(
        agent_history=history,
        message="",
        reason_phrase="a gateway restart",
    )

    # 1. The history object is byte-identical after the call (no mutation).
    assert history == before, (
        "resume-note builder mutated the cached history prefix — cache-break!"
    )
    # 2. The nudge lives in the returned CURRENT-TURN message, not the history.
    assert _CLOSEOUT_TELL in message, message
    assert not any(_CLOSEOUT_TELL in str(turn.get("content", "")) for turn in history), (
        "nudge leaked into cached history turns — must ride the current turn only"
    )


def test_note_identical_bytes_across_repeated_builds() -> None:
    """The note is deterministic: same inputs → byte-identical note. A resume
    note whose bytes drifted run-to-run would itself thrash any prefix caching
    of the note. Same history + reason → identical string.
    """
    h1 = _flagged_interrupt_history()
    h2 = _flagged_interrupt_history()
    m1, _ = _build_resume_pending_message(
        agent_history=h1, message="", reason_phrase="a gateway restart"
    )
    m2, _ = _build_resume_pending_message(
        agent_history=h2, message="", reason_phrase="a gateway restart"
    )
    assert m1 == m2, "resume note is not byte-stable across identical builds"
