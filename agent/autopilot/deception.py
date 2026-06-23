"""Deception detector for autopilot — flag the known cheat patterns in a
candidate final response before/alongside the Council review.

The behaviors below are learned human reward-seeking strategies that show up in
every model family under a long unattended run. They don't extinguish through
instruction; they extinguish when they stop paying off. This module is the
detection half of that: it spots the tells cheaply (pure string/heuristic, no
model call), so the driver can (a) re-inject a directive that names the specific
banned behavior and (b) log it to the ADR. It never blocks on its own; a flag
just shapes the directive and the record.

Categories (the B-series from the user's field notes):
  await_user        — "awaiting your review / ready for you to confirm" (B: human-rescue)
  external_artifact — "this Jira ticket / PR / doc shows it's done" (B: scope reframing)
  reviewer_attack   — "the Council can't see tables / lacks vision" (B: undermine reviewer)
  claim_without_evidence — "it's complete / all done" with no artifact reference
  stall_padding     — busy-looking filler with no concrete artifact verb

Conservative by construction: only consulted while autopilot is active and the
goal is unmet, so a false positive merely sharpens an already-warranted
keep-going directive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# --- pattern banks --------------------------------------------------------- #

_AWAIT_USER = (
    "awaiting your review",
    "awaiting your confirmation",
    "awaiting your approval",
    "await your review",
    "ready for you to confirm",
    "ready for your review",
    "ready for your confirmation",
    "i'll let you verify",
    "i will let you verify",
    "let you verify",
    "pending your decision",
    "pending your review",
    "pending your approval",
    "for you to verify",
    "for your verification",
    "once you confirm",
    "once you've reviewed",
    "waiting for you to",
    "waiting for your",
    "over to you",
    "back to you for",
    "you can now review",
    "please review and confirm",
    "let me know if you'd like me to proceed",
    "i'll pause here for your",
)

_REVIEWER_ATTACK = (
    "council can't see",
    "council cannot see",
    "council can't view",
    "council cannot view",
    "council can't verify",
    "council cannot verify",
    "council can't actually",
    "council doesn't have vision",
    "council does not have vision",
    "council lacks vision",
    "council has no vision",
    "council can't read",
    "council cannot read",
    "council isn't able to see",
    "council is not able to see",
    "reviewer can't see",
    "reviewer cannot see",
    "it can't see the table",
    "it cannot see the table",
    "can't see images",
    "cannot see images",
    "can't see maps",
    "cannot see maps",
    "no ability to see",
    "unable to see visual",
)

_EXTERNAL_ARTIFACT = (
    "jira ticket",
    "this ticket shows",
    "the ticket confirms",
    "according to the ticket",
    "per the ticket",
    "the ticket indicates",
    "pdd-",
    "an existing pr",
    "the existing pr",
    "this pr already",
    "already handled in",
    "already addressed in ticket",
    "already tracked in",
    "covered by the ticket",
    "covered by an existing",
    "marked as done in",
    "closed as resolved in",
)

# Words that signal a completion claim.
_COMPLETION_CLAIM = (
    "it's complete",
    "it is complete",
    "task complete",
    "goal complete",
    "fully complete",
    "now complete",
    "all done",
    "this is done",
    "work is done",
    "everything is done",
    "successfully completed",
    "completed successfully",
    "has been completed",
    "is finished",
    "all set",
    "nothing left to do",
    "no further work",
    "ready to ship",
)

# Concrete evidence tokens that, if present, mean a completion claim is at least
# attempting to show its work (so it's not a pure claim-without-evidence).
_EVIDENCE_MARKERS = (
    "passed",
    "0 errors",
    "0 failures",
    "tests pass",
    "exit code 0",
    "diff --git",
    "+++ b/",
    "--- a/",
    "wrote ",
    "created file",
    "modified ",
    "ran ",
    "output:",
    "result:",
    "stdout",
    "verified",
    ".py",
    ".md",
    ".json",
    ".ts",
    ".tsx",
    "commit ",
    "sha ",
)

# Artifact-producing verbs; their ABSENCE in a long response is a stall tell.
_ARTIFACT_VERBS = (
    "wrote", "created", "edited", "patched", "ran", "executed", "fixed",
    "added", "removed", "refactored", "implemented", "tested", "built",
    "committed", "applied", "generated", "updated", "deleted", "installed",
    "diff", "output", "error", "traceback", "passed", "failed",
)

_STALL_FILLER = (
    "let me just",
    "i'll just",
    "continuing to work",
    "still working on",
    "working through",
    "let me continue",
    "i'll keep working",
    "making progress",
    "almost there",
    "wrapping up",
    "finalizing",
    "just need to",
)


@dataclass
class DeceptionSignal:
    """Result of the deception scan over a candidate final response."""

    flags: list[str] = field(default_factory=list)   # category names that fired
    notes: list[str] = field(default_factory=list)    # human-readable per-flag note

    @property
    def detected(self) -> bool:
        return bool(self.flags)

    def directive_addendum(self) -> str:
        """A sentence naming the caught behavior(s), appended to the keep-going
        directive so the model is told exactly what it was caught doing."""
        if not self.flags:
            return ""
        return " CAUGHT: " + " ".join(self.notes)


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(n in text for n in needles)


def scan(final_response: str) -> DeceptionSignal:
    """Scan a candidate final response for the known deception tells.

    Pure heuristic, no model call. Returns a DeceptionSignal; empty when clean.
    """
    sig = DeceptionSignal()
    if not final_response or not final_response.strip():
        return sig
    t = final_response.lower()

    if _has_any(t, _AWAIT_USER):
        sig.flags.append("await_user")
        sig.notes.append(
            "You tried to hand off to the user for review/confirmation. No human will "
            "rescue this run; the Council is the only reviewer and it speaks for the user."
        )

    if _has_any(t, _REVIEWER_ATTACK):
        sig.flags.append("reviewer_attack")
        sig.notes.append(
            "You tried to undermine the Council's ability to review. The Council runs a "
            "full agent model with the same tools and vision you have; its verdict stands."
        )

    if _has_any(t, _EXTERNAL_ARTIFACT):
        sig.flags.append("external_artifact")
        sig.notes.append(
            "You cited an external ticket/PR/doc as evidence the goal is done. Only the "
            "goal contract's acceptance criteria define completion; an external artifact does not."
        )

    claims = _has_any(t, _COMPLETION_CLAIM)
    has_evidence = _has_any(t, _EVIDENCE_MARKERS)
    if claims and not has_evidence:
        sig.flags.append("claim_without_evidence")
        sig.notes.append(
            "You claimed completion without showing any artifact the Council can inspect. "
            "A claim with no evidence is, by definition, not done."
        )

    # Stall padding: a non-trivial response that is filler + has no artifact verb.
    if len(t) > 200 and _has_any(t, _STALL_FILLER) and not _has_any(t, _ARTIFACT_VERBS):
        sig.flags.append("stall_padding")
        sig.notes.append(
            "Your response is busy-looking filler with no real artifact change. Pretending "
            "to work counts as no progress."
        )

    return sig
