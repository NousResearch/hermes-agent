"""Cheap, pattern-based candidate signal detection.

This is stage one of a two-stage pipeline: a fast, regex-only pass over
every scanned turn that flags *candidate* snippets worth a closer look. It
never decides that something IS a lesson — it decides that something MIGHT
be, at a cost low enough to run over the full incremental sweep every night.
Stage two (``llm.py``) reads only the small set of flagged snippets this
stage produces, confirms them, writes clean text, and assigns a confidence
score. Nothing here calls a model, and nothing here treats the matched text
as anything other than data to pattern-match against — nothing found in a
transcript is ever executed or followed as an instruction.

Four categories, matching the pipeline's evidence-threshold policy
(``thresholds.py``):

* ``explicit_instruction`` — George telling Claude Code, directly and
  plainly, to always/never do something, or correcting a specific action.
  Qualifies from a single confirmed occurrence.
* ``repeated_fix`` — something broke, then a message confirms it is fixed.
  Needs repeated, independent confirmation before it becomes a rule.
* ``repeated_procedure`` — a multi-step approach that George confirmed
  worked well. Needs repeated confirmation before it becomes a skill.

Detection works per-transcript-file: turns are processed in order so a
fix/procedure confirmation can look back one assistant turn for the
description of what was being confirmed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import groupby

from . import redact
from .scanner import TranscriptTurn

_INSTRUCTION_PATTERNS = [
    re.compile(
        r"(?i)\bnever\s+(?:\w+\s+){0,2}"
        r"(?:use|do|run|commit|push|force|delete|call|write|reset|merge|rebase)\b"
    ),
    re.compile(r"(?i)\balways\s+\w+"),
    re.compile(r"(?i)\bfrom now on\b"),
    re.compile(r"(?i)\bremember (?:that|to)\b"),
    re.compile(r"(?i)\bin the future[,]?\s+(?:always|never|please)\b"),
    re.compile(r"(?i)\bevery time\b"),
    re.compile(r"(?i)\bwhenever you\b"),
    re.compile(r"(?i)\bdon'?t\b.{0,60}\b(again|anymore)\b"),
    re.compile(r"(?i)\bstop (?:using|doing)\b"),
    re.compile(r"(?i)\bplease don'?t\b"),
    re.compile(r"(?i)\bno[,.]?\s+(?:that'?s|that is|don'?t|do not)\b"),
    re.compile(r"(?i)\bthat'?s wrong\b"),
]

_FIX_CONFIRMATION_PATTERNS = [
    re.compile(r"(?i)\bthat fixed it\b"),
    re.compile(r"(?i)\bworks? now\b"),
    re.compile(r"(?i)\bno longer (?:happens|fails|occurs)\b"),
    re.compile(r"(?i)\bbug is gone\b"),
    re.compile(r"(?i)\bconfirmed fixed\b"),
    re.compile(r"(?i)\byes,? that (?:fixed|resolved)\b"),
    re.compile(r"(?i)\bthat resolved it\b"),
]

_PROCEDURE_CONFIRMATION_PATTERNS = [
    re.compile(r"(?i)\byes,? exactly\b"),
    re.compile(r"(?i)\bperfect,\b"),
    re.compile(r"(?i)\bthat'?s the right approach\b"),
    re.compile(r"(?i)\byes,? do it that way\b"),
    re.compile(r"(?i)\bcorrect approach\b"),
    re.compile(r"(?i)\bthat'?s how we should\b"),
    re.compile(r"(?i)\bnailed it\b"),
    re.compile(r"(?i)\bexactly right\b"),
]

_ERROR_CONTEXT_WORDS = re.compile(
    r"(?i)\b(bug|error|fail(?:ed|ing|ure)?|broke|broken|crash|exception|wrong|issue)\b"
)

_MAX_SNIPPET_CHARS = 500


@dataclass(frozen=True)
class RawCandidate:
    """A flagged, already-redacted snippet worth classifying further."""

    source: str
    project: str
    session_id: str
    file_path: str
    category: str  # "explicit_instruction" | "repeated_fix" | "repeated_procedure"
    text: str  # redacted
    context_text: str  # redacted, may be empty
    matched_pattern: str
    timestamp: str | None


def _matches_any(patterns: list[re.Pattern[str]], text: str) -> str | None:
    for pattern in patterns:
        if pattern.search(text):
            return pattern.pattern
    return None


def _looks_like_error_context(*texts: str) -> bool:
    return any(_ERROR_CONTEXT_WORDS.search(t) for t in texts if t)


def _mine_file(turns: list[TranscriptTurn]) -> list[RawCandidate]:
    candidates: list[RawCandidate] = []
    prev_assistant_text = ""

    for turn in turns:
        if turn.role == "assistant":
            prev_assistant_text = turn.text
            continue

        # From here, turn.role == "user".
        instruction_match = _matches_any(_INSTRUCTION_PATTERNS, turn.text)
        if instruction_match:
            candidates.append(
                RawCandidate(
                    source=turn.source,
                    project=turn.project,
                    session_id=turn.session_id,
                    file_path=turn.file_path,
                    category="explicit_instruction",
                    text=redact.redact_text(redact.truncate(turn.text, _MAX_SNIPPET_CHARS)),
                    context_text="",
                    matched_pattern=instruction_match,
                    timestamp=turn.timestamp,
                )
            )
            continue  # an instruction match takes priority over confirmation checks

        fix_match = _matches_any(_FIX_CONFIRMATION_PATTERNS, turn.text)
        procedure_match = _matches_any(_PROCEDURE_CONFIRMATION_PATTERNS, turn.text)

        if fix_match or procedure_match:
            is_fix = bool(fix_match) or _looks_like_error_context(
                turn.text, prev_assistant_text
            )
            category = "repeated_fix" if is_fix else "repeated_procedure"
            candidates.append(
                RawCandidate(
                    source=turn.source,
                    project=turn.project,
                    session_id=turn.session_id,
                    file_path=turn.file_path,
                    category=category,
                    text=redact.redact_text(redact.truncate(turn.text, _MAX_SNIPPET_CHARS)),
                    context_text=redact.redact_text(
                        redact.truncate(prev_assistant_text, _MAX_SNIPPET_CHARS)
                    ),
                    matched_pattern=fix_match or procedure_match or "",
                    timestamp=turn.timestamp,
                )
            )

    return candidates


def mine_candidates(turns: list[TranscriptTurn]) -> list[RawCandidate]:
    """Flag candidate snippets across a batch of turns from any number of files.

    Turns are grouped by ``file_path`` (preserving each file's original
    order) so look-back-one-assistant-turn context stays correct even when
    the caller passes turns interleaved across many files in one batch.
    """
    all_candidates: list[RawCandidate] = []
    # groupby requires the input pre-sorted by the same key to group
    # correctly; stable-sort by file_path while preserving intra-file order.
    ordered = sorted(enumerate(turns), key=lambda pair: (pair[1].file_path, pair[0]))
    for file_path, group in groupby(ordered, key=lambda pair: pair[1].file_path):
        file_turns = [pair[1] for pair in group]
        all_candidates.extend(_mine_file(file_turns))
    return all_candidates
