# Placeholder-title rejection predicate (spec §4, ticket t_a642ab85).
#
# Pure classification: does a candidate session title consist entirely of a
# machine-authored STT placeholder, so the derived-title lane must refuse to
# name a session after it?  Sister of `_MACHINE_PREFIXES`
# (agent/title_generator.py), which these notes currently slip past — every
# marker here carries a drifting detail (cache path, duration suffix), so a
# literal blacklist would rot on the next path change.  Bracket shape + marker
# survives the drift (spec §4.2).
#
# Scope (spec §4.4): titling input ONLY.  Never suppress the note itself from
# the turn or history; never suppress a tag-only recompose.

from __future__ import annotations

# Raw STT-JSON leak: a transcription provider result swallowed whole into the
# title (observed live 2026-09-30: {"text":" Go to all my recent projects…").
_STT_JSON_LEAK_PREFIX = '{"text"'

# Known interior markers of Hermes-authored bracket notes (case-insensitive).
# "The user sent a voice message" is subsumed by "voice message".
_INTERIOR_MARKERS = (
    "voice message",
    "audio message",
    "could not be transcribed",
)


def is_rejected_placeholder(title: str | None) -> bool:
    """``True`` when *title* is a rejected STT placeholder, not a real subject.

    Matches (spec §4.1): the whole titleable content is one bracket note
    ``[...]`` whose interior carries a known voice/audio marker, or a raw
    STT JSON leak starting ``{"text"``.

    Explicitly NOT rejected (spec §4.3): ``Transcribe voice message
    audio_*.ogg`` — a real opener typed by the user; brackets alone are not
    enough (``[no marker here]`` survives); empty/``None`` is absent, not a
    placeholder.  No I/O; no dependence on the exact placeholder wording —
    only on bracket shape + marker.
    """
    if not title:
        return False
    stripped = title.strip()
    if not stripped:
        return False
    if stripped.startswith(_STT_JSON_LEAK_PREFIX):
        return True
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return False
    interior = stripped[1:-1].strip().lower()
    return any(marker in interior for marker in _INTERIOR_MARKERS)
