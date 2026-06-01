from __future__ import annotations

from .types import InterruptionAction, InterruptionDecision, InterruptionSignal, TurnEventKind


def _word_count(signal: InterruptionSignal) -> int:
    p = signal.latest_partial
    return len((p.text or "").split()) if p else 0


class InterruptionPolicy:
    """Pure, deterministic barge-in policy. No I/O, no clock."""

    def decide(self, signal: InterruptionSignal) -> InterruptionDecision:
        at = signal.at_ms
        if not signal.assistant_speaking:
            return InterruptionDecision(InterruptionAction.WAIT, "assistant_idle", at)

        te = signal.turn_event
        if te is not None and te.kind is TurnEventKind.POSSIBLE_BACKCHANNEL:
            return InterruptionDecision(InterruptionAction.IGNORE, "backchannel", at)

        sustained = signal.ms_since_speech_start >= signal.params.min_speech_ms
        enough_words = _word_count(signal) >= signal.params.min_words
        if sustained and enough_words:
            return InterruptionDecision(InterruptionAction.INTERRUPT, "real_barge_in", at)

        if signal.ms_since_speech_start >= signal.params.false_interruption_timeout_ms:
            return InterruptionDecision(InterruptionAction.RESUME, "false_interruption_timeout", at)

        return InterruptionDecision(InterruptionAction.WAIT, "insufficient_evidence", at)
