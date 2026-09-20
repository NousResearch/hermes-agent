"""Operational notifications must not anchor compaction or auto-focus (#92703).

Kanban/background completion wakes are persisted as ``role="user"`` rows typed
with ``display_kind="internal_notification"`` (run.py -- the synthetic-wake
path). The model-payload builder already strips ``display_kind`` before the
request, and ``is_user_originated_turn`` already ignores it, but the compaction
tail-anchor / auto-focus scans must do the same. Otherwise routine machine
traffic becomes the "last user turn" the compressor protects and the focus
hint derives from, displacing the user's real objective.

These are behavior contracts against the real compressor functions, not mocks:
feed a transcript of 1,000 operational notifications around a single human turn
and assert the operational rows are invisible to the anchor/focus logic.
"""
import pytest

from agent.context_compressor import ContextCompressor


def _compressor() -> ContextCompressor:
    cc = ContextCompressor(
        model="test-model",
        threshold_percent=0.75,
        protect_first_n=5,
        protect_last_n=20,
        quiet_mode=True,
        config_context_length=40960,
        provider="test",
    )
    cc._generate_summary = lambda *a, **k: "Summary of earlier turns."
    return cc


def _ops_notice(text: str) -> dict:
    """A Kanban/background completion wake, as persisted by the wake path."""
    return {
        "role": "user",
        "content": text,
        "display_kind": "internal_notification",
    }


def _human(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _transcript_with_n_ops(n: int, human_text: str = "Actually deploy the fix") -> list:
    """1,000 operational notifications with a single real human turn at the end."""
    msgs: list = [{"role": "system", "content": "sys"}]
    for i in range(n):
        msgs.append(_ops_notice(f"✔ Kanban T-{i} done — worker summary line {i}"))
        msgs.append(_assistant(f"Noted task {i} completion."))
    msgs.append(_human(human_text))
    msgs.append(_assistant("On it."))
    return msgs


def test_ops_notice_is_not_actionable_user_turn():
    cc = _compressor()
    assert cc._is_actionable_user_turn(_ops_notice("✔ Kanban T-1 done")) is False
    # A real human turn still is.
    assert cc._is_actionable_user_turn(_human("deploy the fix")) is True


def test_ops_notices_do_not_anchor_compaction_tail():
    cc = _compressor()
    msgs = _transcript_with_n_ops(1000)
    # The tail anchor must land on the human turn, not on any operational notice.
    idx = cc._find_last_user_message_idx(msgs, head_end=0)
    assert idx >= 0
    assert msgs[idx]["role"] == "user"
    assert msgs[idx].get("display_kind") is None, (
        "compaction tail anchored on an operational notification, "
        "not the user's real objective"
    )
    assert msgs[idx]["content"] == "Actually deploy the fix"


def test_ops_notices_do_not_become_auto_focus_source():
    cc = _compressor()
    msgs = _transcript_with_n_ops(1000, human_text="Summarize the Q3 roadmap")
    focus = cc._derive_auto_focus_topic(msgs)
    assert focus is not None
    assert "Q3 roadmap" in focus, (
        f"auto-focus derived from operational traffic instead of the user: {focus!r}"
    )
    assert "Kanban T-" not in focus, (
        f"auto-focus leaked an operational notification: {focus!r}"
    )


def test_conversational_user_count_unchanged_by_ops_notices():
    """1,000 notifications must not be counted as actionable user turns."""
    cc = _compressor()
    msgs = _transcript_with_n_ops(1000)
    actionable = [m for m in msgs if cc._is_actionable_user_turn(m)]
    # Exactly one: the human turn at the end.
    assert len(actionable) == 1
    assert actionable[0]["content"] == "Actually deploy the fix"
