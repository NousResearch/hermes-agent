"""Live reasoning deduplicates against delivered text across plain and detail fields."""

from types import SimpleNamespace

from agent.chat_completion_helpers_reasoning import stream_reasoning_delta
from agent.reasoning_summaries import ReasoningDeltaAccumulator


class _Watch:
    def __init__(self):
        self.seen = []

    def feed(self, text):
        self.seen.append(text)
        return False


class _Carriers:
    def feed(self, _delta, text):
        return text


def _stream(chunks):
    delivered = []
    owner = SimpleNamespace(_emit_reasoning=delivered.append)
    reasoning = ReasoningDeltaAccumulator()
    display_parts = []
    details = []
    plain_watch, detail_watch = _Watch(), _Watch()
    for plain, detail in chunks:
        delta = SimpleNamespace(reasoning_content=plain, reasoning=None,
                                reasoning_details=detail, model_extra=None)
        assert not stream_reasoning_delta(owner, delta, reasoning, display_parts, details,
                                          plain_watch, detail_watch, _Carriers())
    return "".join(delivered), "".join(reasoning.parts), details, plain_watch.seen, detail_watch.seen


def test_cumulative_plain_and_incremental_detail_share_delivered_prefix():
    prefix = "P" * 30
    detail = lambda text: [{"type": "reasoning.text", "text": text}]
    shown, persisted, details, plain_raw, detail_raw = _stream([
        (prefix, detail(prefix + "B")),
        (prefix + "BC", detail("C")),
    ])
    assert shown == prefix + "BC"
    assert persisted == prefix + "BC"
    assert details == [{"type": "reasoning.text", "text": prefix + "BC"}]
    assert plain_raw == [prefix, prefix + "BC"]
    assert detail_raw == [prefix + "B", "C"]


def test_repeated_plain_snapshot_does_not_hide_new_detail():
    prefix = "P" * 30
    detail = lambda text: [{"type": "reasoning.text", "text": text}]
    shown, persisted, details, plain_raw, detail_raw = _stream([
        (prefix, detail(prefix + "B")),
        (prefix, detail("C")),
    ])
    assert shown == prefix + "BC"
    assert persisted == prefix
    assert details == [{"type": "reasoning.text", "text": prefix + "BC"}]
    assert plain_raw == [prefix, prefix]
    assert detail_raw == [prefix + "B", "C"]


def test_detail_choice_keeps_opaque_replay_and_short_repetition():
    shown, persisted, details, _, detail_raw = _stream([
        ("C", [{"type": "reasoning.text", "text": "Complete thought"},
               {"type": "reasoning.encrypted", "data": "secret"}]),
    ])
    assert shown == "Complete thought"
    assert persisted == "C"
    assert details == [{"type": "reasoning.text", "text": "Complete thought"},
                       {"type": "reasoning.encrypted", "data": "secret"}]
    assert detail_raw == ["Complete thought"]

    shown, _, _, _, _ = _stream([
        ("the", [{"type": "reasoning.text", "text": "the"}]),
        ("the", [{"type": "reasoning.text", "text": "the"}]),
    ])
    assert shown == "thethe"


def test_detail_summary_boundary_is_exact():
    shown, _, _, _, _ = _stream([
        (None, [{"type": "reasoning.summary", "summary": "**One**"}]),
        (None, [{"type": "reasoning.summary", "summary": "**Two**"}]),
    ])
    assert shown == "**One**\n\n**Two**"

    first = "**Inspecting the first summary part**"
    second = "**Checking the second summary part**"
    shown, persisted, _, _, _ = _stream([
        (first, [{"type": "reasoning.summary", "summary": first}]),
        (first + second, [{"type": "reasoning.summary", "summary": second}]),
    ])
    assert shown == first + "\n\n" + second
    assert persisted == shown


def test_plain_repetition_after_intervening_text_is_real():
    prefix = "abcdefghijklmnopqrstuvwxyz0123456789"
    shown, persisted, _, _, _ = _stream([
        (prefix + " middle ", []),
        (prefix, []),
    ])
    assert shown == prefix + " middle " + prefix
    assert persisted == shown
