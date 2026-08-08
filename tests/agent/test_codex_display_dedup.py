"""Regression tests for the display-path snapshot-replay guard.

Sibling to test_codex_stream_cumulative_dedup.py. That test protects the FINAL
assembled output by collapsing cumulative ``message`` output-items. This one
protects the LIVE display path: on the Bedrock "mantle" Responses endpoint the
``output_text.delta`` stream is re-emitted from the top of the answer on every
new ``output_item`` for long, reasoning-interleaved replies, so a naive consumer
forwards the whole answer to ``on_text_delta`` N times (observed 27x). The
``_DisplayDeltaDeduper`` state machine forwards only the genuinely-new suffix.

Deterministic — no network. Run with:
    pytest tests/agent/test_codex_display_dedup.py -xvs
"""
import sys
import pathlib
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.codex_runtime import _DisplayDeltaDeduper, _consume_codex_event_stream


# ---------------------------------------------------------------------------
# Unit tests for the state machine
# ---------------------------------------------------------------------------

def test_true_incremental_is_passthrough():
    """A well-behaved endpoint (real incremental deltas) is unchanged."""
    d = _DisplayDeltaDeduper()
    assert d.feed("Hello") == "Hello"
    assert d.feed(", ") == ", "
    assert d.feed("world") == "world"


def test_single_item_progressive_snapshot_suppressed():
    """Within one item, incremental chunks accumulate and pass through.

    Deltas are incremental *chunks* that accumulate in the per-item buffer;
    the deduper forwards each genuinely-new suffix. This models the normal
    within-item case (no restart).
    """
    d = _DisplayDeltaDeduper()
    assert d.feed("Hel") == "Hel"
    assert d.feed("lo") == "lo"
    assert d.feed(" wo") == " wo"
    assert d.feed("rld") == "rld"
    # Visible text == "Hello world" exactly once.


def test_full_replay_across_items_suppressed():
    """New item replays the ENTIRE answer from char 0 — all suppressed.

    The endpoint restarts the delta stream on each new output_item. reset_item()
    clears the per-item accumulator; the displayed watermark keeps growing, so
    the replayed full answer in item 2 surfaces nothing new.
    """
    answer = "The quick brown fox jumps over the lazy dog."
    d = _DisplayDeltaDeduper()
    # Item 1: stream the answer as incremental chunks.
    out1 = []
    step = 10
    for s in range(0, len(answer), step):
        out1.append(d.feed(answer[s:s + step]))
    d.reset_item()
    # Item 2: endpoint restarts and replays the FULL answer from char 0
    # (again as incremental chunks). All of it is already displayed.
    out2 = []
    for s in range(0, len(answer), step):
        out2.append(d.feed(answer[s:s + step]))
    visible = "".join(out1) + "".join(out2)
    assert visible == answer, f"expected 1x answer, got {len(visible)} vs {len(answer)}"
    assert "".join(out2) == "", "second item full replay must be fully suppressed"


def test_divergent_content_not_dropped():
    """Genuinely different text (not a snapshot) is forwarded, never eaten."""
    d = _DisplayDeltaDeduper()
    assert d.feed("Answer A") == "Answer A"
    # A completely different continuation that is not a prefix relationship.
    got = d.feed("XYZ")
    assert "XYZ" in got, "divergent content must still reach the user"


# ---------------------------------------------------------------------------
# End-to-end: feed a synthetic mantle-style event stream and assert the
# display callback receives the answer exactly once.
# ---------------------------------------------------------------------------

def _ev(type_, **kw):
    return SimpleNamespace(type=type_, **kw)


def _message_item(text):
    return SimpleNamespace(
        type="message",
        role="assistant",
        status="completed",
        content=[SimpleNamespace(type="output_text", text=text)],
    )


def test_e2e_mantle_snapshot_stream_displays_once():
    """43-frame-style cumulative replay: display callback sees answer ~1x."""
    answer = "".join(f"section-{i} " for i in range(1, 40))  # ~ multi-hundred chars
    displayed_parts = []

    def make_stream():
        # Simulate N cumulative message items, each restarting the delta
        # stream from the top of the full answer-so-far.
        cuts = [len(answer) // 8 * k for k in range(1, 9)]
        cuts[-1] = len(answer)
        prev_snapshot = ""
        for idx, cut in enumerate(cuts):
            snapshot = answer[:cut]
            # output_item.added → new item boundary (delta restarts from 0)
            yield _ev("response.output_item.added", item=_message_item(""))
            # The endpoint re-emits the ENTIRE snapshot as deltas from char 0.
            # Emit it in a few chunks to mimic SSE granularity.
            step = max(1, len(snapshot) // 3)
            for s in range(0, len(snapshot), step):
                yield _ev("response.output_text.delta", delta=snapshot[s:s + step])
            yield _ev("response.output_item.done", item=_message_item(snapshot))
            prev_snapshot = snapshot
        yield _ev("response.completed", response=SimpleNamespace(output=[], status="completed"))

    final = _consume_codex_event_stream(
        make_stream(),
        model="openai.gpt-5.5",
        on_text_delta=lambda s: displayed_parts.append(s),
    )

    displayed = "".join(displayed_parts)
    # The user should see the answer exactly once — not 8x.
    assert displayed == answer, (
        f"display inflated: showed {len(displayed)} chars, answer is {len(answer)}"
    )
    # Sanity: naive (buggy) behavior would have shown sum of all snapshots.
    naive = sum(len(answer[:len(answer) // 8 * k]) for k in range(1, 9))
    assert len(displayed) < naive, "fix must beat naive concatenation"

    # --- The returned response object must ALSO be deduplicated, not just the
    # live display. This is the half the display-only fix originally missed:
    # final.output_text (raw delta join) and final.output (cumulative message
    # items) both re-inflated the same snapshots the display path collapsed.
    assert final.output_text == answer, (
        f"final.output_text inflated: {len(final.output_text)} chars vs "
        f"answer {len(answer)}"
    )
    # Cumulative message items collapse to a single most-complete message.
    assert len(final.output) == 1, (
        f"expected 1 collapsed message item, got {len(final.output)}"
    )
    assert final.output[0].type == "message"
    assert final.output[0].content[0].text == answer, (
        "collapsed output item must carry the answer exactly once"
    )


def test_e2e_final_output_text_deduped_on_tool_call_turn():
    """Even when nothing is displayed (tool-call turn), final.output_text is deduped.

    The deduper is fed on every delta regardless of has_tool_calls, so a
    snapshot-replaying endpoint can't smuggle the replays into the returned
    output_text via the tool-call path (where the display gate is closed).
    """
    answer = "".join(f"tok{i} " for i in range(1, 25))
    displayed_parts = []

    def make_stream():
        # A function_call item flips has_tool_calls True, closing the display gate.
        yield _ev("response.output_item.added",
                  item=SimpleNamespace(type="function_call", name="t", arguments="{}"))
        yield _ev("response.output_item.done",
                  item=SimpleNamespace(type="function_call", name="t", arguments="{}"))
        # Now a message that replays the full answer twice across two items.
        for _ in range(2):
            yield _ev("response.output_item.added", item=_message_item(""))
            step = max(1, len(answer) // 4)
            for s in range(0, len(answer), step):
                yield _ev("response.output_text.delta", delta=answer[s:s + step])
            yield _ev("response.output_item.done", item=_message_item(answer))
        yield _ev("response.completed", response=SimpleNamespace(output=[], status="completed"))

    final = _consume_codex_event_stream(
        make_stream(),
        model="openai.gpt-5.5",
        on_text_delta=lambda s: displayed_parts.append(s),
    )

    # Display gate closed on a tool-call turn → nothing streamed to screen.
    assert "".join(displayed_parts) == "", "no live display on a tool-call turn"
    # But the final assembled text is still deduped to the answer exactly once.
    assert final.output_text == answer, (
        f"final.output_text inflated on tool-call turn: {len(final.output_text)} "
        f"vs {len(answer)}"
    )
    # function_call preserved; two cumulative messages collapsed to one.
    types = [getattr(it, "type", None) for it in final.output]
    assert types == ["function_call", "message"], types
    assert final.output[1].content[0].text == answer


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
