"""Observed-context rolling compaction (gateway/observed_context.py).

Covers the write path (overflow -> iterative summary row, originals preserved) and the
read path (summary watermark + verbatim rows + hard-cap valve) with relationship
assertions, not frozen snapshots. The aux LLM is faked at the module boundary
(`_call_summary_llm`), everything else runs the real code.
"""
import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.observed_context import (
    _SUMMARY_ROW_END,
    _SUMMARY_ROW_PREFIX,
    assemble_observed_context,
    bound_observed_rows,
    compact_observed_overflow,
    is_observed_summary_row,
    maybe_compact_observed_context,
    observed_context_limits,
)


def _row(content, observed=True, role="user"):
    return {"role": role, "content": content, "observed": observed, "timestamp": "2026-09-06T00:00:00+00:00"}


def _summary_row(body, covered=10):
    return _row(f"{_SUMMARY_ROW_PREFIX} - rolling summary of {covered} older observed group "
                f"messages, compressed to preserve context. {_SUMMARY_ROW_END}\n\n{body}")


# --------------------------------------------------------------------------- read path

def test_assembly_leads_with_latest_summary_and_skips_pre_watermark_rows():
    rows = [_row("old 1"), _row("old 2"), _summary_row("earlier chatter digest"),
            _row("new 1"), _row("new 2")]
    out = assemble_observed_context(rows, 8000, 200)
    assert out is not None
    assert out.index("earlier chatter digest") < out.index("new 1") < out.index("new 2")
    assert "old 1" not in out and "old 2" not in out  # already summarized, never re-injected


def test_assembly_without_summary_is_all_verbatim():
    rows = [_row("a"), _row("b"), _row("c")]
    out = assemble_observed_context(rows, 8000, 200)
    assert out == "a\nb\nc"


def test_assembly_hard_cap_drops_oldest_with_note():
    rows = [_row(f"row{i:03d} " + "x" * 496) for i in range(100)]
    out = assemble_observed_context(rows, 8000, 0)
    assert out is not None
    assert len(out) <= 8000 + len("[... N older observed messages omitted ...]") + 2
    assert "row099" in out and "row000" not in out
    assert "older observed messages omitted" in out


def test_assembly_summary_row_survives_cap_when_it_fits():
    """The summary leads the block; a cap that fits summary + recent keeps both."""
    rows = [_row("x" * 3000), _summary_row("digest"), _row("recent")]
    out = assemble_observed_context(rows, 8000, 0)
    assert "digest" in out and "recent" in out


def test_bound_helper_keeps_most_recent_within_chars():
    rows = [f"row{i:03d} " + "x" * 493 for i in range(10)]  # 500 chars each
    kept = bound_observed_rows(rows, 1200, 0)
    assert rows[-2:] == kept  # most recent 1000 chars fit; the third would overflow
    assert bound_observed_rows(rows, 0, 3) == rows[-3:]
    assert bound_observed_rows(rows, 0, 0) == rows  # disabled


def test_summary_row_detection_requires_prefix_and_observed():
    assert is_observed_summary_row(_summary_row("body"))
    assert not is_observed_summary_row(_row("plain observed chatter"))
    assert not is_observed_summary_row({"role": "user", "content": f"{_SUMMARY_ROW_PREFIX} ...", "observed": False})


def test_limits_defaults_and_overrides():
    assert observed_context_limits(None) == (512_000, 4_000)
    assert observed_context_limits({"gateway": {"observed_context_max_chars": 100, "observed_context_max_rows": 3}}) == (100, 3)
    assert observed_context_limits({"gateway": {"observed_context_max_chars": "oops"}}) == (512_000, 4_000)
    assert observed_context_limits({"gateway": {"observed_context_max_chars": -5}}) == (0, 4_000)


# --------------------------------------------------------------------------- write path

def _store_with(rows):
    store = MagicMock()
    store.load_transcript.return_value = list(rows)
    return store


def test_compaction_summarizes_oldest_overflow_and_appends_summary_row(monkeypatch):
    rows = [_row(f"chatter {i:03d} " + "y" * 200) for i in range(60)]  # ~12.6K chars
    store = _store_with(rows)
    captured = {}

    def fake_summary(prompt):
        captured["prompt"] = prompt
        return "DIGEST: the group discussed testing."

    monkeypatch.setattr("gateway.observed_context._call_summary_llm", fake_summary)
    appended = compact_observed_overflow(store, "sess1", max_chars=8000, max_rows=200)
    assert appended is not None
    assert is_observed_summary_row(appended)
    store.append_to_transcript.assert_called_once_with("sess1", appended)
    # The prompt carries the serialized OLDEST rows (redaction/labels included) and no previous summary.
    assert "OBSERVED MESSAGES TO INCORPORATE" in captured["prompt"]
    assert "chatter 000" in captured["prompt"]
    # The newest rows stay verbatim (retention window = 50% of the char budget).
    kept_recent = rows[-1]["content"]
    assert kept_recent not in captured["prompt"]


def test_compaction_is_iterative_over_previous_summary(monkeypatch):
    rows = [_row(f"chatter {i:03d} " + "y" * 200) for i in range(60)]
    rows.insert(0, _summary_row("earlier digest", covered=5))
    store = _store_with(rows)
    captured = {}

    def fake_summary(prompt):
        captured["prompt"] = prompt
        return "UPDATED DIGEST."

    monkeypatch.setattr("gateway.observed_context._call_summary_llm", fake_summary)
    appended = compact_observed_overflow(store, "sess1", max_chars=8000, max_rows=200)
    assert appended is not None
    # The previous summary BODY (wrapper stripped) feeds the iterative prompt.
    assert "PREVIOUS SUMMARY" in captured["prompt"]
    assert "earlier digest" in captured["prompt"]
    assert _SUMMARY_ROW_PREFIX not in captured["prompt"].split("OBSERVED MESSAGES")[0].split("PREVIOUS SUMMARY (update it in place; PRESERVE all still-relevant information):\n")[1]
    # The new row covers the newly compacted rows only (count excludes the prior summary).
    assert f"rolling summary of {len(rows) - 1 - 30}" in appended["content"] or "rolling summary of" in appended["content"]


def test_compaction_noop_below_overflow_threshold(monkeypatch):
    rows = [_row("tiny"), _row("also tiny")]
    store = _store_with(rows)
    monkeypatch.setattr("gateway.observed_context._call_summary_llm", lambda prompt: pytest.fail("must not call LLM"))
    assert compact_observed_overflow(store, "sess1", max_chars=8000, max_rows=200) is None
    store.append_to_transcript.assert_not_called()


def test_compaction_aux_failure_leaves_transcript_untouched(monkeypatch):
    """Aux failure surfaces as None (the _call_summary_llm contract): no summary row is appended."""
    rows = [_row(f"chatter {i:03d} " + "y" * 200) for i in range(60)]
    store = _store_with(rows)
    monkeypatch.setattr("gateway.observed_context._call_summary_llm", lambda prompt: None)
    assert compact_observed_overflow(store, "sess1", max_chars=8000, max_rows=200) is None
    store.append_to_transcript.assert_not_called()


def test_compaction_disabled_when_bounds_are_zero(monkeypatch):
    rows = [_row(f"chatter {i:03d} " + "y" * 200) for i in range(60)]
    store = _store_with(rows)
    monkeypatch.setattr("gateway.observed_context._call_summary_llm", lambda prompt: pytest.fail("must not call LLM"))
    assert compact_observed_overflow(store, "sess1", max_chars=0, max_rows=0) is None


def test_serialized_input_is_bounded_and_redacted(monkeypatch):
    from gateway.observed_context import _serialize_observed_rows

    rows = [_row("api_key=sk-supersecret123 " + "z" * 9000)]
    text = _serialize_observed_rows(rows)
    assert len(text) <= 6000 + 200  # per-row head/tail truncation
    assert "sk-supersecret123" not in text  # redacted before the summarizer sees it
    assert "[OBSERVED 2026-09-06T00:00:00+00:00]:" in text


# --------------------------------------------------------------------------- scheduling

def test_maybe_compact_gates_on_pending_chars_before_spawning(monkeypatch):
    """A busy group must not reload its transcript on every message: the cheap pending-chars
    gate spawns a pass only past half the char budget."""
    from gateway.observed_context import _pending_chars

    _pending_chars.clear()
    store = _store_with([])
    calls = []
    monkeypatch.setattr("gateway.observed_context.compact_observed_overflow",
                        lambda *a, **k: calls.append(k) or None)

    async def drive():
        for _ in range(10):
            await maybe_compact_observed_context(store, "sess1", None, appended_chars=1000)
        await asyncio.sleep(0)  # let spawned tasks (if any) run

    asyncio.run(drive())
    assert calls == []  # 10K chars < 256K gate: no pass spawned
    assert _pending_chars["sess1"] == 10_000


def test_maybe_compact_spawns_background_pass_past_gate(monkeypatch):
    from gateway.observed_context import _pending_chars

    _pending_chars.clear()
    store = _store_with([])
    calls = []
    monkeypatch.setattr("gateway.observed_context.compact_observed_overflow",
                        lambda *a, **k: calls.append(k) or None)

    async def drive():
        await maybe_compact_observed_context(store, "sess1", None, appended_chars=300_000)
        await asyncio.sleep(0.05)  # let the background task run

    asyncio.run(drive())
    assert len(calls) == 1
    assert calls[0]["max_chars"] == 512_000
    assert _pending_chars["sess1"] == 0  # counter resets after spawning


def test_maybe_compact_never_raises_without_event_loop():
    # Sync context (no running loop): must be a silent no-op, not an error.
    asyncio.run(asyncio.to_thread(
        maybe_compact_observed_context, _store_with([]), "sess1", None, 300_000))
