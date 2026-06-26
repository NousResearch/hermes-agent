"""Generate the in-turn-reconcile sanitizer-derived fixture.

Builds a realistic tool-heavy in-turn compaction shape (600 folded + 32 kept
tail) whose ``compressed`` kept tail is produced by the REAL LCM sanitizer
(``_sanitize_active_context_messages``), so the comp-vs-pre divergence is the
genuine strip/drop/stub delta — not a hand-rolled one. Records:

  * ``true_kept_count`` — an INDEPENDENT integer (how many raw rows were placed
    in the tail BEFORE sanitize), so the test oracle isn't circular.
  * ``sanitizer_source_sha1`` — a provenance hash; a parity test fails if the
    live sanitizer source diverges from what generated the fixture.

Asserts (so a bad fixture can't be committed):
  * comp-kept overshoots pre-kept with the LIVE sign + ~1101-class magnitude;
  * >=1 stub AND >=1 dropped raw row are present;
  * no post-sanitize content collision among kept rows;
  * the cut is UNIQUE: sanitize(messages[cut+-1:]) != comp_kept (neighbor reject).

Run:  python tests/agent/fixtures/gen_inturn_reconcile_fixture.py
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys

# Make the repo importable when run directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ.setdefault("HERMES_HOME", os.path.expanduser("~/.hermes"))

from agent.model_metadata import estimate_messages_tokens_rough as est  # noqa: E402
from plugins.context_engine.lcm.engine import LCMEngine  # noqa: E402

FIXTURE_PATH = os.path.join(_HERE, "inturn_reconcile_sanitized_shape.json")
FRESH_TAIL = 32


def _sanitizer():
    eng = LCMEngine.__new__(LCMEngine)
    return lambda ms: eng._sanitize_active_context_messages(list(ms))


def _sanitizer_source_sha1() -> str:
    """Hash the sanitizer + its two tail-global helpers, so a behavioral change
    to any of them invalidates the fixture provenance."""
    parts = []
    for name in (
        "_sanitize_active_context_messages",
        "_sanitize_tool_pairs",
        "_clean_active_assistant_message",
        "_sanitize_active_assistant_content",
    ):
        fn = getattr(LCMEngine, name, None)
        if fn is not None:
            parts.append(inspect.getsource(fn))
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


def _at(i):
    return {"role": "assistant",
            "content": f"<thinking>internal reasoning {i} {'x' * 120}</thinking>Visible answer {i}."}


def _atc(i):
    return {"role": "assistant",
            "content": f"<thinking>deciding {i}</thinking>",
            "tool_calls": [{"id": f"call_{i}", "type": "function",
                            "function": {"name": "terminal", "arguments": "{}"}}]}


def _tr(i):
    return {"role": "tool", "tool_call_id": f"call_{i}", "content": f"tool output {i} {'y' * 150}"}


def _usr(i):
    return {"role": "user", "content": f"user message {i} {'w' * 30}"}


def build():
    san = _sanitizer()
    # 600 folded rows (tool-heavy mix).
    pre = [_atc(i) if i % 3 == 0 else (_tr(i) if i % 3 == 1 else _usr(i)) for i in range(600)]
    # 32-row kept tail with an INTERIOR drop (empty-thinking assistant, no tool_calls)
    # and an INTERIOR stub source (tool-call whose result is absent) — positioned so a
    # neighbor cut produces a different sanitized slice (uniqueness).
    kept_raw = []
    for i in range(600, 632):
        if i == 610:
            kept_raw.append({"role": "assistant", "content": "<thinking>only internal</thinking>"})
        elif i == 615:
            kept_raw.append(_atc(i))  # missing result → stub inserted
        else:
            kept_raw.append(_at(i) if i % 2 else _usr(i))
    kept_raw = kept_raw[:FRESH_TAIL]
    pre = pre[:600] + kept_raw
    true_kept_count = len(kept_raw)

    anchor = {"role": "system", "content": "objective anchor"}
    summary = {"role": "assistant",
               "content": "[Recent Summary (d0, node 5)] folded chat condensed "
                          + ("z" * 200) + " [Expand for details: x]",
               "_lcm_summary": True}
    comp_kept = san(kept_raw)
    comp = [anchor, summary] + comp_kept

    # ── Assertions: the fixture must genuinely stress the mechanism ──
    cut = len(pre) - true_kept_count
    # comp-kept heavier than pre-kept (live sign), ~1101-class magnitude isn't reachable
    # at 32 rows, but the SIGN and a non-trivial overshoot must hold.
    pre_kept_tok = est(kept_raw)
    comp_kept_tok = est(comp_kept)
    assert comp_kept_tok != pre_kept_tok, "kept tail must diverge under sanitize"
    stubs = [m for m in comp_kept if m.get("role") == "tool"
             and "earlier conversation" in str(m.get("content", ""))]
    assert len(stubs) >= 1, "fixture must contain >=1 inserted stub"
    # the interior empty-thinking assistant row (index 610) must have been DROPPED:
    # its sanitized content is empty → no visible content + no tool_calls → dropped.
    dropped_present = any(
        isinstance(m.get("content"), str) and m["content"].strip() == "" and not m.get("tool_calls")
        for m in comp_kept
    )
    assert not dropped_present, "the empty-thinking row must be dropped, not kept"
    # and the drop must have actually shrunk the real-rows count (excluding the +1 stub):
    real_kept = [m for m in comp_kept if m not in stubs]
    assert len(real_kept) < len(kept_raw), "a raw row must have been dropped by sanitize"
    # no post-sanitize content collision among kept rows (normalize would be ambiguous)
    from agent.compaction_stats import _inturn_norm_row
    norms = [_inturn_norm_row(m) for m in comp_kept]
    assert len(norms) == len(set(norms)), "post-sanitize kept rows must not collide"
    # uniqueness: neighbor cuts must NOT reproduce comp_kept
    from agent.compaction_stats import _inturn_norm
    tgt = _inturn_norm(comp_kept)
    assert _inturn_norm(san(pre[cut - 1:])) != tgt, "cut-1 must NOT match (uniqueness)"
    assert _inturn_norm(san(pre[cut + 1:])) != tgt, "cut+1 must NOT match (uniqueness)"

    return {
        "messages": pre,
        "compressed": comp,
        "true_kept_count": true_kept_count,
        "fresh_tail_count": FRESH_TAIL,
        "sanitizer_source_sha1": _sanitizer_source_sha1(),
        "_note": "Generated by gen_inturn_reconcile_fixture.py; comp kept tail is real "
                 "_sanitize_active_context_messages output. Regenerate if the parity test fails.",
    }


if __name__ == "__main__":
    data = build()
    with open(FIXTURE_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    print(f"wrote {FIXTURE_PATH}")
    print(f"  messages={len(data['messages'])} compressed={len(data['compressed'])} "
          f"true_kept_count={data['true_kept_count']} sha1={data['sanitizer_source_sha1'][:12]}")
