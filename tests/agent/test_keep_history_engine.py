"""Tests for the keep_history context engine (request-only compaction).

``keep_history`` never rewrites the persisted transcript: it compacts
background tool context (tool logs, terminal dumps, file arrays) only in
the per-request message list via ``select_context()``, so the visible chat
history in UIs that render the session DB stays fully intact while the
model still receives a bounded, compacted context.
"""

import copy

import pytest

from plugins.context_engine import discover_context_engines, load_context_engine


@pytest.fixture()
def engine():
    eng = load_context_engine("keep_history")
    assert eng is not None, "keep_history engine must be discoverable"
    eng.update_model(
        model="deepseek/deepseek-v4-flash-0731",
        context_length=1_048_576,
        provider="nous",
    )
    return eng


def _bulky_session(n_dumps: int = 20, dump_chars: int = 150_000):
    """System + user + N terminal tool_call/tool_result pairs + chat tail."""
    msgs = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Set up the proxy stack"},
    ]
    big = "y" * dump_chars
    for i in range(n_dumps):
        msgs.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"ls -la"}',
                    },
                }
            ],
        })
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": big})
    msgs += [
        {"role": "assistant", "content": "All green"},
        {"role": "user", "content": "Ship it"},
        {"role": "assistant", "content": "Done, released v2"},
    ]
    return msgs


def test_engine_discoverable_and_named():
    names = [name for name, _desc, _avail in discover_context_engines()]
    assert "keep_history" in names
    assert load_context_engine("keep_history").name == "keep_history"


def test_never_triggers_destructive_compression(engine):
    # The whole point: the LLM-summary + archive_and_compact path must
    # never auto-fire, even with a nearly full context.
    assert engine.should_compress(999_999) is False
    assert engine.should_compress_info(999_999) == (False, None)
    assert engine.should_compress_preflight([]) is False


def test_loop_proactive_prune_is_noop(engine):
    # prune_tool_results_only commits to the session DB via
    # archive_and_compact; keep_history must not participate.
    msgs = _bulky_session()
    passed = copy.deepcopy(msgs)
    out, n = engine.prune_tool_results_only(passed, current_tokens=900_000)
    assert out is passed  # input object returned unchanged (no-op contract)
    assert n == 0


def test_select_context_prunes_request_only(engine):
    msgs = _bulky_session()
    orig = copy.deepcopy(msgs)

    selected = engine.select_context(copy.deepcopy(msgs), budget_tokens=1_048_576)

    assert selected is not None
    assert selected is not msgs  # replacement list, never in-place mutation
    # Chat turns (user/assistant text) survive verbatim.
    chat = [
        (m["role"], m.get("content", ""))
        for m in selected
        if m["role"] in ("user", "assistant")
    ]
    assert chat[0] == ("user", "Set up the proxy stack")
    assert chat[-1] == ("assistant", "Done, released v2")
    # Bulk tool results are replaced with one-line summaries. The dedup
    # pass is intentionally lossless: the NEWEST full copy survives while
    # older duplicates become short back-references, so at most one tool
    # result keeps its full payload and the rest collapse.
    tool_lens = sorted(
        len(m.get("content", "")) for m in selected if m.get("role") == "tool"
    )
    assert tool_lens
    assert sum(1 for l in tool_lens if l <= 200) >= len(tool_lens) - 1
    assert tool_lens[0] <= 200
    # Persisted transcript is untouched (request-only contract).
    assert msgs == orig


def test_select_context_small_request_noop(engine):
    small = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert engine.select_context(copy.deepcopy(small), budget_tokens=1_048_576) is None


def test_select_context_falls_back_when_budget_missing(engine):
    # Safety valve: a call site that supplies neither a model context
    # length nor budget_tokens must not silently disable request
    # compaction — the engine falls back to a conservative default window.
    engine.context_length = 0
    msgs = _bulky_session()
    selected = engine.select_context(copy.deepcopy(msgs), budget_tokens=0)
    assert selected is not None
    # Fallback window is far below the ~1M-token session bulk, so the
    # hard-ceiling trim must also have engaged and bounded the request.
    est = sum(
        len(str(m.get("content", ""))) // 4 + len(str(m.get("tool_calls", ""))) // 4
        for m in selected
    )
    assert est < 128_000


def test_select_context_rearm_gate(engine):
    # After a committed prune, an immediate re-call must no-op so prompt
    # cache breaks stay episodic (same contract as the built-in rearm).
    msgs = _bulky_session()
    first = engine.select_context(copy.deepcopy(msgs), budget_tokens=1_048_576)
    assert first is not None
    second = engine.select_context(copy.deepcopy(first), budget_tokens=1_048_576)
    assert second is None


def test_compress_preserves_all_chat_turns(engine):
    msgs = _bulky_session()
    out = engine.compress(copy.deepcopy(msgs))
    roles = [m["role"] for m in out if m["role"] in ("user", "assistant")]
    # 2 user turns + 20 tool-call assistant rows + 2 chat assistant turns
    # — every chat turn survives; nothing is summarized away.
    assert roles.count("user") == 2
    assert roles.count("assistant") == 22
    # No summary card is produced.
    joined = " ".join(str(m.get("content", "")) for m in out)
    assert "CONTEXT COMPACTION" not in joined
    assert "Earlier turns were compacted" not in joined
    # Tool bulk is still deterministically pruned (dedup back-references).
    tool_lens = sorted(
        len(m.get("content", "")) for m in out if m.get("role") == "tool"
    )
    assert tool_lens
    assert tool_lens[0] <= 200
