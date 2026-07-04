"""Tests for the opt-in recency-weighted "brief" advisory view (moa_loop)."""

from agent.moa_loop import (
    _brief_frame,
    _brief_reference_messages,
    _compress_cold_result,
    _reference_messages,
    _render_tool_calls,
)


def _transcript(depth: int, *, result_chars: int = 6000, arg_chars: int = 3000):
    msgs = [
        {"role": "system", "content": "boilerplate " * 400},
        {"role": "user", "content": "Fix the failing reconciliation tests without changing the public API."},
    ]
    for i in range(depth):
        msgs.append({
            "role": "assistant",
            "content": f"step {i}",
            "tool_calls": [{"function": {"name": "edit_file",
                                         "arguments": '{"new_content":"' + ("Y" * arg_chars) + '"}'}}],
        })
        msgs.append({"role": "tool", "content": f"result {i}\n" + ("X" * result_chars)})
    msgs.append({"role": "assistant", "content": "one test still failing"})
    return msgs


_TOOLS = [{"function": {"name": "read_file"}},
          {"function": {"name": "edit_file"}},
          {"function": {"name": "run"}}]


def test_brief_is_budget_bounded_regardless_of_depth():
    """Baseline grows with depth; brief stays clamped to its total budget."""
    budget = 24000
    prev = None
    for depth in (2, 8, 32):
        brief = _brief_reference_messages(_transcript(depth), total_budget=budget, tools=_TOOLS)
        size = sum(len(m["content"]) for m in brief)
        # Allow a small overshoot for the (never-dropped) hot window + frame.
        assert size <= budget + 6000, (depth, size)
        prev = size
    # And the deep brief is far smaller than the deep baseline.
    deep = _transcript(32)
    assert sum(len(m["content"]) for m in _brief_reference_messages(deep, tools=_TOOLS)) \
        < 0.3 * sum(len(m["content"]) for m in _reference_messages(deep))


def test_brief_matches_baseline_size_on_short_turns():
    """No penalty when there is nothing to overload (short conversation)."""
    short = _transcript(1)
    base = sum(len(m["content"]) for m in _reference_messages(short))
    brief = sum(len(m["content"]) for m in _brief_reference_messages(short, tools=_TOOLS))
    assert brief <= base * 1.15  # frame adds a little; must not balloon


def test_brief_ends_on_user_turn_and_alternates():
    for depth in (0, 1, 5):
        brief = _brief_reference_messages(_transcript(depth), tools=_TOOLS)
        assert brief, depth
        assert brief[-1]["role"] == "user", depth
        # no two consecutive same-role turns (coalesced)
        for a, b in zip(brief, brief[1:]):
            assert a["role"] != b["role"], (depth, a["role"])


def test_brief_frame_derived_from_goal_and_tools():
    frame = _brief_frame(_transcript(3), _TOOLS, "Do not change the public API.")
    assert frame is not None
    assert "GOAL:" in frame and "reconciliation" in frame
    assert "read_file" in frame and "edit_file" in frame
    assert "CONSTRAINTS: Do not change the public API." in frame


def test_brief_frame_none_when_no_goal_or_tools():
    assert _brief_frame([{"role": "system", "content": "x"}], None, None) is None


def test_render_tool_calls_arg_budget_truncates():
    calls = [{"function": {"name": "edit_file", "arguments": "A" * 5000}}]
    full = _render_tool_calls(calls)                 # default: no cap (shipping behaviour)
    capped = _render_tool_calls(calls, arg_budget=80)
    assert len(full) > 4000
    assert len(capped) < 200 and "+4920 chars" in capped


def test_compress_cold_result_keeps_salient_lines():
    log = ("# ran: pytest -q\n"
           + "\n".join(f"pass line {i} boring output" for i in range(400))
           + "\nFAILED tests/test_x.py::test_rounding — AssertionError: 0.01 != 0.00\n"
           + "\n".join(f"more noise {i}" for i in range(400)))
    out = _compress_cold_result(log, 320)
    assert len(out) <= 320 + 40
    assert "FAILED" in out and "0.01 != 0.00" in out      # salient survives
    assert "boring output" not in out                      # noise dropped
    assert "ran: pytest -q" in out                         # header kept


def test_compress_cold_result_falls_back_to_head_when_no_signal():
    dump = "\n".join(f"benign config key_{i} = {i}" for i in range(500))
    out = _compress_cold_result(dump, 300)
    assert len(out) <= 300 + 60
    assert out.startswith("benign config key_0")


def test_brief_distributed_failures_survive_deep_history():
    """The recon-regression guard: on a deep transcript whose diagnosis is smeared
    across many early failing-test logs, those FAILED lines must survive cold
    compression rather than being elided wholesale."""
    depth = 16
    t = _transcript(depth, result_chars=6000)
    # make each tool result carry a distinct failing-assertion breadcrumb
    n = 0
    for msg in t:
        if msg["role"] == "tool":
            msg["content"] = f"FAILED test_step_{n} — AssertionError: mismatch {n}\n" + msg["content"]
            n += 1
    brief = _brief_reference_messages(t, recent_turns=4, total_budget=24000, tools=_TOOLS)
    blob = "\n".join(m["content"] for m in brief)
    survived = sum(1 for i in range(n) if f"FAILED test_step_{i}" in blob)
    # the vast majority of early breadcrumbs should survive (not just the hot 4)
    assert survived >= n - 2, f"only {survived}/{n} failing breadcrumbs survived"


def test_brief_preserves_recent_state_fidelity():
    """The most recent tool result must survive at high fidelity; an ancient one
    is compressed to a gist."""
    depth = 20
    brief = _brief_reference_messages(_transcript(depth, result_chars=6000),
                                      recent_turns=3, tools=_TOOLS)
    blob = "\n".join(m["content"] for m in brief)
    assert f"result {depth - 1}" in blob            # latest step present
    # an early step is either elided or gisted, never present at full 6000 width
    assert "result 0\n" + ("X" * 6000) not in blob
