"""Pre/post compression figures must be compared like-for-like (#135992).

A committed compression pass clears the usage anchor, so the post-pass preflight figure is a
rough whole-request estimate (messages + system prompt + tool schemas) while the pre-pass
figure was provider usage plus a rough delta for the new rows: two different units. Comparing
them directly reads a real cut as growth and fails a fitting turn closed, e.g.
``~56,427 -> ~65,606 request tokens`` on a 65,536 window (schema-heavy route).

``_run_preflight_passes`` now prices the pass input with the same estimator that produces the
post-pass figure and projects the post-pass rough figure onto the anchored scale before it
judges progress, the next-pass decision, both fail-close calls and the post-pass threshold
test. The projection is an estimate, never provider usage, and it falls back to the raw
figure whenever the inputs are not sound (no anchor before the pass, anchor still present
after, no committed compression, non-positive rough figure).

Case letters (A1/A2/A3/B/C/D/E/F) match the approved plan. Synthetic messages and real token
estimators only: no provider calls, no production session data, and ``_preflight_request_tokens``
is never patched in the cases that exercise the comparison.
"""

from unittest.mock import patch

import pytest

from agent import model_metadata
from agent import turn_context as tc
from agent.turn_context import PreflightCompressionTimedOut, project_rough_onto_anchored
from agent.turn_context_compaction import (
    CompactionOutcome,
    _preflight_compression,
    _run_preflight_passes,
)
from agent.usage_anchor import capture_usage_anchor, set_usage_anchor

WINDOW = 65_536
THRESHOLD = 55_705                        # 0.85 x window: the engine trigger from the issue
BIG_WINDOW = 131_072
BIG_THRESHOLD = 111_411                   # 0.85 x BIG_WINDOW
ANCHOR_PROMPT, ANCHOR_COMPLETION = 56_027, 400      # anchored pre-pass figure 56,427
HEAVY_ANCHOR_PROMPT = 355_713                       # anchored pre-pass figure 356,113
SYSTEM_PROMPT = "s" * 14_800              # ~3.7K rough tokens
ROW_CHARS = 1_320


def _transcript(count: int, chars: int) -> list[dict]:
    return [
        {"role": "user", "content": ("x" * chars) + f"|row{i}"} for i in range(count)
    ]


def _tool_schemas(count: int = 12, desc: int = 1_200, param: int = 1_000) -> list[dict]:
    """~8.8K rough tokens of schema-heavy tools, the shape that made the units diverge."""
    return [
        {"type": "function", "function": {
            "name": f"tool_{i}", "description": "d" * desc,
            "parameters": {"type": "object", "properties": {
                "p1": {"type": "string", "description": "s" * param},
                "p2": {"type": "string", "description": "s" * param},
            }},
        }}
        for i in range(count)
    ]


MSGS = _transcript(160, ROW_CHARS)
TOOLS = _tool_schemas()
# Genuinely over a 131K window with no anchor: the #116472 shape.
HEAVY = _transcript(300, 2_000)


def _rough(messages, *, system_prompt: str = SYSTEM_PROMPT, tools=TOOLS) -> int:
    return model_metadata.estimate_request_tokens_rough(
        messages, system_prompt=system_prompt, tools=tools
    )


def _halved(messages, rows: int) -> list[dict]:
    """Content-only rewrite: same row count, first ``rows`` rows truncated by half."""
    return [
        dict(m, content=m["content"][: len(m["content"]) // 2]) if i < rows else m
        for i, m in enumerate(messages)
    ]


class _Compressor:
    """Compressor double: thresholds and predicates only, no token accounting of its own."""

    def __init__(self, window: int = WINDOW, threshold: int = THRESHOLD) -> None:
        self.context_length = window
        self.threshold_tokens = threshold
        self.protect_first_n = 3
        self.protect_last_n = 3
        self.summary_target_ratio = 0.5
        self.emit_automatic_compaction_status = False
        self.awaiting_real_usage_after_compression = False
        self.last_compression_rough_tokens = 0
        self.compression_count = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def should_compress(self, tokens: int) -> bool:
        return tokens >= self.threshold_tokens


class _Agent:
    def __init__(
        self, *, window: int = WINDOW, threshold: int = THRESHOLD, anchored: bool = True,
        messages=None, compress=None, tools=TOOLS,
    ) -> None:
        self.context_compressor = _Compressor(window, threshold)
        self.session_id = "s1"
        self.model = "m"
        self.provider = "p"
        self.api_mode = "chat_completions"
        self.base_url = ""
        self.compression_enabled = True
        self.max_compression_attempts = 3
        self.tools = tools
        self.messages = MSGS if messages is None else messages
        self._usage_anchor = None
        self._turn_base_usage_anchor = None
        self._request_pressure_anchored = False
        self._persist_disabled = True
        self._compression_blocked_transient = None
        self._compress_calls = 0
        self._compress = compress

    def _emit_status(self, message) -> None:  # pragma: no cover - status surface only
        pass

    def _compress_context(self, messages, system_message, approx_tokens=None, task_id=None):
        self._compress_calls += 1
        return self._compress(self, messages, system_message)


def _rows_dropped(count: int):
    """A pass that commits: fewer rows, usage anchor cleared (the real commit contract)."""
    def impl(agent, messages, system_message):
        new = [dict(m) for m in messages[:-count]]
        agent.context_compressor.awaiting_real_usage_after_compression = True
        agent.context_compressor.last_compression_rough_tokens = _rough(new)
        set_usage_anchor(agent, None)
        return new, system_message
    return impl


def _rows_shortened(count: int):
    """A pass that commits by summarising content only: same rows, fewer tokens."""
    def impl(agent, messages, system_message):
        new = _halved(messages, count)
        agent.context_compressor.awaiting_real_usage_after_compression = True
        agent.context_compressor.last_compression_rough_tokens = _rough(new)
        set_usage_anchor(agent, None)
        return new, system_message
    return impl


def _noop(agent, messages, system_message):
    """A pass that reclaims nothing: the input list comes back unchanged."""
    return messages, system_message


def _anchor_and_measure(agent, messages, *, prompt_tokens: int = ANCHOR_PROMPT,
                        system_prompt: str = SYSTEM_PROMPT) -> int:
    """Install real provider usage, then measure the pass input through the real estimator."""
    set_usage_anchor(agent, capture_usage_anchor(prompt_tokens, ANCHOR_COMPLETION, messages))
    pre = tc._preflight_request_tokens(agent, messages, system_prompt)
    assert agent._request_pressure_anchored is True, "precondition: the pre-pass figure is anchored"
    return pre


def _drive(agent, preflight_tokens, *, messages=None, system_prompt: str = SYSTEM_PROMPT):
    """Run the real pass loop; returns ``(outcome, raised_message_or_None)``."""
    out = CompactionOutcome(
        messages=agent.messages if messages is None else messages,
        active_system_prompt=system_prompt, conversation_history=None,
        current_turn_user_idx=0,
    )
    raised = None
    try:
        _run_preflight_passes(
            agent, out, agent.context_compressor, preflight_tokens, system_prompt, "t"
        )
    except PreflightCompressionTimedOut as exc:
        raised = str(exc)
    return out, raised


# ── A. The reported false fail-close ──


def test_A1_anchored_pre_with_a_rough_post_that_fits_after_projection_survives():
    agent = _Agent(compress=_rows_dropped(2))
    pre = _anchor_and_measure(agent, MSGS)
    raw = _rough(MSGS[:-2])
    projected = project_rough_onto_anchored(pre, _rough(MSGS), raw)
    assert raw > WINDOW, "precondition: the raw post-pass figure alone looks over-window"
    assert THRESHOLD <= projected < WINDOW, "precondition: like-for-like it fits, still over threshold"

    out, raised = _drive(agent, pre)

    assert raised is None, raised          # RED before the fix: PreflightCompressionTimedOut
    assert out.compressed is True
    assert len(out.messages) == len(MSGS) - 2
    assert out.blocked is True             # over threshold but fitting: stop passes, send it
    assert agent._compress_calls == 1


def test_A2_committed_pass_with_rows_removed_is_not_reported_as_no_progress():
    agent = _Agent(compress=_rows_dropped(16))
    pre = _anchor_and_measure(agent, MSGS)
    raw = _rough(MSGS[:-16])
    projected = project_rough_onto_anchored(pre, _rough(MSGS), raw)
    assert THRESHOLD <= raw < WINDOW, "precondition: raw keeps it over threshold, inside the window"
    assert projected < THRESHOLD, "precondition: like-for-like it is back under threshold"

    out, raised = _drive(agent, pre)

    assert raised is None
    assert out.blocked is False            # RED before the fix: spurious "insufficient progress"
    assert agent._compress_calls == 1


def test_A3a_content_only_pass_projecting_a_material_cut_counts_as_progress():
    halved = _halved(MSGS, 60)
    agent = _Agent(compress=_rows_shortened(60))
    pre = _anchor_and_measure(agent, MSGS)
    raw = _rough(halved)
    projected = project_rough_onto_anchored(pre, _rough(MSGS), raw)
    assert raw >= THRESHOLD, "precondition: the raw figure still crosses the threshold"
    assert projected < pre * 0.95, "precondition: the like-for-like cut is material (>5%)"

    out, raised = _drive(agent, pre)

    assert raised is None
    assert out.blocked is False            # RED before the fix: 160 -> 160 rows read as no progress
    assert len(out.messages) == len(MSGS), "the pass was content-only: rows are unchanged"


def test_A3b_content_only_pass_under_five_percent_keeps_the_materiality_rule():
    halved = _halved(MSGS, 3)
    agent = _Agent(compress=_rows_shortened(3))
    pre = _anchor_and_measure(agent, MSGS)
    raw = _rough(halved)
    projected = project_rough_onto_anchored(pre, _rough(MSGS), raw)
    assert raw > WINDOW, "precondition: the raw figure alone looks over-window"
    assert projected >= pre * 0.95, "precondition: under 5% of the anchored figure"

    out, raised = _drive(agent, pre)

    assert raised is None                  # like-for-like it fits: no fail-close
    assert out.blocked is True             # the >5% materiality rule still refuses a retry


# ── B. The over-window protection the projection must not weaken ──


def test_B_over_window_session_still_fails_closed_through_the_projection():
    agent = _Agent(window=BIG_WINDOW, threshold=BIG_THRESHOLD, compress=_rows_dropped(2))
    pre = _anchor_and_measure(agent, MSGS, prompt_tokens=HEAVY_ANCHOR_PROMPT)
    raw = _rough(MSGS[:-2])
    projected = project_rough_onto_anchored(pre, _rough(MSGS), raw)
    assert raw < BIG_WINDOW, "precondition: the raw figure alone would look harmless"
    assert projected > BIG_WINDOW, "precondition: like-for-like it is still over the window"

    out, raised = _drive(agent, pre)

    assert raised is not None and "too large to compress further" in raised
    assert out.blocked is True


def test_B_unanchored_116472_shape_still_fails_closed():
    agent = _Agent(window=BIG_WINDOW, threshold=BIG_THRESHOLD, messages=HEAVY, compress=_noop)
    pre = tc._preflight_request_tokens(agent, HEAVY, SYSTEM_PROMPT)
    assert agent._request_pressure_anchored is False, "precondition: no usage anchor"
    assert pre > BIG_WINDOW, "precondition: the transcript alone is over the window (#116472)"

    out, raised = _drive(agent, pre, messages=HEAVY)

    assert raised is not None and "too large to compress further" in raised
    assert out.messages is agent.messages, "the pass reclaimed nothing: the raw input is kept"


# ── C. Unanchored pre/post: unchanged ──


def test_C_unanchored_pre_and_post_keep_the_raw_comparison():
    # Over the threshold but inside the window: sent as-is, no fail-close.
    fitting = _Agent(window=1_000_000, threshold=850_000, messages=MSGS, compress=_noop)
    pre = tc._preflight_request_tokens(fitting, MSGS, SYSTEM_PROMPT)
    assert fitting._request_pressure_anchored is False
    out, raised = _drive(fitting, pre)
    assert raised is None
    assert out.blocked is True
    assert fitting._compress_calls == 1

    # A real cut on the same raw scale: progress, no block, no projection involved.
    shrinking = _Agent(compress=_rows_dropped(40))
    pre = tc._preflight_request_tokens(shrinking, MSGS, SYSTEM_PROMPT)
    assert shrinking._request_pressure_anchored is False
    out, raised = _drive(shrinking, pre)
    assert raised is None
    assert out.blocked is False


# ── D. No-op and lock-skip contracts ──


def test_D_noop_pass_keeps_the_anchor_and_still_fails_closed_over_window():
    agent = _Agent(window=BIG_WINDOW, threshold=BIG_THRESHOLD, messages=MSGS, compress=_noop)
    pre = _anchor_and_measure(agent, MSGS, prompt_tokens=HEAVY_ANCHOR_PROMPT)

    out, raised = _drive(agent, pre)

    assert raised is not None and "too large to compress further" in raised
    assert agent._usage_anchor is not None, "no commit: the usage anchor is untouched"
    assert out.messages is agent.messages, "no-op contract: the input list is returned"


def test_D_lock_skipped_pass_defers_without_arming_the_blocker():
    agent = _Agent(window=BIG_WINDOW, threshold=BIG_THRESHOLD, messages=MSGS, compress=_noop)
    pre = _anchor_and_measure(agent, MSGS, prompt_tokens=HEAVY_ANCHOR_PROMPT)

    with patch(
        "agent.turn_context_compaction.compression_skipped_due_to_lock", return_value=True
    ):
        out, raised = _drive(agent, pre)

    assert raised is None
    assert out.blocked is False, "a lock DEFER is not proof of incompressibility"
    assert agent._compress_calls == 1, "passes stop for this turn"


# ── E. Unsound rough_before ──


def test_E_projection_helper_refuses_unsound_inputs():
    assert project_rough_onto_anchored(56_427, 0, 67_365) is None
    assert project_rough_onto_anchored(56_427, -5, 67_365) is None
    assert project_rough_onto_anchored(0, 68_045, 67_365) is None
    assert project_rough_onto_anchored(56_427, 68_045, None) is None
    assert project_rough_onto_anchored(True, 68_045, 67_365) is None
    assert project_rough_onto_anchored(56_427, 68_045, 67_365) == 55_863


def test_E_zero_rough_before_preserves_the_raw_comparison():
    agent = _Agent(
        window=BIG_WINDOW, threshold=BIG_THRESHOLD, messages=MSGS, compress=_rows_dropped(2)
    )
    pre = _anchor_and_measure(agent, MSGS, prompt_tokens=HEAVY_ANCHOR_PROMPT)
    raw = _rough(MSGS[:-2])
    assert raw < BIG_THRESHOLD, "precondition: the raw figure is under the threshold"
    assert project_rough_onto_anchored(pre, _rough(MSGS), raw) > BIG_WINDOW, (
        "precondition: a sound rough_before would have projected an over-window figure"
    )
    real_rough = tc._rough_request_tokens
    # A real transcript never rough-estimates to 0 (per-message overhead), so force the guard on
    # the pass input only; the post-pass estimate and the estimator under test stay real.
    forced = lambda a, m, s: 0 if m is MSGS else real_rough(a, m, s)  # noqa: E731

    with patch("agent.turn_context._rough_request_tokens", side_effect=forced):
        out, raised = _drive(agent, pre)

    assert raised is None, "no division by zero and no invented scale"
    assert out.compressed is True
    assert out.blocked is False, "the raw comparison governs, exactly as before the fix"


# ── F. The real caller path ──


def test_F_real_preflight_gate_projects_and_keeps_the_over_window_guard():
    # (1) Anchored pre-pass figure, committed pass, raw post-pass figure over the window:
    #     the real gate must complete the turn instead of failing it closed.
    agent = _Agent(compress=_rows_dropped(2))
    set_usage_anchor(agent, capture_usage_anchor(ANCHOR_PROMPT, ANCHOR_COMPLETION, MSGS))
    out = CompactionOutcome(
        messages=MSGS, active_system_prompt=SYSTEM_PROMPT, conversation_history=None,
        current_turn_user_idx=0,
    )
    try:
        _preflight_compression(agent, out, SYSTEM_PROMPT, MSGS[0]["content"], "t")
    except PreflightCompressionTimedOut as exc:  # pragma: no cover - the RED path
        pytest.fail(f"the real gate failed the turn closed: {exc}")

    assert out.compressed is True
    assert len(out.messages) == len(MSGS) - 2
    assert agent._usage_anchor is None, "the committed pass cleared the anchor"
    assert agent._compress_calls == 1

    # (2) Same gate, a genuinely over-window transcript that reclaims nothing: still guarded.
    heavy = _Agent(
        window=BIG_WINDOW, threshold=BIG_THRESHOLD, messages=HEAVY, compress=_noop
    )
    heavy_out = CompactionOutcome(
        messages=HEAVY, active_system_prompt=SYSTEM_PROMPT, conversation_history=None,
        current_turn_user_idx=0,
    )
    with pytest.raises(PreflightCompressionTimedOut, match="too large to compress further"):
        _preflight_compression(heavy, heavy_out, SYSTEM_PROMPT, HEAVY[0]["content"], "t")
