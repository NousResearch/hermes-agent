# tests/agent/benchmarks/test_tier3_trajectory_replay.py
"""Trajectory replay differential.

Marked ``integration`` so pyproject's default ``addopts = "-m 'not integration'"``
keeps these out of plain CI runs. They depend on captured real-session
trajectories under ``trajectories/``; opt in with ``-m integration``.
"""
import hashlib
import json
import pytest
from pathlib import Path

from tests.agent.benchmarks.trajectory import load, TRAJECTORY_DIR

# Module-level marker — every test in this file is integration.
pytestmark = pytest.mark.integration


def _fingerprint(messages: list[dict]) -> str:
    """Stable hash of message-list shape (role + content + tool_call_ids)."""
    canonical = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c_str = json.dumps(
                [{"type": p.get("type"),
                  "text_len": len(p.get("text", ""))} for p in c if isinstance(p, dict)],
                sort_keys=True,
            )
        else:
            c_str = f"<{type(c).__name__}:{len(c or '')}>"
        tcs = [tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
               for tc in (m.get("tool_calls") or [])]
        canonical.append({"role": m.get("role"), "c": c_str, "tcs": tcs})
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode()
    ).hexdigest()[:16]


def _trajectories():
    if not TRAJECTORY_DIR.exists():
        return []
    return sorted(p.name for p in TRAJECTORY_DIR.glob("*.json"))


@pytest.mark.parametrize("name", _trajectories())
def test_3_1_baseline_replay_is_stable(name, compressor_pair, stub_summarizer):
    """Sanity: baseline compaction on a captured trajectory is deterministic."""
    baseline, _ = compressor_pair
    msgs, state = load(name)

    out1 = baseline.compress(msgs.copy(), current_tokens=state.get("tokens", 200_000))
    out2 = baseline.compress(msgs.copy(), current_tokens=state.get("tokens", 200_000))
    assert _fingerprint(out1) == _fingerprint(out2)


@pytest.mark.parametrize("name", _trajectories())
def test_3_2_candidate_diff_only_in_documented_fields(name, compressor_pair, stub_summarizer):
    """Compare baseline vs candidate output. Flags-on can differ — but
    the only allowed differences are:
      - content of older tool results (Pass 1.5 supersession)
      - boundary index (anchor)
      - summary text (different inputs → different summary)
    Anything else is a regression (e.g. roles changed, tool_call_ids
    corrupted, image parts dropped)."""
    baseline, with_flags = compressor_pair
    msgs, state = load(name)
    pre_tokens = state.get("tokens", 200_000)

    out_b = baseline.compress(msgs.copy(), current_tokens=pre_tokens)
    out_c = with_flags.compress(msgs.copy(), current_tokens=pre_tokens)

    # Roles in same order (modulo length difference at the head/tail)
    roles_b = [m.get("role") for m in out_b]
    roles_c = [m.get("role") for m in out_c]

    # Strict: every role in candidate must be a valid OpenAI role
    assert set(roles_c) <= {"system", "user", "assistant", "tool"}

    # No orphaned tool_call_ids in either output
    from tests.agent.benchmarks.test_tier1_atomicity import _orphan_tool_ids
    assert not _orphan_tool_ids(out_b)
    assert not _orphan_tool_ids(out_c)

    # If both produced the same number of messages AND the same role
    # sequence, the only acceptable per-message diffs are content
    # (specifically: older tool results becoming "[Superseded ..." or
    # the summary message body).
    if len(out_b) == len(out_c) and roles_b == roles_c:
        for i, (mb, mc) in enumerate(zip(out_b, out_c)):
            if mb.get("role") != "tool":
                continue
            cb = mb.get("content") or ""
            cc = mc.get("content") or ""
            if cb != cc:
                # Only acceptable: candidate replaced with "[Superseded ..."
                assert isinstance(cc, str) and (
                    cc.startswith("[Superseded by later")
                    or cc.startswith("[Duplicate tool output")
                ), (
                    f"Unexpected tool-content diff at idx {i}: "
                    f"{cb[:80]} → {cc[:80]}"
                )


@pytest.mark.parametrize("name", _trajectories())
def test_3_3_real_session_replay_does_not_raise(name, compressor_pair, stub_summarizer):
    """Smoke: every captured trajectory should compact without raising."""
    baseline, with_flags = compressor_pair
    msgs, state = load(name)
    for c in (baseline, with_flags):
        out = c.compress(msgs.copy(), current_tokens=state.get("tokens", 200_000))
        # Final shape is OpenAI-valid
        for m in out:
            assert m.get("role") in {"system", "user", "assistant", "tool"}
            if m.get("role") == "tool":
                assert "tool_call_id" in m
