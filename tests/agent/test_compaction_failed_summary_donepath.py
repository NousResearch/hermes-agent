"""Phase D — D-6: a failed/placeholder summary that leaves the request over
threshold is counted as ineffective on the REAL done-path.

With ``compression.abort_on_summary_failure = false`` (the fleet default), a
summary failure inserts a deterministic placeholder and compaction "completes".
Before the fix, the done-site never recorded this as ineffective at the request
level, so a failed-summary loop could thrash invisibly. This drives the REAL
``compress_context`` done-site (not a unit stub) with a real ContextCompressor
whose summary generation fails, and asserts the agent-visible
``_ineffective_compression_count`` incremented.

This is the negative/adversarial gate for Phase D (the matched-pair #61 fix is
verified separately by tests/agent/test_auxiliary_main_first.py and the live
runtime probe at deploy).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from hermes_state import SessionDB


def _build_real_agent(db: SessionDB, session_id: str):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    return agent


def _thrash_transcript() -> list:
    """A transcript whose PROTECTED TAIL is essentially the whole request: a
    tiny summarizable middle + an 8-message huge recent tail. Summarizing the
    middle sheds ~0% of the request, so a failed/placeholder summary that stays
    over threshold is the real thrash condition (the protected tail alone
    exceeds the trigger)."""
    msgs = []
    # Tiny middle (cheap to summarize, ~zero payoff).
    for i in range(4):
        msgs.append({"role": "user", "content": f"mid {i}"})
        msgs.append({"role": "assistant", "content": "ok"})
    # 8-message huge recent tail (protected; dominates the request).
    for i in range(4):
        msgs.append({"role": "user", "content": "recent " * 5000})
        msgs.append({"role": "assistant", "content": "reply " * 5000})
    return msgs


def test_failed_summary_over_threshold_counts_ineffective_on_real_donepath(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    parent = "PARENT_D6"
    db.create_session(parent, source="discord")
    agent = _build_real_agent(db, parent)

    cc = agent.context_compressor
    cc.protect_last_n = 8
    cc.abort_on_summary_failure = False
    # Force the summary to FAIL -> placeholder path.
    cc._generate_summary = lambda *a, **k: None

    messages = _thrash_transcript()
    from agent.model_metadata import estimate_request_tokens_rough
    pre_req = estimate_request_tokens_rough(messages, system_prompt="system prompt", tools=agent.tools or None)
    # Pin the trigger just under pre: the protected 8-msg tail keeps the post
    # request over threshold AND sheds ~0% — the real thrash condition.
    cc.threshold_tokens = int(pre_req * 0.8)

    assert cc._ineffective_compression_count == 0
    agent._compress_context(messages, "system prompt", approx_tokens=pre_req)

    assert cc._ineffective_compression_count >= 1, (
        "a failed/placeholder summary that stays over threshold with <10% shed "
        "must count as ineffective so the anti-thrash guard can eventually fire"
    )
