"""Behavior contract: a completed pure-text assistant turn is durable before
the conversation loop yields control (#81641).

Neighbouring turn exits already carry this guarantee. The tool-call exit
flushes the assistant(tool_calls) block before handing control to
``_execute_tool_calls`` (#49045), and the verify-on-stop / pre_verify exits
flush ``final_msg`` before appending their nudge (#65919 §7). The ordinary
``finish_reason=stop`` exit — the common case — did not.

Its only durable write was ``finalize_turn`` -> ``_persist_session``, which
runs after the loop exits and after post-turn work (interrupt tail repair,
persist-override rewrite, micro-compaction — which can issue its own aux-LLM
call). Anything that ended the process or tore the session down inside that
window lost a reply the user had already been shown: the text reaches the UI
through the streaming/interim display path, which is display-only and never
writes to ``state.db``.

Reported against v0.20.0 on a remote (non-loopback) backend, where WS ``1006``
closures drive ``ws_orphan_reap`` teardown and widen that window: affected
sessions held user rows with zero assistant rows in ``state.db``.

These tests pin the fix:

1. The completed assistant row is flushed to the session DB *before* the
   turn's post-loop finalization runs.
2. A failing flush must not abort the turn — the answer is already produced
   and delivered, so ``_persist_session`` stays the retry.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client (mirrors test_run_agent's fixture)."""
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        return agent


def _run_text_turn(agent, answer: str, *, flush_side_effect=None):
    """Drive one clean finish_reason=stop turn, recording persistence calls.

    Returns ``(result, events)`` where ``events`` is the ordered log of
    persistence calls. Flush entries carry a snapshot of the transcript taken
    *at call time*, so the assertion cannot be satisfied by a later mutation
    of the same live list.
    """
    from tests.run_agent.test_run_agent import _mock_response

    agent.client.chat.completions.create.side_effect = [
        _mock_response(content=answer, finish_reason="stop"),
    ]

    events: list[tuple[str, object]] = []

    def _record_flush(messages, conversation_history=None):
        events.append((
            "flush",
            [
                (m.get("role"), m.get("content"))
                for m in messages
                if isinstance(m, dict)
            ],
        ))
        if flush_side_effect is not None:
            raise flush_side_effect
        return True

    def _record_persist(messages, conversation_history=None):
        events.append(("persist_session", None))

    with (
        patch.object(
            agent, "_flush_messages_to_session_db", side_effect=_record_flush
        ),
        patch.object(agent, "_persist_session", side_effect=_record_persist),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("what is 2 + 2?")

    return result, events


class TestCompletedTextTurnIncrementalPersistence:
    def test_completed_text_turn_is_flushed_before_finalization(self, loop_agent):
        """The assistant row must reach the session DB before the loop exits.

        Asserting on the snapshot taken at flush time (not on the final list)
        is what makes this mutation-survivable: removing the production flush
        leaves ``finalize_turn``'s ``_persist_session`` as the first
        persistence event and the test fails.
        """
        answer = "2 + 2 is 4."
        result, events = _run_text_turn(loop_agent, answer)

        assert result["final_response"] == answer

        # The turn-start crash-resilience write of the user message is itself a
        # _persist_session call, so "first event" is not the contract. The
        # contract is that a flush carrying the assistant answer lands before
        # the FINAL persist_session — the one finalize_turn issues after the
        # loop exits and after post-turn work.
        assert any(kind == "flush" for kind, _ in events), (
            "A completed text turn must flush the assistant row to the session "
            "DB; no _flush_messages_to_session_db call was observed."
        )

        answer_flushes = [
            i
            for i, (kind, snapshot) in enumerate(events)
            if kind == "flush" and snapshot and snapshot[-1] == ("assistant", answer)
        ]
        assert answer_flushes, (
            "A flush must carry the completed assistant answer as its "
            f"transcript tail; observed events: {events!r}"
        )

        persist_indices = [
            i for i, (kind, _) in enumerate(events) if kind == "persist_session"
        ]
        assert persist_indices, (
            "finalize_turn must call _persist_session after the loop exits; "
            f"observed events: {events!r}"
        )
        final_persist = persist_indices[-1]
        assert answer_flushes[0] < final_persist, (
            "The assistant row must be durable BEFORE post-loop finalization, "
            f"but the answer flush at index {answer_flushes[0]} did not precede "
            f"finalize_turn's _persist_session at index {final_persist}."
        )

    def test_flush_failure_does_not_abort_the_completed_turn(self, loop_agent):
        """A transient SQLite failure must not swallow a produced answer.

        The tool-call exit treats a failed flush as fatal because side-effecting
        tools must not run on state that exists only in this process. Here the
        turn is already over and the text already delivered, so the turn must
        still return its answer and still reach ``_persist_session``, which
        retries the write.
        """
        answer = "Still answered."
        result, events = _run_text_turn(
            loop_agent, answer, flush_side_effect=RuntimeError("database is locked")
        )

        assert result["final_response"] == answer, (
            "A failed incremental flush must not discard the produced answer."
        )
        assert ("persist_session", None) in events, (
            "finalize_turn's _persist_session must still run so the failed "
            "flush gets retried."
        )
