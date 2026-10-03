"""A raising ``step_callback`` must be visible at WARNING, not swallowed at DEBUG (regression for #33023).

Issue #33023 lost ACP tool-completion events behind two silent layers: ``_send_update``
logged its failure at DEBUG only, and the core's ``step_callback`` invocation caught the
whole callback crash and logged it at DEBUG only. The first layer is covered by
``tests/acp_adapter/test_events.py``; this file covers the second.
"""

from __future__ import annotations

import logging


def _agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "agent:\n  budget_warning_ratio: 0.75\n", encoding="utf-8"
    )
    from hermes_state import SessionDB
    from model_tools import _clear_tool_defs_cache
    from run_agent import AIAgent
    from tools.registry import invalidate_check_fn_cache

    # Cases may run in separate workers; their availability caches must not
    # survive a change from ordinary to dispatcher-owned construction.
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    return AIAgent(
        session_db=SessionDB(db_path=tmp_path / "proof.db"),
        model="test-model",
        provider="openai-compat",
        api_key="test",
        base_url="http://127.0.0.1:1/v1",
        max_iterations=4,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def test_raising_step_callback_is_logged_at_warning(tmp_path, monkeypatch, caplog):
    """The ACP completion delivery lives inside ``step_callback``; when it raises, DEBUG means
    neither the agent log nor anyone debugging a stuck ACP client ever sees why."""
    from agent.turn_iteration_prep import prepare_iteration

    agent = _agent(tmp_path, monkeypatch)
    try:
        def _boom(_api_call_count, _prev_tools):
            raise RuntimeError("callback exploded")

        agent.step_callback = _boom
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "t0", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "t0", "content": "result"},
        ]
        with caplog.at_level(logging.DEBUG, logger="agent.turn_iteration_prep"):
            prepare_iteration(agent, messages=messages, api_call_count=2)

        surfaced = [
            record
            for record in caplog.records
            if record.levelno >= logging.WARNING and "step_callback error" in record.getMessage()
        ]
        assert surfaced, "a step_callback crash must surface above DEBUG"
        assert "callback exploded" in surfaced[0].getMessage()
        assert "2" in surfaced[0].getMessage()  # the iteration it happened on
    finally:
        agent._session_db.close()
