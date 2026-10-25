"""The codex app-server runtime runs ``transform_llm_output`` like the generic loop.

That runtime returns before ``finalize_turn``, so a rewriting plugin (redaction) must be applied
to the codex reply before its rows are persisted: the user (who, with ``hold_stream``, saw nothing
stream) receives the rewrite, and the stored assistant row is the rewrite too.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.codex_runtime import run_codex_app_server_turn
from tests.fakes.redactor_plugin import RAW, REWRITE, redactor_home


def _codex_agent(final_text: str):
    agent = MagicMock()
    agent._codex_session = MagicMock()  # a seeded session: no subprocess spawn
    agent._codex_session.run_turn.return_value = SimpleNamespace(
        interrupted=False, error=None, thread_id="thread-1", turn_id="turn-1", tool_iterations=0,
        projected_messages=[{"role": "assistant", "content": final_text}], final_text=final_text,
        should_retire=False)
    agent._codex_session_prompt = None
    agent.tool_progress_callback = None
    agent._iters_since_skill = agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent._session_db = None
    agent._llm_output_transform = None
    agent.session_id = "sess-codex"
    agent.platform = "cli"
    return agent


def test_codex_reply_is_rewritten_before_it_is_returned_or_stored():
    from hermes_cli import plugins

    redactor_home(Path(os.environ["HERMES_HOME"]), "http://127.0.0.1:9/v1", hold=True)
    plugins._reset_plugin_managers_for_tests()
    try:
        result = run_codex_app_server_turn(
            _codex_agent(RAW), user_message="config?", original_user_message="config?",
            messages=[{"role": "user", "content": "config?"}], effective_task_id="task-1", turn_id="t1")
    finally:
        plugins._reset_plugin_managers_for_tests()

    assert result["final_response"] == REWRITE
    assert result["response_transformed"] is True and result["pre_transform_response"] == RAW
    assert [m["content"] for m in result["messages"] if m["role"] == "assistant"] == [REWRITE]
