"""A cron run that the tool-loop guardrail halts is a failed run (#160062).

Cron is an unattended platform, so hard stops are on by default. The halted turn ends with
"send `continue`", which nobody can answer in a cron run, and its result carries
``failed=False`` / ``completed=True``: the scheduler recorded ``last_status: ok`` and
``failure_deliver`` never fired.
"""

import os
from pathlib import Path

import pytest

from cron.scheduler import _final_response_from_result, _summarize_cron_failure_for_delivery
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home


def _halted_cron_turn() -> tuple[dict, type]:
    """The result the real turn loop returns when a cron agent keeps repeating a failing call."""
    from run_agent import AIAgent

    home = Path(os.environ["HERMES_HOME"])
    failing = ToolCall("read_file", {"path": str(home / "missing" / "report.txt")})
    with FakeLLMServer([failing] * 8 + [Text("never reached")]) as srv:
        write_hermes_home(home, srv.base_url)
        agent = AIAgent(base_url=srv.base_url, api_key="sk-fake-e2e", provider="custom", model="fake-model",
                        platform="cron", quiet_mode=True, skip_context_files=True, skip_memory=True,
                        enabled_toolsets=["file"])
        return agent.run_conversation("read the report"), AIAgent


def test_a_guardrail_halted_cron_run_fails_the_job_and_the_notice_names_the_tool():
    result, agent_cls = _halted_cron_turn()
    # The turn loop's own verdict: a controlled halt, not a failure.
    assert result["turn_exit_reason"] == "guardrail_halt"
    assert result["failed"] is False and result["completed"] is True

    with pytest.raises(RuntimeError) as raised:
        _final_response_from_result(result, "job1", "Morning brief", agent_cls)

    notice = _summarize_cron_failure_for_delivery(
        {"id": "job1", "name": "Morning brief"}, f"RuntimeError: {raised.value}")
    assert "read_file" in notice
    assert "continue" not in notice  # nobody can answer the halt reply's prompt in a cron run
