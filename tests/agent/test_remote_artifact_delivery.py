"""Final replies preserve terminal references without eagerly fetching attachments."""

import os
from pathlib import Path

import pytest

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home
from tools.environments.base import BaseEnvironment


class UnreadEnvironment(BaseEnvironment):
    def execute(self, command, **kwargs):
        pytest.fail("Finalization must not read terminal attachments")

    def fetch_file(self, *args, **kwargs):
        pytest.fail("Finalization must not export terminal attachments")

    def cleanup(self):
        pass


@pytest.mark.parametrize("remote_path", ["/workspace/chart.svg", "/workspace/my script.py"])
def test_remote_reply_is_unchanged_without_eager_fetch(tmp_path, monkeypatch, request, remote_path):
    from hermes_cli import plugins
    from run_agent import AIAgent
    from hermes_state import SessionDB
    from tools import terminal_tool

    task_id = "remote-artifact-turn"
    monkeypatch.setitem(terminal_tool._active_environments, task_id, UnreadEnvironment("/workspace", 30))
    home = Path(os.environ["HERMES_HOME"])
    db = SessionDB(db_path=home / "state.db")
    request.addfinalizer(db.close)
    raw = f"Here is the file.\n\nMEDIA:{remote_path}"
    with FakeLLMServer([Text(raw, chunk_chars=5)]) as server:
        write_hermes_home(home, server.base_url, extra_config=(
            "updates:\n  check: false\nmodels_dev:\n  url: http://127.0.0.1:9/api.json\n"
            "skills:\n  creation_nudge_interval: 0\nmemory:\n  nudge_interval: 0\n"
        ))
        plugins._reset_plugin_managers_for_tests()
        try:
            agent = AIAgent(base_url=server.base_url, api_key="sk-fake-e2e", provider="custom",
                            model="fake-model", quiet_mode=True, skip_context_files=True, skip_memory=True,
                            enabled_toolsets=["todo"], platform="desktop", session_id="artifact-session", session_db=db)
            result = agent.run_conversation("Send the chart", task_id=task_id)
        finally:
            plugins._reset_plugin_managers_for_tests()
    assert result["final_response"] == raw
    assert not result["response_transformed"]
    stored = db.get_messages(agent.session_id)
    assert any(row["role"] == "assistant" and row["content"] == raw for row in stored)
    assert not (home / "cache/generated/files").exists()
