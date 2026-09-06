"""Full-content fallback and desktop delegation exports share private storage."""
import json
import os
from datetime import datetime
from types import SimpleNamespace

import pytest


@pytest.mark.skipif(os.name != "posix", reason="POSIX file-mode contract")
def test_fallback_and_spawn_tree_artifacts_are_private(tmp_path, monkeypatch):
    from hermes_state import divert_session_transcript_jsonl
    from tui_gateway import server

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    message = {"role": "user", "content": "private full transcript"}
    fallback = divert_session_transcript_jsonl("sibling-test", [message])
    response = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "spawn_tree.save",
                                      "params": {"session_id": "sibling-test",
                                                 "subagents": [{"text": "private full transcript"}]}})
    assert "result" in response, response
    from pathlib import Path
    snapshot = Path(response["result"]["path"])
    for path in (fallback, snapshot, snapshot.parent / "_index.jsonl"):
        assert path.is_file()
        assert not path.stat().st_mode & 0o077
    assert json.loads(fallback.read_text(encoding="utf-8")) == message
    assert json.loads(snapshot.read_text(encoding="utf-8"))["subagents"][0]["text"] == message["content"]
    from run_agent import AIAgent
    agent = object.__new__(AIAgent)
    agent._session_json_enabled = True
    agent.logs_dir = tmp_path / "sessions"
    agent.session_id = "legacy"
    agent.model, agent.base_url, agent.platform = "m", "http://localhost", "cli"
    agent.session_start = datetime.now()
    agent._cached_system_prompt, agent.tools = "system", []
    agent.verbose_logging = True
    legacy = agent.logs_dir / "session_legacy.json"
    legacy.write_text('{}', encoding="utf-8")
    legacy.chmod(0o644)
    agent._save_session_log([message])
    assert not legacy.stat().st_mode & 0o077
    assert json.loads(legacy.read_text(encoding="utf-8"))["messages"][0]["content"] == message["content"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX file-mode contract")
def test_combined_batch_transcript_is_private(tmp_path):
    from batch_runner import BatchRunner, _append_jsonl

    row = {"conversations": [{"from": "human", "value": "private"}], "tool_stats": {}}
    _append_jsonl(tmp_path / "batch_1.jsonl", row)
    BatchRunner._combine_batch_files(SimpleNamespace(output_dir=tmp_path))
    combined = tmp_path / "trajectories.jsonl"
    assert not combined.stat().st_mode & 0o077
    assert json.loads(combined.read_text(encoding="utf-8")) == row
