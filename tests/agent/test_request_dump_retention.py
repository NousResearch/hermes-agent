"""HERMES_DUMP_RETENTION_DAYS prunes old request dumps at dump-write time."""

import json
import os
import time


def _make_agent(tmp_path):
    import run_agent

    import model_tools

    model_tools.get_tool_definitions = lambda **kwargs: []
    model_tools.check_toolset_requirements = dict
    agent = run_agent.AIAgent(
        model="gpt-4o",
        base_url="http://127.0.0.1:9208/v1",
        api_key="test-key",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.logs_dir = tmp_path
    return agent


def _old_dump(logs_dir, name):
    path = logs_dir / name
    path.write_text(json.dumps({"reason": "old"}), encoding="utf-8")
    stale = time.time() - 90 * 86400
    os.utime(path, (stale, stale))
    return path


def test_dump_write_prunes_dumps_older_than_retention(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_DUMP_RETENTION_DAYS", "30")
    agent = _make_agent(tmp_path)
    stale = _old_dump(tmp_path, "request_dump_sid_old_20250101_000000_000000.json")
    fresh = tmp_path / "request_dump_sid_new_20260101_000000_000000.json"
    fresh.write_text(json.dumps({"reason": "fresh"}), encoding="utf-8")

    path = agent._dump_api_request_debug(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        reason="preflight",
    )

    assert path is not None and path.exists()
    assert not stale.exists()
    assert fresh.exists()


def test_zero_retention_keeps_every_dump(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_DUMP_RETENTION_DAYS", "0")
    agent = _make_agent(tmp_path)
    stale = _old_dump(tmp_path, "request_dump_sid_old_20250101_000000_000000.json")

    path = agent._dump_api_request_debug(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        reason="preflight",
    )

    assert path is not None and path.exists()
    assert stale.exists()


def test_unrelated_files_in_logs_dir_are_untouched(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_DUMP_RETENTION_DAYS", "30")
    agent = _make_agent(tmp_path)
    other = _old_dump(tmp_path, "agent.log")

    agent._dump_api_request_debug(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        reason="preflight",
    )

    assert other.exists()
