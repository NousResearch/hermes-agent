"""Hermes scope cwd export through the real NeMo Relay ATOF plugin."""

from __future__ import annotations

import json

import pytest


def test_run_conversation_exports_session_and_turn_cwds(tmp_path, monkeypatch):
    relay = pytest.importorskip("nemo_relay")
    if getattr(relay, "_native", None) is None:
        pytest.skip("NeMo Relay native binding is unavailable on this platform")

    from agent import relay_runtime
    from hermes_cli.lifecycle import finalize_session
    from run_agent import AIAgent
    from tools.terminal_tool import clear_session_cwd, record_session_cwd

    hermes_home = tmp_path / "hermes-home"
    session_cwd = tmp_path / "session"
    turn_cwd = tmp_path / "task"
    atof_dir = tmp_path / "atof"
    for directory in (hermes_home, session_cwd, turn_cwd, atof_dir):
        directory.mkdir()

    config = tmp_path / "plugins.toml"
    config.write_text(
        f"""version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 4

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
output_directory = {json.dumps(str(atof_dir))}
filename = "events.jsonl"
mode = "overwrite"
""",
        encoding="utf-8",
    )
    session_id = "cwd-e2e-session"
    task_id = "cwd-e2e-task"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, str(config))
    monkeypatch.chdir(session_cwd)
    monkeypatch.setattr(
        "agent.conversation_loop.run_conversation",
        lambda *_args, **_kwargs: {"final_response": "ok", "completed": True},
    )

    relay_runtime._reset_for_tests()
    relay.plugin.clear()
    record_session_cwd(task_id, str(turn_cwd))
    agent = None
    try:
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai",
            model="test-model",
            session_id=session_id,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            disabled_toolsets=["*"],
        )
        result = agent.run_conversation("hello", task_id=task_id)
        assert result["final_response"] == "ok"
        finalize_session(session_id=session_id)
    finally:
        if agent is not None:
            agent.close()
        clear_session_cwd(task_id)
        relay_runtime._reset_for_tests()

    events = [
        json.loads(line)
        for line in (atof_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    def only_event(name: str, category: str) -> dict:
        matches = [
            event
            for event in events
            if event.get("name") == name and event.get("scope_category") == category
        ]
        assert len(matches) == 1
        return matches[0]

    session_start = only_event("hermes.session", "start")
    turn_start = only_event("hermes.turn", "start")
    session_end = only_event("hermes.session", "end")
    turn_end = only_event("hermes.turn", "end")

    assert session_start["data"] == {"cwd": str(session_cwd)}
    assert turn_start["data"] == {"cwd": str(turn_cwd)}
    assert turn_start["parent_uuid"] == session_start["uuid"]
    assert "cwd" not in session_end["data"]
    assert "cwd" not in turn_end["data"]
