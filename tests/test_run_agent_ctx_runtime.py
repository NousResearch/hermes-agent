from hermes_cli.ctx_runtime import CtxBinding


def test_aiagent_includes_ctx_note_in_system_prompt(monkeypatch):
    import run_agent

    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **_kwargs: [])
    monkeypatch.setattr(
        run_agent,
        "maybe_bind_ctx_session",
        lambda **kwargs: CtxBinding(
            active=True,
            reason="ctx task bound",
            session_id=kwargs["session_id"],
            platform=kwargs.get("platform") or "cli",
            workspace_id="ws-1",
            task_id="task-1",
            worktree_id="wt-1",
            worktree_path="/tmp/ctx-worktree",
        ),
    )

    agent = run_agent.AIAgent(
        model="test/model",
        api_key="test-key",
        base_url="https://example.com/v1",
        enabled_toolsets=["terminal"],
        session_id="sess-1",
        platform="cli",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    prompt = agent._build_system_prompt()
    assert "ctx task `task-1`" in prompt
    assert "/tmp/ctx-worktree" in prompt
    assert "When dispatching Codex workers" in prompt
    assert "Slack" in prompt
    assert "Moshi" in prompt
    assert "webhook URLs" in prompt
    assert "external status-report/notification endpoints" in prompt
    assert "unless the task explicitly requires notification work" in prompt
