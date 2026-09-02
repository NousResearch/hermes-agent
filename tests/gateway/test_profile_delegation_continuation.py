def test_profile_delegation_notification_formatter():
    from tools.process_registry import format_process_notification

    text = format_process_notification({
        "type": "profile_delegation",
        "delegation_id": "pd_123",
        "task_id": "t_1",
        "session_key": "agent:main:discord:channel:1",
        "requester_profile": "cmo",
        "executor_profile": "cto",
        "capability": "mcp:vercel",
        "risk": "READ",
        "status": "completed",
        "summary": "Vercel inspected.",
        "result": {"project": "ConnectMe", "status": "ok"},
    })
    assert "INTERNAL PROFILE DELEGATION RESULT" in text
    assert "executor_profile: cto" in text
    assert "mcp:vercel" in text
    assert "Continue the original task" in text
