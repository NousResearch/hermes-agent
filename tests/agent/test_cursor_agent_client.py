from __future__ import annotations

from agent.cursor_agent_client import CursorAgentClient, _build_subprocess_env


def test_cursor_agent_client_does_not_put_api_key_in_argv() -> None:
    client = CursorAgentClient(
        api_key="crsr_secret_should_not_leak",
        command="cursor-agent",
        base_url="cursor://agent",
    )
    argv = client._build_argv(model="composer-2.5", workspace="/tmp")

    assert "--api-key" not in argv
    assert all("crsr_secret_should_not_leak" not in part for part in argv)
    assert _build_subprocess_env(client.api_key)["CURSOR_API_KEY"] == "crsr_secret_should_not_leak"


def test_cursor_agent_client_treats_login_sentinel_as_no_key() -> None:
    client = CursorAgentClient(
        api_key="cursor-agent-login",
        command="cursor-agent",
        base_url="cursor://agent",
    )

    assert client.api_key is None
    assert "--api-key" not in client._build_argv(model="composer-2.5", workspace="/tmp")
