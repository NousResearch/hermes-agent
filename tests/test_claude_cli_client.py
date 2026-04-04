from types import SimpleNamespace
from unittest.mock import patch

from agent.claude_cli_client import ClaudeCLIClient, _CLAUDE_CLI_DISABLED_TOOLS


def _completed(stdout: str):
    return SimpleNamespace(stdout=stdout, stderr="", returncode=0)


def test_first_call_uses_session_id_then_resume_on_followup():
    client = ClaudeCLIClient(command="claude", session_id="hermes-session-1")

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            return _completed('{"result":"first","session_id":"claude-session-abc","usage":{"input_tokens":10,"output_tokens":5}}')
        return _completed('{"result":"second","session_id":"claude-session-abc","usage":{"input_tokens":8,"output_tokens":4}}')

    with patch("agent.claude_cli_client.subprocess.run", side_effect=fake_run):
        client.chat.completions.create(model="claude-cli/claude-sonnet-4-6", messages=[{"role": "user", "content": "hi"}])
        client.chat.completions.create(model="claude-cli/claude-sonnet-4-6", messages=[{"role": "user", "content": "again"}])

    assert "--session-id" in calls[0]
    assert "--resume" not in calls[0]
    assert "--resume" in calls[1]
    assert "claude-session-abc" in calls[1]



def test_existing_seeded_session_id_retries_with_resume_when_cli_reports_in_use():
    client = ClaudeCLIClient(command="claude", session_id="hermes-session-1")

    calls = []
    seeded_id = client._hermes_session_uuid

    from types import SimpleNamespace
    def fake_invoke(cmd, *, timeout_seconds):
        calls.append(cmd)
        if len(calls) == 1:
            return SimpleNamespace(stdout="", stderr=f"Error: Session ID {seeded_id} is already in use.", returncode=1)
        return SimpleNamespace(stdout=f'{{"result":"resumed","session_id":"{seeded_id}","usage":{{"input_tokens":6,"output_tokens":3}}}}', stderr="", returncode=0)

    with patch.object(client, "_invoke", side_effect=fake_invoke):
        resp = client.chat.completions.create(model="claude-cli/claude-sonnet-4-6", messages=[{"role": "user", "content": "again"}])

    assert "--session-id" in calls[0]
    assert "--resume" in calls[1]
    assert seeded_id in calls[1]
    assert resp.choices[0].message.content == "resumed"


def test_build_command_disables_claude_builtin_tools():
    client = ClaudeCLIClient(command="claude", session_id="hermes-session-1")

    cmd = client._build_command(
        model="claude-sonnet-4-6",
        prompt_text="hi",
    )

    pairs = list(zip(cmd, cmd[1:]))
    disabled_tools = [value for flag, value in pairs if flag == "--disallowedTools"]

    assert disabled_tools == list(_CLAUDE_CLI_DISABLED_TOOLS)
    assert cmd[:5] == ["claude", "-p", "--output-format", "json", "--model"]
