from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.claude_cli_client import ClaudeCLIClient, _normalize_claude_cli_model


def test_normalize_claude_cli_model_strips_anthropic_prefix_and_dots():
    assert _normalize_claude_cli_model("anthropic/claude-sonnet-4.6") == "claude-sonnet-4-6"


def test_chat_completion_runs_claude_print_mode(tmp_path):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="hello from claude\n", stderr="")

    client = ClaudeCLIClient(
        command="/usr/local/bin/claude",
        args=["--no-session-persistence", "--tools", ""],
        cwd=str(tmp_path),
    )

    with patch("agent.claude_cli_client.subprocess.run", side_effect=fake_run):
        response = client.chat.completions.create(
            model="anthropic/claude-sonnet-4.6",
            messages=[{"role": "user", "content": "Say hello"}],
        )

    assert captured["cmd"][:6] == [
        "/usr/local/bin/claude",
        "--no-session-persistence",
        "--tools",
        "",
        "-p",
        "--model",
    ]
    assert captured["cmd"][6] == "claude-sonnet-4-6"
    assert captured["kwargs"]["cwd"] == str(tmp_path)
    assert response.choices[0].message.content == "hello from claude"
    assert response.choices[0].finish_reason == "stop"


def test_chat_completion_raises_with_cli_stderr(tmp_path):
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="Not logged in")

    client = ClaudeCLIClient(command="claude", args=[], cwd=str(tmp_path))

    with patch("agent.claude_cli_client.subprocess.run", side_effect=fake_run):
        with pytest.raises(RuntimeError, match="Not logged in"):
            client.chat.completions.create(
                model="sonnet",
                messages=[{"role": "user", "content": "hello"}],
            )
