from agent.grok_cli_client import GrokCliClient


def test_grok_cli_client_runs_single_prompt(monkeypatch):
    calls = []

    class FakeProcess:
        returncode = 0

        def communicate(self, timeout=None):
            return "GROK_OK\n", ""

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return FakeProcess()

    monkeypatch.setattr("agent.grok_cli_client.subprocess.Popen", fake_popen)

    client = GrokCliClient(
        command="/usr/local/bin/grok",
        args=["--no-memory", "--output-format", "plain"],
    )
    response = client.chat.completions.create(
        model="grok-build",
        messages=[{"role": "user", "content": "Reply with GROK_OK"}],
        timeout=5,
    )

    assert response.choices[0].message.content == "GROK_OK"
    cmd, kwargs = calls[0]
    assert cmd[0] == "/usr/local/bin/grok"
    assert "--prompt-file" in cmd
    assert "--model" in cmd
    assert "grok-build" in cmd
    assert kwargs["stdout"] is not None
    assert kwargs["stderr"] is not None


def test_grok_cli_client_preserves_tool_calls(monkeypatch):
    raw = (
        '<tool_call>{"id":"call_1","type":"function",'
        '"function":{"name":"terminal","arguments":"{\\"cmd\\":\\"pwd\\"}"}}'
        "</tool_call>"
    )

    class FakeProcess:
        returncode = 0

        def communicate(self, timeout=None):
            return raw, ""

    monkeypatch.setattr(
        "agent.grok_cli_client.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(),
    )

    client = GrokCliClient(command="/usr/local/bin/grok", args=["--no-memory"])
    response = client.chat.completions.create(
        model="grok-build",
        messages=[{"role": "user", "content": "Use terminal"}],
    )

    message = response.choices[0].message
    assert response.choices[0].finish_reason == "tool_calls"
    assert message.content == ""
    assert message.tool_calls[0].function.name == "terminal"
