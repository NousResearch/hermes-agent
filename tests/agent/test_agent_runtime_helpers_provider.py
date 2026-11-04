from types import SimpleNamespace

from agent import agent_runtime_helpers


def test_external_process_profile_receives_resolved_command_and_args(monkeypatch):
    received = {}

    class Profile:
        name = "oc-acp"
        auth_type = "external_process"

        def create_client(self, **kwargs):
            received.update(kwargs)
            return "client"

    monkeypatch.setattr(
        "providers.get_provider_profile",
        lambda name: Profile() if name == "oc-acp" else None,
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_external_process_provider_credentials",
        lambda name: {
            "command": "/home/user/.local/bin/opencode",
            "args": ["acp"],
        },
    )

    result = agent_runtime_helpers._provider_supplied_client(
        SimpleNamespace(provider="oc-acp"),
        {"base_url": "acp://opencode", "model": "opencode/big-pickle"},
    )

    assert result == "client"
    assert received["command"].endswith("/opencode")
    assert received["args"] == ["acp"]
