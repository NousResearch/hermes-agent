from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli import memory_setup


class FakeConfidenceProvider:
    name = "confidence"

    def __init__(self):
        self.initialized = False
        self.calls = []

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.initialized = True

    def handle_tool_call(self, tool_name, args):
        self.calls.append((tool_name, args))
        return '{"success": true, "items": [{"id": "m1", "statement": "remembered"}]}'

    def shutdown(self):
        pass


def _run(command, **kwargs):
    provider = FakeConfidenceProvider()
    args = SimpleNamespace(memory_command=command, **kwargs)
    with patch("hermes_cli.config.load_config", return_value={"memory": {"provider": "confidence"}}), \
         patch("plugins.memory.load_memory_provider", return_value=provider):
        memory_setup.memory_command(args)
    return provider


def test_memory_review_routes_to_active_provider(capsys):
    provider = _run("review", include_inactive=False, limit=50)

    assert provider.initialized
    assert provider.calls == [("confidence_memory", {"action": "list", "include_inactive": False, "limit": 50})]
    assert "remembered" in capsys.readouterr().out


def test_memory_confirm_routes_to_active_provider(capsys):
    provider = _run("confirm", id="m1", source_excerpt="user confirmed")

    assert provider.calls == [("confidence_memory", {
        "action": "confirm",
        "id": "m1",
        "source_kind": "user_confirmed",
        "source_excerpt": "user confirmed",
    })]
    assert "success" in capsys.readouterr().out


def test_memory_delete_routes_to_active_provider(capsys):
    provider = _run("delete", id="m1")

    assert provider.calls == [("confidence_memory", {"action": "delete", "id": "m1"})]
    assert "success" in capsys.readouterr().out
