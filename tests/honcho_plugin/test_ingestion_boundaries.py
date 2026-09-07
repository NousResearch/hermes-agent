"""Configuration and message-origin boundaries for durable Honcho memory."""

import json
from types import SimpleNamespace

import pytest

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSession
from tools.process_registry_notifications import format_process_notification


def _configured_provider(tmp_path, monkeypatch, root=None, host=None):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "honcho.json").write_text(json.dumps({
        "enabled": True,
        **(root or {}),
        "hosts": {"hermes": host or {}},
    }))
    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig.from_global_config(host="hermes")
    session = HonchoSession("audit", "human", "assistant", "audit")
    provider._manager = SimpleNamespace(get_or_create=lambda key: session, save=lambda session: None)
    provider._session_key = session.key
    provider._session_initialized = True
    # Exercise the production sync body without any client or background I/O.
    monkeypatch.setattr(provider, "_spawn_write", lambda fn, *args: fn())
    return provider, session


@pytest.mark.parametrize(("root", "host", "expected_roles"), [
    ({}, {}, ["user", "assistant"]),
    ({"saveAssistantMessages": False}, {}, ["user"]),
    ({"saveAssistantMessages": True}, {"saveAssistantMessages": False}, ["user"]),
    ({"saveAssistantMessages": False}, {"saveAssistantMessages": True}, ["user", "assistant"]),
    ({"saveAssistantMessages": False}, {"saveAssistantMessages": None}, ["user"]),
    ({"saveMessages": False}, {"saveAssistantMessages": True}, []),
])
def test_resolved_assistant_policy_controls_persisted_roles(tmp_path, monkeypatch, root, host, expected_roles):
    provider, session = _configured_provider(tmp_path, monkeypatch, root, host)

    provider.sync_turn("A human preference.", "An assistant suggestion.")

    assert [m["role"] for m in session.messages] == expected_roles


@pytest.mark.parametrize("message", [
    "Message from 🤖 Hyde (@Hyde): A teammate's result.",
    *[format_process_notification({
        "session_id": "proc_a1234567890b", "command": "synthetic-command",
        "exit_code": exit_code, "completion_reason": reason, "output": "synthetic result",
    }) for exit_code, reason in [(0, "exited"), (1, "exited"), (None, "failed_start"), (None, "lost"), (-15, "killed")]],
    format_process_notification({"type": "watch_match", "session_id": "proc_a1234567890b", "pattern": "ready"}),
    '[IMPORTANT: Background process 123 matched watch pattern "ready".\nOutput: done]',
    "[IMPORTANT: 2 background processes completed for this session.\nSynthetic batch.]",
], ids=["bot-relay", "process-success", "process-failure", "process-failed-start", "process-lost",
        "process-killed", "process-watch", "legacy-process-watch", "process-batch"])
def test_machine_envelopes_are_excluded_but_human_discussion_is_retained(tmp_path, monkeypatch, message):
    provider, session = _configured_provider(tmp_path, monkeypatch)

    provider.sync_turn("\n " + message, "The assistant's machine-event reply.")
    assert session.messages == []

    human_message = "Please explain this notification: " + message
    provider.sync_turn(human_message, "A human-requested explanation.")
    assert [(m["role"], m["content"]) for m in session.messages] == [
        ("user", human_message), ("assistant", "A human-requested explanation."),
    ]
