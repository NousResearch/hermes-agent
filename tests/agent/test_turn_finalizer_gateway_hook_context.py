"""Gateway ownership must survive normalization of local sessions to the CLI label.

The Gateway reports its local source as ``Platform.LOCAL``, while the agent's
surface label is normalized to ``cli``. Observer plugins need the separate
ownership signal to avoid competing with gateway event hooks.
"""

import logging
from types import SimpleNamespace

from agent import turn_finalizer


def _capture_post_hook(monkeypatch, agent):
    calls = []

    def invoke(name, logger, **kwargs):
        calls.append((name, kwargs))
        return []

    monkeypatch.setattr(turn_finalizer, "_invoke_hook_safely", invoke)
    turn_finalizer._apply_output_hooks(
        agent,
        "done",
        logging.getLogger("test.gateway-hook-context"),
        platform="cli",
        effective_task_id="task-1",
        turn_id="turn-1",
        original_user_message="hello",
        messages=[],
    )
    return next(kwargs for name, kwargs in calls if name == "post_llm_call")


def test_post_llm_call_marks_gateway_managed_local_session(monkeypatch):
    agent = SimpleNamespace(
        session_id="session-1",
        model="test/model",
        _persist_disabled=False,
        _llm_output_transform=("turn-1", False, None),
        _gateway_session_key="gateway-session-1",
    )

    event = _capture_post_hook(monkeypatch, agent)

    assert event["platform"] == "cli"
    assert event["is_gateway_session"] is True


def test_post_llm_call_marks_standalone_cli_session_as_non_gateway(monkeypatch):
    agent = SimpleNamespace(
        session_id="session-2",
        model="test/model",
        _persist_disabled=False,
        _llm_output_transform=("turn-1", False, None),
        _gateway_session_key=None,
    )

    event = _capture_post_hook(monkeypatch, agent)

    assert event["platform"] == "cli"
    assert event["is_gateway_session"] is False
