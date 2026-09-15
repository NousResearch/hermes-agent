"""Title-only input must not change history, eligibility, or fallback behavior."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.turn_context import _maybe_title_session_at_turn_start


@pytest.mark.parametrize("override, content, expected", [
    (None, "Ordinary request", "Ordinary request"),
    ("Actual question", "Injected model context", "Actual question"),
    ("  Actual question  ", "Injected model context", "Actual question"),
    ("", "Injected model context", None),
    ("   ", "Injected model context", None),
    (None, [{"type": "text", "text": "Image caption"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}], "Image caption"),
    (None, [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}], None),
])
def test_title_input_override_is_optional_and_history_is_unchanged(monkeypatch, override, content, expected):
    agent = SimpleNamespace(_session_db=Mock(), session_id="test", _session_db_created=True)
    messages = [{"role": "user", "content": content}]
    original = deepcopy(messages)
    title = Mock()
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", title)

    _maybe_title_session_at_turn_start(agent, messages, override)

    assert messages == original
    if expected is None:
        title.assert_not_called()
    else:
        assert title.call_args.args[2] == expected
        assert title.call_args.kwargs["conversation_history"] is messages


@pytest.mark.parametrize("text, titleable", [("/help", True), ("[CONTEXT COMPACTION] synthetic handoff", False)])
def test_title_override_preserves_existing_control_message_eligibility(monkeypatch, text, titleable):
    agent = SimpleNamespace(_session_db=Mock(), session_id="test", _session_db_created=True)
    start = Mock()
    monkeypatch.setattr("agent.title_generator.threading", SimpleNamespace(Thread=start))

    _maybe_title_session_at_turn_start(agent, [{"role": "user", "content": "injected context"}], text)

    assert start.called is titleable


@pytest.mark.parametrize("platform", ["cron", "subagent", "CRON"])
def test_title_override_cannot_enable_titling_for_machine_runs(monkeypatch, platform):
    agent = SimpleNamespace(_session_db=Mock(), session_id="test", platform=platform)
    title = Mock()
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", title)

    _maybe_title_session_at_turn_start(agent, [], "Actual question")

    title.assert_not_called()
