"""Behavioral tests for the single-card Feishu per-turn progress rail."""
import queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


def _ctx():
    ctx = TurnContext(source=SimpleNamespace(platform=Platform.FEISHU, chat_id="oc_test"))
    ctx.feishu_progress_card_enabled = True
    ctx.progress_queue = queue.Queue()
    ctx._run_still_current = lambda: True
    ctx._progress_metadata = None
    ctx._progress_reply_to = None
    return ctx


def test_card_keeps_current_operation_prominent_and_history_collapsed():
    card = TurnRunner._build_feishu_progress_card(
        "terminal — git status", ["✅ terminal — git status && git diff --stat", "❌ search_files — TODO"], 17)
    assert "Current operation" in card["elements"][0]["content"]
    assert "terminal — git status" in card["elements"][0]["content"]
    panel = card["elements"][1]
    assert panel["tag"] == "collapsible_panel"
    assert panel["expanded"] is False
    assert panel["header"]["title"]["tag"] == "plain_text"
    first, second = panel["elements"]
    assert first["tag"] == second["tag"] == "collapsible_panel"
    assert first["expanded"] is second["expanded"] is False
    assert first["header"]["title"]["content"] == "✅ terminal"
    assert first["elements"][0]["content"] == "git status && git diff --stat"
    assert second["header"]["title"]["content"] == "❌ search_files"
    assert second["elements"][0]["content"] == "TODO"


def test_progress_card_does_not_synthesize_completion_or_duplicate_history():
    card = TurnRunner._build_feishu_progress_card("terminal — git status", [], 1)
    assert len(card["elements"]) == 1
    card = TurnRunner._build_feishu_progress_card(None, ["❌ terminal"], 2)
    item = card["elements"][1]["elements"][0]
    assert item["header"]["title"]["content"] == "❌ terminal"
    assert item["elements"][0]["content"] == "❌ terminal"


def test_progress_card_final_outcomes_are_not_running():
    for outcome, title, template in (("success", "Hermes finished", "green"),
                                     ("failed", "Hermes failed", "red"),
                                     ("interrupted", "Hermes interrupted", "orange")):
        card = TurnRunner._build_feishu_progress_card(None, [], 4, outcome=outcome)
        assert card["header"]["title"]["content"] == title
        assert card["header"]["template"] == template
        assert "Current operation" not in card["elements"][0]["content"]


def test_progress_card_history_is_bounded():
    card = TurnRunner._build_feishu_progress_card(None, [f"op {i}" for i in range(100)], 10)
    entries = card["elements"][1]["elements"]
    assert len(entries) == 40
    assert entries[-1]["header"]["title"]["content"] == "op 99"
    assert entries[0]["header"]["title"]["content"] == "op 60"


def test_native_callbacks_preserve_full_command_in_started_event():
    ctx = _ctx()
    runner = TurnRunner(MagicMock(), ctx)
    command = "git status && git diff --stat"
    runner.native_tool_start_callback("call-1", "terminal", {"command": command})
    event = ctx.progress_queue.get_nowait()
    assert event["tool_call_id"] == "call-1"
    assert event["history_preview"] == command
    runner.native_tool_complete_callback("call-1", "terminal", {"command": command}, {"success": True})
    assert ctx.progress_queue.get_nowait()["tool_call_id"] == "call-1"


def test_progress_callback_preserves_log_mode_events_for_feishu():
    ctx = _ctx()
    ctx.log_queue = queue.Queue()
    runner = TurnRunner(MagicMock(), ctx)

    runner.progress_callback("tool.started", "terminal", "git status", {})

    assert ctx.log_queue.get_nowait().endswith('terminal: "git status"')
    assert ctx.progress_queue.empty()  # the ID-bearing callback alone drives the card


@pytest.mark.parametrize(
    ("display", "expected_mode", "expected_card"),
    [
        ({}, "new", True),
        ({"tool_progress": "log"}, "log", True),
        ({"tool_progress": "off"}, "off", False),
        ({"platforms": {"feishu": {"tool_progress": "off"}}}, "off", False),
        ({"tool_progress_overrides": {"feishu": "off"}}, "off", False),
    ],
)
def test_feishu_card_follows_operator_progress_intent(monkeypatch, display, expected_mode, expected_card):
    from gateway import run as gateway_run

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"display": display})
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {Platform.FEISHU: SimpleNamespace(platform=Platform.FEISHU)}
    runner.config = SimpleNamespace()
    runner._session_db = None
    source = SessionSource(platform=Platform.FEISHU, chat_id="oc_test", chat_type="dm", thread_id=None)

    settings = runner._run_agent_display_settings(source)

    assert settings.progress_mode == expected_mode
    assert settings.feishu_progress_card_enabled is expected_card
    if expected_mode == "log":
        assert settings.log_queue is not None
    if not expected_card:
        assert settings.needs_progress_queue is False
