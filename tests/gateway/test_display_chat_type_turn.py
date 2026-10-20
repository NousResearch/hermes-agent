"""Turn-level integration tests for chat-type display policy."""

import importlib

import hermes_yaml as yaml

from gateway.config import Platform
from gateway.platforms.base import SessionSource
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter, _make_runner


def test_chat_type_preview_and_labels_are_turn_local(monkeypatch, tmp_path):
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "display": {
            "tool_preview_length": 80,
            "friendly_tool_labels": True,
            "platforms": {"telegram": {"chat_types": {
                "group": {"tool_preview_length": 7, "friendly_tool_labels": False},
            }}},
        },
    }), encoding="utf-8")
    runner = _make_runner(ProgressCaptureAdapter(platform=Platform.TELEGRAM))
    group = SessionSource(platform=Platform.TELEGRAM, chat_id="G1", chat_type="group")
    dm = SessionSource(platform=Platform.TELEGRAM, chat_id="D1", chat_type="dm")

    group_display = runner._run_agent_display_settings(group)
    dm_display = runner._run_agent_display_settings(dm)

    assert (group_display.tool_preview_length, group_display.friendly_tool_labels) == (7, False)
    assert (dm_display.tool_preview_length, dm_display.friendly_tool_labels) == (80, True)
    assert (group_display.tool_preview_length, group_display.friendly_tool_labels) == (7, False)