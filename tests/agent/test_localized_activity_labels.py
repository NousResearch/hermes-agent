"""Localized rendering of engine activity stamps.

``_touch_activity`` stores the engine's own English bookkeeping; the gateway heartbeat renders
that field straight into chat. These tests pin the read-side rendering contract: plain language
for zh, canonical English everywhere else, and never an empty string.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture()
def zh_home(tmp_path, monkeypatch):
    """A profile home whose display.language is zh."""
    (tmp_path / "config.yaml").write_text("display:\n  language: zh\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent import i18n

    i18n.reset_language_cache()
    yield tmp_path
    monkeypatch.delenv("HERMES_HOME", raising=False)
    i18n.reset_language_cache()


@pytest.fixture()
def en_home(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("display:\n  language: en\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent import i18n

    i18n.reset_language_cache()
    yield tmp_path
    monkeypatch.delenv("HERMES_HOME", raising=False)
    i18n.reset_language_cache()


@pytest.mark.parametrize("stamp,expected", [
    ("receiving stream response", "正在接收模型回复"),
    ("waiting for provider response (streaming)", "正在等待模型回复"),
    ("waiting for non-streaming API response", "正在等待模型回复"),
    ("starting API call #3", "正在发起第 3 次模型调用"),
    ("context compression started", "正在压缩上下文"),
    ("turn end logged", "本轮已结束"),
    ("initializing", "正在初始化"),
    ("executing tool: terminal", "正在运行命令"),
    ("executing tool: read_file", "正在读取文件"),
])
def test_zh_renders_plain_language(zh_home, stamp, expected):
    from agent.session_activity import localized_activity_label

    assert localized_activity_label(stamp) == expected


def test_unknown_tool_falls_back_to_raw_name_not_empty(zh_home):
    from agent.session_activity import localized_activity_label

    out = localized_activity_label("executing tool: some_new_tool_xyz")
    assert out == "正在运行：some_new_tool_xyz"


def test_unmatched_stamp_passes_through_unchanged(zh_home):
    """A new writer must degrade to the raw label, never to nothing."""
    from agent.session_activity import localized_activity_label

    assert localized_activity_label("brand new unknown stamp") == "brand new unknown stamp"


def test_empty_stamp_is_empty(zh_home):
    from agent.session_activity import localized_activity_label

    assert localized_activity_label("") == ""
    assert localized_activity_label(None) == ""


def test_non_chinese_locale_keeps_canonical_english(en_home):
    """Diagnostics and non-zh users must keep the canonical English stamp."""
    from agent.session_activity import localized_activity_label

    assert localized_activity_label("receiving stream response") == "receiving stream response"
    assert localized_activity_label("executing tool: terminal") == "executing tool: terminal"


def test_heartbeat_composes_localized_detail(zh_home):
    """The full heartbeat line reads as plain language, not engine jargon."""
    from agent.i18n import t
    from agent.session_activity import format_iteration_progress, localized_activity_label

    detail = (" — " + format_iteration_progress(27, 1000)
              + t("gateway.busy.ack_detail_joiner")
              + localized_activity_label("receiving stream response"))
    line = t("gateway.progress.working_heartbeat", minutes=6, detail=detail)
    assert "receiving stream response" not in line
    assert "正在接收模型回复" in line
    assert "第 27/1000 轮迭代" in line
