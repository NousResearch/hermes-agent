"""plan-secretary tests: precise capture filter + confirm/reminder flow."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

core = importlib.import_module("plugins.plan-secretary.core")


POSITIVE = [
    "小墨接下来会检查 logs/plan_secretary_test.jsonl，并把误抓规则写进过滤器。",
    "小墨稍后启动 5 分钟 watcher 短测，验证 pending capture 是否只抓真实计划。",
    "小墨接下来会检查 scripts/build.sh，并写入测试结果。",
    "I\u2019ll check the config file next and fix the parser.",
]

NEGATIVE = [
    "/new 名称 可以让新会话后续更好恢复。",
    "下一步一般可以考虑把会话拆短。",
    "这个设计后续会更清晰。",
    "种子池下一步应该进入二阶增强。",
    "新会话开头只带 5 件事：小秘书目标、当前 watcher 文件、测试规则、已知问题、下一步验证清单。",
    "这个功能用于文档说明，后续可以看看。",
]


@pytest.mark.parametrize("sentence", POSITIVE)
def test_precise_capture_accepts_commitments(sentence: str) -> None:
    assert core.is_precise_capture(sentence), sentence


@pytest.mark.parametrize("sentence", NEGATIVE)
def test_precise_capture_rejects_noise(sentence: str) -> None:
    assert not core.is_precise_capture(sentence), sentence


def test_scan_text_only_creates_precise_pending(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(core, "state_dir", lambda: tmp_path)
    captures = core.scan_text(
        "\n".join(POSITIVE + NEGATIVE),
        source="test",
        source_id="test:1",
        source_session_id="session-A",
        source_message_id="42",
    )
    assert len(captures) == len(POSITIVE)
    assert all(c["status"] == "pending" for c in captures)
    assert all(c["source_session_id"] == "session-A" for c in captures)


def test_confirm_and_ignore_flow(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(core, "state_dir", lambda: tmp_path)
    captures = core.scan_text("小墨接下来会检查 logs/a.jsonl，并把结果写进汇报。",
                              source="test", source_id="t:1", source_session_id="S1", source_message_id="1")
    cid = captures[0]["id"]
    plan = core.confirm_capture(cid, due="10m", mode="parallel", owner="小墨", worker="agent")
    assert plan["status"] == "active"
    assert plan["source_session_id"] == "S1"
    assert core.parse_plan_time(plan["due"]) is not None
    msgs = core.notify(session_id="S1", state_path=tmp_path / "notify.json", repeat_pending=True)
    assert any("确认是否登记" in m for m in msgs)
    msgs_other = core.notify(session_id="S2", state_path=tmp_path / "notify2.json")
    assert msgs_other == []


def test_capture_session_isolation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(core, "state_dir", lambda: tmp_path)
    core.scan_text("小墨接下来会检查 logs/b.jsonl，并把结果写进汇报。",
                   source="test", source_id="t:1", source_session_id="S1", source_message_id="1")
    core.scan_text("小墨接下来会检查 logs/c.jsonl，并把结果写进汇报。",
                   source="test", source_id="t:2", source_session_id="S2", source_message_id="2")
    msgs1 = core.notify(session_id="S1", state_path=tmp_path / "n1.json", repeat_pending=True)
    msgs2 = core.notify(session_id="S2", state_path=tmp_path / "n2.json", repeat_pending=True)
    assert all("session: S1" in m for m in msgs1)
    assert all("session: S2" in m for m in msgs2)


def test_parse_dt_relative() -> None:
    base = core.now_local()
    assert (core.parse_dt("10m") - base).total_seconds() == pytest.approx(600, abs=5)
    assert (core.parse_dt("2h") - base).total_seconds() == pytest.approx(7200, abs=5)
