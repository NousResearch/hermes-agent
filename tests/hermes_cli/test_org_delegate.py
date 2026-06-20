"""定向委派任务给某个下级 agent(协同地基 Phase 4 · ②面3)。复用 ask/collect 中继,这里只验编排。"""

from __future__ import annotations

import pytest

from hermes_cli import org_client as oc


@pytest.fixture
def relay(monkeypatch):
    """打桩 ask/collect/subtree_capabilities,记录调用。"""
    state = {"ask_args": None, "asked": [101]}
    monkeypatch.setattr(oc, "subtree_capabilities", lambda: [{"user_id": "child-1", "name": "子A", "email": "c@b"}])

    def fake_ask(query, targets=None):
        state["ask_args"] = (query, targets)
        return state["asked"]

    monkeypatch.setattr(oc, "ask", fake_ask)
    return state


def test_delegate_targets_single_subordinate(relay, monkeypatch):
    monkeypatch.setattr(
        oc, "collect",
        lambda ids: {"answers": [{"target_id": "child-1", "status": "done", "result": {"answer": "做完了"}}], "all_done": True},
    )

    out = oc.delegate_to("child-1", "整理这周财务", wait_seconds=1.0)

    assert relay["ask_args"] == ("整理这周财务", ["child-1"])  # 定向到单个目标
    assert out["target"] == "子A"  # 用能力简介里的展示名
    assert out["status"] == "done"
    assert out["answer"] == "做完了"
    assert out["all_done"] is True


def test_delegate_no_target_when_ask_empty(relay, monkeypatch):
    relay["asked"] = []  # 目标不可达/非下级 → ask 返回空
    monkeypatch.setattr(oc, "collect", lambda ids: {"answers": [], "all_done": True})

    out = oc.delegate_to("ghost", "x")

    assert out["status"] == "no-target"
    assert out["answer"] is None


def test_delegate_bad_request_skips_ask(monkeypatch):
    called = {"ask": False}
    monkeypatch.setattr(oc, "subtree_capabilities", lambda: [])
    monkeypatch.setattr(oc, "ask", lambda *a, **k: called.__setitem__("ask", True) or [1])

    assert oc.delegate_to("", "task")["status"] == "bad-request"  # 空 target
    assert oc.delegate_to("child-1", "  ")["status"] == "bad-request"  # 空 task
    assert called["ask"] is False  # 非法入参根本不打中继


def test_delegate_pending_when_not_done(relay, monkeypatch):
    # 超时未答完:all_done=False、无 answers → status 'pending'。模拟时钟避免真 sleep。
    monkeypatch.setattr(oc.time, "sleep", lambda *_a: None)
    times = [1000.0, 1000.5, 1002.0]  # deadline=1001;第二次检查已过点 → 退出
    monkeypatch.setattr(oc.time, "time", lambda: times.pop(0) if times else 1002.0)
    monkeypatch.setattr(oc, "collect", lambda ids: {"answers": [], "all_done": False})

    out = oc.delegate_to("child-1", "慢活", wait_seconds=1.0, poll_interval=0.5)

    assert out["all_done"] is False
    assert out["status"] == "pending"
