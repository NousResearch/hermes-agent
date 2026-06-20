"""本节点 agent 作为可授权/可委派资源(协同地基 Phase 4 · ②面3 委派任务)。"""

from __future__ import annotations

from hermes_cli import org_client as oc


def test_gather_agent_resource_from_capabilities(monkeypatch):
    monkeypatch.setattr(
        oc,
        "gather_capabilities",
        lambda: {"host": "mac-1", "model": "极致", "summary": "财务助手", "toolsets": ["bash", "read"]},
    )

    items = oc.gather_agent_resources()

    assert len(items) == 1
    a = items[0]
    assert a["resource_id"] == "agent"  # 每节点恰好一个 agent
    assert a["name"] == "财务助手"  # 优先用能力简介当展示名
    assert a["meta"] == {"summary": "财务助手", "model": "极致", "host": "mac-1", "toolsets": ["bash", "read"]}


def test_gather_agent_resource_falls_back_to_host_name(monkeypatch):
    # 没简介时退回 host 当名字;空字段不塞进 meta。
    monkeypatch.setattr(oc, "gather_capabilities", lambda: {"host": "mac-1", "summary": "", "model": None})

    a = oc.gather_agent_resources()[0]

    assert a["name"] == "mac-1"
    assert a["meta"] == {"host": "mac-1"}  # summary 空 / model None 不进 meta


def test_gather_agent_resource_always_present(monkeypatch):
    # agent 不依赖 langflow,任何情况都返回一条(总会登记+上报,绝不 None)。
    monkeypatch.setattr(oc, "gather_capabilities", lambda: {})
    items = oc.gather_agent_resources()
    assert len(items) == 1
    assert items[0]["resource_id"] == "agent"
    assert items[0]["name"] == "本地爱马仕"


def test_report_agent_stores_own_and_reports(monkeypatch):
    monkeypatch.setattr(oc, "_cloud", lambda: ("http://hub.test", "tok"))
    monkeypatch.setattr(oc, "gather_capabilities", lambda: {"host": "h", "summary": "s"})
    stored: list = []
    reported: list = []
    monkeypatch.setattr(oc, "_store_own_resources", lambda kind, items: stored.append((kind, items)))
    monkeypatch.setattr(oc, "report_resources", lambda kind, items: reported.append((kind, items)) or (True, {"ok": True}))

    ok, _ = oc.report_agent_resources()

    assert ok is True
    assert stored and stored[0][0] == "agent"  # 自有也登本地注册表
    assert reported and reported[0][0] == "agent"  # 同时上报云端
    assert stored[0][1] == reported[0][1]  # 同一批 items
