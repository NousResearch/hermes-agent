"""自有资源也登记本地注册表(协同地基 Phase 4)——主账号能授权自己的 KB/工作流,不只下级的。

self_user_id():用 org 同一份 workspace token 调 /account/me 解析自身 uid 并缓存。
_store_own_resources():把自己 gather 的资源整体登进本地注册表(node_uid=自身 uid)。
"""

from __future__ import annotations

import pytest

from hermes_cli import org_client as oc


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    monkeypatch.setenv("KARI_RESOURCES_DB", str(tmp_path / "kari_resources.sqlite"))
    monkeypatch.setattr(oc, "_cloud", lambda: ("http://hub.test", "tok-1"))
    monkeypatch.setattr(oc, "_self_uid_cache", {})  # 隔离模块级缓存
    from tools import kari_resources

    return kari_resources


def test_self_user_id_resolves_and_caches(fresh, monkeypatch):
    calls = {"n": 0}

    def fake_call(method, path, token, body=None, timeout=30.0):
        calls["n"] += 1
        assert path == "/account/me"
        return 200, {"user_id": "root-9", "email": "a@b.com"}

    monkeypatch.setattr(oc, "_call", fake_call)

    assert oc.self_user_id() == "root-9"
    assert oc.self_user_id() == "root-9"  # 第二次走缓存
    assert calls["n"] == 1  # 只打了一次 /account/me


def test_self_user_id_none_when_not_logged_in(monkeypatch):
    monkeypatch.setattr(oc, "_cloud", lambda: (None, None))
    monkeypatch.setattr(oc, "_self_uid_cache", {})
    assert oc.self_user_id() is None


def test_self_user_id_none_on_http_failure(fresh, monkeypatch):
    monkeypatch.setattr(oc, "_call", lambda *a, **k: (401, {"error": "bad token"}))
    assert oc.self_user_id() is None


def test_store_own_resources_registers_under_self_uid(fresh, monkeypatch):
    kari_resources = fresh
    monkeypatch.setattr(oc, "self_user_id", lambda: "root-9")

    oc._store_own_resources(
        "workflow",
        [{"resource_id": "flow-1", "name": "客服问答", "meta": {"mcp_enabled": True}}],
    )

    rows = kari_resources.list_resources(kind="workflow", node_uid="root-9")
    assert [r["resource_id"] for r in rows] == ["flow-1"]
    assert rows[0]["meta"] == {"mcp_enabled": True}


def test_store_own_resources_skips_on_none_or_no_uid(fresh, monkeypatch):
    kari_resources = fresh
    monkeypatch.setattr(oc, "self_user_id", lambda: "root-9")

    oc._store_own_resources("knowledge", None)  # 读失败 None → 不动注册表
    assert kari_resources.list_resources() == []

    monkeypatch.setattr(oc, "self_user_id", lambda: None)  # 拿不到自身 uid → 跳过
    oc._store_own_resources("knowledge", [{"resource_id": "kb_a", "name": "KB"}])
    assert kari_resources.list_resources() == []


def test_own_and_pulled_resources_coexist(fresh, monkeypatch):
    """自有(直接登)与下级(经 pull 存)在同一注册表里按 node_uid 并存,不互相覆盖。"""
    kari_resources = fresh
    monkeypatch.setattr(oc, "self_user_id", lambda: "root-9")
    # 自有
    oc._store_own_resources("knowledge", [{"resource_id": "own_kb", "name": "主KB"}])
    # 下级(模拟 1d pull 存)
    kari_resources.replace_node_resources("child-1", "knowledge", [{"resource_id": "sub_kb", "name": "子KB"}])

    all_kb = kari_resources.list_resources(kind="knowledge")
    assert {(r["node_uid"], r["resource_id"]) for r in all_kb} == {("root-9", "own_kb"), ("child-1", "sub_kb")}
