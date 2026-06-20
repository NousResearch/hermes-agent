"""主本地授权策略(角色 → 资源 grant)CRUD —— 协同地基 Phase 2a。

grant = (role, node_uid, kind, resource_id):「角色被授权可用某节点上的某资源」。
"""

from __future__ import annotations

import pytest


@pytest.fixture
def reg(monkeypatch, tmp_path):
    monkeypatch.setenv("KARI_RESOURCES_DB", str(tmp_path / "kari_resources.sqlite"))
    from tools import kari_resources

    return kari_resources


def test_add_list_and_is_granted(reg):
    assert reg.add_grant("财务", "child-1", "knowledge", "kb_a") is True
    assert reg.add_grant("财务", "child-1", "knowledge", "kb_b") is True
    assert reg.add_grant("客服", "child-2", "knowledge", "kb_c") is True

    assert reg.is_granted("财务", "child-1", "knowledge", "kb_a") is True
    assert reg.is_granted("财务", "child-1", "knowledge", "kb_zzz") is False
    assert reg.is_granted("客服", "child-1", "knowledge", "kb_a") is False  # 别的角色不串

    fin = reg.list_grants(role="财务")
    assert {g["resource_id"] for g in fin} == {"kb_a", "kb_b"}
    assert {g["role"] for g in reg.list_grants()} == {"财务", "客服"}


def test_add_is_idempotent_keeps_created_ts(reg):
    assert reg.add_grant("财务", "child-1", "knowledge", "kb_a") is True
    ts1 = reg.list_grants(role="财务")[0]["created_ts"]
    assert reg.add_grant("财务", "child-1", "knowledge", "kb_a") is True  # 重复授权
    rows = reg.list_grants(role="财务")
    assert len(rows) == 1
    assert rows[0]["created_ts"] == ts1  # created_ts 不被覆盖


def test_invalid_grant_rejected(reg):
    assert reg.add_grant("", "child-1", "knowledge", "kb_a") is False  # 空 role
    assert reg.add_grant("财务", "", "knowledge", "kb_a") is False  # 空 node
    assert reg.add_grant("财务", "child-1", "bogus", "kb_a") is False  # 非法 kind
    assert reg.add_grant("财务", "child-1", "knowledge", "") is False  # 空 resource_id
    assert reg.list_grants() == []


def test_remove_grant_and_role_cleanup(reg):
    reg.add_grant("财务", "child-1", "knowledge", "kb_a")
    reg.add_grant("财务", "child-1", "knowledge", "kb_b")
    reg.add_grant("客服", "child-2", "knowledge", "kb_c")

    assert reg.remove_grant("财务", "child-1", "knowledge", "kb_a") is True
    assert reg.remove_grant("财务", "child-1", "knowledge", "kb_a") is False  # 已不在
    assert {g["resource_id"] for g in reg.list_grants(role="财务")} == {"kb_b"}

    # 删角色 → 清其全部授权,不动别的角色。
    assert reg.remove_role_grants("财务") == 1
    assert reg.list_grants(role="财务") == []
    assert {g["resource_id"] for g in reg.list_grants(role="客服")} == {"kb_c"}


def test_grant_independent_of_resource_table(reg):
    # 授权不要求资源此刻在注册表里(子可能离线/未上报),也不被资源删除级联清掉。
    reg.add_grant("财务", "child-1", "knowledge", "ghost_kb")
    assert reg.is_granted("财务", "child-1", "knowledge", "ghost_kb") is True
    assert reg.list_resources(node_uid="child-1") == []  # 注册表里根本没有这条资源


def test_list_authorized_resources_excludes_stale_grants(reg):
    """解析「角色→实际可用资源」= grant ∩ 现存资源,带 name/meta;失效授权被排除(Phase 3a)。"""
    reg.replace_node_resources(
        "child-1",
        "knowledge",
        [
            {"resource_id": "kb_a", "name": "财务KB", "meta": {"chunks": 9}},
            {"resource_id": "kb_b", "name": "客服KB"},
        ],
    )
    reg.add_grant("财务", "child-1", "knowledge", "kb_a")  # 资源存在 → 可用
    reg.add_grant("财务", "child-1", "knowledge", "ghost")  # 失效:授权在但资源不在注册表
    reg.add_grant("客服", "child-1", "knowledge", "kb_b")  # 别的角色

    fin = reg.list_authorized_resources("财务")
    assert [r["resource_id"] for r in fin] == ["kb_a"]  # ghost 被排除
    assert fin[0]["name"] == "财务KB"
    assert fin[0]["meta"] == {"chunks": 9}  # 带上注册表里的 name/meta(非裸 grant)

    assert reg.list_authorized_resources("客服")[0]["resource_id"] == "kb_b"
    assert reg.list_authorized_resources("不存在的角色") == []
    assert reg.list_authorized_resources("") == []


def test_list_authorized_resources_kind_filter(reg):
    reg.replace_node_resources("child-1", "knowledge", [{"resource_id": "kb_a", "name": "KB"}])
    reg.replace_node_resources("child-1", "workflow", [{"resource_id": "wf_a", "name": "对话流"}])
    reg.add_grant("研发", "child-1", "knowledge", "kb_a")
    reg.add_grant("研发", "child-1", "workflow", "wf_a")

    assert {r["resource_id"] for r in reg.list_authorized_resources("研发")} == {"kb_a", "wf_a"}
    assert [r["resource_id"] for r in reg.list_authorized_resources("研发", kind="workflow")] == ["wf_a"]
