"""主端拉取下级资源上报 → 存主本地注册表 → ack 清云缓冲(协同地基 Phase 1d)。

只测 org_client.pull_and_store_resource_reports 的编排正确性:
  - 合法上报存进本地、并按拉到的 updated_ts 水位线 ack;
  - 报文异常(缺 user_id/kind、items 不是 list)跳过、且**不 ack**(下轮重试);
  - 子重报走整体替换(删掉已不存在的、更新留下的),不残留旧条目。
云端传输层用打桩替身,不依赖真实 hub。
"""

from __future__ import annotations

import pytest

from hermes_cli import org_client
from tools import kari_resources


@pytest.fixture
def isolated_registry(monkeypatch, tmp_path):
    """把主本地注册表指到临时 sqlite,并打桩云端凭据 + ack(记录调用)。"""
    monkeypatch.setenv("KARI_RESOURCES_DB", str(tmp_path / "kari_resources.sqlite"))
    monkeypatch.setattr(org_client, "_cloud", lambda: ("http://hub.test", "tok"))
    acks: list[tuple[str, str | None, float | None]] = []
    monkeypatch.setattr(
        org_client,
        "ack_resource_reports",
        lambda uid, kind=None, up_to_ts=None: (acks.append((uid, kind, up_to_ts)), 1)[1],
    )
    return acks


def _stub_pull(monkeypatch, reports):
    monkeypatch.setattr(org_client, "pull_resource_reports", lambda since=None: reports)


def test_pull_stores_valid_and_acks_to_watermark(isolated_registry, monkeypatch):
    acks = isolated_registry
    _stub_pull(
        monkeypatch,
        [
            {
                "user_id": "child-1",
                "kind": "knowledge",
                "items": [
                    {"resource_id": "kb_a", "name": "财务KB", "meta": {"chunks": 12}},
                    {"resource_id": "kb_b", "name": "客服KB", "meta": {"chunks": 3}},
                ],
                "updated_ts": 111.0,
            },
            {"user_id": "child-2", "kind": "knowledge", "items": [], "updated_ts": 222.0},
        ],
    )

    summary = org_client.pull_and_store_resource_reports()

    assert summary == {"pulled": 2, "stored_nodes": 2, "items": 2}
    # 各节点都按拉到的 updated_ts ack(水位线防 lost-update)。
    assert acks == [("child-1", "knowledge", 111.0), ("child-2", "knowledge", 222.0)]
    rows = kari_resources.list_resources(kind="knowledge", node_uid="child-1")
    assert {r["resource_id"] for r in rows} == {"kb_a", "kb_b"}
    assert kari_resources.list_resources(kind="knowledge", node_uid="child-2") == []


def test_pull_skips_malformed_reports_without_ack(isolated_registry, monkeypatch):
    acks = isolated_registry
    _stub_pull(
        monkeypatch,
        [
            {"user_id": "", "kind": "knowledge", "items": [{"resource_id": "x"}], "updated_ts": 333.0},
            {"user_id": "child-9", "kind": "knowledge", "items": None, "updated_ts": 444.0},
            {"user_id": "child-9", "kind": "", "items": [], "updated_ts": 555.0},
        ],
    )

    summary = org_client.pull_and_store_resource_reports()

    assert summary == {"pulled": 3, "stored_nodes": 0, "items": 0}
    # 异常报文绝不 ack —— 否则会丢掉云缓冲里这份(没存进本地却清了云)。
    assert acks == []
    assert kari_resources.list_resources() == []


def test_pull_skips_unknown_kind_without_ack(isolated_registry, monkeypatch):
    """云端允许但本地不认的 kind:绝不 ack(否则没存却清云=丢数据),也不存。"""
    acks = isolated_registry
    _stub_pull(
        monkeypatch,
        [{"user_id": "child-1", "kind": "secret-vault", "items": [{"resource_id": "z"}], "updated_ts": 9.0}],
    )

    summary = org_client.pull_and_store_resource_reports()

    assert summary == {"pulled": 1, "stored_nodes": 0, "items": 0}
    assert acks == []
    assert kari_resources.list_resources() == []


def test_re_report_replaces_node_resources(isolated_registry, monkeypatch):
    _stub_pull(
        monkeypatch,
        [
            {
                "user_id": "child-1",
                "kind": "knowledge",
                "items": [
                    {"resource_id": "kb_a", "name": "财务KB"},
                    {"resource_id": "kb_b", "name": "客服KB"},
                ],
                "updated_ts": 10.0,
            }
        ],
    )
    org_client.pull_and_store_resource_reports()

    # 子重报:kb_a 删除、kb_b 改名 → 主端整体替换跟手,不残留 kb_a。
    _stub_pull(
        monkeypatch,
        [
            {
                "user_id": "child-1",
                "kind": "knowledge",
                "items": [{"resource_id": "kb_b", "name": "客服KB v2"}],
                "updated_ts": 20.0,
            }
        ],
    )
    org_client.pull_and_store_resource_reports()

    rows = kari_resources.list_resources(kind="knowledge", node_uid="child-1")
    assert {(r["resource_id"], r["name"]) for r in rows} == {("kb_b", "客服KB v2")}


def test_no_cloud_creds_is_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("KARI_RESOURCES_DB", str(tmp_path / "kari_resources.sqlite"))
    monkeypatch.setattr(org_client, "_cloud", lambda: (None, None))
    # 未登录直接空转,不抛、不拉、不存。
    assert org_client.pull_and_store_resource_reports() == {"pulled": 0, "stored_nodes": 0, "items": 0}
    assert kari_resources.list_resources() == []
