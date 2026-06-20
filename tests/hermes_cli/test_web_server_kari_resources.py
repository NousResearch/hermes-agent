"""Dashboard 读主本地资源注册表端点(GET /api/kari/resources)—— 协同地基 Phase 1d。

团队账号 UI 据此展示「每个子有哪些知识库」。数据源是**主本地** kari_resources.sqlite
(子上报、主拉取存档的 authoritative 副本),所以端点直接读本地、按 node_uid 分组。
"""

import pytest


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_empty_registry_returns_empty(client):
    resp = client.get("/api/kari/resources")
    assert resp.status_code == 200
    data = resp.json()
    assert data["resources"] == [] and data["by_node"] == {}
    # 端点带「本节点是否有 langflow 能力」标志(权限面板据此门控配 MCP);值随环境,只断类型。
    assert isinstance(data["langflow_capable"], bool)


def test_lists_and_groups_by_node(client):
    from tools import kari_resources

    kari_resources.replace_node_resources(
        "child-1",
        "knowledge",
        [
            {"resource_id": "kb_a", "name": "财务KB", "meta": {"chunks": 12}},
            {"resource_id": "kb_b", "name": "客服KB", "meta": {"chunks": 3}},
        ],
    )
    kari_resources.replace_node_resources(
        "child-2", "knowledge", [{"resource_id": "kb_c", "name": "研发KB"}]
    )

    body = client.get("/api/kari/resources").json()

    assert {r["resource_id"] for r in body["resources"]} == {"kb_a", "kb_b", "kb_c"}
    assert set(body["by_node"]) == {"child-1", "child-2"}
    assert {r["resource_id"] for r in body["by_node"]["child-1"]} == {"kb_a", "kb_b"}
    assert body["by_node"]["child-2"][0]["name"] == "研发KB"
    # meta 透传给 UI 做悬停摘要。
    kb_a = next(r for r in body["by_node"]["child-1"] if r["resource_id"] == "kb_a")
    assert kb_a["meta"] == {"chunks": 12}


def test_kind_filter(client):
    from tools import kari_resources

    kari_resources.replace_node_resources("child-1", "knowledge", [{"resource_id": "kb_a", "name": "KB"}])
    kari_resources.replace_node_resources("child-1", "workflow", [{"resource_id": "wf_a", "name": "对话流"}])

    body = client.get("/api/kari/resources", params={"kind": "workflow"}).json()

    assert {r["resource_id"] for r in body["resources"]} == {"wf_a"}
    assert set(body["by_node"]) == {"child-1"}


# --------------------------- 授权策略端点(Phase 2a)---------------------------
def _grant(role="财务", node_uid="child-1", kind="knowledge", resource_id="kb_a"):
    return {"role": role, "node_uid": node_uid, "kind": kind, "resource_id": resource_id}


def test_grants_empty(client):
    assert client.get("/api/kari/grants").json() == {"grants": []}


def test_grant_add_list_delete_roundtrip(client):
    assert client.post("/api/kari/grants", json=_grant()).json() == {"ok": True}
    assert client.post("/api/kari/grants", json=_grant(resource_id="kb_b")).json() == {"ok": True}

    grants = client.get("/api/kari/grants", params={"role": "财务"}).json()["grants"]
    assert {g["resource_id"] for g in grants} == {"kb_a", "kb_b"}

    deleted = client.post("/api/kari/grants/delete", json=_grant()).json()
    assert deleted == {"ok": True, "removed": True}
    remaining = client.get("/api/kari/grants", params={"role": "财务"}).json()["grants"]
    assert {g["resource_id"] for g in remaining} == {"kb_b"}
    # 撤销不存在的授权不报错,removed=False。
    assert client.post("/api/kari/grants/delete", json=_grant()).json() == {"ok": True, "removed": False}


def test_grant_add_invalid_rejected(client):
    resp = client.post("/api/kari/grants", json=_grant(kind="bogus"))
    assert resp.status_code == 400
    assert client.get("/api/kari/grants").json() == {"grants": []}


def test_grant_clear_role(client):
    client.post("/api/kari/grants", json=_grant(role="财务", resource_id="kb_a"))
    client.post("/api/kari/grants", json=_grant(role="财务", resource_id="kb_b"))
    client.post("/api/kari/grants", json=_grant(role="客服", resource_id="kb_c"))

    cleared = client.post("/api/kari/grants/clear-role", json={"role": "财务"}).json()
    assert cleared == {"ok": True, "cleared": 2}
    # 只清「财务」,客服不动 → 防同名新角色继承旧授权。
    assert client.get("/api/kari/grants", params={"role": "财务"}).json() == {"grants": []}
    assert {g["resource_id"] for g in client.get("/api/kari/grants", params={"role": "客服"}).json()["grants"]} == {"kb_c"}


def test_authorized_resolves_grant_intersect_registry(client):
    from tools import kari_resources

    kari_resources.replace_node_resources(
        "child-1", "knowledge", [{"resource_id": "kb_a", "name": "财务KB", "meta": {"chunks": 9}}]
    )
    client.post("/api/kari/grants", json=_grant(role="财务", resource_id="kb_a"))
    client.post("/api/kari/grants", json=_grant(role="财务", resource_id="ghost"))  # 失效授权

    body = client.get("/api/kari/authorized", params={"role": "财务"}).json()
    assert body["role"] == "财务"
    assert [r["resource_id"] for r in body["resources"]] == ["kb_a"]  # ghost 被排除
    assert body["resources"][0]["meta"] == {"chunks": 9}
    # 没授权的角色 → 空。
    assert client.get("/api/kari/authorized", params={"role": "客服"}).json() == {"role": "客服", "resources": []}
