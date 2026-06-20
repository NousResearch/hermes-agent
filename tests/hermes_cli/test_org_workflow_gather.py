"""子端采集本机工作流 → 资源目录条目(协同地基 Phase 4)。

**硬约定**:只有「ChatInput + ChatOutput」一进一出对话流才进可注册工作流池。
只测纯映射逻辑(_is_chat_flow / _workflow_items_from_flows),不打 langflow HTTP。
"""

from __future__ import annotations

from hermes_cli import org_client as oc


def _node(type_=None, node_id=None):
    n = {}
    if type_ is not None:
        n["data"] = {"type": type_}
    if node_id is not None:
        n["id"] = node_id
    return n


def _flow(fid="f1", name="对话流", nodes=None, **extra):
    return {"id": fid, "name": name, "data": {"nodes": nodes or []}, **extra}


def test_workflow_items_filters_and_maps():
    flows = [
        _flow(fid="chat-1", name="客服问答", nodes=[_node("ChatInput"), _node("ChatOutput")],
              mcp_enabled=True, action_name="kefu_qa"),
        _flow(fid="batch-1", name="批处理", nodes=[_node("TextInput"), _node("TextOutput")]),  # 非对话→排除
        _flow(fid="", name="无id", nodes=[_node("ChatInput"), _node("ChatOutput")]),  # 无 id→排除
    ]

    items = oc._workflow_items_from_flows(flows)

    assert [it["resource_id"] for it in items] == ["chat-1"]
    only = items[0]
    assert only["name"] == "客服问答"
    assert only["meta"] == {"mcp_enabled": True, "action_name": "kefu_qa"}


def test_workflow_items_none_when_not_a_list():
    # langflow 不可达/读失败上游会给 None 或非列表 → 返回 None(跳过,绝不上报 [] 误清缓冲)。
    assert oc._workflow_items_from_flows(None) is None
    assert oc._workflow_items_from_flows({"oops": 1}) is None
    assert oc._workflow_items_from_flows([]) == []


def test_workflow_items_skip_unregistered():
    # 没 mcp_enabled(还没注册成 MCP 工具)的对话流不进可授权池 —— 也天然排除 langflow 只读 starter 示例。
    items = oc._workflow_items_from_flows([_flow(fid="c1", nodes=[_node("ChatInput"), _node("ChatOutput")])])
    assert items == []
