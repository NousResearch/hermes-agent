"""Kari 画布 MCP(stdio)—— 爱马仕(EasyHermes)的 agent 经此操作本地 Langflow 画布。

每个工具把一次画布操作 POST 到本机 langflow 的 ``/kari/canvas/op``,由 langflow 经 WS 桥
``/kari/bridge`` 转发到浏览器(canvasBridgeClient 执行,画布是真相源),回结果。

爱马仕在 ``mcp_servers`` 里以 command/args 方式 spawn 本脚本(stdio transport)。
画布地址用 env ``KARI_CANVAS_OP_URL`` 覆盖(默认本机 langflow 7860)。
"""

from __future__ import annotations

import os

import httpx
from mcp.server.fastmcp import FastMCP

OP_URL = os.environ.get("KARI_CANVAS_OP_URL", "http://127.0.0.1:7860/kari/canvas/op")

mcp = FastMCP("kari_canvas")


async def _op(op: str, args: dict) -> dict:
    async with httpx.AsyncClient(timeout=90.0) as client:
        r = await client.post(OP_URL, json={"op": op, "args": args})
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(data.get("error") or f"{op} 失败")
    return data.get("result")


@mcp.tool()
async def list_components(query: str = "") -> dict:
    """搜索可用的 Langflow 组件,返回 componentType、输入字段(含可连接的 input 字段名)与输出字段名。
    add_node / connect 前先用它查准 componentType 和字段名,别猜。"""
    return await _op("list_components", {"query": query})


@mcp.tool()
async def add_node(componentType: str, params: dict | None = None, position: dict | None = None) -> dict:  # noqa: N803
    """在画布添加一个组件节点。componentType 必须来自 list_components;params 预设字段值;position 可省(自动居中)。"""
    args: dict = {"componentType": componentType, "params": params or {}}
    if position:
        args["position"] = position
    return await _op("add_node", args)


@mcp.tool()
async def add_python_node(code: str, position: dict | None = None) -> dict:
    """添加一个自定义 Python 组件节点(完整的 Langflow Component 子类源码,服务端校验)。"""
    args: dict = {"code": code}
    if position:
        args["position"] = position
    return await _op("add_python_node", args)


@mcp.tool()
async def connect(source: str, sourceOutput: str, target: str, targetInput: str) -> dict:  # noqa: N803
    """连接两个节点。sourceOutput/targetInput 必须是 list_components 给出的准确字段名。"""
    return await _op(
        "connect", {"source": source, "sourceOutput": sourceOutput, "target": target, "targetInput": targetInput}
    )


@mcp.tool()
async def update_node(nodeId: str, params: dict) -> dict:  # noqa: N803
    """更新某节点的字段值。"""
    return await _op("update_node", {"nodeId": nodeId, "params": params})


@mcp.tool()
async def delete_node(nodeId: str) -> dict:  # noqa: N803
    """删除某节点。"""
    return await _op("delete_node", {"nodeId": nodeId})


@mcp.tool()
async def get_flow() -> dict:
    """读取当前画布:节点(含已设字段值)+ 连线。规划/连线前先看一眼。"""
    return await _op("get_flow", {})


if __name__ == "__main__":
    mcp.run()  # stdio transport
