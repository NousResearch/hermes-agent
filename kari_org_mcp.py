"""Kari 组织协同 MCP(stdio)—— 主账号爱马仕的 agent 经此询问下级子爱马仕并收齐回报。

两个工具:
  - ``list_subordinate_capabilities()`` 看子树内各(子)爱马仕上报到管理端的能力(先看谁能干什么)。
  - ``query_subordinates(query)``        把问题扇出给下级 → 各自用本地知识库作答后收齐回报。

主账号 agent 拿到回报后,结合**自身知识库**把多方汇报归纳成给用户的答案
(即 "问各个子爱马仕,让他们汇报上来,再总结")。

云端地址 + token 复用 ``~/.hermes/workflow-secrets.json``(或 env ``KARI_HUB_URL`` /
``KARI_WORKSPACE_TOKEN``)。爱马仕在 ``mcp_servers`` 里以 hermes venv python spawn 本脚本(stdio)。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

# 让 `import hermes_cli.org_client` 可用(脚本在仓库根)。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from hermes_cli import org_client  # noqa: E402
from tools import knowledge_tool as _kt  # noqa: E402 — 复用知识源管理逻辑(同一份 JSON + langflow)

mcp = FastMCP("kari_org")


@mcp.tool()
async def list_subordinate_capabilities() -> list:
    """列出本账号子树内各(子)爱马仕上报的能力(user_id / name / summary / toolsets / mcp_servers 等)。
    在 query_subordinates 之前先看一眼:哪些下级跟这个问题相关、值得问。"""
    return await asyncio.to_thread(org_client.subtree_capabilities)


@mcp.tool()
async def query_subordinates(query: str, wait_seconds: float = 30.0) -> dict:
    """把一个问题分发给本账号所有下级子爱马仕,等它们各自用本地知识库作答后收齐回报。

    返回 {answers:[{subordinate, answer, summary, status}], asked, all_done, pending}。
    随后请你结合**主账号自身知识库**,把这些下级回报归纳、去重、总结成给用户的最终答案
    (注明哪条来自哪个子爱马仕)。all_done=false 表示还有下级未答完(可提示用户稍后再问)。"""
    return await asyncio.to_thread(org_client.query_subordinates, query, wait_seconds)


@mcp.tool()
async def delegate_task(target_user_id: str, task: str, wait_seconds: float = 60.0) -> dict:
    """把一个任务**定向委派给某一个下级子爱马仕**(②面3 委派任务),等它作答后收回。

    与 query_subordinates(问全部)不同:这里只派给一个目标。先用 list_subordinate_capabilities
    看有哪些下级及其能力简介,挑一个 user_id 传进来。返回 {target, status, answer, all_done};
    status='no-target' 表示目标不可达或非本账号下级。"""
    return await asyncio.to_thread(org_client.delegate_to, target_user_id, task, wait_seconds)


# --------------------------- 本机知识源管理(让会话自己也能维护知识库)---------------------------
# 操作的是和桌面端「知识库」页**同一份状态**(knowledge_sources.json + 本机 langflow KB):
# UI 改了会话看得到、会话改了 UI 刷新也看得到;改完即时刷新协同注册表(团队面板秒同步)。
@mcp.tool()
async def list_knowledge_sources() -> dict:
    """列出本机知识库里已加入的知识源(文件夹/文件):名称、类型、路径、已索引数、上次同步时间。"""
    return await asyncio.to_thread(lambda: json.loads(_kt._handle_list({})))  # noqa: SLF001


@mcp.tool()
async def sync_knowledge_source(source: str) -> dict:
    """同步一个知识源:重新扫描它的文件夹/文件,把新增/改动同步进知识库(纯新增=追加,改/删=重建)。
    source 传知识源的名字或路径(先用 list_knowledge_sources 看)。员工更新了文件夹里的文件后用它保持最新。"""
    return await asyncio.to_thread(lambda: json.loads(_kt._handle_sync({"source": source})))  # noqa: SLF001


@mcp.tool()
async def remove_knowledge_source(source: str) -> dict:
    """从本机知识库移除一个知识源(同时删掉它对应的知识库)。source 传名字或路径。"""
    return await asyncio.to_thread(lambda: json.loads(_kt._handle_remove({"source": source})))  # noqa: SLF001


@mcp.tool()
async def search_local_knowledge(query: str, source: str = "") -> dict:
    """检索**本机**知识源:grep 本地登记的知识源文件,返回关键词命中的 {file, line, snippet}(+文件名命中)。
    严格 **local-only**,不跨节点。给「个人 / 日常 / 探索」类问题用——拿到命中后用 Read 打开 file 看完整内容、
    可换关键词再搜。source 选填(传知识源名字/路径只搜那一个;不传搜全部)。

    ⚠️ 「部门 / 受治理 / 需权威或最新」的数据**别用这个**:本工具只看本机、不保证拿到部门最新数据,
    那类问题走受授权的查询工作流(query_authorized_knowledge,按角色授权 + 审计)。"""
    return await asyncio.to_thread(
        lambda: json.loads(_kt._handle_search({"query": query, "source": source}))  # noqa: SLF001
    )


if __name__ == "__main__":
    mcp.run()  # stdio transport
