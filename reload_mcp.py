#!/usr/bin/env python3
"""MCP 热重载脚本：在不重启 Agent 的情况下刷新所有 MCP servers。

用法:
    cd /home/renyh/.hermes/hermes-agent
    source venv/bin/activate
    python reload_mcp.py

逻辑:
    1. 注销所有已注册的 mcp_* 工具
    2. 关闭所有 MCP 连接并清空 _servers
    3. 重新读取 config.yaml 并发现新 tools
    4. 打印变化对比
"""

import sys
import os

# 确保 hermes-agent 根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.registry import registry
from tools.mcp_tool import discover_mcp_tools, shutdown_mcp_servers, _servers


def get_mcp_tool_names():
    """获取当前注册的所有 MCP 工具名称。"""
    return sorted(name for name in registry._tools if name.startswith("mcp_"))


def main():
    old_tools = get_mcp_tool_names()
    print(f"📦 重新加载前: {len(old_tools)} 个 MCP 工具")
    if old_tools:
        for t in old_tools:
            print(f"   - {t}")

    # 1. 注销所有旧 MCP 工具
    print("\n🗑️  注销旧工具...")
    for name in old_tools:
        registry.deregister(name)

    # 2. 关闭所有连接 + 清空 _servers
    print("🔌 关闭 MCP 连接...")
    shutdown_mcp_servers()

    # 3. 重新发现（会重新读取 config.yaml）
    print("🔍 重新读取配置并连接...\n")
    discover_mcp_tools()

    # 4. 打印结果对比
    new_tools = get_mcp_tool_names()
    added = sorted(set(new_tools) - set(old_tools))
    removed = sorted(set(old_tools) - set(new_tools))

    print(f"📦 重新加载后: {len(new_tools)} 个 MCP 工具")
    if added:
        print(f"\n✅ 新增 ({len(added)}):")
        for t in added:
            print(f"   + {t}")
    if removed:
        print(f"\n❌ 移除 ({len(removed)}):")
        for t in removed:
            print(f"   - {t}")
    if not added and not removed and len(new_tools) == len(old_tools):
        print("ℹ️  工具列表无变化")

    print("\n✅ 热重载完成。注意：当前对话的工具列表会在下一轮 API 调用时生效。")


if __name__ == "__main__":
    main()
