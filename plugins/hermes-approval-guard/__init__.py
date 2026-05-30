"""Hermes Approval Guard — 两阶段工具调用审批插件。

阶段一：规则引擎（<1ms）+ LLM 快速分类（~500ms）
阶段二：ACP Agent 深度审查（3-8s）

覆盖所有工具，提供结构化拒绝反馈，Hindsight 记忆学习。
默认关闭，需 plugin_guard.enabled: true 启用。
"""

from __future__ import annotations

from .guard import pre_tool_call_handler


def register(ctx) -> None:
    """注册 pre_tool_call 钩子。"""
    ctx.register_hook("pre_tool_call", pre_tool_call_handler)
