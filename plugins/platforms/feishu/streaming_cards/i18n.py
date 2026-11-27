"""Feishu card i18n — bilingual (en/zh) text mapping."""

from __future__ import annotations

__all__ = [
    "_LOCALES",
    "_T",
    "_i18n",
    "_t",
]

_LOCALES = ["zh_cn", "en_us"]

_T: dict[str, tuple[str, str]] = {
    "status_completed": ("✅ Completed", "✅ 已完成"),
    "status_error": ("❌ Error", "❌ 出错"),
    "status_stopped": ("🛑 Stopped", "🛑 已停止"),
    "elapsed": ("Elapsed {}", "耗时 {}"),
    "speed": ("Speed {}", "速度 {}"),
    "context": ("Context {}", "上下文 {}"),
    "processing": ("Processing...", "处理中..."),
    "processing_prefix": ("💭 Processing...", "💭 处理中..."),
    "tool_use": ("Tool use", "工具执行"),
    "tool_pending": ("🛠️ Tool use pending", "🛠️ 等待工具执行"),
    "tool_pending_hint": ("Tool activity will appear here", "工具执行动态将显示在这里"),
    "steps": ("{} step{}", "{} 步"),
    # Multi-panel titles (tool→text→tool) carry their global step range so panels
    # are distinguishable
    "steps_range": ("steps {}–{}", "第 {}–{} 步"),
    # 折叠态失败信号：不展开也能发现回合中出过错（标题同时转红）
    "steps_failed": ("⚠️ {} failed", "⚠️ {} 步失败"),
    "thought": ("Thought", "思考"),
    "thinking_panel": ("Thinking", "思考中"),
    "thought_for": ("Thought for {}", "思考了 {}"),
    "done": ("Done.", "完成。"),
}


def _i18n(en: str, zh: str) -> dict[str, str]:
    return {"zh_cn": zh, "en_us": en}


def _t(key: str) -> dict[str, str]:
    """简写: _t("processing") → _i18n(*_T["processing"])。"""
    return _i18n(*_T[key])
