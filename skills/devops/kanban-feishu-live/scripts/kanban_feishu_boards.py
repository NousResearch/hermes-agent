"""Board configs for skill-layer Feishu live Kanban updates."""

from __future__ import annotations

BOARD_CONFIG: dict[str, dict] = {
    "paper-nexus": {
        "header": "Paper Nexus",
        "header_icon": "📄",
        "entity_key": "canonical_id",
        "entity_label": "论文",
        "stages": ("T0", "T1", "T2", "T3", "T4", "T5", "T6"),
        "stage_labels": {
            "T0": "论点与阅读地图",
            "T1": "主张-证据链 CEL",
            "T2": "方法与复现要点",
            "T3": "对标与开源地图",
            "T4": "实验审计与局限",
            "T5": "飞书精读文档",
            "T6": "QA 门禁",
        },
    },
    "paper-search": {
        "header": "Paper Search",
        "header_icon": "📚",
        "entity_key": "query_slug",
        "entity_label": "检索",
        "stages": ("T0", "T1", "T2"),
        "stage_labels": {
            "T0": "文献检索",
            "T1": "综合排序",
            "T2": "飞书投递",
        },
    },
    "stock-nexus": {
        "header": "Stock Nexus",
        "header_icon": "📈",
        "entity_key": "symbol",
        "entity_label": "标的",
        "stages": ("T0", "T1", "T2", "T_deep", "T3", "T4", "T5", "T6"),
        "stage_labels": {
            "T0": "上下文时间线",
            "T1": "宏观/行业/资金",
            "T2": "技术面/量价",
            "T_deep": "TradingAgents深度",
            "T3": "避雷审查",
            "T4": "规则校准",
            "T5": "投资决策合成",
            "T6": "QA门禁",
        },
    },
}


def get_board_config(board: str) -> dict:
    key = (board or "paper-nexus").strip()
    if key not in BOARD_CONFIG:
        raise ValueError(f"unsupported board: {key!r}")
    return BOARD_CONFIG[key]
