#!/usr/bin/env python3
"""
PATCH ① SCENARIO-ROUTER — 用户输入 → 场景参数块
v1.0 2026-06-23

用法: python3 scenario_router.py "<用户原始消息>"
输出: 追加到 source.md 文末的 🎨 场景路由注入 段
"""

import sys, re

# ═══════════════════════════════════════════
# 表 A：场景关键词 → 场景参数
# 来源: 场景参数速查表_v2_2026-06-22.md
# ═══════════════════════════════════════════

SCENARIOS = {
    "高管汇报": {
        "keywords": ["高管", "老板", "董事会", "决策", "汇报", "领导", "向上"],
        "page_count": "≤8",
        "content_divergence": "结论先行——每页标题=该页结论，正文≤3 bullet，细节放附录",
        "mode": "pyramid",
        "visual_style": "swiss-minimal",
        "color_primary": "#1A56DB",
        "color_bg": "#FFFFFF",
        "color_secondary": "#6B7280",
        "typography": "微软雅黑 + Arial",
        "images": "placeholder 或 none（高管汇报少用图）",
        "executor": "顶级专业咨询型",
        "design_template": "每页标题=该页结论（非话题标签）。正文≤3 bullet，每 bullet≤10 字。\n所有图表必须有\"so what\"标注。禁止：正文超3行、纯文字页面、装饰性元素。\n色调：冷静、自信、无废话。",
        "dont_list": "装饰性图标、渐变色背景、超过 8 页、正文超过 3 行、无结论的标题"
    },
    "产品发布": {
        "keywords": ["产品", "发布", "上市", "新品", "路演", "营销", "品牌"],
        "page_count": "10-15",
        "content_divergence": "情绪驱动——先造悬念，后揭示答案。视觉冲击为主，文字为辅",
        "mode": "showcase 或 narrative",
        "visual_style": "dark-tech 或 glassmorphism",
        "color_primary": "品牌主色",
        "color_bg": "#F5F5F7",
        "color_secondary": "#86868B",
        "typography": "微软雅黑 + Segoe UI",
        "images": "AI-generated（氛围图为主）",
        "executor": "发布会视觉型",
        "design_template": "每页一个核心画面+一句标题。图片占≥60%面积。文字=标注。\n节奏：悬念页→揭示页→冲击数据页→CTA页。禁止：纯文字页、密集表格。",
        "dont_list": "bullet列表、Excel表格、小于24pt的文字、保守配色"
    },
    "数据报告": {
        "keywords": ["数据", "报告", "分析", "指标", "趋势", "财报", "季度", "KPI"],
        "page_count": "15-25",
        "content_divergence": "数据驱动——先给核心发现，再展开数据论证",
        "mode": "pyramid 或 briefing",
        "visual_style": "data-journalism 或 editorial",
        "color_primary": "#003366",
        "color_bg": "#F8F9FA",
        "color_secondary": "#E63946",
        "typography": "Georgia + 微软雅黑",
        "images": "placeholder（图表为主，AI 图少用）",
        "executor": "咨询数据可视化型",
        "design_template": "图表为主角——每页≤1个核心图表+2-3行解读。数据必须标注来源和年份。\n禁止：装饰性图片、无来源的数据、超过4种图表配色。",
        "dont_list": "装饰图、无数据来源标注、饼图（用柱状代替）、超过4色图表"
    },
    "培训教学": {
        "keywords": ["培训", "教学", "课程", "学习", "教程", "讲解", "教育"],
        "page_count": "10-20",
        "content_divergence": "教学结构——概念→示例→练习→总结",
        "mode": "instructional",
        "visual_style": "sketch-notes 或 soft-rounded",
        "color_primary": "#2E86AB",
        "color_bg": "#FFF8F0",
        "color_secondary": "#F18F01",
        "typography": "微软雅黑",
        "images": "AI-generated（插图为主）",
        "executor": "教学亲和型",
        "design_template": "每页一个概念+一个示例。互动练习页≥20%。文字友好、不学术。\n禁止：密集表格、专业术语不加解释、全是文字无图。",
        "dont_list": "学术术语、无图纯文字页、没有练习/互动环节"
    },
    "创意提案": {
        "keywords": ["创意", "提案", "方案", "建议", "pitch", "融资", "BP"],
        "page_count": "8-12",
        "content_divergence": "故事驱动——问题→洞察→方案→愿景",
        "mode": "narrative",
        "visual_style": "memphis 或 brutalist",
        "color_primary": "#FF6D00",
        "color_bg": "#FFFFFF",
        "color_secondary": "#1A73E8",
        "typography": "微软雅黑 + Impact",
        "images": "AI-generated（大胆配色）",
        "executor": "创意视觉型",
        "design_template": "大胆配色+非常规布局。每页一个核心概念。封面=冲击力标题。\n禁止：保守配色、模板化布局、超过5行的正文。",
        "dont_list": "保守配色、模板化布局、多文字、图表优先于故事"
    }
}

# ═══════════════════════════════════════════
# 表 B：风格关键词 → visual_style 映射
# 来源: 风格映射表_v2_2026-06-22.md
# ═══════════════════════════════════════════

STYLE_MAP = {
    "麦肯锡": ("swiss-minimal", "pyramid"),
    "咨询": ("swiss-minimal", "pyramid"),
    "Apple": ("swiss-minimal", "showcase"),
    "极简": ("swiss-minimal", "showcase"),
    "高冷": ("swiss-minimal", "showcase"),
    "互联网": ("glassmorphism", "narrative"),
    "大厂": ("glassmorphism", "narrative"),
    "科技": ("dark-tech", "narrative"),
    "未来": ("dark-tech", "narrative"),
    "杂志": ("editorial", "pyramid"),
    "编辑": ("editorial", "pyramid"),
    "学术": ("editorial", "briefing"),
    "论文": ("editorial", "briefing"),
    "手绘": ("sketch-notes", "instructional"),
    "亲和": ("sketch-notes", "instructional"),
    "温暖": ("ink-notes", "instructional"),
    "复古": ("vintage-poster", "narrative"),
    "老字号": ("vintage-poster", "narrative"),
    "水墨": ("ink-wash", "narrative"),
    "东方": ("ink-wash", "narrative"),
    "国风": ("ink-wash", "narrative"),
    "粗野": ("brutalist", "showcase"),
    "前卫": ("brutalist", "showcase"),
    "设计感": ("zine", "showcase"),
    "蓝图": ("blueprint", "instructional"),
    "工程": ("blueprint", "instructional"),
    "架构": ("blueprint", "instructional"),
    "数据报告": ("data-journalism", "pyramid"),
    "财经": ("data-journalism", "pyramid"),
    "SaaS": ("soft-rounded", "narrative"),
    "产品": ("soft-rounded", "narrative"),
    "现代": ("soft-rounded", "narrative"),
    "培训": ("sketch-notes", "instructional"),
    "教学": ("sketch-notes", "instructional"),
}


def detect_scenario(user_text: str) -> tuple:
    """返回 (场景名, 匹配关键词数, 场景dict)"""
    scores = {}
    for name, sc in SCENARIOS.items():
        score = sum(1 for kw in sc["keywords"] if kw in user_text)
        if score > 0:
            scores[name] = (score, sc)
    if not scores:
        return ("通用", 0, None)
    best = max(scores, key=lambda k: scores[k][0])
    return (best, scores[best][0], scores[best][1])


def detect_style_override(user_text: str):
    """检测用户是否明确说了风格关键词"""
    for kw, (vs, mode) in STYLE_MAP.items():
        if kw in user_text:
            return vs, mode
    return None, None


def generate_routing_block(user_text: str) -> str:
    scenario_name, score, scenario = detect_scenario(user_text)
    vs_override, mode_override = detect_style_override(user_text)
    
    if scenario is None:
        return "# 🎨 场景路由注入（自动生成）\n\n未能自动识别场景，将使用 PPT Master 默认参数。\n"
    
    vs = vs_override or scenario["visual_style"]
    mode = mode_override or scenario["mode"]
    
    # 用户有风格覆盖时标注
    override_note = ""
    if vs_override:
        override_note = f"\n- ⚠️ 检测到风格关键词，已覆盖场景默认 visual_style: {scenario['visual_style']} → {vs_override}"
    if mode_override:
        override_note += f"\n- ⚠️ 检测到模式关键词，已覆盖场景默认 mode: {scenario['mode']} → {mode_override}"
    
    block = f"""# 🎨 场景路由注入（自动生成——PPT Master 前置预处理）

> 检测场景：**{scenario_name}**（匹配关键词 {score} 个）
> 以下参数为建议默认值，PPT Master 的 Eight Confirmations 阶段会再次让你确认——你可随时修改任何参数。{override_note}

## 场景参数

| 参数 | 建议值 |
|------|--------|
| Page Count | {scenario['page_count']} |
| Content Divergence | {scenario['content_divergence']} |
| Mode | **{mode}** |
| Visual Style | **{vs}** |
| Color | 主色 {scenario['color_primary']} / 底色 {scenario['color_bg']} / 辅色 {scenario['color_secondary']} |
| Typography | {scenario['typography']} |
| Images | {scenario['images']} |
| Executor Type | {scenario['executor']} |

## 设计要求

{scenario['design_template']}

## Don't 清单

{scenario['dont_list']}

---
"""
    return block


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 scenario_router.py \"<用户原始消息>\"")
        sys.exit(1)
    
    user_text = sys.argv[1]
    print(generate_routing_block(user_text))
