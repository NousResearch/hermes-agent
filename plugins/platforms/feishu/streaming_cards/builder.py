"""CardKit v2.0 卡片构建器 — i18n、元素构建、卡片组装."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .segments import Segment, SegmentType
from .tooluse import ToolDisplayStep
from .i18n import _LOCALES, _T, _i18n, _t
from .markdown import (
    _MAX_CHUNK_CHARS,
    _downgrade_tables,
    _split_long_text,
    optimize_markdown_style,
)

STREAMING_ELEMENT_ID = "streaming_content"
REASONING_ELEMENT_ID = "reasoning_content"
REASONING_TEXT_ELEMENT_ID = "reasoning_text"
TOOL_PANEL_ELEMENT_ID = "tool_panel"
HEARTBEAT_ELEMENT_ID = "heartbeat_status"
_LOADING_ELEMENT_ID = "loading_icon"
_LOADING_IMG_KEY = "img_v3_02vb_496bec09-4b43-4773-ad6b-0cdd103cd2bg"


def _collapsible_panel(
    *,
    expanded: bool,
    title_el: dict,
    elements: list[dict],
    vertical_spacing: str = "4px",
    icon_position: str = "right",
) -> dict:
    icon_el = {
        "tag": "standard_icon",
        "token": "down-small-ccm_outlined",
        "size": "16px 16px",
    }
    if icon_position == "right":
        icon_el["color"] = "grey"
    return {
        "tag": "collapsible_panel",
        "expanded": expanded,
        "header": {
            "title": title_el,
            "vertical_align": "center",
            "icon": icon_el,
            "icon_position": icon_position,
            "icon_expanded_angle": -180,
        },
        "border": {"color": "grey", "corner_radius": "5px"},
        "vertical_spacing": vertical_spacing,
        "padding": "8px 8px 8px 8px",
        "elements": elements,
    }


def _streaming_element(
    content: str = "",
    *,
    element_id: str = STREAMING_ELEMENT_ID,
    text_size: str = "normal_v2",
) -> dict:
    return {
        "tag": "markdown",
        "content": content,
        "text_align": "left",
        "text_size": text_size,
        "margin": "0px 0px 0px 0px",
        "element_id": element_id,
    }


_HEADER_STATES: dict[str, dict[str, str]] = {
    "streaming": {"template": "blue", "i18n_key": "processing_prefix"},
    "completed": {"template": "green", "i18n_key": "status_completed"},
    "error": {"template": "red", "i18n_key": "status_error"},
    "stopped": {"template": "red", "i18n_key": "status_stopped"},
}


def _build_header(status: str) -> dict[str, Any]:
    """构建卡片级 header — 流式蓝 / 完成绿 / 停止红."""
    cfg = _HEADER_STATES.get(status, _HEADER_STATES["completed"])
    en_text, zh_text = _T[cfg["i18n_key"]]
    return {
        "title": {
            "tag": "plain_text",
            "content": en_text,
            "i18n_content": _i18n(en_text, zh_text),
        },
        "template": cfg["template"],
    }


def _loading_element() -> dict:
    return {
        "tag": "markdown",
        "content": " ",
        "icon": {
            "tag": "custom_icon",
            "img_key": _LOADING_IMG_KEY,
            "size": "16px 16px",
        },
        "element_id": _LOADING_ELEMENT_ID,
    }


def _build_heartbeat_element(content: str = " ") -> dict:
    """卡片末尾的长回合心跳状态行（loading 图标之后 = 真正底部）。

    element_id 固定，流式期由 controller 对同一元素反复 cardkit_stream_element
    更新内容；complete 重建的最终卡片不含此元素 → 回合结束状态行自然消失。
    小号灰色文本，尽量不干扰正文。
    """
    return {
        "tag": "markdown",
        "content": content,
        "text_align": "left",
        "text_size": "notation",
        "text_color": "grey",
        "margin": "4px 0px 0px 0px",
        "element_id": HEARTBEAT_ELEMENT_ID,
    }


# 工具面板显示步数上限：长 agent 回合动辄几十步，全量渲染会撑爆卡片体积
_TOOL_STEPS_SHOWN = 15

# 完成卡整卡重渲时的动态放宽上限：流式期增量更新预算紧（固定 15 封顶），完成
# 卡按元素预算（本地镜像 segment_helper.ELEMENT_THRESHOLD=180，避免循环导入）与
# 折叠区字符量双约束摊给各工具面板——长回合前段步骤不再永久丢失，仍防 200860
_COMPLETE_ELEMENT_BUDGET = 180
_TOOL_SECTION_CHAR_BUDGET = 40_000
_MAX_TOOL_STEPS_COMPLETE = 50


def _tool_step_element_cost(step: ToolDisplayStep) -> int:
    """单步元素成本（口径同 segment_helper.estimate_tool_elements）."""
    cost = 3  # 标题行 div + standard_icon + lark_md
    if step.get("detail"):
        cost += 2  # div + plain_text
    if step.get("result_block") or step.get("error_block"):
        cost += 2  # div + lark_md
    return cost


def _tool_step_char_cost(step: ToolDisplayStep) -> int:
    """单步折叠区字符量（detail + 结果/错误块正文，防体积上限 200860）."""
    chars = len(str(step.get("detail") or ""))
    block: Any = step.get("error_block") or step.get("result_block") or {}
    chars += len(str(block.get("content") or block.get("fenced") or ""))
    return chars


def _complete_tool_max_steps(
    steps: list[ToolDisplayStep], el_budget: int, char_budget: int,
) -> int:
    """完成卡单面板可显示步数：双预算内尽量多；下限 1（不渲染空面板），上限 50."""
    used_el = used_char = n = 0
    for s in steps:
        el = _tool_step_element_cost(s)
        ch = _tool_step_char_cost(s)
        if (n >= _MAX_TOOL_STEPS_COMPLETE or used_el + el > el_budget
                or used_char + ch > char_budget):
            break
        used_el += el
        used_char += ch
        n += 1
    return max(n, 1)


def _build_tool_panel(
    steps: list[ToolDisplayStep],
    elapsed_ms: float = 0,
    *,
    expanded: bool = False,
    element_id: str | None = TOOL_PANEL_ELEMENT_ID,
    step_offset: int = 0,
    max_steps: int = _TOOL_STEPS_SHOWN,
) -> dict:
    en_t, zh_t = _T["tool_use"]
    # 折叠态标题：有 running 步骤时直接显示「动作标签」（📖 Reading 幼儿园与学习.md），
    # 让用户不展开就能看到 agent 正在操作什么；无 running（含空/全完成）退回
    # 「🛠️ Tool use · N steps」骨架。label 自带 emoji，故 running 态不加 🛠️ 前缀。
    # 有失败步骤时折叠态必须可见（⚠️ N failed + 标题转红），不展开也能发现回合出过错。
    failed = sum(1 for s in steps if s.get("status") == "error")
    running = next((s for s in reversed(steps) if s.get("status") == "running"), None)
    if running:
        prefix = ""
        en_parts, zh_parts = [running.get("label") or running.get("title") or en_t], [
            running.get("label") or running.get("title") or zh_t
        ]
    else:
        prefix = "🛠️ "
        en_parts, zh_parts = [en_t], [zh_t]
    total_steps = len(steps)
    if total_steps > max_steps:
        # 步数封顶：只渲染最近 N 步（running 步骤恒在末尾），标题计数仍是全量
        hidden = total_steps - max_steps
        steps = steps[-max_steps:]
    else:
        hidden = 0
    if total_steps:
        if step_offset > 0:
            tpl_en, tpl_zh = _T["steps_range"]
            en_parts.append(tpl_en.format(step_offset + 1, step_offset + total_steps))
            zh_parts.append(tpl_zh.format(step_offset + 1, step_offset + total_steps))
        else:
            tpl_en, tpl_zh = _T["steps"]
            en_parts.append(tpl_en.format(total_steps, "s" if total_steps > 1 else ""))
            zh_parts.append(tpl_zh.format(total_steps, ""))
    if failed:
        tpl_en, tpl_zh = _T["steps_failed"]
        en_parts.append(tpl_en.format(failed))
        zh_parts.append(tpl_zh.format(failed))
    if elapsed_ms > 0:
        en_parts.append(f"({_format_elapsed(elapsed_ms)})")
        zh_parts.append(f"({_format_elapsed(elapsed_ms)})")

    children: list[dict] = []
    if hidden:
        children.append({
            "tag": "markdown",
            "content": f"…（已折叠前 {hidden} 步，共 {total_steps} 步）…",
            "text_size": "notation",
            "text_color": "grey",
        })
    for s in steps:
        children.extend(_build_tool_step_elements(s))

    panel = _collapsible_panel(
        expanded=expanded,
        title_el={
            "tag": "plain_text",
            "content": f"{prefix}{' · '.join(en_parts)}",
            "i18n_content": _i18n(f"{prefix}{' · '.join(en_parts)}", f"{prefix}{' · '.join(zh_parts)}"),
            "text_color": "red" if failed else "grey",
            "text_size": "notation",
        },
        elements=children,
    )
    if element_id:
        panel["element_id"] = element_id
    return panel


def _build_tool_step_elements(step: ToolDisplayStep) -> list[dict]:
    elements: list[dict] = [_build_tool_step_title(step)]
    detail = _build_tool_step_detail(step)
    if detail:
        elements.append(detail)
    output = _build_tool_step_output(step)
    if output:
        elements.append(output)
    return elements


def _build_tool_step_title(step: ToolDisplayStep) -> dict:
    status = step.get("status", "running")
    status_info = _tool_status_info(status)
    title = step.get("title", step.get("name", "tool"))
    content = f"**{_escape_md(title)}** · <font color='{status_info['color']}'>{status_info['label']}</font>"
    return {
        "tag": "div",
        "icon": {
            "tag": "standard_icon",
            "token": step.get("icon", "tool_02"),
            "color": "grey",
        },
        "text": {
            "tag": "lark_md",
            "content": content,
            "text_size": "notation",
        },
    }


def _build_tool_step_detail(step: ToolDisplayStep) -> dict | None:
    detail = step.get("detail", "").strip()
    if not detail:
        return None
    return {
        "tag": "div",
        "margin": "0px 0px 0px 22px",
        "text": {
            "tag": "plain_text",
            "content": detail,
            "text_color": "grey",
            "text_size": "notation",
        },
    }


def _build_tool_step_output(step: ToolDisplayStep) -> dict | None:
    error_block = step.get("error_block")
    result_block = step.get("result_block")

    lines: list[str] = []
    if error_block:
        lines.append("**Error**")
        lines.append(
            error_block.get("fenced")
            or _format_code_block(error_block.get("content", ""), error_block.get("language", "text"))
        )
    elif result_block:
        lines.append("**Result**")
        lines.append(
            result_block.get("fenced")
            or _format_code_block(result_block.get("content", ""), result_block.get("language", "json"))
        )

    if not lines:
        return None

    return {
        "tag": "div",
        "margin": "0px 0px 0px 22px",
        "text": {
            "tag": "lark_md",
            "content": "\n".join(lines),
            "text_size": "notation",
        },
    }


def _tool_status_info(status: str) -> dict[str, str]:
    return {
        "running": {"label": "Running", "color": "turquoise"},
        "success": {"label": "Succeeded", "color": "green"},
        "error": {"label": "Failed", "color": "red"},
    }.get(status, {"label": status.capitalize(), "color": "grey"})


def _format_code_block(content: str, language: str) -> str:
    normalized = content.replace("\r\n", "\n").strip()
    fence = "`" * max(3, _longest_backtick_run(normalized) + 1)
    return f"{fence}{language}\n{normalized}\n{fence}"


def _longest_backtick_run(value: str) -> int:
    matches = re.findall(r"`+", value)
    return max((len(m) for m in matches), default=0)


def _escape_md(value: str) -> str:
    return re.sub(r"([`*_{}\[\]<>])", r"\\\1", value.replace("\\", "\\\\"))


# 思考面板摘录上限：飞书卡片无滚动组件，长思考全文填充会撑爆卡片体积
# （工具面板 200860 前车之鉴），折叠区只保留头尾摘录 + 总量标注
_REASONING_HEAD_CHARS = 600
_REASONING_TAIL_CHARS = 300


def cap_reasoning_text(text: str, *, head: int = _REASONING_HEAD_CHARS,
                       tail: int = _REASONING_TAIL_CHARS) -> str:
    """长思考截断为头尾摘录；短文原样返回（流式与完成重渲共用同一口径）."""
    if len(text) <= head + tail + 80:
        return text
    omitted = len(text) - head - tail
    return (f"{text[:head]}\n\n…（思考原文共 {len(text)} 字，"
            f"中间省略 {omitted} 字）…\n\n{text[-tail:]}")


def _build_reasoning_panel(
    text: str, elapsed_ms: float = 0, *, expanded: bool = False, element_id: str | None = None,
    text_element_id: str | None = REASONING_TEXT_ELEMENT_ID,
) -> dict:
    if elapsed_ms > 0:
        d = _format_elapsed(elapsed_ms)
        en_label, zh_label = _T["thought_for"][0].format(d), _T["thought_for"][1].format(d)
    elif not text.strip():
        en_label, zh_label = _T["thinking_panel"]
    else:
        en_label, zh_label = _T["thought"]
    # 摘录封顶后恒 ≤ head+tail+标注，单 markdown 元素足够；原 2400 字符分块
    # 仅服务无上限全文，随摘录方案移除
    chunks = [cap_reasoning_text(text)] if text.strip() else [text]
    inner_elements: list[dict] = []
    for i, chunk in enumerate(chunks):
        el: dict = {"tag": "markdown", "content": chunk, "text_size": "notation"}
        if text_element_id and i == 0:
            el["element_id"] = text_element_id
        inner_elements.append(el)
    panel = _collapsible_panel(
        expanded=expanded,
        title_el={
            "tag": "plain_text",
            "content": f"💭 {en_label}",
            "i18n_content": _i18n(f"💭 {en_label}", f"💭 {zh_label}"),
            "text_color": "grey",
            "text_size": "notation",
        },
        elements=inner_elements,
        vertical_spacing="8px",
    )
    if element_id:
        panel["element_id"] = element_id
    return panel


def _notice_markdown(text: str) -> str:
    """非对话通知的灰色紧凑渲染（bg watcher 完成通知等），截断防长输出刷屏."""
    compact = " ".join(text.split())
    if len(compact) > 300:
        compact = compact[:300] + "…"
    compact = compact.replace("[", "\\[").replace("]", "\\]")
    return f"<font color='grey'>{compact}</font>"


_MODEL_BUTTONS_PER_ROW = 3
MAX_FAVORITE_MODELS = 8
# 管理卡候选池按钮上限：防超多 provider 全量平铺把卡片推到飞书体积上限；
# 常用清单本身不受此限（已加入的始终完整显示）
MAX_ADMIN_CANDIDATES = 60


def _model_button_rows(buttons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按钮按每行 3 个排成 action 行（飞书 action 容器一行一组）。"""
    return [{"tag": "action", "actions": buttons[i:i + _MODEL_BUTTONS_PER_ROW]}
            for i in range(0, len(buttons), _MODEL_BUTTONS_PER_ROW)]


def build_model_picker_card(current: str, models: list[str]) -> dict[str, Any]:
    """模型选择卡（原生 interactive 消息，非 cardkit 实体）— 🧠⇄ 点击后补发.

    原生消息路径支持 action 容器（clarify 同款），按钮名 = 模型名渲染有保证；
    选中经回调合成 `/model <name>`。当前模型打 ✅。末行「⚙ 管理常用」进管理卡
    （常用清单存盘、跨回合生效；为空时清单即 default+fallback 推导）。"""
    buttons = [
        {"tag": "button",
         "text": {"tag": "plain_text", "content": (f"✅ {m}" if m == current else m)},
         "type": "default",
         "value": {"hermes_model_action": "switch", "target": m}}
        for m in models
    ]
    elements = _model_button_rows(buttons)
    elements.append({"tag": "action", "actions": [
        {"tag": "button",
         "text": {"tag": "plain_text", "content": "⚙ 管理常用模型"},
         "type": "default",
         "value": {"hermes_model_action": "admin"}},
    ]})
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text",
                             "content": f"🧠 切换模型（当前：{current}）"},
                   "template": "blue"},
        "elements": elements,
    }


def build_model_admin_card(current: str, favorites: list[str],
                           providers: list[dict[str, Any]],
                           notice: str = "") -> dict[str, Any]:
    """常用模型管理卡 — picker 卡「⚙ 管理常用模型」点击后补发，toggle 即存盘.

    providers = [{"slug","name","models"}]（hermes list_picker_providers 同构）。
    常用清单单列一节排最前（含不在候选池里的手工配置项，保证可移出）；候选池
    按 provider 分节，已在常用的打 ✅。所有按钮点击 → toggle → 整卡原地刷新
    （_card_response 替换，clarify/switch ack 同款）。"""
    fav_set = set(favorites)
    merged: list[tuple[str, list[str]]] = []
    if favorites:
        merged.append(("★ 常用（点击移出）", list(favorites)))
    budget = MAX_ADMIN_CANDIDATES
    for p in providers or []:
        if budget <= 0:
            break
        rows = [m for m in (p.get("models") or [])
                if str(m).strip() and str(m).strip() not in fav_set]
        rows = rows[:budget]
        if not rows:
            continue
        budget -= len(rows)
        label = str(p.get("name") or p.get("slug") or "models")
        shown_here = len(rows) + len([m for m in (p.get("models") or []) if m in fav_set])
        if (p.get("total_models") or 0) > shown_here:
            label += f"（{shown_here}/{p['total_models']}）"
        merged.append((label, rows))
    truncated = budget <= 0 and any(
        (p.get("total_models") or 0) > len([m for m in (p.get("models") or [])
                                            if m in fav_set])
        for p in (providers or []))
    elements: list[dict[str, Any]] = []
    if notice:
        elements.append({"tag": "markdown", "content": f"⚠️ {notice}"})
    hint = ("点击模型加入/移出常用（✅ = 已常用，即点即存）。"
            f"常用上限 {MAX_FAVORITE_MODELS} 个，选择卡（footer 🧠⇄）只展示常用。")
    if truncated:
        hint += f" 候选池较多，仅列前 {MAX_ADMIN_CANDIDATES} 个可加项。"
    elements.append({"tag": "markdown", "content": hint})
    for title, models in merged:
        elements.append({"tag": "markdown",
                         "content": f"**{title}**"})
        elements.extend(_model_button_rows([
            {"tag": "button",
             "text": {"tag": "plain_text",
                      "content": (f"✅ {m}" if m in fav_set
                                  else ("➕ " + m if title.startswith("★") else m))},
             "type": "primary" if m in fav_set else "default",
             "value": {"hermes_model_action": "toggle", "target": m}}
            for m in models]))
    if not any(models for _, models in merged):
        elements.append({"tag": "markdown",
                         "content": "候选池为空（hermes provider 目录不可用）——请检查凭据，"
                                    "或在 config.yaml 配 `streaming.footer.model_cycle`。"})
    elements.append({"tag": "action", "actions": [
        {"tag": "button", "text": {"tag": "plain_text", "content": "✅ 完成"},
         "type": "primary", "value": {"hermes_model_action": "admin_done"}},
    ]})
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text",
                             "content": f"⚙ 管理常用模型（当前模型：{current or '未知'}）"},
                   "template": "blue"},
        "elements": elements,
    }


def build_model_switch_ack_card(model: str) -> dict[str, Any]:
    """模型选择卡点击后的同步确认卡（替换选择卡本体）."""
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"✅ 已切换到 {model}"},
                   "template": "green"},
        "elements": [{"tag": "markdown",
                      "content": "下一回合起使用该模型（以 footer 显示为准）。"}],
    }


def build_model_admin_done_card(favorites: list[str]) -> dict[str, Any]:
    """管理卡「✅ 完成」后的收尾卡（替换管理卡本体）."""
    names = "、".join(favorites) if favorites else "（空——选择卡回退默认清单）"
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "✅ 常用模型已更新"},
                   "template": "green"},
        "elements": [{"tag": "markdown",
                      "content": f"当前常用（{len(favorites)} 个）：{names}\n"
                                 "任意完成卡 footer 点 🧠⇄ 即按此清单出选择卡。"}],
    }


def _build_footer_elements(
    footer_data: dict | None,
    is_error: bool = False,
    is_aborted: bool = False,
    fields: list[list[str]] | None = None,
    show_label: bool = False,
    text_size: str = "notation",
) -> list[dict]:
    if fields is None:
        fields = [["elapsed", "model", "context"]]

    data = footer_data or {}
    en_lines: list[str] = []
    zh_lines: list[str] = []
    for row in fields:
        en_parts: list[str] = []
        zh_parts: list[str] = []
        for field in row:
            en, zh = _render_footer_field(field, data, is_error, is_aborted, show_label)
            if en:
                en_parts.append(en)
                if zh:
                    zh_parts.append(zh)
        if en_parts:
            en_lines.append(" ｜ ".join(en_parts))
            zh_lines.append(" ｜ ".join(zh_parts))

    if not en_lines:
        return []

    en_content = "\n".join(en_lines)
    zh_content = "\n".join(zh_lines)
    if is_error:
        en_content = f"<font color='red'>{en_content}</font>"
        zh_content = f"<font color='red'>{zh_content}</font>"

    return [
        {"tag": "hr"},
        {
            "tag": "markdown",
            "content": en_content,
            "i18n_content": _i18n(en_content, zh_content),
            "text_size": text_size,
        },
    ]


def _render_footer_field(
    name: str,
    data: dict,
    is_error: bool,
    is_aborted: bool,
    show_label: bool,
) -> tuple[str | None, str | None]:
    if name == "status":
        if is_error:
            return _T["status_error"]
        if is_aborted:
            return _T["status_stopped"]
        return _T["status_completed"]

    if name == "elapsed":
        duration = data.get("duration", 0)
        if isinstance(duration, (int, float)) and duration > 0:
            val = _format_elapsed(duration * 1000)
            if show_label:
                return _T["elapsed"][0].format(val), _T["elapsed"][1].format(val)
            return f"⏱ {val}", f"⏱ {val}"
        return None, None

    if name == "model":
        v = data.get("model") or None
        if v:
            return f"🧠 {v}", f"🧠 {v}"
        return None, None

    if name == "tokens":
        input_t = data.get("input_tokens", 0) or 0
        output_t = data.get("output_tokens", 0) or 0
        if input_t or output_t:
            v = f"↑ {_compact(input_t)} ↓ {_compact(output_t)}"
            return v, v
        return None, None

    if name == "speed":
        tps = data.get("tps")
        if isinstance(tps, (int, float)) and tps > 0:
            # 短回答常见 0.x t/s——取整成 0 看着像坏了（2026-10-09 跨机部署实测）
            val = "<1 t/s" if tps < 1 else f"{tps:.0f} t/s"
            if show_label:
                return _T["speed"][0].format(val), _T["speed"][1].format(val)
            return f"⚡ {val}", f"⚡ {val}"
        return None, None

    if name == "context":
        used = data.get("context_used", 0) or 0
        max_c = data.get("context_max", 0) or 0
        if max_c:
            pct = int(used / max_c * 100)
            val = f"{pct}%/{_compact(max_c)}"
            if show_label:
                return _T["context"][0].format(val), _T["context"][1].format(val)
            return f"📊 {val}", f"📊 {val}"
        return None, None

    return None, None


def _compact(n: int) -> str:
    if n >= 1_000_000:
        m = n / 1_000_000
        return f"{int(m)}M" if m >= 100 else f"{m:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _format_elapsed(ms: float) -> str:
    seconds = ms / 1000
    return f"{seconds:.1f}s" if seconds < 60 else f"{int(seconds // 60)}m {int(seconds % 60)}s"


def build_streaming_tool_use_pending_panel() -> dict[str, Any]:
    # 展开态提示行：面板点开不再是空白（markdown 元素不带 i18n_content，与
    # running 动作标签同口径先英文）
    return _collapsible_panel(
        expanded=False,
        title_el={
            "tag": "plain_text",
            "content": _T["tool_pending"][0],
            "i18n_content": _t("tool_pending"),
            "text_color": "grey",
            "text_size": "notation",
        },
        elements=[{
            "tag": "markdown",
            "content": _T["tool_pending_hint"][0],
            "text_size": "notation",
            "text_color": "grey",
        }],
    )


def build_streaming_card_v2(
    *,
    tool_steps: list[ToolDisplayStep] | None = None,
    elapsed_ms: float = 0,
    show_tool_use: bool = True,
    show_reasoning: bool = False,
    show_streaming_element: bool = True,
    header_enabled: bool = False,
    text_size: str = "normal_v2",
    heartbeat_enabled: bool = False,
    width_mode: str = "default",
) -> dict[str, Any]:
    """CardKit 2.0 流式占位卡片 — 含工具面板 + streaming + loading 元素."""
    elements: list[dict] = []

    if show_reasoning:
        elements.append(
            _build_reasoning_panel(" ", expanded=False, element_id=REASONING_ELEMENT_ID)
        )

    if show_tool_use:
        if tool_steps:
            elements.append(_build_tool_panel(tool_steps, elapsed_ms))
        else:
            elements.append(build_streaming_tool_use_pending_panel())

    if show_streaming_element:
        elements.append(_streaming_element(text_size=text_size))
    elements.append(_loading_element())
    # 心跳状态行放 loading 之后 = 卡片真正末尾（不会被 insert_before 新内容挤走）。
    # 空内容占位，首次心跳到达时由 controller 更新；回合结束 complete 重建不带它。
    if heartbeat_enabled:
        elements.append(_build_heartbeat_element())

    card = {
        "schema": "2.0",
        "config": {
            "width_mode": width_mode,
            "streaming_mode": True,
            "streaming_config": {
                "print_frequency_ms": {"default": 120},
                "print_step": {"default": 6},
                "print_strategy": "fast",
            },
            "locales": _LOCALES,
            "summary": {
                "content": _T["processing"][0],
                "i18n_content": _t("processing"),
            },
        },
        "body": {"elements": elements},
    }
    if header_enabled:
        card["header"] = _build_header("streaming")
    return card


def build_complete_card(
    *,
    segments: list[Segment],
    all_tool_steps: list[ToolDisplayStep],
    footer_data: dict | None = None,
    image_keys: list[str] | None = None,
    is_error: bool = False,
    is_aborted: bool = False,
    footer_fields: list[list[str]] | None = None,
    footer_show_label: bool = True,
    footer_enabled: bool = True,
    footer_text_size: str = "notation",
    tool_panel_expanded: bool = False,
    reasoning_panel_expanded: bool = False,
    header_enabled: bool = False,
    body_text_size: str = "normal_v2",
    show_tool_use: bool = True,
    width_mode: str = "default",
    model_switch: dict | None = None,
) -> dict[str, Any]:
    """完成态流式卡片 — 按 segments 顺序渲染."""
    elements: list[dict] = []
    has_answer = False

    # 工具步骤显示预算：扣除非工具元素与 footer 的估算占用后，剩余在元素/字符
    # 双上限内摊给各工具面板（流式期固定 15 封顶，完成卡动态放宽最多 50）
    tool_el_budget = _COMPLETE_ELEMENT_BUDGET - 4  # 基础波动余量
    if footer_enabled:
        tool_el_budget -= 4  # hr + footer 文本 + 按钮列等
    for seg in segments:
        if seg.type == SegmentType.REASONING:
            tool_el_budget -= 4
        elif seg.type == SegmentType.ANSWER:
            tool_el_budget -= len(seg.text) // _MAX_CHUNK_CHARS + 1
        elif seg.type == SegmentType.NOTICE:
            tool_el_budget -= 1
        elif seg.type == SegmentType.TOOL and show_tool_use:
            tool_el_budget -= 3  # 面板壳（panel + header 子节点）
    tool_char_budget = _TOOL_SECTION_CHAR_BUDGET

    for seg in segments:
        if seg.type == SegmentType.REASONING:
            if seg.text:
                elements.append(_build_reasoning_panel(
                    seg.text, seg.elapsed_ms, expanded=reasoning_panel_expanded,
                    element_id=None, text_element_id=None,
                ))
        elif seg.type == SegmentType.TOOL:
            if not show_tool_use:
                continue
            start = seg.tool_offset
            end = seg.tool_end_offset if seg.tool_end_offset else len(all_tool_steps)
            steps = all_tool_steps[start:end]
            if steps:
                max_steps = _complete_tool_max_steps(steps, tool_el_budget, tool_char_budget)
                elements.append(_build_tool_panel(
                    steps, expanded=tool_panel_expanded, element_id=None,
                    step_offset=start, max_steps=max_steps))
                shown = steps[-max_steps:]
                tool_el_budget -= sum(_tool_step_element_cost(s) for s in shown)
                tool_char_budget -= sum(_tool_step_char_cost(s) for s in shown)
        elif seg.type == SegmentType.ANSWER and seg.text:
            has_answer = True
            content = _downgrade_tables(optimize_markdown_style(seg.text))
            for chunk in _split_long_text(content):
                elements.append({"tag": "markdown", "content": chunk, "text_size": body_text_size})
        elif seg.type == SegmentType.NOTICE and seg.text:
            elements.append({
                "tag": "markdown",
                "content": _notice_markdown(seg.text),
                "text_size": "notation",
            })

    # 无回答时的「Done.」占位——带 NOTICE 的卡（redirect 收尾/后台通知）本身已有
    # 状态说明，再补一句 Done. 会隔断「结果见下方新卡片」的指向（2026-09-22 实测）
    if not has_answer and not any(seg.type == SegmentType.NOTICE for seg in segments):
        elements.append({"tag": "markdown", "content": _T["done"][0], "text_size": body_text_size})

    # image_generate 产物图（用 markdown 图片语法 ![alt](img_key)，复用 ImageResolver 已验证的渲染方式）
    for img_key in (image_keys or []):
        elements.append({"tag": "markdown", "content": f"![image]({img_key})"})

    if footer_enabled:
        footer_elems = _build_footer_elements(
            footer_data,
            is_error,
            is_aborted,
            fields=footer_fields,
            show_label=footer_show_label,
            text_size=footer_text_size,
        )
        # 最右极简按钮（🧠⇄ tiny）：点击 → 机器人弹出模型选择卡（原生 interactive
        # 消息、clarify 同款 action 容器，按钮名渲染有保证——v2 select_static
        # 选项名端上渲染空白已弃用）。behaviors 携带 value——v2 实体卡不支持
        # action 容器（200861）。
        if footer_enabled and model_switch and model_switch.get("current"):
            btn = {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "🧠⇄"},
                "type": "default", "size": "tiny",
                "behaviors": [{"type": "callback",
                               "value": {"hermes_model_action": "pick",
                                         "from": str(model_switch["current"])}}],
            }
            text_elems = [e for e in footer_elems if e.get("tag") == "markdown"]
            if text_elems:
                elements.append(footer_elems[0])  # hr
                elements.append({
                    "tag": "column_set", "flex_mode": "none",
                    "background_style": "default",
                    "columns": [
                        # weighted 撑满 → 按钮列贴最右
                        {"tag": "column", "width": "weighted", "weight": 1,
                         "elements": text_elems},
                        {"tag": "column", "width": "auto", "elements": [btn]},
                    ],
                })
            else:
                elements.extend(footer_elems)
                elements.append(btn)
        else:
            elements.extend(footer_elems)

    summary_text = ""
    for seg in reversed(segments):
        if seg.type in (SegmentType.ANSWER, SegmentType.REASONING) and seg.text:
            summary_text = seg.text
            break
    if not summary_text:
        # 无回答的卡（redirect 早期收尾）：会话列表摘要用重启提示，一眼识别作废旧卡
        for seg in reversed(segments):
            if seg.type == SegmentType.NOTICE and seg.text:
                summary_text = seg.text
                break
    summary = summary_text[:120].replace("\n", " ").replace("```", "").strip()

    card: dict[str, Any] = {
        "schema": "2.0",
        "config": {
            "width_mode": width_mode,
            "wide_screen_mode": True,
            "update_multi": True,
            "locales": _LOCALES,
        },
    }
    if summary:
        card["config"]["summary"] = {"content": summary}
    card["body"] = {"elements": elements}
    if header_enabled:
        header_status = "error" if is_error else "stopped" if is_aborted else "completed"
        card["header"] = _build_header(header_status)
    return card


def _format_run_time(run_time: str) -> str:
    """将 ISO 时间戳格式化为可读日期时间，失败则原样返回."""
    if not run_time:
        return ""
    try:
        dt = datetime.fromisoformat(run_time)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return run_time


def build_cron_card(
    content: str, *, task_name: str = "", run_time: str = "",
    image_keys: list[str] | None = None, template: str = "blue",
) -> dict[str, Any]:
    """Cron 推送用的极简静态卡片 — schema 2.0，可选 header + markdown 内容 + 图片.

    ``template`` 是 header 配色（飞书 header template 名）——失败通知传 "red"。
    """
    card: dict[str, Any] = {
        "schema": "2.0",
        "config": {"wide_screen_mode": True, "locales": _LOCALES},
        "body": {"elements": []},
    }
    header_parts = [p for p in (task_name, _format_run_time(run_time)) if p]
    if header_parts:
        card["header"] = {
            "title": {"tag": "lark_md", "content": ":Alarm: " + " · ".join(header_parts)},
            "template": template,
        }
    if not content.strip():
        return card
    summary = content[:120].replace("\n", " ").replace("```", "").strip()
    if summary:
        card["config"]["summary"] = {"content": summary}
    for chunk in _split_long_text(optimize_markdown_style(content)):
        if chunk.strip():
            card["body"]["elements"].append({"tag": "markdown", "content": chunk})
    # image_generate 产物图（markdown 图片语法，同 build_complete_card）
    for img_key in (image_keys or []):
        card["body"]["elements"].append({"tag": "markdown", "content": f"![image]({img_key})"})
    return card


def build_background_card(preview: str, content: str) -> dict[str, Any]:
    """Background 任务完成推送卡片 — schema 2.0，header + markdown."""
    card: dict[str, Any] = {
        "schema": "2.0",
        "config": {"wide_screen_mode": True, "locales": _LOCALES},
        "header": {
            "title": {"tag": "plain_text", "content": f"✅ Background: \"{preview}\""},
        },
        "body": {"elements": []},
    }
    body = content if content.strip() else "(No response generated)"
    summary = body[:120].replace("\n", " ").replace("```", "").strip()
    if summary:
        card["config"]["summary"] = {"content": summary}
    for chunk in _split_long_text(optimize_markdown_style(body)):
        if chunk.strip():
            card["body"]["elements"].append({"tag": "markdown", "content": chunk})
    return card
