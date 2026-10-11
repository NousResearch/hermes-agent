"""CardKit v2.0 card builder — i18n, element construction, card assembly."""

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
    """Card-level header — blue while streaming / green completed / red stopped."""
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
    """Heartbeat status line at the true bottom of the card (after the loading icon).

    The element_id is fixed; while streaming the controller repeatedly updates
    the same element via cardkit_stream_element. The final card rebuilt by
    complete omits it, so the line disappears with the turn. Small grey text
    to stay out of the body's way.
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


# Tool-panel step cap: long agent turns run dozens of steps; rendering them all
# overflows the card
_TOOL_STEPS_SHOWN = 15

# Relaxed dynamic cap when the complete card re-renders wholesale: incremental
# streaming updates run a tight budget (hard cap 15); the complete card spreads
# an element budget (mirroring segment_helper.ELEMENT_THRESHOLD=180 locally to
# avoid a circular import) and a collapsed-character budget across panels — early
# steps of long turns are no longer lost forever, while still guarding 200860
_COMPLETE_ELEMENT_BUDGET = 180
_TOOL_SECTION_CHAR_BUDGET = 40_000
_MAX_TOOL_STEPS_COMPLETE = 50


def _tool_step_element_cost(step: ToolDisplayStep) -> int:
    """Per-step element cost (same basis as segment_helper.estimate_tool_elements)."""
    cost = 3  # title-row div + standard_icon + lark_md
    if step.get("detail"):
        cost += 2  # div + plain_text
    if step.get("result_block") or step.get("error_block"):
        cost += 2  # div + lark_md
    return cost


def _tool_step_char_cost(step: ToolDisplayStep) -> int:
    """Per-step collapsed-region characters (detail + result/error bodies, guarding the 200860 size limit)."""
    chars = len(str(step.get("detail") or ""))
    block: Any = step.get("error_block") or step.get("result_block") or {}
    chars += len(str(block.get("content") or block.get("fenced") or ""))
    return chars


def _complete_tool_max_steps(
    steps: list[ToolDisplayStep], el_budget: int, char_budget: int,
) -> int:
    """Steps displayable in one complete-card panel: as many as the dual budgets allow; floor 1 (no empty panel), ceiling 50."""
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
    # Collapsed title: with a running step show the action label (📖 Reading foo.md)
    # so users see what the agent is doing without expanding; otherwise (empty or
    # all-done) fall back to the "🛠️ Tool use · N steps" skeleton. Labels carry
    # their own emoji, so the running state adds no 🛠️ prefix. Failures must be
    # visible while collapsed (⚠️ N failed + red title).
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
        # Step cap: render only the most recent N steps (running steps are always last);
    # the title count stays at the full total
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
            "content": f"…({hidden} earlier steps collapsed, {total_steps} total)…",
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


# Reasoning excerpt cap: Feishu cards have no scrollable component; a full
# thinking stream overflows the card (the tool panel's 200860 lesson), so the
# collapsed region keeps head/tail excerpts plus a total marker
_REASONING_HEAD_CHARS = 600
_REASONING_TAIL_CHARS = 300


def cap_reasoning_text(text: str, *, head: int = _REASONING_HEAD_CHARS,
                       tail: int = _REASONING_TAIL_CHARS) -> str:
    """Truncate long thinking to head/tail excerpts; short text passes through (streaming and complete re-render share this basis)."""
    if len(text) <= head + tail + 80:
        return text
    omitted = len(text) - head - tail
    return (f"{text[:head]}\n\n…(thinking was {len(text)} chars, "
            f"{omitted} omitted)…\n\n{text[-tail:]}")


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
    # Capped excerpts are always ≤ head+tail+marker, so one markdown element
    # suffices; the old 2400-char chunking served uncapped full text and is gone
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
    """Grey compact rendering for non-conversational notices (bg watcher completions); truncated so long output cannot flood the card."""
    compact = " ".join(text.split())
    if len(compact) > 300:
        compact = compact[:300] + "…"
    compact = compact.replace("[", "\\[").replace("]", "\\]")
    return f"<font color='grey'>{compact}</font>"


_MODEL_BUTTONS_PER_ROW = 3
MAX_FAVORITE_MODELS = 8
# Candidate-pool button cap: many providers fully tiled would push the card
# past Feishu's size limit; the favorites list itself is uncapped (added items
# always display)
MAX_ADMIN_CANDIDATES = 60


def _model_button_rows(buttons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Buttons arranged three per action row (one group per row in the action container)."""
    return [{"tag": "action", "actions": buttons[i:i + _MODEL_BUTTONS_PER_ROW]}
            for i in range(0, len(buttons), _MODEL_BUTTONS_PER_ROW)]


def build_model_picker_card(current: str, models: list[str]) -> dict[str, Any]:
    """Model picker card (a native interactive message, not a cardkit entity) —
    sent after a 🧠⇄ click.

    The native message path supports action containers (like clarify), so button
    names render reliably; a selection synthesizes `/model <name>` via callback.
    The current model is ✅-marked; the trailing ⚙ row opens the management card
    (favorites persist across turns; empty falls back to default+fallback)."""
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
         "text": {"tag": "plain_text", "content": "⚙ Manage favorites"},
         "type": "default",
         "value": {"hermes_model_action": "admin"}},
    ]})
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text",
                             "content": f"🧠 Switch model (current: {current})"},
                   "template": "blue"},
        "elements": elements,
    }


def build_model_admin_card(current: str, favorites: list[str],
                           providers: list[dict[str, Any]],
                           notice: str = "") -> dict[str, Any]:
    """Favorite-model management card — sent after the picker's ⚙ click; toggles
    persist immediately.

    providers = [{"slug","name","models"}] (same shape as list_picker_providers).
    Favorites get their own leading section (including manually-configured items
    absent from the pool, so they can be removed); the pool is grouped by
    provider with ✅ on favorites. Every button toggles and refreshes the card
    in place (_card_response replacement, like clarify/switch acks)."""
    fav_set = set(favorites)
    merged: list[tuple[str, list[str]]] = []
    if favorites:
        merged.append(("★ Favorites (click to remove)", list(favorites)))
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
    hint = ("Click a model to add/remove favorites (✅ = favorited; saved on click). "
            f"Up to {MAX_FAVORITE_MODELS} favorites; the picker (footer 🧠⇄) lists only favorites.")
    if truncated:
        hint += f" Large pool: only the first {MAX_ADMIN_CANDIDATES} addable items are listed."
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
                         "content": "Candidate pool is empty (provider catalog unavailable) — check "
                                    "credentials or set streaming.footer.model_cycle in config.yaml."})
    elements.append({"tag": "action", "actions": [
        {"tag": "button", "text": {"tag": "plain_text", "content": "✅ Done"},
         "type": "primary", "value": {"hermes_model_action": "admin_done"}},
    ]})
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text",
                             "content": f"⚙ Manage favorites (current: {current or 'unknown'})"},
                   "template": "blue"},
        "elements": elements,
    }


def build_model_switch_ack_card(model: str) -> dict[str, Any]:
    """Synchronous acknowledgement card after a picker selection (replaces the picker card)."""
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"✅ Switched to {model}"},
                   "template": "green"},
        "elements": [{"tag": "markdown",
                      "content": "Applies from the next turn (the footer shows the active model)."}],
    }


def build_model_admin_done_card(favorites: list[str]) -> dict[str, Any]:
    """Closing card after the management card's ✅ Done (replaces it in place)."""
    names = ", ".join(favorites) if favorites else "(empty — picker falls back to the default list)"
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "✅ Favorites updated"},
                   "template": "green"},
        "elements": [{"tag": "markdown",
                      "content": f"Favorites ({len(favorites)}): {names}\n"
                                 "Tap 🧠⇄ on any completed card's footer to pick from this list."}],
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
            # Short answers commonly land at 0.x t/s — flooring to 0 looks broken
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
    # Expanded-state hint so an opened panel is never blank (markdown elements
    # carry no i18n_content; English-first like the running action labels)
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
    """CardKit 2.0 streaming placeholder — tool panel + streaming + loading elements."""
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
    # The heartbeat line sits after loading = the true bottom (insert_before on new
    # content cannot displace it). Empty placeholder until the first heartbeat;
    # the complete rebuild at turn end omits it.
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
    """Completed streaming card — rendered in segment order."""
    elements: list[dict] = []
    has_answer = False

    # Tool-step display budget: subtract estimated non-tool elements and footer,
    # spread the rest across tool panels under the element/character ceilings
    # (streaming hard-caps at 15; the complete card relaxes up to 50)
    tool_el_budget = _COMPLETE_ELEMENT_BUDGET - 4  # base slack
    if footer_enabled:
        tool_el_budget -= 4  # hr + footer text + button column
    for seg in segments:
        if seg.type == SegmentType.REASONING:
            tool_el_budget -= 4
        elif seg.type == SegmentType.ANSWER:
            tool_el_budget -= len(seg.text) // _MAX_CHUNK_CHARS + 1
        elif seg.type == SegmentType.NOTICE:
            tool_el_budget -= 1
        elif seg.type == SegmentType.TOOL and show_tool_use:
            tool_el_budget -= 3  # panel shell (panel + header children)
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

    # "Done." placeholder when there is no answer — cards carrying a NOTICE
    # (redirect closures / background notices) already state their outcome, and an
    # extra Done. would break the "see the new card below" pointer
    if not has_answer and not any(seg.type == SegmentType.NOTICE for seg in segments):
        elements.append({"tag": "markdown", "content": _T["done"][0], "text_size": body_text_size})

    # image_generate artifacts (markdown image syntax ![alt](img_key), the render
    # path ImageResolver already proved)
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
        # Minimal trailing button (🧠⇄, tiny): click → the bot sends a native picker
        # card (interactive message with the clarify-style action container, where
        # button names render reliably — v2 select_static option names render
        # blank on some clients and were dropped). The value rides behaviors —
        # v2 card entities reject the action container (200861).
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
                        # weighted column stretches; the button column hugs the right edge
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
        # No-answer cards (early redirect closure): the session-list summary uses the
        # restart hint so voided cards are recognizable at a glance
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
    """Format an ISO timestamp for display; pass through unchanged on failure."""
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
    """Minimal static card for cron pushes — schema 2.0, optional header + markdown + images.

    ``template`` is the header color (a Feishu template name) — failure notices
    pass "red".
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
    # image_generate artifacts (markdown image syntax, as in build_complete_card)
    for img_key in (image_keys or []):
        card["body"]["elements"].append({"tag": "markdown", "content": f"![image]({img_key})"})
    return card


def build_background_card(preview: str, content: str) -> dict[str, Any]:
    """Background-task completion card — schema 2.0, header + markdown."""
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
