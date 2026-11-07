"""A stored conversation as readable entries for the classic CLI's subagent viewer.

``build_blocks`` turns the rows ``SessionDB.get_messages_as_conversation`` returns into entries a
person can scan: a subagent's brief, what it said, each tool call as one line (its result shown
when expanded), each ``delegate_task`` task with its outcome, timeline events and the final
result. ``render`` lays the entries out as prompt_toolkit fragments, one list per screen line,
wrapped to the window width. Stored text is untrusted: every string is sanitized before display.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from prompt_toolkit.utils import get_cwidth

from agent.i18n import t

# A tool result is shown in full only up to this many characters; the rest is elided.
_MAX_RESULT_CHARS = 6000
_TEXT_KINDS = frozenset({"brief", "user", "assistant", "final"})


@dataclass
class Block:
    kind: str  # brief | user | assistant | tool | delegation | event | final
    title: str = ""
    body: str = ""
    collapsible: bool = False
    tool_name: str = ""
    goal: str = ""
    outcome: str = ""
    link: Optional[str] = None  # a delegation task's child session id, resolved by the viewer
    call_id: str = ""
    task_index: int = 0


@dataclass
class Rendered:
    lines: list = field(default_factory=list)  # one list of (style, text) fragments per screen line
    starts: list = field(default_factory=list)  # first line index of each block


def _clean(text: Any) -> str:
    from tools.ansi_strip import sanitize_display_text
    return sanitize_display_text("" if text is None else str(text))


def _assistant_text(content: Any) -> str:
    from hermes_cli.cli_render import _assistant_content_as_text, _strip_reasoning_tags
    return _clean(_strip_reasoning_tags(_assistant_content_as_text(content))).strip()


def _user_text(content: Any) -> str:
    if isinstance(content, list):
        return _clean("\n".join(str(p.get("text") or "") for p in content if isinstance(p, dict))).strip()
    return _clean(content).strip()


_EVENT_BODY_CHARS = 1200


def _event_body(kind: str, message: dict) -> str:
    """What a timeline event says: a steer's own text (without the delivery marker), otherwise
    the row's stored text, capped. Events that only carry a display line have no body."""
    from agent.prompt_builder import STEER_DISPLAY_KIND, STEER_MARKER_CLOSE, STEER_MARKER_OPEN
    text = _user_text(message.get("content"))
    if kind == STEER_DISPLAY_KIND:
        text = text.replace(STEER_MARKER_OPEN, "").replace(STEER_MARKER_CLOSE, "").strip()
    if len(text) > _EVENT_BODY_CHARS:
        text = text[:_EVENT_BODY_CHARS].rstrip() + "…"
    return text


def _loads(value: Any, depth: int = 3) -> Any:
    """Decode stored JSON that may have been encoded more than once (delegate_task results are)."""
    while isinstance(value, str) and depth:
        depth -= 1
        try:
            value = json.loads(value)
        except ValueError:
            break
    return value


def _tool_calls(message: dict) -> list[dict]:
    calls = _loads(message.get("tool_calls"), 1)
    return [c for c in calls if isinstance(c, dict)] if isinstance(calls, list) else []


def _call_parts(call: dict) -> tuple[str, str, dict]:
    function = call.get("function") or {}
    args = _loads(function.get("arguments"), 1)
    return str(call.get("id") or call.get("call_id") or ""), str(function.get("name") or "tool"), (
        args if isinstance(args, dict) else {})


def _tool_title(name: str, args: dict) -> str:
    from agent.display import build_tool_preview, get_tool_emoji
    try:
        preview = build_tool_preview(name, args, max_len=0)
    except Exception:  # health: allow BLE001 -- a preview is decoration; an odd argument shape must not hide the entry
        preview = None
    preview = " ".join(_clean(preview).split()) if preview else ""
    return f"{get_tool_emoji(name)} {name}" + (f": {preview}" if preview else "")


def _result_text(content: Any) -> str:
    data = _loads(content, 1)
    if isinstance(data, dict) and "output" in data:
        text = str(data.get("output") or "")
        if data.get("exit_code") not in (None, 0):
            text += f"\n[exit {data.get('exit_code')}]"
        if data.get("error"):
            text += f"\n{data['error']}"
    elif isinstance(data, (dict, list)):
        text = json.dumps(data, indent=2, ensure_ascii=False)
    else:
        text = "" if content is None else str(content)
    text = _clean(text).rstrip()
    if len(text) > _MAX_RESULT_CHARS:
        text = text[:_MAX_RESULT_CHARS] + "\n" + t("cli.subagent_view.result_truncated")
    return text


def _delegation_goals(args: dict) -> list[str]:
    tasks = args.get("tasks")
    if isinstance(tasks, list) and tasks:
        return [_clean(task.get("goal") if isinstance(task, dict) else task).strip() for task in tasks]
    return [_clean(args.get("goal")).strip()]


def delegation_title(goal: str, outcome: str) -> str:
    return t("cli.subagent_view.delegation_title", goal=" ".join(goal.split()),
             outcome=t(f"cli.subagent_view.outcome_{outcome}") if outcome in _OUTCOMES else outcome)


_OUTCOMES = frozenset({"running", "completed", "failed", "interrupted", "timeout", "error"})


def _apply_delegation_result(tasks: list[Block], content: Any) -> None:
    data = _loads(content)
    if not isinstance(data, dict) or data.get("status") == "dispatched":
        return  # a background dispatch handle: the result arrives later as its own event
    results = data.get("results")
    if not isinstance(results, list):
        results = [data]
    for entry in results:
        if not isinstance(entry, dict):
            continue
        index = entry.get("task_index", 0)
        if isinstance(index, int) and 0 <= index < len(tasks):
            task = tasks[index]
            task.outcome = str(entry.get("status") or "completed")
            task.body = _clean(entry.get("summary") or entry.get("error") or "").strip()
            task.title = delegation_title(task.goal, task.outcome)


def _assistant_blocks(message: dict, pending: dict) -> list[Block]:
    blocks = []
    text = _assistant_text(message.get("content"))
    if text:
        blocks.append(Block("assistant", body=text))
    for call in _tool_calls(message):
        call_id, name, args = _call_parts(call)
        if name == "delegate_task":
            tasks = [Block("delegation", title=delegation_title(goal, "running"), goal=goal, outcome="running",
                           collapsible=True, call_id=call_id, task_index=i)
                     for i, goal in enumerate(_delegation_goals(args))]
            pending[call_id] = tasks
            blocks.extend(tasks)
        else:
            block = Block("tool", title=_tool_title(name, args), tool_name=name, collapsible=True, call_id=call_id,
                          body=t("cli.subagent_view.waiting_for_result"))
            pending[call_id] = [block]
            blocks.append(block)
    return blocks


def build_blocks(messages: list[dict], *, subagent: bool) -> list[Block]:
    """Readable entries for ``messages``; ``subagent`` frames the first user message as the brief
    and the last assistant reply as the result."""
    blocks: list[Block] = []
    pending: dict[str, list[Block]] = {}
    for message in messages or []:
        role, kind = message.get("role"), message.get("display_kind")
        if kind == "hidden" or role in ("system", "session_meta"):
            continue
        if kind:
            metadata = message.get("display_metadata") or {}
            text = _clean(metadata.get("display_text") or kind.replace("_", " ")).strip()
            blocks.append(Block("event", title=" ".join(text.split()), body=_event_body(kind, message)))
        elif role == "user":
            first = subagent and not any(b.kind == "brief" for b in blocks)
            blocks.append(Block("brief" if first else "user", body=_user_text(message.get("content"))))
        elif role == "assistant":
            blocks.extend(_assistant_blocks(message, pending))
        elif role == "tool":
            # delegate_task results are applied per call below; other tools fill their own entry.
            for block in pending.pop(str(message.get("tool_call_id") or ""), []):
                if block.kind == "tool":
                    block.body = _result_text(message.get("content"))
    _link_delegation_results(blocks, messages)
    if subagent:
        last = next((b for b in reversed(blocks) if b.kind == "assistant"), None)
        if last is not None and last is blocks[-1]:
            last.kind = "final"
    return blocks


def _link_delegation_results(blocks: list[Block], messages: list[dict]) -> None:
    """Apply each delegate_task result to every task of its call (one result row per call)."""
    by_call: dict[str, list[Block]] = {}
    for block in blocks:
        if block.kind == "delegation":
            by_call.setdefault(block.call_id, []).append(block)
    for message in messages or []:
        if message.get("role") == "tool" and str(message.get("tool_call_id") or "") in by_call:
            _apply_delegation_result(by_call[str(message["tool_call_id"])], message.get("content"))


# ── layout ────────────────────────────────────────────────────────────────────────────────────


def wrap(text: str, width: int) -> list[str]:
    """Wrap ``text`` to ``width`` terminal cells, keeping its own line breaks."""
    width = max(8, width)
    out: list[str] = []
    for paragraph in text.split("\n"):
        line, cells = "", 0
        for word in paragraph.split(" "):
            word_cells = get_cwidth(word)
            if line and cells + 1 + word_cells > width:
                out.append(line)
                line, cells = "", 0
            while word_cells > width:  # a word longer than the line is split by cells
                head = ""
                for char in word:
                    if get_cwidth(head + char) > width - cells - (1 if line else 0):
                        break
                    head += char
                head = head or word[0]
                out.append((line + " " + head) if line else head)
                line, cells = "", 0
                word = word[len(head):]
                word_cells = get_cwidth(word)
            if line:
                line, cells = f"{line} {word}", cells + 1 + word_cells
            else:
                line, cells = word, word_cells
        out.append(line)
    return out


_LABELS = {"brief": "cli.subagent_view.label_brief", "user": "cli.subagent_view.label_user",
           "assistant": "cli.subagent_view.label_assistant", "final": "cli.subagent_view.label_result"}


def _header(block: Block, expanded: bool, subagent: bool) -> tuple[str, str]:
    if block.kind == "assistant" and subagent:
        return "class:subagent-view.label-assistant", t("cli.subagent_view.label_subagent")
    if block.kind in _TEXT_KINDS:
        return f"class:subagent-view.label-{block.kind}", t(_LABELS[block.kind])
    if block.kind == "event":
        return "class:subagent-view.event", f"◈ {block.title}"
    marker = "▾" if expanded else "▸"
    return f"class:subagent-view.{block.kind}", f"{marker} {block.title}"


def render(blocks: list[Block], *, width: int, expanded: set, cursor: Optional[int] = None,
           subagent: bool = False) -> Rendered:
    """Screen lines for ``blocks`` at ``width`` cells; ``expanded`` holds the indexes of open
    entries and ``cursor`` the selected one (its header line carries the cursor style)."""
    rendered = Rendered()
    width = max(16, width)
    for index, block in enumerate(blocks):
        if index:
            rendered.lines.append([])
        rendered.starts.append(len(rendered.lines))
        is_open = index in expanded
        style, header = _header(block, is_open, subagent)
        if index == cursor:
            style += " class:subagent-view.cursor"
        for line in wrap(header, width):
            rendered.lines.append([(style, line)])
        show_body = block.kind in _TEXT_KINDS or block.kind == "event" or (block.collapsible and is_open)
        if show_body and block.body:
            indent = "    " if block.collapsible else "  "
            body_style = "class:subagent-view.body" if block.kind in _TEXT_KINDS else "class:subagent-view.detail"
            for line in wrap(block.body, width - len(indent)):
                rendered.lines.append([(body_style, indent + line)])
    return rendered
