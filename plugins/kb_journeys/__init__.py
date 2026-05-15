"""Telegram KB journey renderer plugin.

Intercepts a small set of Telegram slash commands and renders concise,
read-only KB status summaries from the configured KB MCP target.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import inspect
import json
import logging
import os
import re
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

DEFAULT_MCP_TARGET = "kb_engine_prod"
MENU_COMMANDS = {"kb"}
LEGACY_COMMANDS = {"kbtoday", "kbstatus", "kbruns", "kbqueue", "kbreview", "kbrun"}
SUPPORTED_COMMANDS = MENU_COMMANDS


def _sanitize_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(value or ""))


def _mcp_target() -> str:
    return os.getenv("HERMES_KB_MCP_TARGET", DEFAULT_MCP_TARGET).strip() or DEFAULT_MCP_TARGET


def _mcp_tool_name(target: str, tool_name: str) -> str:
    return f"mcp_{_sanitize_component(target)}_{_sanitize_component(tool_name)}"


def _platform_name(platform: Any) -> str:
    return str(getattr(platform, "value", platform) or "").lower()


def _command_from_text(text: str) -> str | None:
    stripped = (text or "").strip()
    if not stripped.startswith("/"):
        return None
    token = stripped.split(maxsplit=1)[0][1:]
    command = token.split("@", 1)[0].lower()
    return command if command in MENU_COMMANDS or command in LEGACY_COMMANDS else None


def _command_args_from_text(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped.startswith("/"):
        return ""
    parts = stripped.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def _short(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value).strip()
    return text if text else default


def _clip(value: Any, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", _short(value, "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return value
    try:
        return json.loads(stripped)
    except Exception:
        return value


def _unwrap_tool_result(raw: Any) -> tuple[Any | None, str | None]:
    parsed = _maybe_json(raw)
    if not isinstance(parsed, dict):
        return parsed, None
    if parsed.get("error"):
        return None, _short(parsed.get("error"))
    payload = parsed.get("structuredContent")
    if payload is None:
        payload = parsed.get("result", parsed)
    payload = _maybe_json(payload)
    return payload, None


def _dispatch_first(
    ctx: Any,
    target: str,
    candidates: Iterable[tuple[str, dict[str, Any]]],
) -> tuple[str | None, Any | None, list[str]]:
    errors: list[str] = []
    for kb_tool, args in candidates:
        registry_name = _mcp_tool_name(target, kb_tool)
        try:
            payload, error = _unwrap_tool_result(ctx.dispatch_tool(registry_name, args))
        except Exception as exc:
            errors.append(f"{registry_name}: {exc}")
            continue
        if error:
            errors.append(f"{registry_name}: {error}")
            continue
        return registry_name, payload, errors
    return None, None, errors


def _get_path(data: Any, *path: str, default: Any = None) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def _count_from(data: Any, *keys: str) -> Any:
    for key in keys:
        value = _get_path(data, key, "count")
        if value is not None:
            return value
        if isinstance(data, dict) and data.get(key) is not None and not isinstance(data.get(key), dict):
            found = data.get(key)
            return len(found) if isinstance(found, list) else found
    return None


def _readiness_status(data: dict[str, Any]) -> Any:
    return (
        _get_path(data, "summary", "readiness_status")
        or _get_path(data, "sections", "readiness", "summary", "status")
        or _get_path(data, "sections", "readiness", "payload", "status")
        or _get_path(data, "readiness", "status")
        or _get_path(data, "readiness", "state")
        or data.get("readiness")
    )


def _publication_status(data: dict[str, Any]) -> Any:
    return (
        _get_path(data, "summary", "publication_status")
        or _get_path(data, "sections", "publication", "summary", "status")
        or _get_path(data, "sections", "publication", "payload", "status")
        or _get_path(data, "publication", "status")
        or _get_path(data, "publication", "state")
        or data.get("publication")
    )


def _item_title(item: Any) -> str:
    if isinstance(item, dict):
        return _short(
            item.get("title")
            or item.get("name")
            or item.get("summary")
            or item.get("id")
            or item.get("proposal_id"),
            "item",
        )
    return _short(item, "item")


def _summary_count(summary: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = summary.get(key)
        if value is not None:
            return value
    return None


def _proposal_count_from_summary(summary: dict[str, Any]) -> Any:
    proposal_count = _summary_count(
        summary,
        "proposal_queue_count",
        "pending_proposal_count",
        "proposal_count",
        "review_proposal_count",
    )
    if proposal_count is not None:
        return proposal_count
    legacy_queue_count = summary.get("queue_item_count")
    todo_count = _summary_count(summary, "active_todo_count", "triage_todo_count", "task_queue_count")
    if legacy_queue_count is not None and todo_count is not None and legacy_queue_count != todo_count:
        return legacy_queue_count
    return None


def _todo_count_from_summary(summary: dict[str, Any]) -> Any:
    return _summary_count(summary, "active_todo_count", "triage_todo_count", "task_queue_count")


def _display_text(value: Any) -> str:
    text = _short(value, "")
    if text == "Review prioritized queue items through workbench.queue.":
        return "Review prioritized TODO items; use /kb queue for proposal review."
    return text


def _dashboard_section_title(section: dict[str, Any], summary: dict[str, Any]) -> str:
    title = _short(section.get("title") or section.get("id"), "Section")
    key = title.strip().lower()
    section_id = str(section.get("id") or "").strip().lower()
    if key == "queue" or section_id == "queue":
        proposal_count = _proposal_count_from_summary(summary)
        todo_count = _todo_count_from_summary(summary)
        legacy_queue_count = summary.get("queue_item_count")
        if proposal_count is None or (todo_count is not None and legacy_queue_count == todo_count):
            return "TODO Focus"
        return "Proposal Queue"
    return title


def _items(data: Any, *paths: tuple[str, ...]) -> list[Any]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for path in paths:
        found = _get_path(data, *path)
        if isinstance(found, list) and found:
            return found
    for key in ("items", "proposals", "queue", "runs", "recent", "active"):
        found = data.get(key)
        if isinstance(found, list) and found:
            return found
    return []


def _public_error(errors: list[str]) -> str:
    if not errors:
        return "No compatible KB MCP tool responded."
    detail = errors[-1]
    if detail.startswith("mcp_") and ": " in detail:
        detail = detail.split(": ", 1)[1]
    return detail or "No compatible KB MCP tool responded."


def _render_error(title: str, target: str, errors: list[str]) -> dict[str, Any]:
    detail = _public_error(errors)
    text = f"{title}\nMCP target: {target}\nKB data is not available yet.\n{detail}"
    return {"title": title, "text": text, "actions": []}


def _render_today(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        text = f"KB Today\n{_short(data, 'No cockpit details returned.')}"
        return {"title": "KB Today", "text": text, "actions": []}

    readiness = _short(_readiness_status(data))
    publication = _short(_publication_status(data))
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    queue_count = _proposal_count_from_summary(summary)
    if queue_count is None:
        queue_count = _count_from(data, "proposals", "proposal_queue")
    todo_count = _count_from(data, "todo", "todos")
    if todo_count is None:
        todo_count = _todo_count_from_summary(summary)

    active_runs = _items(data, ("runs", "active"), ("active_runs",))
    recent_runs = _items(data, ("runs", "recent"), ("recent_runs",))
    run_bits: list[str] = []
    for run in active_runs[:2]:
        if isinstance(run, dict):
            run_bits.append(f"{_item_title(run)} {_short(run.get('status') or run.get('state'))}")
        else:
            run_bits.append(_short(run))
    for run in recent_runs[:1]:
        if isinstance(run, dict):
            run_bits.append(f"recent {_item_title(run)} {_short(run.get('status') or run.get('state'))}")
        else:
            run_bits.append(f"recent {_short(run)}")

    next_actions = _items(data, ("next_actions",), ("actions",))
    lines = [
        "KB Today",
        f"Readiness: {readiness}",
        f"Publication: {publication}",
    ]
    if queue_count is not None or todo_count is not None:
        count_bits = []
        if queue_count is not None:
            count_bits.append(f"Proposals: {_short(queue_count, 'unknown')}")
        if todo_count is not None:
            count_bits.append(f"TODOs: {_short(todo_count, 'unknown')}")
        lines.append(" · ".join(count_bits))
    if run_bits:
        lines.append("Runs: " + " · ".join(run_bits[:3]))
    if next_actions:
        lines.append("Next: " + "; ".join(_item_title(a) for a in next_actions[:3]))
    return {"title": "KB Today", "text": "\n".join(lines), "actions": []}


def _render_dashboard(data: Any, *, ctx: Any, target: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"title": "KB Dashboard", "text": f"KB Dashboard\n{_short(data, 'No dashboard details returned.')}", "actions": []}

    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    readiness = _short(
        summary.get("readiness_status")
        or _readiness_status(data)
    )
    publication = _short(
        summary.get("publication_status")
        or _publication_status(data)
    )
    sections = data.get("sections") if isinstance(data.get("sections"), list) else []
    queue_count = _proposal_count_from_summary(summary)
    todo_count = _todo_count_from_summary(summary)
    active_runs = summary.get("active_run_count")
    lines = [
        "KB Dashboard",
        f"Readiness: {readiness}",
        f"Publication: {publication}",
    ]
    counts: list[str] = []
    if queue_count is not None:
        counts.append(f"Proposals {queue_count}")
    if todo_count is not None:
        counts.append(f"TODOs {todo_count}")
    if active_runs is not None:
        counts.append(f"Runs {active_runs}")
    if counts:
        lines.append(" · ".join(counts))
    for section in sections[:4]:
        if not isinstance(section, dict):
            continue
        cards = section.get("cards") if isinstance(section.get("cards"), list) else []
        if not cards:
            continue
        lines.append("")
        lines.append(_dashboard_section_title(section, summary))
        for card in cards[:3]:
            if not isinstance(card, dict):
                continue
            detail = _display_text(card.get("detail"))
            suffix = f" — {detail}" if detail else ""
            lines.append(f"- {_display_text(card.get('title') or 'item')}{suffix}")
    next_actions = data.get("next_actions") if isinstance(data.get("next_actions"), list) else []
    if next_actions and not any(
        isinstance(section, dict) and str(section.get("id") or "").strip().lower() == "next"
        for section in sections
    ):
        lines.append("")
        lines.append("Next Actions")
        for action in next_actions[:3]:
            lines.append(f"- {_display_text(action)}")
    warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
    if warnings:
        lines.append("")
        lines.append(f"Warnings: {len(warnings)}")
    refresh = data.get("refresh") if isinstance(data.get("refresh"), dict) else {}
    if refresh:
        lines.append(f"Refresh: every {_short(refresh.get('ttl_seconds'), '60')}s target")
    lines.append("")
    lines.append("Commands: /kb queue · /kb status · /kb runs · /kb today")
    return {"title": "KB Dashboard", "text": "\n".join(lines), "actions": []}


def _config_snapshot() -> dict[str, str]:
    config: dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config

        loaded = load_config()
        if isinstance(loaded, dict):
            config = loaded
    except Exception:
        config = {}

    model_cfg = config.get("model")
    agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
    if isinstance(model_cfg, dict):
        model = model_cfg.get("default") or model_cfg.get("name") or model_cfg.get("model")
        provider = model_cfg.get("provider")
        api_mode = model_cfg.get("api_mode")
        reasoning = (
            agent_cfg.get("reasoning_effort")
            or model_cfg.get("reasoning_effort")
            or model_cfg.get("reasoning")
        )
    else:
        model = model_cfg
        provider = config.get("provider")
        api_mode = None
        reasoning = None

    provider = provider or os.getenv("HERMES_PROVIDER") or os.getenv("MODEL_PROVIDER")
    model = model or os.getenv("HERMES_MODEL") or os.getenv("MODEL")
    reasoning = reasoning or os.getenv("HERMES_REASONING_EFFORT") or os.getenv("OPENAI_REASONING_EFFORT")
    api_mode = api_mode or os.getenv("HERMES_MODEL_API_MODE") or os.getenv("HERMES_API_MODE")

    api_envs = [
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
    ]
    configured = [name.removesuffix("_API_KEY") for name in api_envs if os.getenv(name)]

    return {
        "lane": os.getenv("HERMES_KB_MODE") or os.getenv("HERMES_KB_LANE") or os.getenv("HERMES_PROFILE") or "unknown",
        "environment": (
            os.getenv("HERMES_ENVIRONMENT")
            or os.getenv("HERMES_ENV")
            or os.getenv("ENVIRONMENT")
            or os.getenv("HERMES_PROFILE")
            or "unknown"
        ),
        "workspace": os.getenv("HERMES_KB_WORKSPACE") or os.getenv("KB_WORKSPACE") or "not set",
        "model": _short(model, "not set"),
        "provider": _short(provider, "not set"),
        "api_mode": _short(api_mode, "not set"),
        "api": ", ".join(configured) if configured else "not detected",
        "reasoning": _short(reasoning, "not set"),
    }


def _render_status(data: Any, target: str) -> dict[str, Any]:
    snap = _config_snapshot()
    readiness = "unknown"
    publication = "unknown"
    if isinstance(data, dict):
        readiness = _short(_readiness_status(data))
        publication = _short(_publication_status(data))
    lines = [
        "KB Status",
        f"Lane: {snap['lane']}",
        f"Environment: {snap['environment']}",
        f"MCP target: {target}",
        f"Workspace: {snap['workspace']}",
        f"Model: {snap['model']}",
        f"Provider/API: {snap['provider']} / {snap['api_mode']} / {snap['api']}",
        f"Reasoning: {snap['reasoning']}",
        f"Readiness: {readiness}",
        f"Publication: {publication}",
    ]
    return {"title": "KB Status", "text": "\n".join(lines), "actions": []}


def _render_runs(data: Any) -> dict[str, Any]:
    if isinstance(data, str):
        return {"title": "KB Runs", "text": f"KB Runs\n{data}", "actions": []}
    if isinstance(data, dict):
        active = _items(data, ("active",), ("runs", "active"))
        recent = _items(data, ("recent",), ("runs", "recent"))
        runs = [*active, *recent] or _items(data, ("runs",))
    else:
        runs = []
    lines = ["KB Runs"]
    if not runs:
        lines.append("No active or recent run details returned.")
    for idx, run in enumerate(runs[:6], start=1):
        if isinstance(run, dict):
            status = _short(run.get("status") or run.get("state") or run.get("phase"))
            detail = _short(run.get("summary") or run.get("message") or run.get("updated_at"), "")
            suffix = f" - {detail}" if detail else ""
            lines.append(f"{idx}. {_item_title(run)}: {status}{suffix}")
        else:
            lines.append(f"{idx}. {_short(run)}")
    return {"title": "KB Runs", "text": "\n".join(lines), "actions": []}


def _proposal_ids_for_item(item: Any) -> list[str]:
    if not isinstance(item, dict):
        return []
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    proposal_ids = raw.get("proposal_ids") or item.get("proposal_ids") or []
    if isinstance(proposal_ids, str):
        proposal_ids = [proposal_ids]
    return [str(pid).strip() for pid in proposal_ids if str(pid).strip()]


def _result_payload(raw: Any) -> Any:
    payload, error = _unwrap_tool_result(raw)
    if error:
        return {"error": error}
    return payload


def _item_kind(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    return _short(item.get("kind") or item.get("type") or raw.get("kind") or raw.get("type"), "")


def _item_target(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    return _short(
        item.get("entity_path")
        or item.get("target")
        or item.get("item_id")
        or raw.get("entity_path")
        or raw.get("target")
        or raw.get("item_id"),
        "",
    )


def _item_detail(item: Any) -> str:
    if not isinstance(item, dict):
        return _short(item, "")
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    return _short(
        item.get("preview")
        or item.get("why")
        or item.get("summary")
        or item.get("description")
        or item.get("detail")
        or raw.get("preview")
        or raw.get("why")
        or raw.get("summary")
        or raw.get("description")
        or raw.get("detail"),
        "",
    )


def _safe_actions_for_item(item: Any) -> list[dict[str, Any]]:
    if not isinstance(item, dict):
        return []
    actions = item.get("safe_actions")
    if not isinstance(actions, list):
        return []
    return [action for action in actions if isinstance(action, dict)]


def _queue_action_decisions(item: dict[str, Any]) -> list[tuple[str, str]]:
    decisions: list[tuple[str, str]] = []
    seen: set[str] = set()
    for action in _safe_actions_for_item(item):
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        decision = str(params.get("decision") or "").strip().lower()
        if not decision or decision in seen:
            continue
        label = _short(action.get("label") or decision.replace("_", " ").title(), "")
        if not label:
            continue
        seen.add(decision)
        decisions.append((decision, label))
    if any(decision in {"complete", "keep", "demote"} for decision, _ in decisions):
        order = {"complete": 0, "keep": 1, "demote": 2, "archive": 3, "skip": 4}
    else:
        order = {"reject": 0, "archive": 1, "approve": 2, "skip": 3}
    decisions.sort(key=lambda pair: (order.get(pair[0], 99), pair[0]))
    return decisions


def _queue_decision_commands(item: dict[str, Any], *, index: int) -> list[str]:
    decisions = _queue_action_decisions(item)
    if not decisions:
        decisions = [
            ("reject", "Reject"),
            ("archive", "Archive"),
            ("approve", "Approve"),
        ]
    lines: list[str] = []
    for decision, label in decisions:
        if decision == "approve":
            label = "Approve proposal"
        elif decision == "reject":
            label = "Reject proposal"
        elif decision == "archive" and not any(d in {"complete", "keep", "demote"} for d, _ in decisions):
            label = "Archive proposal"
        lines.append(f"- {label}: /kb queue {decision} {index}")
    if decisions:
        example_decision = decisions[0][0]
        lines.append(f"Confirm after preview: /kb queue {example_decision} {index} confirm")
    return lines


def _queue_item_text(item: dict[str, Any], *, index: int) -> str:
    proposal_ids = _proposal_ids_for_item(item)
    lines = [
        f"Queue Item {index}",
        f"Title: {_item_title(item)}",
    ]
    kind = _item_kind(item)
    target = _item_target(item)
    detail = _item_detail(item)
    if kind:
        lines.append(f"Type: {kind}")
    if target:
        lines.append(f"Target: {target}")
    if detail:
        lines.append("")
        lines.append("Summary: " + _clip(detail, 420))
    if proposal_ids:
        lines.append("")
        lines.append(f"Proposal ids: {', '.join(proposal_ids[:5])}")
        lines.append("Available actions:")
        lines.extend(_queue_decision_commands(item, index=index))
    else:
        lines.append("")
        lines.append("This item did not include proposal ids, so Telegram cannot apply a decision yet. Use the KB workbench for details.")
    return "\n".join(lines)


def _selection_lines(selection: list[tuple[int, dict[str, Any]]]) -> list[str]:
    lines: list[str] = []
    for index, item in selection:
        lines.append(f"{index}. {_item_title(item)}")
        target = _item_target(item)
        kind = _item_kind(item)
        detail = _item_detail(item)
        if target:
            lines.append(f"   Target: {target}")
        if kind:
            lines.append(f"   Type: {kind}")
        if detail:
            lines.append(f"   Summary: {_clip(detail, 180)}")
    return lines


def _format_indices(indices: list[int]) -> str:
    return ",".join(str(index) for index in indices)


def _proposal_ids_for_selection(selection: list[tuple[int, dict[str, Any]]]) -> list[str]:
    proposal_ids: list[str] = []
    seen: set[str] = set()
    for _, item in selection:
        for proposal_id in _proposal_ids_for_item(item):
            if proposal_id not in seen:
                seen.add(proposal_id)
                proposal_ids.append(proposal_id)
    return proposal_ids


def _preview_text(
    decision: str,
    proposal_ids: list[str],
    payload: Any,
    *,
    selection: list[tuple[int, dict[str, Any]]] | None = None,
    item: dict[str, Any] | None = None,
) -> str:
    if selection is None:
        selection = [(0, item)] if isinstance(item, dict) else []
    if isinstance(payload, dict) and payload.get("error"):
        return f"Queue {decision} preview failed\n{payload['error']}"
    if isinstance(payload, dict):
        status = _short(payload.get("status"))
        ok = _short(payload.get("ok"))
        preview = payload.get("preview") if isinstance(payload.get("preview"), dict) else {}
        summary = _short(
            preview.get("summary")
            or _get_path(payload, "plan", "summary")
            or f"{decision.title()} {len(proposal_ids)} proposal(s).",
        )
        lines = [f"Queue {decision} preview"]
        if selection:
            lines.append(f"Items: {len(selection)}")
            lines.extend(_selection_lines(selection))
        lines.extend(
            [
                f"Status: {status} · ok: {ok}",
                f"Proposal ids: {', '.join(proposal_ids[:5])}",
                "Plan: " + _clip(summary, 260),
                "Confirm only if this item and decision match what you intend.",
            ]
        )
        return "\n".join(lines)
    lines = [f"Queue {decision} preview"]
    if selection:
        lines.extend(_selection_lines(selection))
    lines.append(f"Proposal ids: {', '.join(proposal_ids[:5])}")
    return "\n".join(lines)


def _preview_allows_confirmation(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("error") or payload.get("isError"):
        return False
    if payload.get("ok") is False:
        return False
    status = str(payload.get("status") or payload.get("state") or "").strip().lower()
    if status in {
        "blocked",
        "error",
        "failed",
        "operator_blocked",
        "validation_failed",
    }:
        return False
    return True


def _git_summary(git_state: dict[str, Any]) -> str:
    after = git_state.get("after") if isinstance(git_state.get("after"), dict) else {}
    before = git_state.get("before") if isinstance(git_state.get("before"), dict) else {}
    branch = _short(after.get("branch") or git_state.get("branch") or before.get("branch"), "")
    changed = after.get("changed_count")
    if changed is None and isinstance(after.get("changes"), list):
        changed = len(after["changes"])
    if changed is None:
        changed = git_state.get("changed_count")
    if changed is not None:
        suffix = f" on {branch}" if branch else ""
        return f"{changed} changed path(s){suffix}"
    return _short(git_state.get("summary") or git_state.get("status"), "")


def _decision_past_tense(decision: str) -> str:
    return {
        "approve": "Approved",
        "reject": "Rejected",
        "archive": "Archived",
        "complete": "Completed",
        "keep": "Kept unchanged",
        "demote": "Demoted",
        "skip": "Skipped",
    }.get(decision, f"{decision.title()}ed")


def _confirmed_text(
    decision: str,
    payload: Any,
    *,
    selection: list[tuple[int, dict[str, Any]]] | None = None,
    proposal_ids: list[str] | None = None,
) -> str:
    if isinstance(payload, dict) and payload.get("error"):
        return f"Queue {decision} failed\n{payload['error']}"
    selection = selection or []
    proposal_ids = proposal_ids or []
    past_tense = _decision_past_tense(decision)
    if isinstance(payload, dict):
        publication = payload.get("publication") if isinstance(payload.get("publication"), dict) else {}
        git_state = payload.get("git") if isinstance(payload.get("git"), dict) else {}
        lines = [
            f"Queue {decision.title()} Applied",
            f"{past_tense} {len(proposal_ids) or len(selection)} proposal(s).",
        ]
        if selection:
            lines.append("")
            lines.append("Changed:")
            lines.extend(_selection_lines(selection))
        if proposal_ids:
            lines.append("")
            lines.append(f"Proposal ids: {', '.join(proposal_ids[:8])}")
        lines.extend(
            [
                f"Status: {_short(payload.get('status'))} · ok: {_short(payload.get('ok'))}",
            ]
        )
        if publication:
            lines.append(
                "Publication: "
                + _short(publication.get("status") or publication.get("state") or publication.get("reason"))
            )
        if git_state:
            git_summary = _git_summary(git_state)
            if git_summary:
                lines.append("Git: " + git_summary)
        lines.append("Next: /kb queue")
        return "\n".join(lines)
    lines = [
        f"Queue {decision.title()} Applied",
        f"{past_tense} {len(proposal_ids) or len(selection)} proposal(s).",
    ]
    if selection:
        lines.extend(["", "Changed:", *_selection_lines(selection)])
    lines.append("Next: /kb queue")
    return "\n".join(lines)


def _queue_summary_payload(ctx: Any, target: str, *, limit: int = 5) -> tuple[Any | None, list[str]]:
    _, data, errors = _dispatch_first(
        ctx,
        target,
        [
            ("queue.summary", {"scope": "proposals", "limit": limit}),
            ("queue.preview", {"limit": limit}),
            ("workbench.queue", {"scope": "proposals", "limit": limit}),
        ],
    )
    return data, errors


def _changed_paths(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    paths = payload.get("changed_paths")
    if paths is None:
        paths = _get_path(payload, "publication", "changed_paths")
    if isinstance(paths, str):
        return [paths]
    if isinstance(paths, list):
        return [str(path).strip() for path in paths if str(path).strip()]
    return []


def _format_changed_paths(paths: list[str], *, limit: int = 10) -> list[str]:
    if not paths:
        return []
    lines = [f"- {path}" for path in paths[:limit]]
    remaining = len(paths) - limit
    if remaining > 0:
        lines.append(f"- ... {remaining} more")
    return lines


def _publish_args(args: str) -> tuple[bool, str]:
    parts = (args or "").strip().split()
    confirm = any(part.lower() in {"confirm", "confirmed", "apply", "publish", "push"} for part in parts)
    message_parts = [part for part in parts if part.lower() not in {"confirm", "confirmed", "apply", "publish", "push"}]
    if message_parts and message_parts[0].lower() in {"message", "msg"}:
        message_parts = message_parts[1:]
    message = " ".join(message_parts).strip() or "Publish KB update"
    return confirm, message


def _publication_git_line(git_state: Any) -> str:
    if not isinstance(git_state, dict):
        return ""
    branch = _short(git_state.get("branch"), "")
    head = _short(git_state.get("head"), "")
    upstream = _short(git_state.get("upstream"), "")
    bits: list[str] = []
    if branch:
        bits.append(branch)
    if head:
        bits.append(head[:12])
    if upstream:
        bits.append(upstream)
    return " · ".join(bits)


def _render_publish_preview(payload: Any, *, confirm_hint: str = "/kb publish confirm") -> dict[str, Any]:
    if isinstance(payload, dict) and payload.get("error"):
        return {"title": "KB Publish", "text": f"KB Publish Preview Failed\n{payload['error']}", "actions": []}
    if not isinstance(payload, dict):
        return {"title": "KB Publish", "text": "KB Publish Preview Failed\nPublication preview returned an unexpected response.", "actions": []}
    changed_paths = _changed_paths(payload)
    status = _short(payload.get("status"))
    message = _short(payload.get("message"), "Publish KB update")
    git_line = _publication_git_line(payload.get("git"))
    if not changed_paths:
        lines = [
            "KB Publish Preview",
            "Nothing to publish.",
            f"Status: {status}",
            f"Message: {message}",
        ]
        if git_line:
            lines.append(f"Git: {git_line}")
        return {"title": "KB Publish", "text": "\n".join(lines), "actions": []}
    lines = [
        "KB Publish Preview",
        f"Status: {status}",
        f"Message: {message}",
        f"Changed paths: {len(changed_paths)}",
    ]
    if git_line:
        lines.append(f"Git: {git_line}")
    lines.append("")
    lines.extend(_format_changed_paths(changed_paths))
    lines.extend(
        [
            "",
            f"To publish: {confirm_hint}",
            "No commit or push has been made.",
        ]
    )
    return {"title": "KB Publish", "text": "\n".join(lines), "actions": []}


def _render_publish_result(preview: Any, commit: Any, push: Any) -> dict[str, Any]:
    changed_paths = _changed_paths(preview)
    if isinstance(commit, dict) and commit.get("error"):
        return {"title": "KB Publish", "text": f"KB Publish Failed\nCommit failed: {commit['error']}", "actions": []}
    if not isinstance(commit, dict):
        return {"title": "KB Publish", "text": "KB Publish Failed\nCommit returned an unexpected response.", "actions": []}
    commit_status = _short(commit.get("status"))
    commit_ok = bool(commit.get("ok"))
    if not commit_ok:
        reason = _short(commit.get("reason") or _get_path(commit, "publication", "reason"), "unknown")
        lines = [
            "KB Publish Blocked",
            f"Committed: {commit_status}",
            f"Reason: {reason}",
        ]
        if changed_paths:
            lines.append(f"Changed paths: {len(changed_paths)}")
            lines.extend(_format_changed_paths(changed_paths))
        lines.append("Next: /kb publish")
        return {"title": "KB Publish", "text": "\n".join(lines), "actions": []}
    push_status = "not run"
    push_ok = False
    if isinstance(push, dict):
        push_status = _short(push.get("status"))
        push_ok = bool(push.get("ok"))
    elif push is not None:
        push_status = "unexpected response"
    publication = commit.get("publication") if isinstance(commit.get("publication"), dict) else {}
    commit_hash = _short(publication.get("commit") or publication.get("head"), "")
    lines = [
        "KB Published",
        f"Committed: {commit_status}",
        f"Pushed: {push_status}",
    ]
    if commit_hash:
        lines.append(f"Commit: {commit_hash[:12]}")
    if changed_paths:
        lines.append(f"Changed paths: {len(changed_paths)}")
        lines.extend(_format_changed_paths(changed_paths))
    if not push_ok:
        lines.append("Warning: commit succeeded but push did not report success.")
        lines.append("Next: /kb publish push confirm")
    else:
        lines.append("Next: /kb status")
    return {"title": "KB Publish", "text": "\n".join(lines), "actions": []}


def _render_publish_command(ctx: Any, target: str, args: str) -> dict[str, Any]:
    confirm, message = _publish_args(args)
    preview_tool = _mcp_tool_name(target, "publication.preview_commit")
    commit_tool = _mcp_tool_name(target, "publication.commit_confirmed")
    push_tool = _mcp_tool_name(target, "publication.push_confirmed")
    actor = "telegram:operator"
    source = "Hermes Telegram"
    session_id = f"telegram-kb-publish-{int(time.time())}"
    preview_payload = _result_payload(ctx.dispatch_tool(preview_tool, {"message": message}))
    if not confirm:
        return _render_publish_preview(preview_payload)
    if not isinstance(preview_payload, dict) or preview_payload.get("error"):
        return _render_publish_preview(preview_payload)
    changed_paths = _changed_paths(preview_payload)
    if not changed_paths:
        return {
            "title": "KB Publish",
            "text": _render_publish_preview(preview_payload)["text"].replace("KB Publish Preview", "KB Publish"),
            "actions": [],
        }
    confirmation = {
        "confirmed": True,
        "surface": "telegram",
        "action": "publication.commit_and_push",
        "preview_required": True,
        "confirmation_text": "/kb publish confirm",
    }
    commit_payload = _result_payload(
        ctx.dispatch_tool(
            commit_tool,
            {
                "message": message,
                "expected_git_head": _short(_get_path(preview_payload, "git", "head"), ""),
                "expected_changed_paths": changed_paths,
                "push": False,
                "actor": actor,
                "source": source,
                "session_id": session_id,
                "user_confirmation": confirmation,
            },
        )
    )
    if not isinstance(commit_payload, dict) or not commit_payload.get("ok"):
        return _render_publish_result(preview_payload, commit_payload, None)
    push_payload = _result_payload(
        ctx.dispatch_tool(
            push_tool,
            {
                "actor": actor,
                "source": source,
                "session_id": session_id,
                "user_confirmation": confirmation,
            },
        )
    )
    return _render_publish_result(preview_payload, commit_payload, push_payload)


def _queue_items_from_payload(data: Any) -> list[Any]:
    return _items(data, ("items",), ("proposals",), ("queue", "items"))


def _queue_item_at(data: Any, index: int) -> dict[str, Any] | None:
    if index < 1:
        return None
    items = _queue_items_from_payload(data)
    if index > len(items):
        return None
    item = items[index - 1]
    return item if isinstance(item, dict) else None


def _parse_queue_indices(tokens: list[str]) -> list[int]:
    text = " ".join(tokens)
    indices: list[int] = []
    seen: set[int] = set()
    for match in re.finditer(r"\d+\s*-\s*\d+|\d+", text):
        token = match.group(0).strip()
        if re.fullmatch(r"\d+\s*-\s*\d+", token):
            start_text, end_text = re.split(r"\s*-\s*", token, maxsplit=1)
            start, end = int(start_text), int(end_text)
            step = 1 if end >= start else -1
            candidates = range(start, end + step, step)
        else:
            candidates = [int(token)]
        for index in candidates:
            if index > 0 and index not in seen:
                seen.add(index)
                indices.append(index)
    return indices


def _queue_items_at(data: Any, indices: list[int]) -> tuple[list[tuple[int, dict[str, Any]]], list[int]]:
    selection: list[tuple[int, dict[str, Any]]] = []
    missing: list[int] = []
    for index in indices:
        item = _queue_item_at(data, index)
        if item is None:
            missing.append(index)
        else:
            selection.append((index, item))
    return selection, missing


def _parse_queue_command_args(args: str, *, command: str) -> tuple[str, list[int], str | None, bool]:
    text = (args or "").strip()
    if not text:
        return "dashboard", [], None, False
    parts = text.split()
    first = parts[0].lower()
    if command == "kbreview":
        if first in {"review", "show", "detail", "details"}:
            indices = _parse_queue_indices(parts[1:])
        else:
            indices = _parse_queue_indices(parts)
        return ("review", indices[:1], None, False) if indices else ("help", [], None, False)
    if first.isdigit():
        return "review", [int(first)], None, False
    if first in {"review", "show", "detail", "details"}:
        indices = _parse_queue_indices(parts[1:])
        return ("review", indices[:1], None, False) if indices else ("help", [], None, False)
    if first in {"approve", "reject", "archive", "skip", "complete", "keep", "demote"}:
        confirm = any(part.lower() in {"confirm", "confirmed", "apply"} for part in parts[1:])
        index_tokens = [part for part in parts[1:] if part.lower() not in {"confirm", "confirmed", "apply"}]
        indices = _parse_queue_indices(index_tokens)
        return ("decision", indices, first, confirm) if indices else ("help", [], None, False)
    return "help", [], None, False


def _queue_command_help() -> dict[str, Any]:
    return {
        "title": "KB Queue",
        "text": "\n".join(
            [
                "KB Queue",
                "Use /kb queue to list proposals.",
                "Use /kb queue review 1 to inspect one item.",
                "Use /kb queue reject 1 to preview a decision.",
                "Use /kb queue complete 1 for a TODO-backed proposal.",
                "Use /kb queue reject 1 confirm to apply it.",
                "Use /kb queue reject 1,2 confirm to apply the same decision to multiple items.",
            ]
        ),
        "actions": [],
    }


def _render_queue_item(data: Any, *, index: int, ctx: Any, target: str) -> dict[str, Any]:
    item = _queue_item_at(data, index)
    if item is None:
        total = len(_queue_items_from_payload(data))
        return {
            "title": "KB Queue",
            "text": f"KB Queue\nNo item {index} in the current queue window ({total} shown). Use /kb queue to refresh.",
            "actions": [],
        }
    return {
        "title": "KB Queue",
        "text": _queue_item_text(item, index=index),
        "actions": [],
    }


def _render_queue_text_decision(
    ctx: Any,
    target: str,
    data: Any,
    *,
    indices: list[int],
    decision: str,
    confirm: bool,
) -> dict[str, Any]:
    selection, missing = _queue_items_at(data, indices)
    if not selection:
        total = len(_queue_items_from_payload(data))
        return {
            "title": "KB Queue",
            "text": f"KB Queue\nNo selected items in the current queue window ({total} shown). Use /kb queue to refresh.",
            "actions": [],
        }
    proposal_ids = _proposal_ids_for_selection(selection)
    if not proposal_ids:
        return {"title": "KB Queue", "text": "No proposal ids were available for the selected queue item(s).", "actions": []}
    selected_titles = ", ".join(_item_title(item) for _, item in selection)
    index_text = _format_indices([index for index, _ in selection])
    preview_tool = _mcp_tool_name(target, "queue.decision_preview")
    confirmed_tool = _mcp_tool_name(target, "queue.batch_decide_confirmed")
    actor = "telegram:operator"
    source = "Hermes Telegram"
    preview_payload = _result_payload(
        ctx.dispatch_tool(
            preview_tool,
            {
                "proposal_ids": proposal_ids,
                "decision": decision,
                "actor": actor,
                "source": source,
                "note": f"Previewed from Telegram /kb queue text command for {selected_titles}",
            },
        )
    )
    if not confirm:
        text = _preview_text(decision, proposal_ids, preview_payload, selection=selection)
        if missing:
            text += "\nMissing queue item(s): " + ", ".join(str(index) for index in missing)
        if _preview_allows_confirmation(preview_payload):
            text += f"\nTo apply: /kb queue {decision} {index_text} confirm"
        return {"title": "KB Queue", "text": text, "actions": []}
    if not _preview_allows_confirmation(preview_payload):
        return {
            "title": "KB Queue",
            "text": _preview_text(decision, proposal_ids, preview_payload, selection=selection),
            "actions": [],
        }
    confirmed_payload = _result_payload(
        ctx.dispatch_tool(
            confirmed_tool,
            {
                "proposal_ids": proposal_ids,
                "decision": decision,
                "actor": actor,
                "source": source,
                "session_id": f"telegram-kb-text-{int(time.time())}",
                "user_confirmation": {
                    "confirmed": True,
                    "surface": "telegram",
                    "action": f"queue.{decision}",
                    "preview_required": True,
                    "confirmation_text": f"/kb queue {decision} {index_text} confirm",
                },
                "note": f"Confirmed from Telegram /kb queue text command for {selected_titles}",
            },
        )
    )
    text = _confirmed_text(decision, confirmed_payload, selection=selection, proposal_ids=proposal_ids)
    if missing:
        text += "\nSkipped missing queue item(s): " + ", ".join(str(index) for index in missing)
    return {"title": "KB Queue", "text": text, "actions": []}


def _workflow_id_from_args(args: str) -> tuple[str, str]:
    text = (args or "").strip()
    lowered = text.lower()
    if not text or lowered in {"sync", "kb sync", "sync kb", "update kb", "update_kb"}:
        return "update_kb", text or "kb sync"
    if lowered.startswith("meeting"):
        return "meeting_process", text
    return text.split(maxsplit=1)[0], text


def _workflow_args_from_text(args: str) -> tuple[str, str, bool]:
    text = (args or "").strip()
    parts = text.split()
    confirm = bool(parts and parts[-1].lower() in {"confirm", "confirmed", "start", "apply"})
    if confirm:
        text = " ".join(parts[:-1]).strip()
    workflow_id, intent = _workflow_id_from_args(text)
    return workflow_id, intent, confirm


def _workflow_envelope(plan: dict[str, Any], callback_ctx: Any) -> dict[str, Any]:
    workflow = plan.get("workflow") if isinstance(plan.get("workflow"), dict) else {}
    request = plan.get("request") if isinstance(plan.get("request"), dict) else {}
    actor_id = _short(getattr(callback_ctx, "actor_id", ""), "unknown")
    actor_name = _short(getattr(callback_ctx, "actor_name", ""), "")
    confirmed_at = _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()
    return {
        "schema_version": int(plan.get("schema_version") or 1),
        "tool": plan.get("tool") or "workflow.start_confirmed",
        "plan": {
            "workflow_id": str(workflow.get("workflow_id") or ""),
            "args": dict(request.get("args") or {}),
            "queue_gate_limit": int(request.get("queue_gate_limit") or 0),
            "force": bool(request.get("force", False)),
            "request_id": str(plan.get("request_id") or ""),
            "idempotency_key": str(plan.get("idempotency_key") or ""),
            "preconditions": list(plan.get("preconditions") or []),
        },
        "provenance": dict(plan.get("provenance") or {}),
        "user_confirmation": {
            "confirmed": True,
            "confirmed_by": actor_name or actor_id,
            "confirmed_at": confirmed_at,
            "confirmation_text": "Confirmed by Telegram text command after workflow preview.",
            "preview_status": _short(plan.get("status")),
            "surface": "telegram",
            "actor_id": actor_id,
            "actor_name": actor_name,
        },
    }


def _workflow_start_text(ctx: Any, target: str, plan: dict[str, Any]) -> str:
    callback_ctx = SimpleNamespace(
        callback_id=f"text-{int(time.time())}",
        actor_id="operator",
        actor_name="Telegram",
    )
    envelope = _workflow_envelope(plan, callback_ctx)
    payload = _result_payload(
        ctx.dispatch_tool(
            _mcp_tool_name(target, "workflow.start_confirmed"),
            {"envelope": envelope},
        )
    )
    text = _workflow_status_text("Workflow start result", payload)
    run_id = _workflow_run_id(payload)
    if run_id:
        text += "\nUse /kb runs for progress. Hermes should also keep watching in the main conversation."
    return text


def _workflow_run_id(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
    return str(payload.get("run_id") or run.get("run_id") or "")


def _workflow_status_text(prefix: str, payload: Any) -> str:
    if isinstance(payload, dict) and payload.get("error"):
        return f"{prefix}\n{payload['error']}"
    if not isinstance(payload, dict):
        return f"{prefix}\n{_short(payload, 'No structured response returned.')}"
    lines = [
        prefix,
        f"Status: {_short(payload.get('status'))}",
    ]
    run_id = _workflow_run_id(payload)
    if run_id:
        lines.append(f"Run: {run_id}")
    if payload.get("started") is not None:
        lines.append(f"Started: {_short(payload.get('started'))}")
    follow = payload.get("followthrough_contract") if isinstance(payload.get("followthrough_contract"), dict) else {}
    if follow:
        lines.append(f"Next: {_short(follow.get('recommended_next_action'))}")
    if isinstance(payload.get("readiness"), dict):
        lines.append("Readiness: " + _short(payload["readiness"].get("status")))
    return "\n".join(lines)


def _render_workflow_plan(
    data: Any,
    *,
    ctx: Any,
    target: str,
    adapter: Any,
    start_hint: str = "/kb run sync confirm",
) -> dict[str, Any]:
    if isinstance(data, dict) and data.get("error"):
        return {"title": "Workflow", "text": f"Workflow plan failed\n{data['error']}", "actions": []}
    if not isinstance(data, dict):
        return {"title": "Workflow", "text": f"Workflow\n{_short(data, 'No plan returned.')}", "actions": []}
    workflow = data.get("workflow") if isinstance(data.get("workflow"), dict) else {}
    request = data.get("request") if isinstance(data.get("request"), dict) else {}
    effect_plan = data.get("effect_plan") if isinstance(data.get("effect_plan"), dict) else {}
    effects = effect_plan.get("effects") if isinstance(effect_plan.get("effects"), list) else []
    lines = [
        "Workflow Preview",
        f"Workflow: {_short(workflow.get('workflow_id'))}",
        f"Status: {_short(data.get('status'))}",
        f"Risk: {_short(workflow.get('risk') or effect_plan.get('risk'))}",
        f"Force: {_short(request.get('force'))}",
    ]
    if data.get("message"):
        lines.append("Message: " + _short(data.get("message")))
    if isinstance(data.get("readiness"), dict):
        lines.append("Readiness: " + _short(data["readiness"].get("status")))
    if effects:
        lines.append("Effects: " + ", ".join(_short(effect.get("id")) for effect in effects[:4] if isinstance(effect, dict)))
    follow = data.get("followthrough_contract") if isinstance(data.get("followthrough_contract"), dict) else {}
    if follow:
        lines.append("Follow-through: " + _short(follow.get("watch_tool")) + " -> " + _short(follow.get("terminal_summary_tool")))
    if data.get("status") == "confirmation_required":
        lines.append(f"To start: {start_hint}")
    return {"title": "Workflow", "text": "\n".join(lines), "actions": []}


def _render_queue(data: Any, *, ctx: Any | None = None, target: str | None = None) -> dict[str, Any]:
    if isinstance(data, str):
        return {"title": "KB Queue", "text": f"KB Queue\n{data}", "actions": []}
    count = None
    if isinstance(data, dict):
        count = data.get("total") or data.get("count") or _count_from(data, "queue", "proposals")
    items = _items(data, ("items",), ("proposals",), ("queue", "items"))
    lines = ["KB Queue"]
    if count is not None:
        lines.append(f"{count} pending")
    if not items:
        lines.append("No proposal previews returned.")
    for idx, item in enumerate(items[:5], start=1):
        if isinstance(item, dict):
            lines.append("")
            lines.append(f"{idx}. {_item_title(item)}")
            target_path = _item_target(item)
            kind = _item_kind(item)
            preview = _item_detail(item)
            if target_path:
                lines.append(f"   Target: {target_path}")
            if kind:
                lines.append(f"   Type: {kind}")
            if preview:
                lines.append(f"   Summary: {_clip(preview, 220)}")
            lines.append(f"   Review: /kb queue review {idx}")
        else:
            lines.append(f"{idx}. {_short(item)}")
    if items:
        lines.append("")
        lines.append("Review one: /kb queue review 1")
        lines.append("Then preview a listed action, for example: /kb queue reject 1")
        lines.append("Confirm after preview: /kb queue reject 1 confirm")
    return {"title": "KB Queue", "text": "\n".join(lines), "actions": []}


def _kb_root_command(args: str) -> tuple[str, str]:
    text = (args or "").strip()
    if not text:
        return "kb", ""
    head, _, tail = text.partition(" ")
    key = head.strip().lower()
    rest = tail.strip()
    if key in {"dashboard", "home"}:
        return "kb", rest
    if key in {"help", "commands"}:
        return "kbhelp", rest
    if key == "today":
        return "kbtoday", rest
    if key in {"status", "info"}:
        return "kbstatus", rest
    if key in {"runs", "runlog", "history"}:
        return "kbruns", rest
    if key in {"queue", "q"}:
        return "kbqueue", rest
    if key == "review":
        return "kbqueue", f"review {rest}".strip()
    if key in {"publish", "publication"}:
        return "kbpublish", rest
    if key in {"run", "workflow"}:
        return "kbrun", rest
    if key == "sync":
        return "kbrun", f"sync {rest}".strip()
    return "kbhelp", text


def _kb_command_help() -> dict[str, Any]:
    return {
        "title": "KB",
        "text": "\n".join(
            [
                "KB Commands",
                "/kb - dashboard",
                "/kb queue - proposal review list",
                "/kb queue review 1 - inspect one queue item",
                "/kb queue reject 1 - preview a decision",
                "/kb queue reject 1 confirm - apply a previewed decision",
                "/kb publish - preview KB Git publication",
                "/kb publish confirm - commit and push after preview",
                "/kb status - lane, model, readiness, publication",
                "/kb runs - active and recent workflow runs",
                "/kb run sync - preview a KB sync",
            ]
        ),
        "actions": [],
    }


def _card_for_command(ctx: Any, command: str, *, args: str = "", adapter: Any = None) -> dict[str, Any]:
    target = _mcp_target()
    cockpit_args = {
        "attention_limit": 5,
        "include_publication": True,
        "include_readiness": True,
        "run_limit": 3,
    }
    if command == "kb":
        routed_command, routed_args = _kb_root_command(args)
        if routed_command == "kbhelp":
            return _kb_command_help()
        if routed_command != "kb":
            return _card_for_command(ctx, routed_command, args=routed_args, adapter=adapter)
        _, data, errors = _dispatch_first(
            ctx,
            target,
            [
                (
                    "dashboard.live",
                    {
                        "limit": 5,
                        "include_feedback": True,
                        "include_publication": True,
                        "include_readiness": True,
                    },
                ),
                ("attention.cockpit", cockpit_args),
            ],
        )
        return _render_error("KB Dashboard", target, errors) if data is None else _render_dashboard(data, ctx=ctx, target=target)
    if command == "kbhelp":
        return _kb_command_help()
    if command == "kbtoday":
        _, data, errors = _dispatch_first(ctx, target, [("attention.cockpit", cockpit_args)])
        return _render_error("KB Today", target, errors) if data is None else _render_today(data)
    if command == "kbstatus":
        _, data, _errors = _dispatch_first(ctx, target, [("attention.cockpit", cockpit_args)])
        return _render_status(data, target)
    if command == "kbruns":
        _, data, errors = _dispatch_first(
            ctx,
            target,
            [
                ("run.health", {}),
                ("run.watch", {"mode": "progress_digest"}),
                ("progress_digest", {}),
            ],
        )
        return _render_error("KB Runs", target, errors) if data is None else _render_runs(data)
    if command in {"kbqueue", "kbreview"}:
        mode, indices, decision, confirm = _parse_queue_command_args(args, command=command)
        if mode == "help":
            return _queue_command_help()
        data, errors = _queue_summary_payload(ctx, target, limit=5)
        if data is None:
            return _render_error("KB Queue", target, errors)
        if mode == "review" and indices:
            return _render_queue_item(data, index=indices[0], ctx=ctx, target=target)
        if mode == "decision" and indices and decision:
            return _render_queue_text_decision(ctx, target, data, indices=indices, decision=decision, confirm=confirm)
        return _render_queue(data, ctx=ctx, target=target)
    if command == "kbpublish":
        return _render_publish_command(ctx, target, args)
    if command == "kbrun":
        workflow_id, intent, confirm = _workflow_args_from_text(args)
        if not workflow_id:
            return {
                "title": "Workflow",
                "text": "Workflow\nSend /kb run sync or /kb run <workflow_id>.",
                "actions": [],
            }
        _, data, errors = _dispatch_first(
            ctx,
            target,
            [
                (
                    "workflow.plan_request",
                    {
                        "workflow_id": workflow_id,
                        "intent": intent,
                        "actor": "telegram:operator",
                        "source": "Hermes Telegram",
                        "session_id": f"telegram-kb-{int(time.time())}",
                    },
                )
            ],
        )
        if data is None:
            return _render_error("Workflow", target, errors)
        if confirm and isinstance(data, dict) and data.get("status") == "confirmation_required":
            return {"title": "Workflow", "text": _workflow_start_text(ctx, target, data), "actions": []}
        hint_args = (args or "sync").strip()
        hint_parts = hint_args.split()
        if hint_parts and hint_parts[-1].lower() in {"confirm", "confirmed", "start", "apply"}:
            hint_args = " ".join(hint_parts[:-1]).strip()
        return _render_workflow_plan(
            data,
            ctx=ctx,
            target=target,
            adapter=adapter,
            start_hint=f"/kb run {hint_args or 'sync'} confirm",
        )
    return {"title": "KB", "text": "Unsupported KB command.", "actions": []}


def _adapter_for(gateway: Any, source: Any) -> Any | None:
    adapters = getattr(gateway, "adapters", {}) or {}
    platform = getattr(source, "platform", None)
    return (
        adapters.get(platform)
        or adapters.get(_platform_name(platform))
        or adapters.get("telegram")
    )


def _authorized_for_gateway(gateway: Any, source: Any) -> bool:
    checker = getattr(gateway, "_is_user_authorized", None)
    if checker is None:
        return True
    try:
        return bool(checker(source))
    except Exception:
        logger.debug("kb_journeys: authorization check failed", exc_info=True)
        return False


def _reply_anchor_and_metadata(event: Any) -> tuple[str | None, dict[str, Any] | None]:
    source = getattr(event, "source", None)
    try:
        from gateway.platforms.base import _reply_anchor_for_event, _thread_metadata_for_source

        return _reply_anchor_for_event(event), _thread_metadata_for_source(source)
    except Exception:
        metadata = None
        if getattr(source, "thread_id", None):
            metadata = {"thread_id": getattr(source, "thread_id")}
        return getattr(event, "message_id", None), metadata


async def _send_card(adapter: Any, event: Any, card: dict[str, Any]) -> None:
    source = getattr(event, "source", None)
    chat_id = getattr(source, "chat_id", None)
    if not chat_id:
        return
    reply_to, metadata = _reply_anchor_and_metadata(event)
    actions = card.get("actions", []) or []
    if actions and hasattr(adapter, "send_kb_actions"):
        result = adapter.send_kb_actions(
            chat_id,
            card["text"],
            actions,
            reply_to=reply_to,
            metadata=metadata,
        )
    else:
        result = adapter.send(chat_id, card["text"], reply_to=reply_to, metadata=metadata)
    if inspect.isawaitable(result):
        await result


def _run_delivery(coro: Any) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return
    loop.create_task(coro)


def build_pre_gateway_dispatch_hook(ctx: Any) -> Callable[..., dict[str, str] | None]:
    def _hook(event: Any = None, gateway: Any = None, session_store: Any = None, **_: Any) -> dict[str, str] | None:
        source = getattr(event, "source", None)
        if _platform_name(getattr(source, "platform", None)) != "telegram":
            return None
        command = _command_from_text(getattr(event, "text", ""))
        if command is None:
            return None
        args = _command_args_from_text(getattr(event, "text", ""))
        if not _authorized_for_gateway(gateway, source):
            return None
        adapter = _adapter_for(gateway, source)
        if adapter is None:
            logger.debug("kb_journeys: no Telegram adapter available")
            return None
        card = _card_for_command(ctx, command, args=args, adapter=adapter)
        _run_delivery(_send_card(adapter, event, card))
        return {"action": "skip", "reason": "kb_journeys"}

    return _hook


def register(ctx: Any) -> None:
    def _command_help(_: str = "") -> str:
        return "Use /kb in Telegram. Try: /kb queue, /kb status, /kb runs, /kb run sync."

    for command in sorted(MENU_COMMANDS):
        try:
            ctx.register_command(
                command,
                _command_help,
                description="KB dashboard, queue, status, runs, and sync.",
            )
        except Exception:
            logger.debug("kb_journeys: failed to register /%s", command, exc_info=True)
    ctx.register_hook("pre_gateway_dispatch", build_pre_gateway_dispatch_hook(ctx))
