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
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

DEFAULT_MCP_TARGET = "kb_engine_prod"
SUPPORTED_COMMANDS = {"dashboard", "kb", "today", "kbstatus", "runs", "kbqueue", "review", "run"}


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
    return command if command in SUPPORTED_COMMANDS else None


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
    queue_count = _count_from(data, "queue", "queues", "proposals")
    if queue_count is None:
        queue_count = _get_path(data, "summary", "queue_item_count")
    todo_count = _count_from(data, "todo", "todos")
    if todo_count is None:
        todo_count = (
            _get_path(data, "summary", "active_todo_count")
            or _get_path(data, "summary", "triage_todo_count")
        )

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
        lines.append(
            "Queue: "
            f"{_short(queue_count, 'unknown')} · TODOs: {_short(todo_count, 'unknown')}"
        )
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
    queue_count = summary.get("queue_item_count")
    todo_count = summary.get("active_todo_count") or summary.get("triage_todo_count")
    active_runs = summary.get("active_run_count")
    lines = [
        "KB Dashboard",
        f"Readiness: {readiness}",
        f"Publication: {publication}",
    ]
    counts: list[str] = []
    if queue_count is not None:
        counts.append(f"Queue {queue_count}")
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
        lines.append(_short(section.get("title") or section.get("id"), "Section"))
        for card in cards[:3]:
            if not isinstance(card, dict):
                continue
            detail = _short(card.get("detail"), "")
            suffix = f" — {detail}" if detail else ""
            lines.append(f"- {_short(card.get('title'), 'item')}{suffix}")
    warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
    if warnings:
        lines.append("")
        lines.append(f"Warnings: {len(warnings)}")
    refresh = data.get("refresh") if isinstance(data.get("refresh"), dict) else {}
    if refresh:
        lines.append(f"Refresh: every {_short(refresh.get('ttl_seconds'), '60')}s target")
    actions = [
        _command_card_action(ctx, target, "Refresh", "dashboard"),
        _command_card_action(ctx, target, "Queue", "kbqueue"),
        _command_card_action(ctx, target, "Runs", "runs"),
        _command_card_action(ctx, target, "Status", "kbstatus"),
    ]
    return {"title": "KB Dashboard", "text": "\n".join(lines), "actions": actions}


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


def _kb_action(label: str, action_id: str, handler: Callable[[Any], Any], metadata: dict[str, Any] | None = None) -> Any:
    try:
        from tools.kb_callback_registry import KbAction

        return KbAction(
            label=label,
            action_id=action_id,
            handler=handler,
            metadata=metadata or {},
        )
    except Exception:
        return {
            "label": label,
            "action_id": action_id,
            "handler": handler,
            "metadata": metadata or {},
        }


def _command_card_action(ctx: Any, target: str, label: str, command: str) -> Any:
    async def _handler(_callback_ctx: Any) -> str:
        card = _card_for_command(ctx, command)
        return str(card.get("text") or card.get("title") or label)

    return _kb_action(
        label,
        f"dashboard.{command}",
        _handler,
        {"command": command, "target": target},
    )


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
        lines.append(detail)
    if proposal_ids:
        lines.append("")
        lines.append(f"Proposal ids: {', '.join(proposal_ids[:5])}")
        lines.append("Decision buttons apply only this item.")
    else:
        lines.append("")
        lines.append("This item did not include proposal ids, so Telegram cannot apply a decision yet. Use the KB workbench for details.")
    return "\n".join(lines)


def _preview_text(decision: str, proposal_ids: list[str], payload: Any, *, item: dict[str, Any] | None = None) -> str:
    item_title = _item_title(item) if isinstance(item, dict) else ""
    item_target = _item_target(item) if isinstance(item, dict) else ""
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
        if item_title:
            lines.append(f"Item: {item_title}")
        if item_target:
            lines.append(f"Target: {item_target}")
        lines.extend(
            [
                f"Status: {status} · ok: {ok}",
                f"Proposal ids: {', '.join(proposal_ids[:5])}",
                summary,
                "Confirm only if this item and decision match what you intend.",
            ]
        )
        return "\n".join(lines)
    lines = [f"Queue {decision} preview"]
    if item_title:
        lines.append(f"Item: {item_title}")
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


def _confirmed_text(decision: str, payload: Any) -> str:
    if isinstance(payload, dict) and payload.get("error"):
        return f"Queue {decision} failed\n{payload['error']}"
    if isinstance(payload, dict):
        publication = payload.get("publication") if isinstance(payload.get("publication"), dict) else {}
        git_state = payload.get("git") if isinstance(payload.get("git"), dict) else {}
        lines = [
            f"Queue {decision} applied",
            f"Status: {_short(payload.get('status'))} · ok: {_short(payload.get('ok'))}",
        ]
        if publication:
            lines.append(
                "Publication: "
                + _short(publication.get("status") or publication.get("state") or publication.get("reason"))
            )
        if git_state:
            lines.append("Git: " + _short(git_state.get("summary") or git_state.get("status") or git_state))
        lines.append("Use /kbqueue to refresh.")
        return "\n".join(lines)
    return f"Queue {decision} applied\nUse /kbqueue to refresh."


def _queue_decision_action(ctx: Any, target: str, item: dict[str, Any], decision: str) -> Any:
    proposal_ids = _proposal_ids_for_item(item)

    async def _preview(callback_ctx: Any) -> dict[str, Any] | str:
        if not proposal_ids:
            return "No proposal ids were available for this queue item."
        actor = f"telegram:{_short(getattr(callback_ctx, 'actor_id', ''), 'unknown')}"
        source = "Hermes Telegram"
        preview_tool = _mcp_tool_name(target, "queue.decision_preview")
        confirmed_tool = _mcp_tool_name(target, "queue.batch_decide_confirmed")
        preview_payload = _result_payload(
            ctx.dispatch_tool(
                preview_tool,
                {
                    "proposal_ids": proposal_ids,
                    "decision": decision,
                    "actor": actor,
                    "source": source,
                    "note": f"Previewed from Telegram /kbqueue for {item.get('item_id') or item.get('entity_path') or item.get('title')}",
                },
            )
        )

        async def _confirm(confirm_ctx: Any) -> str:
            confirmed_payload = _result_payload(
                ctx.dispatch_tool(
                    confirmed_tool,
                    {
                        "proposal_ids": proposal_ids,
                        "decision": decision,
                        "actor": f"telegram:{_short(getattr(confirm_ctx, 'actor_id', ''), 'unknown')}",
                        "source": source,
                        "session_id": f"telegram-kb-{_short(getattr(confirm_ctx, 'callback_id', ''), str(int(time.time())))}",
                        "user_confirmation": {
                            "confirmed": True,
                            "surface": "telegram",
                            "action": f"queue.{decision}",
                            "actor_id": _short(getattr(confirm_ctx, "actor_id", ""), ""),
                            "actor_name": _short(getattr(confirm_ctx, "actor_name", ""), ""),
                            "preview_required": True,
                        },
                        "note": f"Confirmed from Telegram /kbqueue for {item.get('item_id') or item.get('entity_path') or item.get('title')}",
                    },
                )
            )
            return _confirmed_text(decision, confirmed_payload)

        actions: list[Any] = []
        if _preview_allows_confirmation(preview_payload):
            actions.append(
                _kb_action(
                    f"Confirm {decision}",
                    f"queue.confirm.{decision}",
                    _confirm,
                    {"proposal_ids": proposal_ids, "decision": decision},
                )
            )
        return {"text": _preview_text(decision, proposal_ids, preview_payload, item=item), "actions": actions}

    return _kb_action(
        f"Preview {decision}",
        f"queue.preview.{decision}",
        _preview,
        {"proposal_ids": proposal_ids, "decision": decision},
    )


def _queue_item_action(ctx: Any, target: str, item: dict[str, Any], index: int) -> Any:
    async def _handler(_callback_ctx: Any) -> dict[str, Any]:
        actions: list[Any] = []
        if _proposal_ids_for_item(item):
            actions = [
                _queue_decision_action(ctx, target, item, "approve"),
                _queue_decision_action(ctx, target, item, "reject"),
                _queue_decision_action(ctx, target, item, "archive"),
            ]
        return {"text": _queue_item_text(item, index=index), "actions": actions}

    return _kb_action(
        f"Review {index}",
        f"queue.item.{index}",
        _handler,
        {
            "index": index,
            "item_id": str(item.get("item_id") or item.get("id") or item.get("entity_path") or ""),
            "proposal_ids": _proposal_ids_for_item(item),
        },
    )


def _workflow_id_from_args(args: str) -> tuple[str, str]:
    text = (args or "").strip()
    lowered = text.lower()
    if not text or lowered in {"sync", "kb sync", "sync kb", "update kb", "update_kb"}:
        return "update_kb", text or "kb sync"
    if lowered.startswith("meeting"):
        return "meeting_process", text
    return text.split(maxsplit=1)[0], text


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
            "confirmation_text": "Confirmed by Telegram button tap after workflow preview.",
            "preview_status": _short(plan.get("status")),
            "surface": "telegram",
            "actor_id": actor_id,
            "actor_name": actor_name,
        },
    }


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


async def _adapter_send_text(adapter: Any, callback_ctx: Any, text: str) -> None:
    chat_id = getattr(callback_ctx, "chat_id", None)
    if not chat_id or not hasattr(adapter, "send"):
        return
    metadata = None
    thread_id = getattr(callback_ctx, "thread_id", None)
    if thread_id:
        metadata = {"thread_id": str(thread_id)}
    result = adapter.send(str(chat_id), text, metadata=metadata)
    if inspect.isawaitable(result):
        await result


async def _watch_run_until_terminal(ctx: Any, target: str, run_id: str, adapter: Any, callback_ctx: Any) -> None:
    if not run_id:
        return
    watch_tool = _mcp_tool_name(target, "run.watch")
    summary_tool = _mcp_tool_name(target, "run.summary")
    last_status = ""
    for attempt in range(24):
        await asyncio.sleep(20 if attempt == 0 else min(60, 20 + attempt * 5))
        try:
            raw = await asyncio.to_thread(
                ctx.dispatch_tool,
                watch_tool,
                {"run_id": run_id, "timeout_seconds": 25, "poll_interval_seconds": 5, "timeline_limit": 5},
            )
            payload = _result_payload(raw)
        except Exception as exc:
            await _adapter_send_text(adapter, callback_ctx, f"KB run watcher failed for {run_id}\n{exc}")
            return
        if isinstance(payload, dict):
            terminal = bool(payload.get("terminal"))
            status = _short(
                payload.get("status")
                or _get_path(payload, "summary", "status")
                or _get_path(payload, "progress_digest", "status"),
                "",
            )
            phase = _short(
                _get_path(payload, "progress_digest", "progress", "current_phase")
                or _get_path(payload, "progress_digest", "current_phase"),
                "",
            )
            current = " · ".join(part for part in (status, phase) if part)
            if current and current != last_status and attempt in {0, 3, 8, 15}:
                last_status = current
                await _adapter_send_text(adapter, callback_ctx, f"KB run {run_id} still running\n{current}")
            if terminal:
                try:
                    summary_payload = _result_payload(
                        await asyncio.to_thread(ctx.dispatch_tool, summary_tool, {"run_id": run_id})
                    )
                except Exception:
                    summary_payload = payload
                await _adapter_send_text(
                    adapter,
                    callback_ctx,
                    _workflow_status_text(f"KB run {run_id} finished", summary_payload),
                )
                return
    await _adapter_send_text(
        adapter,
        callback_ctx,
        f"KB run {run_id} is still active after the watcher window. Use /runs for the latest status.",
    )


def _workflow_start_action(ctx: Any, target: str, plan: dict[str, Any], adapter: Any) -> Any:
    async def _start(callback_ctx: Any) -> str:
        envelope = _workflow_envelope(plan, callback_ctx)
        payload = _result_payload(
            ctx.dispatch_tool(
                _mcp_tool_name(target, "workflow.start_confirmed"),
                {"envelope": envelope},
            )
        )
        run_id = _workflow_run_id(payload)
        if run_id:
            _run_delivery(_watch_run_until_terminal(ctx, target, run_id, adapter, callback_ctx))
        text = _workflow_status_text("Workflow start result", payload)
        if run_id:
            text += "\nI will watch this run and send the terminal summary here."
        return text

    workflow = plan.get("workflow") if isinstance(plan.get("workflow"), dict) else {}
    workflow_id = _short(workflow.get("workflow_id"), "workflow")
    return _kb_action(
        f"Start {workflow_id}",
        f"workflow.start.{workflow_id}",
        _start,
        {"workflow_id": workflow_id, "request_id": plan.get("request_id")},
    )


def _render_workflow_plan(data: Any, *, ctx: Any, target: str, adapter: Any) -> dict[str, Any]:
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
    actions = []
    if data.get("status") == "confirmation_required":
        actions = [_workflow_start_action(ctx, target, data, adapter)]
        lines.append("Tap Start only after this preview matches what you intend.")
    return {"title": "Workflow", "text": "\n".join(lines), "actions": actions}


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
            meta = _item_kind(item)
            preview = _item_detail(item)
            line = f"{idx}. {_item_title(item)}"
            if meta:
                line += f" ({meta})"
            if preview:
                line += f"\n   {preview}"
            lines.append(line)
        else:
            lines.append(f"{idx}. {_short(item)}")
    actions: list[Any] = []
    if ctx is not None and target and items:
        actions = [
            _queue_item_action(ctx, target, item, idx)
            for idx, item in enumerate(items[:5], start=1)
            if isinstance(item, dict)
        ]
        if actions:
            lines.append("")
            lines.append("Tap Review N to inspect one item before choosing a decision.")
    return {"title": "KB Queue", "text": "\n".join(lines), "actions": actions}


def _card_for_command(ctx: Any, command: str, *, args: str = "", adapter: Any = None) -> dict[str, Any]:
    target = _mcp_target()
    cockpit_args = {
        "attention_limit": 5,
        "include_publication": True,
        "include_readiness": True,
        "run_limit": 3,
    }
    if command == "dashboard":
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
    if command in {"kb", "today"}:
        _, data, errors = _dispatch_first(ctx, target, [("attention.cockpit", cockpit_args)])
        return _render_error("KB Today", target, errors) if data is None else _render_today(data)
    if command == "kbstatus":
        _, data, _errors = _dispatch_first(ctx, target, [("attention.cockpit", cockpit_args)])
        return _render_status(data, target)
    if command == "runs":
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
    if command in {"kbqueue", "review"}:
        _, data, errors = _dispatch_first(
            ctx,
            target,
            [
                ("queue.summary", {"scope": "proposals", "limit": 5}),
                ("queue.preview", {"limit": 5}),
                ("workbench.queue", {"scope": "proposals", "limit": 5}),
            ],
        )
        return _render_error("KB Queue", target, errors) if data is None else _render_queue(data, ctx=ctx, target=target)
    if command == "run":
        workflow_id, intent = _workflow_id_from_args(args)
        if not workflow_id:
            return {
                "title": "Workflow",
                "text": "Workflow\nSend /run kb sync or /run <workflow_id>.",
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
        return _render_error("Workflow", target, errors) if data is None else _render_workflow_plan(data, ctx=ctx, target=target, adapter=adapter)
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
    if hasattr(adapter, "send_kb_actions"):
        result = adapter.send_kb_actions(
            chat_id,
            card["text"],
            card.get("actions", []),
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
        return "Use this command from Telegram to render the native KB card."

    for command in sorted(SUPPORTED_COMMANDS):
        try:
            ctx.register_command(
                command,
                _command_help,
                description="Render a Telegram-native KB journey card.",
            )
        except Exception:
            logger.debug("kb_journeys: failed to register /%s", command, exc_info=True)
    ctx.register_hook("pre_gateway_dispatch", build_pre_gateway_dispatch_hook(ctx))
