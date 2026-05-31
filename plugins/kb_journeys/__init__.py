"""Telegram KB journey renderer plugin.

Intercepts a small set of Telegram slash commands and renders concise,
read-only KB status summaries from the configured KB MCP target.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
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
KB_REASONING_LEVELS = {"none", "minimal", "low", "medium", "high", "xhigh"}
QUEUE_REPLY_DECISIONS = {"approve", "reject", "archive", "skip", "complete", "keep", "demote", "detail"}
QUEUE_REPLY_TOOL_DECISIONS = {"approve", "reject", "archive", "skip", "complete", "keep", "demote"}
QUEUE_REPLY_STATE_TTL_SECONDS = 15 * 60
MEETING_HANDOFF_STATE_TTL_SECONDS = 15 * 60


def _sanitize_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(value or ""))


def _mcp_target() -> str:
    return os.getenv("HERMES_KB_MCP_TARGET", DEFAULT_MCP_TARGET).strip() or DEFAULT_MCP_TARGET


def _mcp_tool_name(target: str, tool_name: str) -> str:
    return f"mcp_{_sanitize_component(target)}_{_sanitize_component(tool_name)}"


def _queue_reply_state_path():
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "state" / "kb_queue_reply_state.json"


def _meeting_handoff_state_path():
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "state" / "kb_meeting_handoff_state.json"


def _load_queue_reply_states() -> dict[str, Any]:
    path = _queue_reply_state_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_queue_reply_states(states: dict[str, Any]) -> None:
    path = _queue_reply_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(states, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        logger.debug("kb_journeys: failed to persist iterative queue state", exc_info=True)


def _load_meeting_handoff_states() -> dict[str, Any]:
    path = _meeting_handoff_state_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_meeting_handoff_states(states: dict[str, Any]) -> None:
    path = _meeting_handoff_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(states, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        logger.debug("kb_journeys: failed to persist meeting handoff state", exc_info=True)


def _clear_meeting_handoff_state(session_id: str) -> None:
    if not session_id:
        return
    states = _load_meeting_handoff_states()
    if session_id in states:
        states.pop(session_id, None)
        _save_meeting_handoff_states(states)


def _clear_iterative_queue_reply_state(session_id: str) -> None:
    if not session_id:
        return
    states = _load_queue_reply_states()
    if session_id in states:
        states.pop(session_id, None)
        _save_queue_reply_states(states)


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
        return {"title": "KB Cockpit", "text": f"KB Cockpit\n{_short(data, 'No cockpit details returned.')}", "actions": []}

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
        "KB Cockpit",
        f"Runtime: {readiness}",
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


def _strip_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _selected_env_values(keys: set[str]) -> dict[str, str]:
    values = {key: value for key in keys if (value := os.getenv(key))}
    missing = keys.difference(values)
    if not missing:
        return values
    try:
        from hermes_cli.config import get_env_path

        env_path = get_env_path()
        lines = env_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except Exception:
        return values
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if key in missing and key not in values:
            values[key] = _strip_env_value(value)
    return values


def _first_env(env: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = env.get(key)
        if value:
            return value
    return None


def _config_snapshot() -> dict[str, str]:
    config: dict[str, Any] = {}
    env_keys = {
        "ANTHROPIC_API_KEY",
        "ENVIRONMENT",
        "HERMES_API_MODE",
        "HERMES_ENV",
        "HERMES_ENVIRONMENT",
        "HERMES_KB_LANE",
        "HERMES_KB_LLM_MODEL",
        "HERMES_KB_LLM_PROVIDER",
        "HERMES_KB_MODE",
        "HERMES_KB_MODEL",
        "HERMES_KB_PROVIDER",
        "HERMES_KB_REASONING_EFFORT",
        "HERMES_KB_WORKSPACE",
        "HERMES_MODEL",
        "HERMES_MODEL_API_MODE",
        "HERMES_PROFILE",
        "HERMES_PROVIDER",
        "HERMES_REASONING_EFFORT",
        "KB_LLM_MODEL",
        "KB_LLM_PROVIDER",
        "KB_LLM_REASONING_EFFORT",
        "KB_OPENAI_COMPAT_MODEL",
        "KB_PROVIDER",
        "KB_WORKSPACE",
        "MODEL",
        "MODEL_PROVIDER",
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_REASONING_EFFORT",
        "OPENROUTER_API_KEY",
    }
    env = _selected_env_values(env_keys)
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

    provider = provider or _first_env(env, "HERMES_PROVIDER", "MODEL_PROVIDER")
    model = model or _first_env(env, "HERMES_MODEL", "MODEL")
    reasoning = reasoning or _first_env(env, "HERMES_REASONING_EFFORT", "OPENAI_REASONING_EFFORT")
    api_mode = api_mode or _first_env(env, "HERMES_MODEL_API_MODE", "HERMES_API_MODE")

    api_envs = [
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
    ]
    configured = [name.removesuffix("_API_KEY") for name in api_envs if env.get(name)]

    return {
        "lane": _first_env(env, "HERMES_KB_MODE", "HERMES_KB_LANE", "HERMES_PROFILE") or "unknown",
        "environment": (
            _first_env(env, "HERMES_ENVIRONMENT", "HERMES_ENV", "ENVIRONMENT", "HERMES_PROFILE")
            or "unknown"
        ),
        "workspace": _first_env(env, "HERMES_KB_WORKSPACE", "KB_WORKSPACE") or "not set",
        "model": _short(model, "not set"),
        "provider": _short(provider, "not set"),
        "api_mode": _short(api_mode, "not set"),
        "api": ", ".join(configured) if configured else "not detected",
        "reasoning": _short(reasoning, "not set"),
        "kb_provider": _short(
            _first_env(env, "HERMES_KB_LLM_PROVIDER", "HERMES_KB_PROVIDER", "KB_LLM_PROVIDER", "KB_PROVIDER"),
            "unknown",
        ),
        "kb_model": _short(
            _first_env(env, "HERMES_KB_LLM_MODEL", "HERMES_KB_MODEL", "KB_LLM_MODEL", "KB_OPENAI_COMPAT_MODEL"),
            "unknown",
        ),
        "kb_reasoning": _short(
            _first_env(env, "HERMES_KB_REASONING_EFFORT", "KB_LLM_REASONING_EFFORT"),
            "unknown",
        ),
    }


def _primary_provider_target(provider_data: Any) -> dict[str, Any]:
    if not isinstance(provider_data, dict):
        return {}
    targets = provider_data.get("targets")
    if not isinstance(targets, list):
        return {}
    dict_targets = [target for target in targets if isinstance(target, dict)]
    for item in dict_targets:
        if _short(item.get("role"), "").lower() == "primary":
            return item
    return dict_targets[0] if dict_targets else {}


def _provider_status_summary(provider_data: Any, fallback: dict[str, str] | None = None) -> dict[str, str]:
    primary = _primary_provider_target(provider_data)
    if not primary:
        fallback = fallback or {}
        return {
            "provider": fallback.get("kb_provider") or "unknown",
            "model": fallback.get("kb_model") or "unknown",
            "reasoning": fallback.get("kb_reasoning") or "unknown",
            "status": "unknown",
        }
    return {
        "provider": _short(primary.get("adapter") or primary.get("provider")),
        "model": _short(primary.get("model")),
        "reasoning": _short(primary.get("reasoning_effort"), "not set"),
        "status": _short(primary.get("status")),
    }


def _live_hermes_reasoning(gateway: Any, source: Any) -> str | None:
    resolver = getattr(gateway, "_resolve_session_reasoning_config", None)
    if not callable(resolver):
        return None
    try:
        reasoning_config = resolver(source=source)
    except TypeError:
        try:
            reasoning_config = resolver()
        except Exception:
            return None
    except Exception:
        return None
    if not isinstance(reasoning_config, dict):
        return None
    if reasoning_config.get("enabled") is False:
        return "none"
    effort = str(reasoning_config.get("effort") or "").strip().lower()
    return effort or None


def _render_status(
    data: Any,
    target: str,
    provider_data: Any | None = None,
    *,
    hermes_reasoning: str | None = None,
) -> dict[str, Any]:
    snap = _config_snapshot()
    if hermes_reasoning:
        snap["reasoning"] = hermes_reasoning
    kb = _provider_status_summary(provider_data, snap)
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
        f"Hermes model: {snap['model']}",
        f"Hermes provider/API: {snap['provider']} / {snap['api_mode']} / {snap['api']}",
        f"Hermes reasoning: {snap['reasoning']}",
        f"KB provider: {kb['provider']}",
        f"KB model: {kb['model']}",
        f"KB reasoning: {kb['reasoning']}",
        f"Readiness: {readiness}",
        f"Publication: {publication}",
    ]
    return {"title": "KB Status", "text": "\n".join(lines), "actions": []}


def _render_runs(data: Any) -> dict[str, Any]:
    if isinstance(data, str):
        return {"title": "KB Runs", "text": f"KB Runs\n{data}", "actions": []}
    if isinstance(data, dict):
        runs = []
        for found in (
            _get_path(data, "active"),
            _get_path(data, "runs", "active"),
            _get_path(data, "recent"),
            _get_path(data, "runs", "recent"),
        ):
            if isinstance(found, list):
                runs.extend(found)
        if not runs:
            found = _get_path(data, "runs")
            runs = found if isinstance(found, list) else []
    else:
        runs = []
    lines = ["KB Runs"]
    if not runs:
        lines.append("No active or recent run details returned.")
    for idx, run in enumerate(runs[:6], start=1):
        if isinstance(run, dict):
            status = _short(run.get("status") or run.get("state") or run.get("phase"))
            detail = _short(run.get("summary") or run.get("message") or run.get("updated_at"), "")
            staleness = run.get("staleness") if isinstance(run.get("staleness"), dict) else {}
            if staleness.get("stale"):
                detail = f"stalled {_short(staleness.get('last_trace_age_seconds'), 'unknown')}s"
                action = _short(run.get("recommended_next_action"), "")
                if action:
                    detail += f"; {action}"
            suffix = f" - {detail}" if detail else ""
            title = _short(run.get("run_id") or run.get("workflow_id") or _item_title(run))
            lines.append(f"{idx}. {title}: {status}{suffix}")
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


def _descriptor_tool_name(target: str, tool_name: Any) -> str:
    value = str(tool_name or "").strip()
    if not value:
        return ""
    if value.startswith("mcp_"):
        return value
    return _mcp_tool_name(target, value)


def _queue_descriptor_actions(ctx: Any, target: str, item: dict[str, Any], *, index: int) -> list[Any]:
    try:
        from tools.kb_callback_registry import KbAction
    except Exception:
        return []

    actions: list[Any] = []
    for descriptor in _safe_actions_for_item(item):
        if descriptor.get("dashboard_owned_write") is True:
            continue
        params = descriptor.get("params") if isinstance(descriptor.get("params"), dict) else {}
        decision = str(params.get("decision") or "").strip().lower()
        preview_tool = _descriptor_tool_name(target, descriptor.get("preview_tool"))
        confirm_tool = _descriptor_tool_name(target, descriptor.get("confirm_tool"))
        if not decision or not preview_tool or not confirm_tool:
            continue
        label = _short(descriptor.get("label") or decision.replace("_", " ").title(), decision.title())
        action_id = _short(descriptor.get("action_id") or f"queue.{decision}", f"queue.{decision}")
        descriptor_copy = dict(descriptor)
        actions.append(
            KbAction(
                label=f"Preview {label}",
                action_id=f"{action_id}.preview",
                handler=lambda callback_ctx, d=descriptor_copy: _render_queue_descriptor_preview(
                    ctx,
                    target,
                    item,
                    index=index,
                    descriptor=d,
                    callback_ctx=callback_ctx,
                ),
                metadata={
                    "target_kind": "proposal_queue",
                    "target_ref": _item_target(item),
                    "decision": decision,
                    "preview_tool": preview_tool,
                    "confirm_tool": confirm_tool,
                },
            )
        )
    return actions[:4]


def _queue_descriptor_call_args(
    descriptor: dict[str, Any],
    item: dict[str, Any],
    *,
    decision: str,
    actor: str,
    source: str,
    note: str,
) -> dict[str, Any]:
    params = descriptor.get("params") if isinstance(descriptor.get("params"), dict) else {}
    proposal_ids = [str(proposal_id) for proposal_id in (params.get("proposal_ids") or []) if str(proposal_id)]
    if not proposal_ids:
        proposal_ids = _proposal_ids_for_item(item)
    args = dict(params)
    args["proposal_ids"] = proposal_ids
    args["decision"] = decision
    args["actor"] = actor
    args["source"] = source
    args["note"] = note
    return args


def _queue_callback_actor(callback_ctx: Any) -> str:
    actor_id = str(getattr(callback_ctx, "actor_id", "") or "").strip()
    return f"telegram:{actor_id}" if actor_id else "telegram:operator"


def _render_queue_descriptor_preview(
    ctx: Any,
    target: str,
    item: dict[str, Any],
    *,
    index: int,
    descriptor: dict[str, Any],
    callback_ctx: Any,
) -> dict[str, Any]:
    try:
        from tools.kb_callback_registry import KbAction
    except Exception:
        return {"title": "KB Queue", "text": "KB Queue\nAction buttons are unavailable. Use /kb queue to refresh.", "actions": []}

    params = descriptor.get("params") if isinstance(descriptor.get("params"), dict) else {}
    decision = str(params.get("decision") or "").strip().lower()
    if not decision:
        return {"title": "KB Queue", "text": "KB Queue\nThis action is missing a proposal decision.", "actions": []}
    proposal_ids = [str(proposal_id) for proposal_id in (params.get("proposal_ids") or []) if str(proposal_id)] or _proposal_ids_for_item(item)
    actor = _queue_callback_actor(callback_ctx)
    source = "Hermes Telegram Action Card"
    preview_tool = _descriptor_tool_name(target, descriptor.get("preview_tool"))
    preview_payload = _result_payload(
        ctx.dispatch_tool(
            preview_tool,
            _queue_descriptor_call_args(
                descriptor,
                item,
                decision=decision,
                actor=actor,
                source=source,
                note=f"Previewed from Telegram action card for {_item_title(item)}",
            ),
        )
    )
    selection = [(index, item)]
    text = _preview_text(decision, proposal_ids, preview_payload, selection=selection)
    if not _preview_allows_confirmation(preview_payload):
        return {"title": "KB Queue", "text": text, "actions": []}
    label = _short(descriptor.get("label") or decision.replace("_", " ").title(), decision.title())
    action_id = _short(descriptor.get("action_id") or f"queue.{decision}", f"queue.{decision}")
    confirm_action = KbAction(
        label=f"Confirm {label}",
        action_id=f"{action_id}.confirm",
        handler=lambda confirm_ctx: _render_queue_descriptor_confirm(
            ctx,
            target,
            item,
            index=index,
            descriptor=descriptor,
            callback_ctx=confirm_ctx,
        ),
        metadata={
            "target_kind": "proposal_queue",
            "target_ref": _item_target(item),
            "decision": decision,
            "preview_required": True,
        },
    )
    return {
        "title": "KB Queue",
        "text": text + "\n\nConfirm with the button below only if the preview matches your intent.",
        "actions": [confirm_action],
    }


def _render_queue_descriptor_confirm(
    ctx: Any,
    target: str,
    item: dict[str, Any],
    *,
    index: int,
    descriptor: dict[str, Any],
    callback_ctx: Any,
) -> dict[str, Any]:
    params = descriptor.get("params") if isinstance(descriptor.get("params"), dict) else {}
    decision = str(params.get("decision") or "").strip().lower()
    if not decision:
        return {"title": "KB Queue", "text": "KB Queue\nThis action is missing a proposal decision.", "actions": []}
    proposal_ids = [str(proposal_id) for proposal_id in (params.get("proposal_ids") or []) if str(proposal_id)] or _proposal_ids_for_item(item)
    actor = _queue_callback_actor(callback_ctx)
    source = "Hermes Telegram Action Card"
    preview_tool = _descriptor_tool_name(target, descriptor.get("preview_tool"))
    confirmed_tool = _descriptor_tool_name(target, descriptor.get("confirm_tool"))
    selection = [(index, item)]
    preview_payload = _result_payload(
        ctx.dispatch_tool(
            preview_tool,
            _queue_descriptor_call_args(
                descriptor,
                item,
                decision=decision,
                actor=actor,
                source=source,
                note=f"Re-previewed before Telegram action-card confirmation for {_item_title(item)}",
            ),
        )
    )
    if not _preview_allows_confirmation(preview_payload):
        return {"title": "KB Queue", "text": _preview_text(decision, proposal_ids, preview_payload, selection=selection), "actions": []}
    confirmed_args = _queue_descriptor_call_args(
        descriptor,
        item,
        decision=decision,
        actor=actor,
        source=source,
        note=f"Confirmed from Telegram action card for {_item_title(item)}",
    )
    confirmed_args["session_id"] = f"telegram-kb-card-{int(time.time())}"
    confirmed_args["user_confirmation"] = {
        "confirmed": True,
        "surface": "telegram",
        "action": f"queue.{decision}",
        "preview_required": True,
        "confirmation_text": str(descriptor.get("confirmation_copy") or f"Confirm {decision}"),
        "proposal_ids": proposal_ids,
    }
    confirmed_payload = _result_payload(ctx.dispatch_tool(confirmed_tool, confirmed_args))
    return {
        "title": "KB Queue",
        "text": _confirmed_text(decision, confirmed_payload, selection=selection, proposal_ids=proposal_ids),
        "actions": [],
    }


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


def _queue_reply_choices_from_text(text: str) -> list[str]:
    lowered = str(text or "").lower()
    lowered = re.sub(r"\bdetails\b", "detail", lowered)
    ordered = ["complete", "keep", "demote", "archive", "reject", "approve", "detail", "skip"]
    return [decision for decision in ordered if re.search(rf"\b{re.escape(decision)}\b", lowered)]


def _infer_iterative_queue_title(body: str) -> str:
    lines = body.splitlines()
    last_proposal_line = -1
    for idx, line in enumerate(lines):
        if re.search(r"(?i)\bproposal ids?\b", line) and re.search(r"\bact_[A-Za-z0-9]+\b", line):
            last_proposal_line = idx
    search_lines = lines[:last_proposal_line] if last_proposal_line >= 0 else lines
    skip_prefixes = (
        "next item",
        "proposal",
        "todo",
        "summary",
        "created",
        "reply",
        "applied",
        "archived proposal",
        "proposal ids",
        "options",
    )
    for raw_line in reversed(search_lines):
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*#+\s*", "", line).strip()
        line = re.sub(r"^\s*\d+[\).]\s*", "", line).strip()
        proposal_title_match = re.match(r"(?i)^proposal\s+\d+\s*[—:-]\s*(.+)$", line)
        if proposal_title_match:
            return proposal_title_match.group(1).strip("`*_ ")
        if line.startswith("-"):
            continue
        if line.lower().startswith(skip_prefixes):
            continue
        return line.strip("`*_ ")
    return ""


def _parse_iterative_queue_reply_state(response_text: str) -> dict[str, Any] | None:
    text = str(response_text or "")
    reply_matches = list(
        re.finditer(
            r"(?im)^\s*(?:Reply(?:\s+with)?|Options(?:\s+presented)?)\s*:\s*(.+)$",
            text,
        )
    )
    if not reply_matches:
        return None
    reply_match = reply_matches[-1]
    choices = _queue_reply_choices_from_text(reply_match.group(1))
    if not choices:
        return None
    body = text[: reply_match.start()].rstrip()
    proposal_ids: list[str] = []
    for line in body.splitlines():
        if re.search(r"(?i)\bproposal ids?\b", line):
            ids = re.findall(r"\bact_[A-Za-z0-9]+\b", line)
            if ids:
                proposal_ids = ids
    if not proposal_ids:
        all_ids = re.findall(r"\bact_[A-Za-z0-9]+\b", body)
        proposal_ids = all_ids[-1:] if all_ids else []
    if not proposal_ids:
        return None
    title = _infer_iterative_queue_title(body)
    return {
        "schema_version": 1,
        "proposal_ids": proposal_ids,
        "title": title,
        "choices": choices,
        "recorded_at": time.time(),
    }


def _record_iterative_queue_reply_state(session_id: str, response_text: str) -> dict[str, Any] | None:
    if not session_id:
        return None
    parsed = _parse_iterative_queue_reply_state(response_text)
    states = _load_queue_reply_states()
    if not parsed:
        if session_id in states:
            states.pop(session_id, None)
            _save_queue_reply_states(states)
        return None
    states[session_id] = parsed
    _save_queue_reply_states(states)
    return parsed


def _get_iterative_queue_reply_state(session_id: str) -> dict[str, Any] | None:
    if not session_id:
        return None
    states = _load_queue_reply_states()
    state = states.get(session_id)
    if not isinstance(state, dict):
        return None
    recorded_at = float(state.get("recorded_at") or 0.0)
    if recorded_at and time.time() - recorded_at > QUEUE_REPLY_STATE_TTL_SECONDS:
        states.pop(session_id, None)
        _save_queue_reply_states(states)
        return None
    proposal_ids = [str(item) for item in (state.get("proposal_ids") or []) if str(item).strip()]
    if not proposal_ids:
        return None
    state["proposal_ids"] = proposal_ids
    return state


def _bare_queue_reply_decision(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = {"details": "detail", "show": "detail"}.get(lowered, lowered)
    return lowered if lowered in QUEUE_REPLY_DECISIONS else ""


def scoped_mcp_tool_allowlist_for_message(
    *,
    session_id: str,
    message: str,
    target: str | None = None,
) -> set[str]:
    """Return exact MCP tools allowed by a matched pending queue action.

    This is the stateful bridge between Telegram action cards and the generic
    MCP posture filter.  A bare reply such as ``Reject`` should not globally
    expose confirmed tools.  It may expose only the queue preview/confirmed
    tools when there is fresh queue state for the current gateway session and
    the reply is one of the choices shown for that visible item.
    """
    decision = _bare_queue_reply_decision(message)
    if decision not in QUEUE_REPLY_TOOL_DECISIONS:
        return set()
    state = _get_iterative_queue_reply_state(session_id)
    if not state:
        return set()
    choices = {str(choice).strip().lower() for choice in (state.get("choices") or []) if str(choice).strip()}
    if choices and decision not in choices:
        return set()
    mcp_target = target or _mcp_target()
    return {
        _mcp_tool_name(mcp_target, "queue.decision_preview"),
        _mcp_tool_name(mcp_target, "queue.decide_confirmed"),
        _mcp_tool_name(mcp_target, "queue.batch_decide_confirmed"),
        _mcp_tool_name(mcp_target, "queue.decisions_confirmed"),
    }


def _session_id_for_queue_reply_state(session_store: Any, source: Any) -> str:
    if session_store is None:
        return ""
    try:
        session_store._ensure_loaded()
    except Exception:
        pass
    try:
        session_key = session_store._generate_session_key(source)
        entry = getattr(session_store, "_entries", {}).get(session_key)
        return str(getattr(entry, "session_id", "") or "")
    except Exception:
        logger.debug("kb_journeys: failed to resolve gateway session id", exc_info=True)
        return ""


def _conversation_state_id(session_store: Any, source: Any) -> str:
    session_id = _session_id_for_queue_reply_state(session_store, source)
    if session_id:
        return session_id
    parts = [
        _platform_name(getattr(source, "platform", None)),
        _short(getattr(source, "chat_id", ""), ""),
        _short(getattr(source, "thread_id", ""), ""),
        _short(getattr(source, "user_id", ""), ""),
    ]
    return ":".join(part for part in parts if part) or "telegram"


def _get_meeting_handoff_state(session_id: str) -> dict[str, Any] | None:
    if not session_id:
        return None
    states = _load_meeting_handoff_states()
    state = states.get(session_id)
    if not isinstance(state, dict):
        return None
    recorded_at = float(state.get("recorded_at") or 0.0)
    if recorded_at and time.time() - recorded_at > MEETING_HANDOFF_STATE_TTL_SECONDS:
        states.pop(session_id, None)
        _save_meeting_handoff_states(states)
        return None
    plan = state.get("plan")
    if not isinstance(plan, dict):
        return None
    return state


def _store_meeting_handoff_state(
    session_id: str,
    *,
    plan: dict[str, Any],
    meeting_file: str,
    notes_text: str,
) -> None:
    if not session_id:
        return
    states = _load_meeting_handoff_states()
    states[session_id] = {
        "schema_version": 1,
        "recorded_at": time.time(),
        "meeting_file": _short(meeting_file, ""),
        "notes_sha256": "sha256:" + hashlib.sha256(str(notes_text or "").encode("utf-8")).hexdigest(),
        "notes_chars": len(str(notes_text or "")),
        "plan": plan,
    }
    _save_meeting_handoff_states(states)


def _queue_count(data: Any) -> int | None:
    if not isinstance(data, dict):
        return None
    counts = data.get("counts") if isinstance(data.get("counts"), dict) else {}
    for value in (counts.get("proposals"), data.get("total"), data.get("count"), _count_from(data, "queue", "proposals")):
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _iterative_state_from_item(item: dict[str, Any]) -> dict[str, Any] | None:
    proposal_ids = _proposal_ids_for_item(item)
    if not proposal_ids:
        return None
    decisions = [decision for decision, _ in _queue_action_decisions(item)]
    for fallback in ("detail", "skip"):
        if fallback not in decisions:
            decisions.append(fallback)
    return {
        "schema_version": 1,
        "proposal_ids": proposal_ids,
        "title": _item_title(item),
        "choices": decisions,
        "recorded_at": time.time(),
    }


def _store_iterative_state_from_item(session_id: str, item: dict[str, Any] | None) -> None:
    if not session_id:
        return
    if not isinstance(item, dict):
        _clear_iterative_queue_reply_state(session_id)
        return
    state = _iterative_state_from_item(item)
    if not state:
        _clear_iterative_queue_reply_state(session_id)
        return
    states = _load_queue_reply_states()
    states[session_id] = state
    _save_queue_reply_states(states)


def _iterative_queue_next_text(data: Any, *, session_id: str = "") -> str:
    items = _queue_items_from_payload(data)
    if not items:
        if session_id:
            _clear_iterative_queue_reply_state(session_id)
        return "Queue is empty."
    item = items[0] if isinstance(items[0], dict) else {}
    if not item:
        return "Next queue item could not be rendered. Use /kb queue to refresh."
    _store_iterative_state_from_item(session_id, item)
    count = _queue_count(data)
    proposal_ids = _proposal_ids_for_item(item)
    decisions = [decision for decision, _ in _queue_action_decisions(item)]
    for fallback in ("detail", "skip"):
        if fallback not in decisions:
            decisions.append(fallback)
    lines: list[str] = []
    if count is not None:
        lines.append(f"Queue now has {count} proposal(s).")
        lines.append("")
    lines.extend(["Next item:", "", _item_title(item)])
    detail = _item_detail(item)
    if detail:
        lines.append("- Summary: " + _clip(detail, 240))
    target = _item_target(item)
    if target:
        lines.append("- Target: " + target)
    if proposal_ids:
        label = "Proposal ids" if len(proposal_ids) != 1 else "Proposal id"
        lines.append(f"- {label}: {', '.join(proposal_ids[:8])}")
    lines.append("")
    lines.append("Reply: " + ", ".join(decisions) + ".")
    return "\n".join(lines)


def _iterative_selection_from_state(state: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    return [
        (
            1,
            {
                "title": _short(state.get("title"), "Current queue item"),
                "raw": {"proposal_ids": list(state.get("proposal_ids") or [])},
            },
        )
    ]


def _render_iterative_queue_reply_decision(
    ctx: Any,
    target: str,
    *,
    session_id: str,
    state: dict[str, Any],
    decision: str,
) -> dict[str, Any]:
    proposal_ids = [str(item) for item in (state.get("proposal_ids") or []) if str(item).strip()]
    title = _short(state.get("title"), "current queue item")
    selection = _iterative_selection_from_state(state)
    if decision == "detail":
        lines = [
            "Queue Item",
            f"Title: {title}",
            f"Proposal ids: {', '.join(proposal_ids)}",
            "Reply: " + ", ".join(state.get("choices") or sorted(QUEUE_REPLY_DECISIONS)) + ".",
        ]
        return {"title": "KB Queue", "text": "\n".join(lines), "actions": []}
    if decision not in QUEUE_REPLY_TOOL_DECISIONS:
        return {"title": "KB Queue", "text": "That queue reply is not supported. Use /kb queue to refresh.", "actions": []}
    actor = "telegram:operator"
    source = "Hermes Telegram iterative queue"
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
                "note": f"Previewed from Telegram iterative queue reply for {title}",
            },
        )
    )
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
                "session_id": session_id or f"telegram-kb-reply-{int(time.time())}",
                "user_confirmation": {
                    "confirmed": True,
                    "surface": "telegram",
                    "action": f"queue.{decision}",
                    "preview_required": True,
                    "confirmation_text": decision,
                    "proposal_ids": proposal_ids,
                },
                "note": f"Confirmed from Telegram iterative queue reply for {title}",
            },
        )
    )
    text = _confirmed_text(decision, confirmed_payload, selection=selection, proposal_ids=proposal_ids)
    text = text.replace("\nNext: /kb queue", "")
    if isinstance(confirmed_payload, dict) and confirmed_payload.get("ok") is False:
        return {"title": "KB Queue", "text": text, "actions": []}
    data, errors = _queue_summary_payload(ctx, target, limit=5)
    next_text = _iterative_queue_next_text(data, session_id=session_id)
    if errors:
        next_text += "\nQueue refresh warning: " + "; ".join(errors[:2])
    return {"title": "KB Queue", "text": text.rstrip() + "\n\n" + next_text, "actions": []}


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
        "actions": _queue_descriptor_actions(ctx, target, item, index=index),
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
    text = _workflow_status_text("Workflow start result", payload, include_run_details=False)
    run_id = _workflow_run_id(payload)
    if run_id:
        progress_text = _workflow_initial_progress_text(ctx, target, run_id, include_run_details=False)
        if progress_text:
            text += "\n" + progress_text
        text += "\nDetails: /kb runs"
    return text


def _meeting_handoff_args(args: str) -> dict[str, Any]:
    text = str(args or "").strip()
    if text.lower() in {"confirm", "confirmed", "start", "apply"}:
        return {"confirm_pending": True}
    if not text:
        return {"error": "meeting_file and notes are required"}
    if text.lower().startswith("process "):
        text = text.split(maxsplit=1)[1].strip()
    if " -- " in text:
        meeting_file, notes_text = text.split(" -- ", 1)
    elif "\n" in text:
        first, notes_text = text.split("\n", 1)
        meeting_file = first
    else:
        return {"error": "separate the meeting file from notes with --"}
    meeting_file = _strip_wrapping_quotes(meeting_file.strip())
    notes_text = notes_text.strip()
    if not meeting_file:
        return {"error": "meeting_file is required"}
    if not notes_text:
        return {"error": "notes are required"}
    return {
        "confirm_pending": False,
        "meeting_file": meeting_file,
        "notes_text": notes_text,
    }


def _strip_wrapping_quotes(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def _telegram_actor(source: Any) -> str:
    name = _short(getattr(source, "user_name", ""), "")
    user_id = _short(getattr(source, "user_id", ""), "")
    return f"telegram:{name or user_id or 'operator'}"


def _render_meeting_handoff_command(
    ctx: Any,
    target: str,
    args: str,
    *,
    source: Any = None,
    session_store: Any = None,
    adapter: Any = None,
) -> dict[str, Any]:
    parsed = _meeting_handoff_args(args)
    session_id = _conversation_state_id(session_store, source)
    if parsed.get("confirm_pending"):
        state = _get_meeting_handoff_state(session_id)
        if not state:
            return {
                "title": "Meeting Notes",
                "text": "Meeting Notes\nNo pending meeting handoff found for this chat.",
                "actions": [],
            }
        plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
        text = _workflow_start_text(ctx, target, plan)
        _clear_meeting_handoff_state(session_id)
        return {"title": "Meeting Notes", "text": text, "actions": []}
    if parsed.get("error"):
        return {"title": "Meeting Notes", "text": f"Meeting Notes\n{parsed['error']}", "actions": []}

    meeting_file = str(parsed.get("meeting_file") or "")
    notes_text = str(parsed.get("notes_text") or "")
    plan_args = {
        "meeting_file": meeting_file,
        "source_kind": "telegram",
        "source_notes_source": "telegram",
        "source_notes_text": notes_text,
        "harness_id": "telegram-hermes",
        "harness_session_id": session_id,
    }
    _, data, errors = _dispatch_first(
        ctx,
        target,
        [
            (
                "workflow.plan_request",
                {
                    "workflow_id": "meeting_process",
                    "args": plan_args,
                    "actor": _telegram_actor(source),
                    "source": "Hermes Telegram",
                    "session_id": session_id or f"telegram-meeting-{int(time.time())}",
                },
            )
        ],
    )
    if data is None:
        return _render_error("Meeting Notes", target, errors)
    if isinstance(data, dict) and data.get("status") == "confirmation_required":
        _store_meeting_handoff_state(
            session_id,
            plan=data,
            meeting_file=meeting_file,
            notes_text=notes_text,
        )
    card = _render_workflow_plan(
        data,
        ctx=ctx,
        target=target,
        adapter=adapter,
        start_hint="/kb meeting confirm",
    )
    card["title"] = "Meeting Notes"
    return card


def _workflow_initial_progress_text(
    ctx: Any,
    target: str,
    run_id: str,
    *,
    include_run_details: bool = True,
) -> str:
    payload = _result_payload(
        ctx.dispatch_tool(
            _mcp_tool_name(target, "run.watch"),
            {
                "run_id": run_id,
                "timeout_seconds": 0,
                "poll_interval_seconds": 1,
                "timeline_limit": 5,
            },
        )
    )
    if isinstance(payload, dict) and payload.get("error"):
        return "Initial progress: unavailable - " + _short(payload.get("error"))
    if not isinstance(payload, dict):
        return ""
    digest = payload.get("progress_digest") if isinstance(payload.get("progress_digest"), dict) else payload
    progress = digest.get("progress") if isinstance(digest.get("progress"), dict) else {}
    stage = digest.get("stage") if isinstance(digest.get("stage"), dict) else {}
    provider = digest.get("provider") if isinstance(digest.get("provider"), dict) else {}
    staleness = digest.get("staleness") if isinstance(digest.get("staleness"), dict) else {}
    phase = _short(progress.get("current_phase") or progress.get("current_step") or digest.get("status"), "")
    detail = _short(progress.get("current_detail") or progress.get("latest_message"), "")
    lines: list[str] = []
    if phase:
        lines.append(
            f"Initial progress: {phase}" + (f" - {detail}" if detail and include_run_details else "")
        )
    if include_run_details:
        stage_id = _short(stage.get("stage_id") or stage.get("call_name"), "")
        total = stage.get("total")
        if stage_id and total not in {None, ""}:
            try:
                failed_count = int(stage.get("failed") or 0)
            except (TypeError, ValueError):
                failed_count = 0
            lines.append(
                f"Stage: {stage_id} {_short(stage.get('completed'), '0')}/{_short(total, '0')}"
                + (f" failed {_short(stage.get('failed'), '0')}" if failed_count else "")
            )
        provider_name = _short(provider.get("provider"), "")
        model = _short(provider.get("model"), "")
        if provider_name or model:
            lines.append(f"Provider: {provider_name or 'unknown'} / {model or 'unknown'}")
    if staleness.get("stale"):
        age = _short(staleness.get("last_trace_age_seconds"), "unknown")
        lines.append(f"Attention: run appears stalled; no trace progress for {age}s")
    if include_run_details:
        if payload.get("terminal") is False:
            lines.append("Watch: still running")
        elif payload.get("terminal") is True:
            lines.append("Watch: terminal")
    return "\n".join(lines)


def _workflow_run_id(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
    return str(payload.get("run_id") or run.get("run_id") or "")


def _workflow_status_text(prefix: str, payload: Any, *, include_run_details: bool = True) -> str:
    if isinstance(payload, dict) and payload.get("error"):
        return f"{prefix}\n{payload['error']}"
    if not isinstance(payload, dict):
        return f"{prefix}\n{_short(payload, 'No structured response returned.')}"
    lines = [
        prefix,
        f"Status: {_short(payload.get('status'))}",
    ]
    run_id = _workflow_run_id(payload)
    if run_id and include_run_details:
        lines.append(f"Run: {run_id}")
    if payload.get("started") is not None:
        lines.append(f"Started: {_short(payload.get('started'))}")
    follow = payload.get("followthrough_contract") if isinstance(payload.get("followthrough_contract"), dict) else {}
    if follow and include_run_details:
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
        lines.append("Batch: /kb queue reject 1,2")
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
    if key in {"reasoning", "reasoning-effort", "kb-reasoning"}:
        return "kbreasoning", rest
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
    if key in {"meeting", "meetings", "notes"}:
        return "kbmeeting", rest
    if key == "sync":
        return "kbrun", f"sync {rest}".strip()
    return "kbhelp", text


def _kb_command_help() -> dict[str, Any]:
    return {
        "title": "KB",
        "text": "\n".join(
            [
                "KB Commands",
                "/kb - compact cockpit/status",
                "/kb queue - proposal review list",
                "/kb queue review 1 - inspect one queue item",
                "/kb queue reject 1 - preview a decision",
                "/kb queue reject 1 confirm - apply a previewed decision",
                "/kb publish - preview KB Git publication",
                "/kb publish confirm - commit and push after preview",
                "/kb status - lane, Hermes/KB reasoning, readiness, publication",
                "/kb reasoning xhigh - set KB engine reasoning and reload MCP",
                "/kb runs - active and recent workflow runs",
                "/kb run sync - preview a KB sync",
                "/kb meeting <meeting-file> -- <notes> - preview Telegram notes handoff",
            ]
        ),
        "actions": [],
    }


def _normalize_kb_reasoning_effort(args: str) -> tuple[str, str]:
    effort = ((args or "").strip().split(maxsplit=1) or [""])[0].lower()
    if not effort:
        return "", f"Send /kb reasoning <level>. Options: {', '.join(sorted(KB_REASONING_LEVELS))}."
    if effort not in KB_REASONING_LEVELS:
        return "", f"Unknown KB reasoning effort '{effort}'. Options: {', '.join(sorted(KB_REASONING_LEVELS))}."
    return effort, ""


def _render_kb_reasoning_command(args: str, *, reload_available: bool) -> dict[str, Any]:
    effort, error = _normalize_kb_reasoning_effort(args)
    if error:
        return {"title": "KB Reasoning", "text": f"KB Reasoning\n{error}", "actions": []}
    try:
        from hermes_cli.config import get_env_path, save_env_value

        save_env_value("HERMES_KB_REASONING_EFFORT", effort)
        env_path = get_env_path()
    except Exception as exc:
        logger.warning("kb_journeys: failed to save KB reasoning effort", exc_info=True)
        return {
            "title": "KB Reasoning",
            "text": f"KB Reasoning\nCould not save KB reasoning effort: {_short(exc)}",
            "actions": [],
        }

    reload_line = "MCP reload started." if reload_available else "Run /reload-mcp to apply it to the KB MCP server."
    return {
        "title": "KB Reasoning",
        "text": "\n".join(
            [
                f"KB reasoning set to {effort}.",
                f"Saved: {env_path}:HERMES_KB_REASONING_EFFORT",
                reload_line,
            ]
        ),
        "actions": [],
        "_reload_mcp": reload_available,
    }


def _card_for_command(
    ctx: Any,
    command: str,
    *,
    args: str = "",
    adapter: Any = None,
    gateway: Any = None,
    source: Any = None,
    session_store: Any = None,
) -> dict[str, Any]:
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
            return _card_for_command(
                ctx,
                routed_command,
                args=routed_args,
                adapter=adapter,
                gateway=gateway,
                source=source,
                session_store=session_store,
            )
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
        _, provider_data, _provider_errors = _dispatch_first(ctx, target, [("provider.status", {})])
        hermes_reasoning = _live_hermes_reasoning(gateway, source)
        return _render_status(data, target, provider_data, hermes_reasoning=hermes_reasoning)
    if command == "kbreasoning":
        reload_available = callable(getattr(gateway, "_execute_mcp_reload", None))
        return _render_kb_reasoning_command(args, reload_available=reload_available)
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
    if command == "kbmeeting":
        return _render_meeting_handoff_command(
            ctx,
            target,
            args,
            source=source,
            session_store=session_store,
            adapter=adapter,
        )
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


async def _send_mcp_reload_result(adapter: Any, event: Any, gateway: Any) -> None:
    reload_fn = getattr(gateway, "_execute_mcp_reload", None)
    if not callable(reload_fn):
        return
    try:
        result = reload_fn(event)
        if inspect.isawaitable(result):
            result = await result
        text = "KB MCP Reload\n" + _short(result, "complete")
    except Exception as exc:
        logger.warning("kb_journeys: MCP reload after KB reasoning change failed", exc_info=True)
        text = f"KB MCP Reload\nReload failed: {_short(exc)}"
    await _send_card(adapter, event, {"title": "KB MCP Reload", "text": text, "actions": []})


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
        text = getattr(event, "text", "")
        command = _command_from_text(text)
        bare_decision = _bare_queue_reply_decision(text)
        if command is None and not bare_decision:
            return None
        if not _authorized_for_gateway(gateway, source):
            return None
        adapter = _adapter_for(gateway, source)
        if adapter is None:
            logger.debug("kb_journeys: no Telegram adapter available")
            return None
        if bare_decision:
            session_id = _session_id_for_queue_reply_state(session_store, source)
            state = _get_iterative_queue_reply_state(session_id)
            if not state:
                return None
            card = _render_iterative_queue_reply_decision(
                ctx,
                _mcp_target(),
                session_id=session_id,
                state=state,
                decision=bare_decision,
            )
        else:
            args = _command_args_from_text(text)
            card = _card_for_command(
                ctx,
                command,
                args=args,
                adapter=adapter,
                gateway=gateway,
                source=source,
                session_store=session_store,
            )
        reload_mcp = bool(card.pop("_reload_mcp", False))
        _run_delivery(_send_card(adapter, event, card))
        if reload_mcp:
            _run_delivery(_send_mcp_reload_result(adapter, event, gateway))
        return {"action": "skip", "reason": "kb_journeys"}

    return _hook


def _on_post_llm_call(
    *,
    session_id: str = "",
    assistant_response: str = "",
    platform: str = "",
    **_: Any,
) -> None:
    if str(platform or "").lower() != "telegram":
        return
    _record_iterative_queue_reply_state(session_id, assistant_response)


def register(ctx: Any) -> None:
    def _command_help(_: str = "") -> str:
        return "Use /kb in Telegram. Try: /kb queue, /kb status, /kb reasoning xhigh, /kb run sync."

    for command in sorted(MENU_COMMANDS):
        try:
            ctx.register_command(
                command,
                _command_help,
                description="KB dashboard, queue, status, reasoning, runs, and sync.",
            )
        except Exception:
            logger.debug("kb_journeys: failed to register /%s", command, exc_info=True)
    ctx.register_hook("pre_gateway_dispatch", build_pre_gateway_dispatch_hook(ctx))
    ctx.register_hook("post_llm_call", _on_post_llm_call)
