#!/usr/bin/env python3
"""Hermes-to-Arcane JSONL adapter.

Normal mode reads one JSON request from stdin and writes one Arcane run event
per stdout line. Use ``--self-test`` to verify imports and JSONL output
without starting a model run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.events import AgentEventSink, safe_debug, safe_preview, utc_now_iso


class StdoutArcaneEventSink(AgentEventSink):
    def __init__(self, session_id: str, run_id: str):
        self.session_id = session_id
        self.run_id = run_id
        self._lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("sessionId", self.session_id)
        payload.setdefault("runId", self.run_id)
        self._normalize_timestamp(payload)
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    @staticmethod
    def _normalize_timestamp(payload: dict[str, Any]) -> None:
        now = utc_now_iso()
        event_type = payload.get("type")
        if event_type == "tool.call.started":
            payload.setdefault("createdAt", now)
        elif event_type in {"tool.call.completed", "tool.call.failed"}:
            payload.setdefault("completedAt", now)
        elif event_type in {"assistant.delta", "assistant.message"}:
            payload.setdefault("createdAt", now)
        else:
            payload.setdefault("updatedAt", now)


class ArcaneStreamEmitter:
    """Emit Arcane assistant.delta events for visible assistant text."""

    def __init__(self, sink: StdoutArcaneEventSink, run_id: str):
        self.sink = sink
        self.message_id = f"assistant-{run_id}"
        self.index = 0

    def emit(self, delta: Any) -> None:
        if not isinstance(delta, str) or not delta:
            return
        self.sink.emit(
            {
                "type": "assistant.delta",
                "messageId": self.message_id,
                "delta": delta,
                "index": self.index,
                "createdAt": utc_now_iso(),
            }
        )
        self.index += 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Hermes for an Arcane event-stream request.")
    parser.add_argument("--self-test", action="store_true", help="emit a small JSONL event stream without a model call")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    sink: StdoutArcaneEventSink | None = None
    try:
        request = read_request()
        session_id = require_text(request, "sessionId")
        run_id = require_text(request, "runId")
        content = require_text(request, "content")
        sink = StdoutArcaneEventSink(session_id, run_id)
        configure_arcane_environment(request, session_id, run_id)

        if handle_arcane_slash_command(content, request, sink):
            return 0

        agent = build_agent(request, sink)
        result = agent.run_conversation(
            content,
            conversation_history=arcane_conversation_history(request.get("messages"), content),
            task_id=run_id,
        )
        if result.get("failed") or (result.get("error") and not result.get("final_response")):
            return 1
        return 0
    except Exception as exc:
        message = safe_preview(exc) or "Hermes Arcane adapter failed."
        print(f"hermes_arcane_adapter: {message}", file=sys.stderr)
        if sink is not None:
            sink.emit(
                {
                    "type": "run.error",
                    "message": "Hermes Arcane adapter failed.",
                    "debug": safe_debug(exc),
                    "updatedAt": utc_now_iso(),
                }
            )
        return 1


def self_test() -> int:
    from run_agent import AIAgent  # noqa: F401 - verifies runtime import path

    sink = StdoutArcaneEventSink("self-test-session", "self-test-run")
    sink.emit({"type": "run.status", "status": "thinking", "message": "adapter self-test"})
    sink.emit({"type": "run.done"})
    return 0


def read_request() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("stdin JSON request is required")
    request = json.loads(raw)
    if not isinstance(request, dict):
        raise ValueError("stdin JSON request must be an object")
    return request


def require_text(request: dict[str, Any], key: str) -> str:
    value = request.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"request.{key} is required")
    return value.strip()


def configure_arcane_environment(request: dict[str, Any], session_id: str, run_id: str) -> None:
    """Normalize Arcane env vars for Arcane tool handlers in this process."""
    arcane_cfg = request.get("arcane") if isinstance(request.get("arcane"), dict) else {}
    base_url = (
        _optional_text(request.get("arcaneBaseUrl"))
        or _optional_text(request.get("baseUrl"))
        or _optional_text(arcane_cfg.get("baseUrl"))
        or os.getenv("ARCANE_BASE_URL", "").strip()
        or "http://127.0.0.1:8787"
    )
    access_token = (
        _optional_text(request.get("arcaneAccessToken"))
        or _optional_text(request.get("accessToken"))
        or _optional_text(arcane_cfg.get("accessToken"))
        or os.getenv("ARCANE_ACCESS_TOKEN", "")
    )

    os.environ["ARCANE_SESSION_ID"] = session_id
    os.environ["ARCANE_RUN_ID"] = run_id
    os.environ["ARCANE_BASE_URL"] = base_url.rstrip("/")
    os.environ["ARCANE_ACCESS_TOKEN"] = access_token


def _optional_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def handle_arcane_slash_command(
    content: str,
    request: dict[str, Any],
    sink: StdoutArcaneEventSink,
) -> bool:
    parsed = parse_slash_command(content)
    if parsed is None:
        return False

    name, args = parsed
    from hermes_cli.commands import resolve_command

    cmd = resolve_command(name)
    if cmd is None:
        emit_slash_response(
            sink,
            f"Unknown Hermes command `/{name}`. Type `/commands` to see commands available in Arcane.",
        )
        return True

    if cmd.name == "help":
        emit_slash_response(sink, arcane_help_text())
        return True
    if cmd.name == "commands":
        emit_slash_response(sink, arcane_commands_text(args))
        return True
    if cmd.name == "profile":
        emit_slash_response(sink, arcane_profile_text())
        return True
    if cmd.name == "status":
        emit_slash_response(sink, arcane_status_text(request))
        return True
    if cmd.name == "model":
        emit_slash_response(sink, arcane_model_text(args))
        return True
    if cmd.name == "reasoning":
        emit_slash_response(sink, arcane_reasoning_text(args))
        return True
    if cmd.name in {"usage", "skills", "tools"}:
        emit_slash_response(sink, arcane_unsupported_command_text(cmd.name, cli_only=cmd.cli_only))
        return True

    emit_slash_response(sink, arcane_unsupported_command_text(cmd.name, cli_only=cmd.cli_only))
    return True


def parse_slash_command(content: str) -> tuple[str, str] | None:
    stripped = content.strip()
    if not stripped.startswith("/") or stripped == "/":
        return None
    token, _, args = stripped[1:].partition(" ")
    name = token.strip()
    if not name or "/" in name:
        return None
    return name, args.strip()


def emit_slash_response(sink: StdoutArcaneEventSink, content: str) -> None:
    sink.emit({"type": "assistant.message", "content": content, "createdAt": utc_now_iso()})
    sink.emit({"type": "run.done", "updatedAt": utc_now_iso()})


def arcane_help_text() -> str:
    from hermes_cli.commands import gateway_help_lines

    lines = [
        "Hermes commands available in Arcane:",
        *gateway_help_lines()[:18],
        "",
        "Use `/commands` for the full paginated list.",
    ]
    return "\n".join(lines)


def arcane_commands_text(args: str) -> str:
    from hermes_cli.commands import gateway_help_lines

    if args:
        try:
            requested_page = int(args.split()[0])
        except ValueError:
            return "Usage: `/commands [page]`"
    else:
        requested_page = 1

    entries = gateway_help_lines()
    if not entries:
        return "No Hermes commands are currently available in Arcane."

    page_size = 20
    total_pages = max(1, (len(entries) + page_size - 1) // page_size)
    page = max(1, min(requested_page, total_pages))
    start = (page - 1) * page_size
    page_entries = entries[start:start + page_size]
    lines = [
        f"Hermes commands ({len(entries)} total, page {page}/{total_pages}):",
        "",
        *page_entries,
    ]
    if total_pages > 1:
        nav = []
        if page > 1:
            nav.append(f"`/commands {page - 1}`")
        if page < total_pages:
            nav.append(f"`/commands {page + 1}`")
        if nav:
            lines.extend(["", "More: " + " | ".join(nav)])
    if page != requested_page:
        lines.append(f"Requested page {requested_page} is out of range; showing page {page}.")
    return "\n".join(lines)


def arcane_profile_text() -> str:
    from hermes_cli.profiles import get_active_profile_name
    from hermes_constants import display_hermes_home

    return "\n".join(
        [
            f"Active Hermes profile: `{get_active_profile_name()}`",
            f"Hermes home: `{display_hermes_home()}`",
        ]
    )


def arcane_status_text(request: dict[str, Any]) -> str:
    from hermes_cli.config import load_config

    cfg = load_config()
    model, provider, _base_url = resolve_configured_model(cfg)
    toolsets = resolve_arcane_toolsets(cfg)
    session_id = _optional_text(request.get("sessionId")) or os.getenv("ARCANE_SESSION_ID", "")
    run_id = _optional_text(request.get("runId")) or os.getenv("ARCANE_RUN_ID", "")
    token_status = "configured" if os.getenv("ARCANE_ACCESS_TOKEN") else "not configured"
    return "\n".join(
        [
            "Arcane Hermes status:",
            f"- Arcane session: `{session_id}`",
            f"- Arcane run: `{run_id}`",
            f"- Hermes session: `{stable_hermes_session_id(session_id)}`",
            f"- Model: `{model or '(auto)'}`",
            f"- Provider: `{provider or '(auto)'}`",
            f"- Arcane API: `{os.getenv('ARCANE_BASE_URL', 'http://127.0.0.1:8787')}`",
            f"- Arcane token: {token_status}",
            f"- Enabled toolsets: {', '.join(f'`{name}`' for name in toolsets) if toolsets else '(default)'}",
        ]
    )


def arcane_model_text(args: str) -> str:
    from hermes_cli.config import load_config

    if args:
        return (
            "`/model` changes are not supported from Arcane yet. "
            "Set `HERMES_INFERENCE_MODEL` or update Hermes config, then start a new Arcane run."
        )
    model, provider, base_url = resolve_configured_model(load_config())
    lines = [
        f"Current model: `{model or '(auto)'}`",
        f"Provider: `{provider or '(auto)'}`",
    ]
    if base_url:
        lines.append(f"Base URL: `{base_url}`")
    return "\n".join(lines)


def arcane_reasoning_text(args: str) -> str:
    from hermes_cli.config import cfg_get, load_config

    if args:
        return (
            "`/reasoning` changes are not supported from Arcane yet. "
            "Set `agent.reasoning_effort` in Hermes config before starting the run."
        )
    cfg = load_config()
    effort = cfg_get(cfg, "agent", "reasoning_effort", default="medium") or "medium"
    show_reasoning = bool(cfg_get(cfg, "display", "show_reasoning", default=False))
    return "\n".join(
        [
            f"Reasoning effort: `{effort}`",
            f"Reasoning display: `{'on' if show_reasoning else 'off'}`",
        ]
    )


def arcane_unsupported_command_text(command: str, *, cli_only: bool = False) -> str:
    scope = "the interactive Hermes CLI" if cli_only else "the long-lived Hermes gateway runtime"
    return (
        f"`/{command}` is registered in Hermes but is not supported on the Arcane adapter. "
        f"It depends on {scope}, so the command was not sent to the model."
    )


def resolve_configured_model(cfg: dict[str, Any]) -> tuple[str, str | None, str | None]:
    from hermes_cli.models import detect_provider_for_model

    model_cfg = cfg.get("model") or {}
    if isinstance(model_cfg, str):
        cfg_model = model_cfg
        cfg_provider = ""
    else:
        cfg_model = model_cfg.get("default") or model_cfg.get("model") or ""
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()

    env_model = os.getenv("HERMES_INFERENCE_MODEL", "").strip()
    effective_model = env_model or cfg_model
    effective_provider = os.getenv("HERMES_INFERENCE_PROVIDER", "").strip().lower() or cfg_provider or None
    explicit_base_url_from_alias = None

    if effective_provider is None and env_model:
        try:
            from hermes_cli import model_switch as _ms

            _ms._ensure_direct_aliases()
            direct = _ms.DIRECT_ALIASES.get(env_model.strip().lower())
        except Exception:
            direct = None
        if direct is not None:
            effective_model = direct.model
            effective_provider = direct.provider
            explicit_base_url_from_alias = direct.base_url.rstrip("/") if direct.base_url else None
        else:
            detected = detect_provider_for_model(env_model, cfg_provider or "auto")
            if detected:
                effective_provider, effective_model = detected

    return effective_model, effective_provider, explicit_base_url_from_alias


def resolve_arcane_toolsets(cfg: dict[str, Any]) -> list[str]:
    from hermes_cli.tools_config import _get_platform_tools

    arcane_toolsets = set(_get_platform_tools(cfg, "arcane"))
    cli_toolsets = set(_get_platform_tools(cfg, "cli"))
    if not arcane_toolsets:
        arcane_toolsets = set(cli_toolsets)
    # Arcane runs should always expose the workspace tools in addition to
    # the normal useful Hermes tool surface.
    arcane_toolsets.add("arcane")
    if arcane_toolsets == {"arcane"} and cli_toolsets:
        arcane_toolsets.update(cli_toolsets)
    return sorted(arcane_toolsets)


def build_agent(request: dict[str, Any], sink: StdoutArcaneEventSink):
    from hermes_cli.config import load_config
    from hermes_cli.oneshot import _create_session_db_for_oneshot
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from run_agent import AIAgent

    cfg = load_config()
    effective_model, effective_provider, explicit_base_url_from_alias = resolve_configured_model(cfg)

    runtime = resolve_runtime_provider(
        requested=effective_provider,
        target_model=effective_model or None,
        explicit_base_url=explicit_base_url_from_alias,
    )

    toolsets = _usable_toolsets(resolve_arcane_toolsets(cfg))

    fallback = cfg.get("fallback_providers") or cfg.get("fallback_model") or []
    if isinstance(fallback, dict):
        fallback = [fallback] if fallback.get("provider") and fallback.get("model") else []

    session_id = require_text(request, "sessionId")
    run_id = require_text(request, "runId")
    hermes_session_id = stable_hermes_session_id(session_id)
    files = request.get("files") if isinstance(request.get("files"), list) else []
    file_list = ", ".join(str(item) for item in files[:100]) or "(none)"
    system_hint = (
        f"You are responding inside Arcane session {session_id}. "
        "Arcane is a local chat and canvas workspace. If Arcane tools are available, "
        "use them to inspect or update the session artifact before replying. "
        "When writing artifact HTML, link CSS/JS with relative paths like styles.css "
        "and script.js. Keep the final reply concise. "
        f"Relevant artifact files: {file_list}."
    )

    stream_emitter = ArcaneStreamEmitter(sink, run_id)

    agent = AIAgent(
        api_key=runtime.get("api_key"),
        base_url=runtime.get("base_url"),
        provider=runtime.get("provider"),
        api_mode=runtime.get("api_mode"),
        model=effective_model,
        enabled_toolsets=toolsets,
        quiet_mode=True,
        platform="arcane",
        session_id=hermes_session_id,
        gateway_session_key=f"arcane:{session_id}",
        chat_id=session_id,
        chat_name="Arcane",
        chat_type="arcane",
        user_id=os.getenv("ARCANE_USER_ID", "arcane-local"),
        session_db=_create_session_db_for_oneshot(),
        credential_pool=runtime.get("credential_pool"),
        fallback_model=fallback or None,
        ephemeral_system_prompt=system_hint,
        event_sink=sink,
        clarify_callback=arcane_clarify_callback,
        stream_delta_callback=stream_emitter.emit,
        status_callback=lambda _kind, message: sink.emit(
            {
                "type": "run.status",
                "status": "thinking",
                "message": str(message),
                "updatedAt": utc_now_iso(),
            }
        ),
    )
    agent.suppress_status_output = True
    agent.tool_gen_callback = None
    return agent


def arcane_conversation_history(messages: Any, latest_content: str) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []

    history: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content:
            continue
        history.append({"role": role, "content": content})

    if history and history[-1]["role"] == "user" and history[-1]["content"].strip() == latest_content.strip():
        history.pop()
    return history[-20:]


def stable_hermes_session_id(arcane_session_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "-", arcane_session_id).strip("-")
    return f"arcane-{cleaned[:100] or 'session'}"


def _usable_toolsets(configured_toolsets: list[str]) -> list[str] | None:
    """Return configured toolsets only if they resolve to at least one schema.

    Arcane can be configured before the Arcane-specific toolset exists. Passing
    only unknown/empty toolsets disables every normal Hermes tool, which makes
    the adapter look alive while quietly amputated. Fallback to default tools in
    that case.
    """
    if not configured_toolsets:
        return None
    try:
        from model_tools import get_tool_definitions

        if get_tool_definitions(enabled_toolsets=configured_toolsets, quiet_mode=True):
            return configured_toolsets
    except Exception:
        return configured_toolsets
    return None


def arcane_clarify_callback(question: str, choices=None) -> str:
    if choices:
        return f"[Arcane adapter: no interactive clarification is available. Choose the best option from {choices} and continue.]"
    return "[Arcane adapter: no interactive clarification is available. Make a reasonable assumption and continue.]"


if __name__ == "__main__":
    raise SystemExit(main())
