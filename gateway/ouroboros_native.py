"""Native core router for the gateway /ooo command family.

This module intentionally contains only parsing, safe native routing, MCP tool
selection, concise response formatting, and recent-ID state updates.  Gateway
slash-command handlers wrap this router instead of falling through to legacy
CLI or skill-bridge paths.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
import shlex
from typing import Any, Callable

from gateway.ouroboros_commands import (
    OOO_NATIVE_COMMANDS,
    OOO_ROUTER_ONLY_SUBCOMMANDS,
    OOO_SUBCOMMANDS,
)
from gateway.ouroboros_state import (
    OooRecentState,
    OooStateContext,
    OooStateStore,
    extract_ids,
)

_DEFAULT_TIMEOUT = 45.0
_DEFAULT_QA_QUALITY_BAR = "General correctness, completeness, and actionable quality."
_BOOLEAN_FLAGS = frozenset(
    {
        "force",
        "skip_qa",
        "skip_run",
        "consensus",
        "complete_product",
        "default_only",
        "is_default",
        "execute",
        "parallel",
        "trigger_consensus",
    }
)
_STOP_GATED_COMMANDS = frozenset({"setup", "config", "update", "publish"})
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SAFE_PAYLOAD_SUMMARY_KEYS = (
    "question",
    "prompt",
    "message",
    "status",
    "summary",
    "score",
    "verdict",
    "passed",
    "result",
    "final_output",
    "output",
)
_SUMMARY_VALUE_LIMIT = 360
_SECRET_PATTERNS = (
    (re.compile(r"(?i)\bsk-[A-Za-z0-9._-]{4,}\b"), "sk-[REDACTED]"),
    (
        re.compile(r"(?i)\b(token|api[_-]?key|secret|password|authorization)\s*=\s*[^\s;,&]+"),
        lambda match: f"{match.group(1)}=[REDACTED]",
    ),
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{6,}"), lambda match: f"{match.group(1)}[REDACTED]"),
)


@dataclass(frozen=True)
class OooCommand:
    name: str
    args_text: str
    tokens: list[str]


@dataclass
class OooNativeContext:
    cwd: str | None = None
    state_context: OooStateContext | None = None
    state_store: OooStateStore | None = None
    mcp_caller: Callable[[str, dict[str, Any] | None, float], dict[str, Any]] | None = None
    allow_mutating_side_effects: bool = False
    idempotency_key: str | None = None


@dataclass
class OooNativeResponse:
    text: str
    payload: dict[str, Any] | None = None
    used_tool: str | None = None
    state_updates: dict[str, str] | None = None


def parse_ooo_command(raw: str) -> OooCommand:
    """Parse a raw /ooo tail into a normalized native command.

    ``raw`` may be either the tail after ``/ooo`` or the full ``/ooo ...`` text.
    Malformed shell quoting is intentionally allowed to raise ``ValueError``;
    :func:`handle_ooo_native` catches it and returns a usage response.
    """

    raw_text = "" if raw is None else str(raw).strip()
    if raw_text.startswith("/ooo"):
        raw_text = raw_text[4:].strip()
    elif raw_text.lower() == "ooo":
        raw_text = ""
    elif raw_text.lower().startswith("ooo "):
        raw_text = raw_text[4:].strip()

    if not raw_text:
        return OooCommand(name="help", args_text="", tokens=[])

    tokens = shlex.split(raw_text)
    if not tokens:
        return OooCommand(name="help", args_text="", tokens=[])

    name = tokens[0].lower().replace("_", "-")
    aliases = {
        "h": "help",
        "?": "help",
        "init": "interview",
        "resume": "resume-session",
    }
    name = aliases.get(name, name)
    command_tokens = tokens[1:]
    return OooCommand(name=name, args_text=" ".join(command_tokens), tokens=command_tokens)


def _strip_ansi(text: Any) -> str:
    return _ANSI_RE.sub("", str(text))


def _redact_sensitive_text(text: Any) -> str:
    """Strip ANSI and redact obvious secret-like substrings for user-visible text."""

    clean = _strip_ansi(text)
    try:
        from agent.redact import redact_sensitive_text
    except Exception:  # pragma: no cover - optional dependency path.
        redacted = clean
    else:
        try:
            redacted = str(redact_sensitive_text(clean))
        except Exception:  # pragma: no cover - defensive optional redactor path.
            redacted = clean

    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _shorten(text: str, limit: int = 1800) -> str:
    clean = _strip_ansi(text).replace("Traceback", "traceback")
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def _normalize_flag_name(token: str) -> str:
    return token.lstrip("-").strip().lower().replace("-", "_")


def _split_flags(tokens: list[str], bool_flags: set[str] | frozenset[str] = _BOOLEAN_FLAGS) -> tuple[dict[str, Any], list[str]]:
    flags: dict[str, Any] = {}
    positionals: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--") and len(token) > 2:
            if "=" in token:
                raw_key, raw_value = token[2:].split("=", 1)
                key = _normalize_flag_name(raw_key)
                flags[key] = True if key in bool_flags and raw_value == "" else raw_value
                index += 1
                continue

            key = _normalize_flag_name(token)
            next_value = tokens[index + 1] if index + 1 < len(tokens) else None
            if key in bool_flags or next_value is None or next_value.startswith("--"):
                flags[key] = True
                index += 1
            else:
                flags[key] = next_value
                index += 2
            continue

        positionals.append(token)
        index += 1
    return flags, positionals


def _flag_value(flags: dict[str, Any], *names: str) -> Any | None:
    """Return the first non-boolean value for value-taking flags."""

    for name in names:
        value = flags.get(name)
        if value is None or value is True:
            continue
        return value
    return None


def _flag_bool(flags: dict[str, Any], name: str) -> bool:
    value = flags.get(name)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _as_int(value: Any, field_name: str) -> tuple[int | None, str | None]:
    try:
        return int(str(value)), None
    except (TypeError, ValueError):
        return None, f"{field_name} 값은 정수여야 합니다."


def _as_float(value: Any, field_name: str) -> tuple[float | None, str | None]:
    try:
        return float(str(value)), None
    except (TypeError, ValueError):
        return None, f"{field_name} 값은 숫자여야 합니다."


def _load_recent(ctx: OooNativeContext) -> tuple[OooRecentState, str | None]:
    if ctx.state_store is None or ctx.state_context is None:
        return OooRecentState(), None
    try:
        return ctx.state_store.load(ctx.state_context), None
    except Exception as exc:  # pragma: no cover - defensive path.
        return OooRecentState(), f"최근 상태를 읽지 못했습니다: {type(exc).__name__}"


def _state_note_suffix(note: str | None) -> str:
    return f"\n참고: {note}" if note else ""


def _usage(message: str, *, payload: dict[str, Any] | None = None) -> OooNativeResponse:
    text = f"사용법: {message}"
    return OooNativeResponse(text=_shorten(text), payload=payload)


def _help_response() -> OooNativeResponse:
    commands = ", ".join(OOO_SUBCOMMANDS)
    text = (
        "네이티브 /ooo 라우터입니다.\n"
        "주요 사용: `/ooo interview <요구사항>`, `/ooo run --seed-path seed.yaml`, "
        "`/ooo status --job <job_id>`, `/ooo status --session <session_id>`.\n"
        f"지원 명령: {commands}. 라우터 전용: job."
    )
    return OooNativeResponse(text=_shorten(text), payload={"commands": list(OOO_SUBCOMMANDS), "router_only": ["job"]})


def _static_response(command: str) -> OooNativeResponse:
    if command in _STOP_GATED_COMMANDS:
        return OooNativeResponse(
            text=(
                f"/ooo {command} 는 변경 가능성이 있는 작업이라 현재 네이티브 라우터에서 중단(stop-gate)되었습니다. "
                "명시적 승인 및 별도 구현 전에는 MCP/CLI를 호출하지 않습니다."
            ),
            payload={"stop_gated": True, "command": command},
        )
    if command == "welcome":
        return OooNativeResponse(
            text="환영합니다. `/ooo help`로 네이티브 Ouroboros 명령 요약을 볼 수 있습니다.",
            payload={"command": command},
        )
    if command == "tutorial":
        return OooNativeResponse(
            text="튜토리얼: 1) `/ooo interview <목표>` 2) `/ooo seed` 3) `/ooo run --seed-path <file>` 4) `/ooo status --job <id>`.",
            payload={"command": command},
        )
    if command == "resume-session":
        return OooNativeResponse(
            text="세션 재개 안내: 최근 ID를 쓰거나 `/ooo status --session <session_id>` 및 각 명령의 `--session <id>`를 사용하세요.",
            payload={"command": command},
        )
    raise ValueError(f"not a static command: {command}")


def _mcp_caller(ctx: OooNativeContext) -> Callable[[str, dict[str, Any] | None, float], dict[str, Any]]:
    if ctx.mcp_caller is not None:
        return ctx.mcp_caller
    from gateway.ouroboros_mcp import call_ouroboros_tool

    return call_ouroboros_tool


def _coerce_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    return {"success": True, "result": payload}


def _display_ids(payload: dict[str, Any]) -> dict[str, str]:
    state_ids = extract_ids(payload)
    display: dict[str, str] = {}
    mapping = {
        "interview_session_id": "interview_session_id",
        "pm_session_id": "pm_session_id",
        "auto_session_id": "auto_session_id",
        "last_session_id": "session_id",
        "last_job_id": "job_id",
        "last_execution_id": "execution_id",
        "last_lineage_id": "lineage_id",
        "last_seed_id": "seed_id",
    }
    for state_key, display_key in mapping.items():
        value = state_ids.get(state_key)
        if value is not None:
            display[display_key] = value

    for key in (
        "session_id",
        "interview_session_id",
        "pm_session_id",
        "auto_session_id",
        "job_id",
        "execution_id",
        "lineage_id",
        "seed_id",
    ):
        value = payload.get(key)
        if value is not None:
            display[key] = str(value)
    return display


def _summary_scalar(value: Any) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, bool) or isinstance(value, (int, float)):
        text = str(value)
    elif value is None:
        text = "None"
    else:
        text = f"{type(value).__name__}"
    return _shorten(_redact_sensitive_text(text).replace("\n", " "), _SUMMARY_VALUE_LIMIT)


def _summary_value(value: Any) -> str:
    if isinstance(value, dict):
        nested = [
            f"{key}={_summary_scalar(value[key])}"
            for key in _SAFE_PAYLOAD_SUMMARY_KEYS
            if key in value and value[key] is not None
        ]
        if nested:
            return _shorten(", ".join(nested), _SUMMARY_VALUE_LIMIT)
        return f"dict({len(value)} keys)"
    if isinstance(value, list | tuple):
        if not value:
            return "[]"
        first = _summary_scalar(value[0])
        suffix = f", +{len(value) - 1} more" if len(value) > 1 else ""
        return _shorten(f"[{first}{suffix}]", _SUMMARY_VALUE_LIMIT)
    return _summary_scalar(value)


def _payload_summary_parts(payload: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    for key in _SAFE_PAYLOAD_SUMMARY_KEYS:
        if key not in payload or payload[key] is None:
            continue
        value = _summary_value(payload[key])
        if value:
            parts.append(f"{key}={value}")
    return parts


def _state_updates_for_tool(tool_name: str, payload: dict[str, Any]) -> dict[str, str]:
    updates = dict(extract_ids(payload))
    session_id = updates.get("last_session_id")
    if tool_name == "ouroboros_interview" and session_id:
        updates["interview_session_id"] = session_id
    elif tool_name == "ouroboros_pm_interview" and session_id:
        updates["pm_session_id"] = session_id
    return updates


def _save_state_updates(
    ctx: OooNativeContext,
    tool_name: str,
    payload: dict[str, Any],
) -> tuple[dict[str, str], str | None]:
    updates = _state_updates_for_tool(tool_name, payload)
    if not updates or ctx.state_store is None or ctx.state_context is None:
        return updates, None
    try:
        ctx.state_store.update(ctx.state_context, **updates)
        return updates, None
    except Exception as exc:  # pragma: no cover - defensive path.
        return updates, f"최근 상태 저장 실패: {type(exc).__name__}"


def _format_tool_response(
    *,
    command_name: str,
    tool_name: str,
    args: dict[str, Any],
    payload: dict[str, Any],
    state_updates: dict[str, str],
    fallback_note: str | None = None,
    state_note: str | None = None,
) -> OooNativeResponse:
    if payload.get("success") is False or payload.get("error"):
        error = _redact_sensitive_text(payload.get("error", "알 수 없는 오류"))
        safe_payload = dict(payload)
        safe_payload["error"] = error
        text = f"오류: {tool_name} 호출 실패 - {error}. 다음: 인자와 최근 ID를 확인한 뒤 다시 실행하세요."
        return OooNativeResponse(
            text=_shorten(text + _state_note_suffix(state_note)),
            payload=safe_payload,
            used_tool=tool_name,
            state_updates=state_updates or None,
        )

    ids = _display_ids(payload)
    parts = [f"완료: {tool_name}"]
    if ids:
        parts.append(", ".join(f"{key}={value}" for key, value in ids.items()))
    summary_parts = _payload_summary_parts(payload)
    if summary_parts:
        parts.append(", ".join(summary_parts))
    if fallback_note:
        parts.append(fallback_note)
    job_id = ids.get("job_id")
    if job_id and tool_name.startswith("ouroboros_start_"):
        parts.append(f"다음: `/ooo status --job {job_id}`")
    if command_name == "qa" and "quality_bar" in args:
        parts.append(f"Quality bar: {args['quality_bar']}")
    text = "; ".join(parts) + _state_note_suffix(state_note)
    return OooNativeResponse(
        text=_shorten(text),
        payload=payload,
        used_tool=tool_name,
        state_updates=state_updates or None,
    )


async def _call_tool(
    *,
    command_name: str,
    tool_name: str,
    args: dict[str, Any],
    ctx: OooNativeContext,
    fallback_note: str | None = None,
    state_note: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> OooNativeResponse:
    caller = _mcp_caller(ctx)
    try:
        raw_payload = await asyncio.to_thread(caller, tool_name, args, timeout)
        payload = _coerce_payload(raw_payload)
    except Exception as exc:
        error = _redact_sensitive_text(f"{type(exc).__name__}: {exc}")
        payload = {"success": False, "error": error, "tool": tool_name}
        return _format_tool_response(
            command_name=command_name,
            tool_name=tool_name,
            args=args,
            payload=payload,
            state_updates={},
            fallback_note=fallback_note,
            state_note=state_note,
        )

    if payload.get("success") is False or payload.get("error"):
        return _format_tool_response(
            command_name=command_name,
            tool_name=tool_name,
            args=args,
            payload=payload,
            state_updates={},
            fallback_note=fallback_note,
            state_note=state_note,
        )

    state_updates, save_note = _save_state_updates(ctx, tool_name, payload)
    merged_note = "; ".join(note for note in (state_note, save_note) if note) or None
    return _format_tool_response(
        command_name=command_name,
        tool_name=tool_name,
        args=args,
        payload=payload,
        state_updates=state_updates,
        fallback_note=fallback_note,
        state_note=merged_note,
    )


def _require_value(value: Any, usage: str) -> OooNativeResponse | None:
    if value is None or value == "":
        return _usage(usage)
    return None


def _join_positionals(positionals: list[str]) -> str:
    return " ".join(positionals).strip()


def _add_cwd(args: dict[str, Any], ctx: OooNativeContext, key: str = "cwd") -> None:
    if ctx.cwd:
        args[key] = ctx.cwd


async def _handle_interview(command: OooCommand, ctx: OooNativeContext, recent: OooRecentState, state_note: str | None) -> OooNativeResponse:
    flags, positionals = _split_flags(command.tokens)
    session_id = _flag_value(flags, "session", "session_id")
    if positionals and positionals[0].lower() == "answer":
        answer = _join_positionals(positionals[1:])
        session_id = session_id or recent.interview_session_id
        if missing := _require_value(session_id, "/ooo interview answer <답변> (최근 interview_session_id가 없으면 --session <id> 필요)"):
            return missing
        if missing := _require_value(answer, "/ooo interview answer <답변>"):
            return missing
        args = {"session_id": str(session_id), "answer": answer}
    elif session_id:
        answer = _join_positionals(positionals)
        args = {"session_id": str(session_id)}
        if answer:
            args["answer"] = answer
    else:
        args = {}
        initial_context = _join_positionals(positionals)
        if initial_context:
            args["initial_context"] = initial_context
        _add_cwd(args, ctx)
    return await _call_tool(
        command_name="interview",
        tool_name="ouroboros_interview",
        args=args,
        ctx=ctx,
        state_note=state_note,
    )


async def _handle_pm(command: OooCommand, ctx: OooNativeContext, recent: OooRecentState, state_note: str | None) -> OooNativeResponse:
    flags, positionals = _split_flags(command.tokens)
    action = str(_flag_value(flags, "action") or "").lower().replace("_", "-")
    if positionals and not action and positionals[0].lower() in {"generate", "answer"}:
        action = positionals[0].lower()
        positionals = positionals[1:]

    session_id = _flag_value(flags, "session", "session_id")
    if action == "generate":
        session_id = session_id or recent.pm_session_id
        if missing := _require_value(session_id, "/ooo pm generate --session <pm_session_id> (또는 최근 PM 세션 필요)"):
            return missing
        args = {"session_id": str(session_id), "action": "generate"}
    elif action == "answer":
        session_id = session_id or recent.pm_session_id
        answer = _join_positionals(positionals)
        if missing := _require_value(session_id, "/ooo pm answer <답변> --session <pm_session_id>"):
            return missing
        if missing := _require_value(answer, "/ooo pm answer <답변>"):
            return missing
        args = {"session_id": str(session_id), "answer": answer}
    elif session_id:
        args = {"session_id": str(session_id)}
        answer = _join_positionals(positionals)
        if answer:
            args["answer"] = answer
    else:
        args = {}
        initial_context = _join_positionals(positionals)
        if initial_context:
            args["initial_context"] = initial_context
        _add_cwd(args, ctx)
    return await _call_tool(
        command_name="pm",
        tool_name="ouroboros_pm_interview",
        args=args,
        ctx=ctx,
        state_note=state_note,
    )


async def _handle_seed(command: OooCommand, ctx: OooNativeContext, recent: OooRecentState, state_note: str | None) -> OooNativeResponse:
    flags, _positionals = _split_flags(command.tokens)
    session_id = _flag_value(flags, "session", "session_id") or recent.interview_session_id or recent.pm_session_id
    if missing := _require_value(session_id, "/ooo seed --session <session_id> [--force] (또는 최근 interview/pm 세션 필요)"):
        return missing
    args: dict[str, Any] = {"session_id": str(session_id)}
    if _flag_bool(flags, "force"):
        args["force"] = True
    ambiguity_score = _flag_value(flags, "ambiguity_score")
    if ambiguity_score is not None:
        value, error = _as_float(ambiguity_score, "ambiguity_score")
        if error:
            return _usage(error)
        args["ambiguity_score"] = value
    return await _call_tool(
        command_name="seed",
        tool_name="ouroboros_generate_seed",
        args=args,
        ctx=ctx,
        state_note=state_note,
    )


async def _handle_run(command: OooCommand, ctx: OooNativeContext, recent: OooRecentState, state_note: str | None) -> OooNativeResponse:
    flags, positionals = _split_flags(command.tokens)
    args: dict[str, Any] = {}
    seed_path = _flag_value(flags, "seed_path")
    seed_content = _flag_value(flags, "seed", "seed_content")
    if not seed_path and not seed_content and positionals:
        seed_path = positionals[0]
    if seed_path:
        args["seed_path"] = str(seed_path)
    if seed_content:
        args["seed_content"] = str(seed_content)
    session_id = _flag_value(flags, "session", "session_id")
    if session_id:
        args["session_id"] = str(session_id)
    elif not seed_path and not seed_content and recent.last_session_id:
        args["session_id"] = recent.last_session_id
    if missing := _require_value(args.get("seed_path") or args.get("seed_content") or args.get("session_id"), "/ooo run --seed-path <file> | --seed <yaml> [--session <id>]"):
        return missing
    _add_cwd(args, ctx)
    max_iterations = _flag_value(flags, "max_iterations")
    if max_iterations is not None:
        value, error = _as_int(max_iterations, "max_iterations")
        if error:
            return _usage(error)
        args["max_iterations"] = value
    if _flag_bool(flags, "skip_qa"):
        args["skip_qa"] = True
    model_tier = _flag_value(flags, "model_tier")
    if model_tier is not None:
        args["model_tier"] = str(model_tier)
    if ctx.idempotency_key:
        args["idempotency_key"] = ctx.idempotency_key
    return await _call_tool(
        command_name="run",
        tool_name="ouroboros_start_execute_seed",
        args=args,
        ctx=ctx,
        state_note=state_note,
    )


async def _handle_evaluate(command: OooCommand, ctx: OooNativeContext, recent: OooRecentState, state_note: str | None) -> OooNativeResponse:
    flags, positionals = _split_flags(command.tokens)
    session_id = _flag_value(flags, "session", "session_id")
    artifact = _flag_value(flags, "artifact")
    if not session_id and positionals:
        session_id = positionals[0]
        positionals = positionals[1:]
    if not artifact and positionals:
        artifact = positionals[0]
    if not session_id and recent.last_session_id and artifact:
        session_id = recent.last_session_id
    if missing := _require_value(session_id, "/ooo evaluate <session_id> <artifact> [--consensus]"):
        return missing
    if missing := _require_value(artifact, "/ooo evaluate <session_id> <artifact> [--consensus]"):
        return missing
    args: dict[str, Any] = {"session_id": str(session_id), "artifact": str(artifact)}
    if _flag_bool(flags, "consensus") or _flag_bool(flags, "trigger_consensus"):
        args["trigger_consensus"] = True
    _add_cwd(args, ctx, key="working_dir")
    return await _call_tool(
        command_name="evaluate",
        tool_name="ouroboros_start_evaluate",
        args=args,
        ctx=ctx,
        state_note=state_note,
    )


async def _handle_qa(command: OooCommand, ctx: OooNativeContext, _recent: OooRecentState, state_note: str | None) -> OooNativeResponse:
    flags, positionals = _split_flags(command.tokens)
    artifact = _flag_value(flags, "artifact") or (_join_positionals(positionals) if positionals else None)
    if missing := _require_value(artifact, "/ooo qa --artifact <text-or-path> [--bar <quality bar>] [--type <type>]"):
        return missing
    args: dict[str, Any] = {
        "artifact": str(artifact),
        "quality_bar": str(_flag_value(flags, "bar", "quality_bar") or _DEFAULT_QA_QUALITY_BAR),
        "artifact_type": str(_flag_value(flags, "type", "artifact_type") or "code"),
    }
    threshold = _flag_value(flags, "threshold")
    if threshold is not None:
        value, error = _as_float(threshold, "threshold")
        if error:
            return _usage(error)
        args["pass_threshold"] = value
    reference = _flag_value(flags, "reference")
    if reference is not None:
        args["reference"] = str(reference)
    return await _call_tool(
        command_name="qa",
        tool_name="ouroboros_qa",
        args=args,
        ctx=ctx,
        state_note=state_note,
    )


async def _handle_status(command: OooCommand, ctx: OooNativeContext, recent: OooRecentState, state_note: str | None) -> OooNativeResponse:
    flags, positionals = _split_flags(command.tokens)
    fallback_note = None
    job_id = _flag_value(flags, "job", "job_id")
    session_id = _flag_value(flags, "session", "session_id")
    if not job_id and not session_id and positionals:
        if positionals[0].lower() == "job" and len(positionals) > 1:
            job_id = positionals[1]
        elif positionals[0].lower() == "session" and len(positionals) > 1:
            session_id = positionals[1]
        else:
            session_id = positionals[0]
    if not job_id and not session_id:
        if recent.last_job_id:
            job_id = recent.last_job_id
            fallback_note = "최근 job_id를 사용했습니다."
        elif recent.last_session_id:
            session_id = recent.last_session_id
            fallback_note = "최근 session_id를 사용했습니다."
    if job_id:
        return await _call_tool(
            command_name="status",
            tool_name="ouroboros_job_status",
            args={"job_id": str(job_id)},
            ctx=ctx,
            fallback_note=fallback_note,
            state_note=state_note,
        )
    if session_id:
        return await _call_tool(
            command_name="status",
            tool_name="ouroboros_session_status",
            args={"session_id": str(session_id)},
            ctx=ctx,
            fallback_note=fallback_note,
            state_note=state_note,
        )
    return _usage("/ooo status --job <job_id> | --session <session_id> (또는 최근 ID 필요)")


async def _handle_job(command: OooCommand, ctx: OooNativeContext, recent: OooRecentState, state_note: str | None) -> OooNativeResponse:
    flags, positionals = _split_flags(command.tokens)
    action = "status"
    if positionals and positionals[0].lower() in {"status", "wait", "result"}:
        action = positionals[0].lower()
        positionals = positionals[1:]
    elif positionals and positionals[0].lower() not in {"status", "wait", "result"}:
        action = str(_flag_value(flags, "action") or "status")

    job_id = _flag_value(flags, "job", "job_id") or (positionals[0] if positionals else None)
    fallback_note = None
    if not job_id and recent.last_job_id:
        job_id = recent.last_job_id
        fallback_note = "최근 job_id를 사용했습니다."
    if missing := _require_value(job_id, "/ooo status --job <job_id> 또는 /ooo job wait|result <job_id> (또는 최근 job_id 필요)"):
        return missing

    tool_map = {
        "status": "ouroboros_job_status",
        "wait": "ouroboros_job_wait",
        "result": "ouroboros_job_result",
    }
    tool_name = tool_map.get(action)
    if tool_name is None:
        return _usage("/ooo status --job <job_id> 또는 /ooo job wait|result <job_id>")
    args: dict[str, Any] = {"job_id": str(job_id)}
    if action == "wait":
        timeout_alias = _flag_value(flags, "timeout")
        if timeout_alias is not None and _flag_value(flags, "timeout_seconds") is None:
            flags["timeout_seconds"] = timeout_alias
        for flag_name in ("cursor", "timeout_seconds"):
            raw_value = _flag_value(flags, flag_name)
            if raw_value is not None:
                value, error = _as_int(raw_value, flag_name)
                if error:
                    return _usage(error)
                args[flag_name] = value
        for flag_name in ("view", "stream", "wait_for"):
            raw_value = _flag_value(flags, flag_name)
            if raw_value is not None:
                args[flag_name] = str(raw_value)
    return await _call_tool(
        command_name="job",
        tool_name=tool_name,
        args=args,
        ctx=ctx,
        fallback_note=fallback_note,
        state_note=state_note,
    )


async def _handle_cancel(command: OooCommand, ctx: OooNativeContext, _recent: OooRecentState, state_note: str | None) -> OooNativeResponse:
    flags, positionals = _split_flags(command.tokens)
    job_id = _flag_value(flags, "job", "job_id")
    execution_id = _flag_value(flags, "execution", "execution_id")
    if not job_id and not execution_id and len(positionals) >= 2:
        kind = positionals[0].lower()
        if kind == "job":
            job_id = positionals[1]
        elif kind in {"execution", "exec"}:
            execution_id = positionals[1]
    if not job_id and not execution_id:
        return OooNativeResponse(
            text="안전 중단: cancel은 최근 ID를 추정하지 않습니다. `/ooo cancel --job <job_id>` 또는 `--execution <execution_id>`를 명시하세요.",
            payload={"blocked": True, "reason": "explicit id required"},
        )
    if job_id and execution_id:
        return _usage("/ooo cancel --job <job_id> 또는 /ooo cancel --execution <execution_id> 중 하나만 지정")
    if job_id:
        return await _call_tool(
            command_name="cancel",
            tool_name="ouroboros_cancel_job",
            args={"job_id": str(job_id)},
            ctx=ctx,
            state_note=state_note,
        )
    return await _call_tool(
        command_name="cancel",
        tool_name="ouroboros_cancel_execution",
        args={"execution_id": str(execution_id)},
        ctx=ctx,
        state_note=state_note,
    )


async def _handle_evolve_like(command: OooCommand, ctx: OooNativeContext, recent: OooRecentState, state_note: str | None) -> OooNativeResponse:
    flags, positionals = _split_flags(command.tokens)
    lineage_id = _flag_value(flags, "lineage", "lineage_id") or (positionals[0] if positionals else None) or recent.last_lineage_id
    if missing := _require_value(lineage_id, f"/ooo {command.name} <lineage_id> [--skip-qa]"):
        return missing
    args: dict[str, Any] = {"lineage_id": str(lineage_id)}
    seed_content = _flag_value(flags, "seed", "seed_content")
    if seed_content is not None:
        args["seed_content"] = str(seed_content)
    if _flag_bool(flags, "skip_qa"):
        args["skip_qa"] = True
    if "max_generations" in flags:
        if command.name != "ralph":
            return _usage("/ooo evolve 는 단일 generation 실행입니다. 여러 generation은 /ooo ralph <lineage_id> --max-generations <n> 을 사용하세요.")
        max_generations = _flag_value(flags, "max_generations")
        if max_generations is None:
            return _usage("max_generations 값은 정수여야 합니다.")
        value, error = _as_int(max_generations, "max_generations")
        if error:
            return _usage(error)
        args["max_generations"] = value
    commit_policy = _flag_value(flags, "commit_policy")
    if commit_policy is not None:
        args["commit_policy"] = str(commit_policy)
    _add_cwd(args, ctx, key="project_dir")
    tool_name = "ouroboros_start_ralph" if command.name == "ralph" else "ouroboros_start_evolve_step"
    return await _call_tool(
        command_name=command.name,
        tool_name=tool_name,
        args=args,
        ctx=ctx,
        state_note=state_note,
    )


async def _handle_auto(command: OooCommand, ctx: OooNativeContext, _recent: OooRecentState, state_note: str | None) -> OooNativeResponse:
    flags, positionals = _split_flags(command.tokens)
    args: dict[str, Any] = {}
    goal = _flag_value(flags, "goal") or _join_positionals(positionals)
    if goal:
        args["goal"] = str(goal)
    _add_cwd(args, ctx)
    for flag_name in ("resume", "attach_execution", "attach_job", "attach_session", "attach_source", "reconcile_source", "domain", "commit_policy", "worktree_policy"):
        raw_value = _flag_value(flags, flag_name)
        if raw_value is not None:
            args[flag_name] = str(raw_value)
    for flag_name in ("skip_run", "complete_product", "reconcile_run"):
        if _flag_bool(flags, flag_name):
            args[flag_name] = True
    for flag_name in ("max_interview_rounds", "max_repair_rounds"):
        raw_value = _flag_value(flags, flag_name)
        if raw_value is not None:
            value, error = _as_int(raw_value, flag_name)
            if error:
                return _usage(error)
            args[flag_name] = value
    pipeline_timeout = _flag_value(flags, "pipeline_timeout_seconds")
    if pipeline_timeout is not None:
        value, error = _as_float(pipeline_timeout, "pipeline_timeout_seconds")
        if error:
            return _usage(error)
        args["pipeline_timeout_seconds"] = value
    return await _call_tool(
        command_name="auto",
        tool_name="ouroboros_start_auto",
        args=args,
        ctx=ctx,
        state_note=state_note,
    )


async def _handle_brownfield(command: OooCommand, ctx: OooNativeContext, _recent: OooRecentState, state_note: str | None) -> OooNativeResponse:
    flags, positionals = _split_flags(command.tokens)
    action = str(_flag_value(flags, "action") or (positionals[0] if positionals else "query")).lower().replace("_", "-")
    remaining = positionals[1:] if positionals else []

    def blocked(reason: str) -> OooNativeResponse:
        return OooNativeResponse(
            text=(
                "중단(stop-gate): brownfield는 기본적으로 조회 전용입니다. "
                f"{reason} MCP를 호출하지 않았습니다."
            ),
            payload={"blocked": True, "stop_gated": True, "command": "brownfield", "action": action, "reason": reason},
        )

    read_only_actions = {"query", "list", "show", "status", "default-only"}
    mutating_allowed_actions = read_only_actions | {"scan", "default"}
    supported_actions = mutating_allowed_actions if ctx.allow_mutating_side_effects else read_only_actions
    if action not in supported_actions:
        return blocked(f"action `{action}` 는 현재 허용 범위가 아닙니다.")

    mutating_flags = ("path", "name", "desc", "is_default", "indices", "scan_root", "root")
    present_mutating_flags = [flag_name for flag_name in mutating_flags if flag_name in flags]
    if present_mutating_flags and not ctx.allow_mutating_side_effects:
        return blocked(f"변경/선택 가능 플래그 {', '.join(sorted(present_mutating_flags))} 가 감지되었습니다.")

    # Ouroboros MCP only accepts scan/register/query/set_default/set_defaults.
    # Keep Discord-friendly read aliases native-side, but route read-only forms
    # to the actual safe MCP action instead of leaking unsupported action names.
    mcp_action = "query" if action in {"list", "show", "status", "default-only"} else action
    if action == "default":
        mcp_action = "query" if not ctx.allow_mutating_side_effects else "set_default"

    args: dict[str, Any] = {"action": mcp_action}
    if action in {"default", "default-only"}:
        args["default_only"] = True
    for flag_name in ("path", "name", "desc"):
        raw_value = _flag_value(flags, flag_name)
        if raw_value is not None:
            args[flag_name] = str(raw_value)
    if _flag_bool(flags, "default_only"):
        args["default_only"] = True
    if _flag_bool(flags, "is_default"):
        args["is_default"] = True
    scan_root = _flag_value(flags, "scan_root", "root")
    if scan_root:
        args["scan_root"] = str(scan_root)
    elif action == "scan" and remaining:
        args["scan_root"] = remaining[0]
    indices = _flag_value(flags, "indices")
    if indices is not None:
        args["indices"] = str(indices)
    for flag_name in ("offset", "limit"):
        raw_value = _flag_value(flags, flag_name)
        if raw_value is not None:
            value, error = _as_int(raw_value, flag_name)
            if error:
                return _usage(error)
            args[flag_name] = value
    return await _call_tool(
        command_name="brownfield",
        tool_name="ouroboros_brownfield",
        args=args,
        ctx=ctx,
        state_note=state_note,
    )


async def _handle_unstuck(command: OooCommand, ctx: OooNativeContext, _recent: OooRecentState, state_note: str | None) -> OooNativeResponse:
    flags, positionals = _split_flags(command.tokens)
    problem = _flag_value(flags, "problem", "problem_context") or _join_positionals(positionals)
    if missing := _require_value(problem, "/ooo unstuck <문제 설명> [--persona hacker|researcher|simplifier|architect|contrarian|all]"):
        return missing
    args: dict[str, Any] = {
        "problem_context": str(problem),
        "current_approach": str(_flag_value(flags, "current_approach", "approach") or "현재 접근 미제공"),
    }
    persona = _flag_value(flags, "persona")
    if persona is not None:
        args["persona"] = str(persona)
    stagnation_pattern = _flag_value(flags, "stagnation_pattern")
    if stagnation_pattern is not None:
        args["stagnation_pattern"] = str(stagnation_pattern)
    return await _call_tool(
        command_name="unstuck",
        tool_name="ouroboros_lateral_think",
        args=args,
        ctx=ctx,
        state_note=state_note,
    )


async def handle_ooo_native(raw: str, ctx: OooNativeContext) -> OooNativeResponse:
    """Handle one native /ooo command without invoking the legacy CLI gateway."""

    try:
        command = parse_ooo_command(raw)
    except ValueError as exc:
        return _usage(f"따옴표/quote 파싱 오류: {exc}. 예: /ooo help")

    if command.name == "help":
        return _help_response()
    if command.name in _STOP_GATED_COMMANDS or command.name in {"welcome", "tutorial", "resume-session"}:
        return _static_response(command.name)
    if command.name not in OOO_NATIVE_COMMANDS:
        return _usage(f"알 수 없는 /ooo 명령 `{command.name}`. `/ooo help`를 확인하세요.")

    recent, state_note = _load_recent(ctx)

    if command.name == "interview":
        return await _handle_interview(command, ctx, recent, state_note)
    if command.name == "pm":
        return await _handle_pm(command, ctx, recent, state_note)
    if command.name == "seed":
        return await _handle_seed(command, ctx, recent, state_note)
    if command.name == "run":
        return await _handle_run(command, ctx, recent, state_note)
    if command.name == "evaluate":
        return await _handle_evaluate(command, ctx, recent, state_note)
    if command.name == "qa":
        return await _handle_qa(command, ctx, recent, state_note)
    if command.name == "status":
        return await _handle_status(command, ctx, recent, state_note)
    if command.name == "job":
        return await _handle_job(command, ctx, recent, state_note)
    if command.name == "cancel":
        return await _handle_cancel(command, ctx, recent, state_note)
    if command.name in {"evolve", "ralph"}:
        return await _handle_evolve_like(command, ctx, recent, state_note)
    if command.name == "auto":
        return await _handle_auto(command, ctx, recent, state_note)
    if command.name == "brownfield":
        return await _handle_brownfield(command, ctx, recent, state_note)
    if command.name == "unstuck":
        return await _handle_unstuck(command, ctx, recent, state_note)

    return _usage(f"/ooo {command.name} 는 아직 네이티브 라우터에 구현되지 않았습니다. `/ooo help`를 확인하세요.")
