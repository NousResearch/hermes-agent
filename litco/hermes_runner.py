"""Production :class:`~litco.turn_server.TurnRunner`: one turn of a Hermes ``AIAgent``.

Agent construction follows the API-server adapter (``gateway/platforms/api_server.py``
``_create_agent``): runtime provider and model from the gateway config, toolsets from
``platform_toolsets.litco_turn``, the profile's SessionDB, fallback chain and reasoning
config. The difference is the event mapping, which targets LitKit's agent2 frames:

========================================  ===========================================
Hermes callback                           Turn-server event
========================================  ===========================================
``stream_delta_callback(text)``           ``assistant_delta{delta}``
``stream_delta_callback(None)`` + more    ``assistant_reset`` before the next delta
``tool_start_callback(id, name, args)``   ``tool_started{call, args}``
``tool_progress_callback("subagent.*")``  ``tool_progress{call, message}``
``tool_complete_callback(id, name, …)``   ``tool_complete{call, result{status,summary}}``
result ``interrupted`` / ``failed``       ``loop_halted`` / ``error_classified``
========================================  ===========================================

Hermes has no plan events, so ``plan_*`` frames are never emitted.

A tool's ``status`` follows Hermes's own failure verdict: the executor classifies every result
(``agent.display._detect_tool_failure``: a terminal command's non-zero exit, an ``error`` field,
``success: false``) and reports it on the ``tool.completed`` progress event just before it calls
``tool_complete_callback``. The mapper keeps that verdict per tool name and applies it, so a
command Hermes logs as "returned error" arrives as ``status: "error"``.

Built-in memory is scoped per thread (:mod:`litco.memory_scope`): a dm turn uses the lawyer's
``users/<userId>/memories/``, a channel turn the matter's ``shared/memories/``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

from litco.homes import safe_segment
from litco.litkit.context import TurnIdentity, bind_turn, reset_turn
from litco.memory_scope import memory_dir, scope_agent_memory
from litco.turn_server import TurnContext, TurnOutcome, build_user_message

logger = logging.getLogger("litco.hermes_runner")

PLATFORM = "litco_turn"
SESSION_MAP_FILE = "litco_sessions.json"
_SUMMARY_MAX = 280


class _SessionMap:
    """LitKit ``sessionId`` -> current Hermes session id (compaction can rotate the Hermes id)."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> Dict[str, str]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def get(self, session_id: str) -> str:
        with self._lock:
            return self._load().get(session_id) or f"litco_{safe_segment(session_id)}"

    def set(self, session_id: str, hermes_id: str) -> None:
        with self._lock:
            data = self._load()
            if data.get(session_id) == hermes_id:
                return
            data[session_id] = hermes_id
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=0, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path)


def _size_label(n: int) -> str:
    return f"{n} chars" if n < 10_000 else f"{n / 1000:.0f}k chars"


def _parse(result: Any) -> Tuple[str, Optional[dict]]:
    text = result if isinstance(result, str) else json.dumps(result, default=str, ensure_ascii=False)
    text = text or ""
    parsed = result if isinstance(result, dict) else None
    if parsed is None and text[:1] == "{":
        try:
            parsed = json.loads(text)
        except Exception:
            # A hint or notice appended after the JSON body: parse the leading object alone.
            try:
                parsed, _ = json.JSONDecoder().raw_decode(text)
            except Exception:
                parsed = None
    return text, parsed if isinstance(parsed, dict) else None


def _exit_code(parsed: Optional[dict]) -> Optional[int]:
    if not isinstance(parsed, dict):
        return None
    for key in ("exit_code", "exitCode", "returncode"):
        value = parsed.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
    return None


def _hermes_verdict(name: Optional[str], result: Any) -> Tuple[bool, str]:
    """Hermes's own failure classification (the one behind its "returned error" log line)."""
    if not name:
        return False, ""
    try:
        from agent.display import _detect_tool_failure
        failed, suffix = _detect_tool_failure(name, result)
    except Exception:
        logger.warning("litco: Hermes tool-failure classification raised for %s; treating the call as ok",
                       name, exc_info=True)
        return False, ""
    return bool(failed), str(suffix or "").strip().strip("[]").strip()


def _summarize_result(result: Any, name: Optional[str] = None, is_error: Optional[bool] = None) -> Dict[str, Any]:
    """``{status, summary}`` for a tool result. The summary never carries the result's content:
    only a tool-supplied ``summary``/``message``, an error message, an exit code, or the size.

    ``is_error`` is Hermes's verdict when the executor reported one; otherwise Hermes's classifier
    is asked directly. Either way a non-zero exit code or an ``error`` field is a failure."""
    text, parsed = _parse(result)
    status, summary = "ok", f"returned {_size_label(len(text))}"
    exit_code = _exit_code(parsed)
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if error or parsed.get("success") is False or parsed.get("ok") is False:
            status = "error"
            summary = str(error or parsed.get("message") or "the tool reported a failure")
        elif exit_code not in (None, 0):
            status = "error"
            summary = f"command failed with exit code {exit_code}"
        else:
            for key in ("summary", "message"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    summary = value
                    break
    elif text.lower().startswith("error"):
        status, summary = "error", text.splitlines()[0]
    if status == "ok":
        failed, detail = (bool(is_error), "") if is_error is not None else _hermes_verdict(name, result)
        if failed:
            status = "error"
            if not detail:
                _, detail = _hermes_verdict(name, result)
            summary = (f"command failed with exit code {exit_code}" if exit_code not in (None, 0)
                       else detail if detail and detail != "error" else "the tool reported an error")
    elif exit_code not in (None, 0) and "exit code" not in summary:
        summary = f"{summary} (exit code {exit_code})"
    summary = " ".join(summary.split())
    try:
        from agent.redact import redact_sensitive_text
        summary = redact_sensitive_text(summary, force=True)
    except Exception:
        pass
    if len(summary) > _SUMMARY_MAX:
        summary = summary[: _SUMMARY_MAX - 3] + "..."
    return {"status": status, "summary": summary}


class _EventMapper:
    """Hermes callbacks -> turn-server events for one turn."""

    def __init__(self, ctx: TurnContext):
        self.ctx = ctx
        self._pending_reset = False
        self._streamed_any = False
        self._starts: Dict[str, float] = {}
        self._open_calls: Dict[str, str] = {}  # tool name -> latest open call id
        # Hermes's verdicts from ``tool.completed`` (fired just before tool_complete_callback), per tool
        # name in completion order; results are committed and announced one call at a time.
        self._verdicts: Dict[str, Deque[Tuple[bool, Any]]] = defaultdict(deque)
        self._verdict_lock = threading.Lock()

    def delta(self, text: Optional[str]) -> None:
        if text is None:
            if self._streamed_any:
                self._pending_reset = True
            return
        if not text:
            return
        if self._pending_reset:
            self.ctx.emit("assistant_reset", reason="iteration_committed")
            self._pending_reset = False
        self._streamed_any = True
        self.ctx.emit("assistant_delta", delta=text)

    def commentary(self, text: str, *, already_streamed: bool = False) -> None:
        if already_streamed or not isinstance(text, str) or not text.strip():
            return
        self.delta(text)
        self.delta(None)

    def tool_start(self, call_id: str, name: str, args: Any) -> None:
        call_id = str(call_id or f"call_{name}_{len(self._starts)}")
        self._starts[call_id] = time.monotonic()
        self._open_calls[name] = call_id
        self.ctx.emit("tool_started", call={"toolCallId": call_id, "name": name},
                      args=args if isinstance(args, dict) else {})

    def tool_complete(self, call_id: str, name: str, args: Any, result: Any) -> None:
        call_id = str(call_id or self._open_calls.get(name) or f"call_{name}")
        started = self._starts.pop(call_id, None)
        if self._open_calls.get(name) == call_id:
            self._open_calls.pop(name, None)
        with self._verdict_lock:
            pending = self._verdicts.get(name)
            verdict = pending.popleft() if pending else None
        if verdict is not None:
            is_error, raw = verdict
            # The raw result is what Hermes classified; the callback's copy may be spilled or carry hints.
            summary = _summarize_result(raw if raw is not None else result, name, is_error)
        else:
            summary = _summarize_result(result, name)
        summary["durationMs"] = int((time.monotonic() - started) * 1000) if started else 0
        self.ctx.emit("tool_complete", call={"toolCallId": call_id, "name": name}, result=summary)

    def progress(self, event_type: str, tool_name: Optional[str] = None, preview: Optional[str] = None,
                 args: Any = None, **kwargs: Any) -> None:
        if event_type == "tool.completed" and tool_name and "is_error" in kwargs:
            with self._verdict_lock:
                self._verdicts[tool_name].append((bool(kwargs.get("is_error")), kwargs.get("result")))
            return
        if not isinstance(event_type, str) or not event_type.startswith("subagent."):
            return  # tool.started/completed duplicate the start/complete hooks; reasoning stays private
        if event_type == "subagent.text" or not preview:
            return
        name = "delegate_task"
        call_id = self._open_calls.get(name) or kwargs.get("tool_call_id") or "delegate"
        self.ctx.emit("tool_progress", call={"toolCallId": str(call_id), "name": name},
                      message=" ".join(str(preview).split())[:_SUMMARY_MAX])


class HermesTurnRunner:
    """Runs each turn as a fresh ``AIAgent`` over the thread's persisted Hermes session."""

    def __init__(self, platform: str = PLATFORM):
        self.platform = platform
        self._session_map: Optional[_SessionMap] = None
        self._db = None
        self._db_lock = threading.Lock()

    # -- profile resources ---------------------------------------------------------
    def _session_db(self):
        with self._db_lock:
            if self._db is None:
                from hermes_constants import get_hermes_home
                from hermes_state_registry import acquire
                self._db = acquire(get_hermes_home() / "state.db")
            return self._db

    def _sessions(self) -> _SessionMap:
        if self._session_map is None:
            from hermes_constants import get_hermes_home
            self._session_map = _SessionMap(get_hermes_home() / SESSION_MAP_FILE)
        return self._session_map

    @staticmethod
    def _turn_prompt(ctx: TurnContext) -> str:
        req = ctx.request
        who = f"the lawyer {req.acting_user or req.user_id}" if req.user_id else "the case team"
        scope = ("a private thread with " + who) if req.kind == "dm" else "a thread in the matter's shared channel"
        memory = ("Your memory in this thread is this lawyer's private memory; nothing you save here is seen in "
                  "other lawyers' threads." if req.kind == "dm" else
                  "Your memory in this thread is the case team's shared matter memory; do not save anything "
                  "one lawyer told you in confidence.")
        return (
            f"Matter {req.matter_id}. This turn arrives over {req.channel} in {scope}. "
            f"Your working directory is {ctx.cwd}. Save the files you produce for the team under "
            f"{ctx.cwd / 'deliverables'} and register each one you mean to hand over with "
            "litco_deliver_local (path, optional name and deliverableClass). Only registered files, and "
            ".docx .xlsx .pptx .pdf .md .txt .csv .png .jpg files in deliverables/, reach the thread; "
            "keep specs, JSON and other scratch files elsewhere. Point to the files instead of pasting long "
            f"text. {memory}")

    def _build_agent(self, ctx: TurnContext, hermes_sid: str, mapper: _EventMapper):
        from run_agent import AIAgent
        from gateway.run import (GatewayRunner, _checkpoint_agent_kwargs, _current_max_iterations,
                                 _load_gateway_config, _resolve_gateway_model, _resolve_runtime_agent_kwargs)
        from hermes_cli.tools_config import _get_platform_tools

        runtime_kwargs = _resolve_runtime_agent_kwargs()
        model = runtime_kwargs.pop("model", None) or _resolve_gateway_model()
        runtime_kwargs.pop("_fallback_notice", None)
        user_config = _load_gateway_config()
        req = ctx.request
        return AIAgent(
            model=model, **runtime_kwargs, **_checkpoint_agent_kwargs(user_config),
            max_iterations=_current_max_iterations(), quiet_mode=True, verbose_logging=False,
            ephemeral_system_prompt=self._turn_prompt(ctx),
            enabled_toolsets=sorted(_get_platform_tools(user_config, self.platform)),
            session_id=hermes_sid, platform=self.platform,
            stream_delta_callback=mapper.delta, tool_progress_callback=mapper.progress,
            tool_start_callback=mapper.tool_start, tool_complete_callback=mapper.tool_complete,
            interim_assistant_callback=mapper.commentary,
            session_db=self._session_db(), fallback_model=GatewayRunner._load_fallback_model(),
            reasoning_config=GatewayRunner._load_reasoning_config(model),
            gateway_session_key=_session_key(req))

    # -- the turn ------------------------------------------------------------------
    def run(self, ctx: TurnContext) -> TurnOutcome:
        from gateway.session_context import clear_session_vars, set_session_vars
        from tools.terminal_tool import clear_task_env_overrides, register_task_env_overrides

        req = ctx.request
        sessions = self._sessions()
        hermes_sid = sessions.get(req.session_id)
        mapper = _EventMapper(ctx)
        tokens = set_session_vars(
            platform=self.platform, chat_id=req.session_id, chat_type=req.kind, thread_id=req.session_id,
            user_id=req.user_id, session_key=_session_key(req),
            session_id=hermes_sid, cwd=str(ctx.cwd), async_delivery=False, session_history_delivery="1")
        register_task_env_overrides(hermes_sid, {"cwd": str(ctx.cwd), "cwd_source": "session"})
        # LitKit tools assert this turn's lawyer (verified by the turn server) on every call;
        # an unasserted turn runs under the Matter Agent user's own role.
        turn_token = bind_turn(TurnIdentity(turn_id=ctx.turn_id, matter_id=req.matter_id,
                                            acting_user=req.acting_user, cwd=ctx.cwd))
        agent = None
        try:
            db = self._session_db()
            history = db.get_messages_as_conversation(hermes_sid) if db is not None else []
            agent = self._build_agent(ctx, hermes_sid, mapper)
            # Before the first prompt is assembled: MEMORY.md / USER.md come from this thread's folder.
            scope_agent_memory(agent, memory_dir(ctx.home, req.kind, req.user_id))
            if ctx.interrupted:
                return TurnOutcome(halted=ctx.interrupt_reason or "interrupted")
            ctx.on_interrupt(lambda reason: agent.interrupt(hard_cancel=True, tool_reason=reason))
            result = agent.run_conversation(user_message=build_user_message(ctx), conversation_history=history,
                                            task_id=hermes_sid)
            result = result if isinstance(result, dict) else {}
            outcome = TurnOutcome(
                text=str(result.get("final_response") or ""),
                input_tokens=int(getattr(agent, "session_prompt_tokens", 0) or 0),
                output_tokens=int(getattr(agent, "session_completion_tokens", 0) or 0),
                cache_read_tokens=getattr(agent, "session_cache_read_tokens", None),
                cache_write_tokens=getattr(agent, "session_cache_write_tokens", None),
                model_used=str(getattr(agent, "model", "") or "") or None)
            if result.get("interrupted") or ctx.interrupted:
                outcome.halted = ctx.interrupt_reason or "interrupted"
            elif result.get("failed"):
                outcome.error = str(result.get("error") or result.get("turn_exit_reason") or "the turn failed")[:500]
            new_sid = getattr(agent, "session_id", None)
            if isinstance(new_sid, str) and new_sid and new_sid != hermes_sid:
                sessions.set(req.session_id, new_sid)
            elif not isinstance(new_sid, str) or new_sid == hermes_sid:
                sessions.set(req.session_id, hermes_sid)
            return outcome
        except Exception as exc:
            logger.exception("litco hermes turn failed (session %s)", req.session_id)
            return TurnOutcome(error=str(exc)[:500] or exc.__class__.__name__, error_category=_classify(exc))
        finally:
            reset_turn(turn_token)
            clear_task_env_overrides(hermes_sid)
            clear_session_vars(tokens)


def _session_key(req) -> str:
    """Gateway-style session key in the active profile's namespace: ``agent:<ns>:litco_turn:<kind>:<sessionId>``."""
    from gateway.session import _session_key_namespace
    try:
        from hermes_cli.profiles import get_active_profile_name
        profile = get_active_profile_name()
    except Exception:
        profile = None
    return f"{_session_key_namespace(profile)}:{PLATFORM}:{req.kind}:{req.session_id}"


def _classify(exc: BaseException) -> str:
    text = f"{exc.__class__.__name__} {exc}".lower()
    if "auth" in text or "api key" in text or "401" in text:
        return "auth"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "rate" in text and "limit" in text or "overloaded" in text or "503" in text:
        return "provider_outage"
    return "unknown"
