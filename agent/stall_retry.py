"""
Agentic stall-retry (dflash Q4 premature-EOS workaround).

dflash (Qwen3.6-27B Q4_K_M, lucebox spec-decode) sometimes emits EOS right
after a short action preamble ("Let me check X:") on agentic decision turns,
ending the turn with NO tool_call -> the agent loop treats it as a final
answer and stops mid-task. Higher-precision weights (the stock Q6 lane on the
same host) continue to a real tool call on the identical prompt.

This module detects that stall signature on a no-tool-call turn and retries
the turn against a higher-quality model lane, with a small recovery nudge that
asks the model to emit the tool call it just promised. If the retry produces
tool_calls, the loop adopts that response and continues; otherwise the caller
should fail the turn as partial rather than persist the planning-only text as
a final assistant message.

Entirely opt-in: does nothing unless ``HERMES_STALL_RETRY_MODEL`` is set
(e.g. ``qwen3.6-27b-256k``). Default-off => zero change to existing behavior.

Env:
  HERMES_STALL_RETRY_MODEL  retry lane/model name (required to enable)
  HERMES_STALL_RETRY_MAX_PER_TURN  max retries per user turn (default 5)
  HERMES_STALL_RETRY_MAX_CHARS  max content length to still count as a stall
                                (default 400; real final answers are longer)
  HERMES_STALL_RETRY_NUDGE  true/false; add a retry-only continuation nudge
                            (default true)
  HERMES_STALL_RETRY_TELEMETRY  true/false; append local NDJSON telemetry
                                (default true)
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

# Action-preamble signature: the turn announced an action but produced no tool
# call. These English phrases match the observed dflash stall corpus; broader
# language-agnostic fallbacks below still catch trailing-colon and incomplete
# final fragments without pretending this regex is multilingual.
_ACTION_RE = re.compile(
    r"(let me\b|let's\b|i'?ll\b|i will\b|i'?m going to\b|i am going to\b|"
    r"now i\b|first,?\s+i\b|next,?\s+i\b|i need to\b|i should\b|"
    r"going to (check|look|run|start|examine|search|read|list|create|write|edit|use))",
    re.IGNORECASE,
)
# Genuine completion signature: the model declared it is done / nothing to do.
# These English phrases must NOT be retried (they are correct no-tool-call
# turns); other languages still rely on the neutral structural checks below.
_COMPLETION_RE = re.compile(
    r"(\bdone\b|\bcomplete(d)?\b|nothing to (do|save|change|report|fix)|"
    r"no changes?\b|no action\b|already (complete|done|finished)|\bfinished\b|"
    r"all set\b|no further\b|nothing left\b|here('?s| is| are)\b|"
    r"in summary\b|to summarize\b|the answer is\b)",
    re.IGNORECASE,
)
_NATURAL_END_CHARS = '.!?:)"\']}。！？：）】」』》^'
_MIN_INCOMPLETE_FINAL_CHARS = 80
_ACTION_TAIL_CHARS = 500
_STALL_RETRY_NUDGE = (
    "Your previous assistant response ended after describing the next action, "
    "but it did not include the required tool call. Continue the same task now "
    "by making the tool call immediately. Do not summarize or apologize; call "
    "the tool that performs the action you just announced."
)


def _as_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _stall_retry_config(agent: Any | None = None) -> Mapping[str, Any]:
    cfg = getattr(agent, "_stall_retry_config", None)
    if isinstance(cfg, Mapping):
        return cfg
    try:
        from hermes_cli.config import load_config

        loaded = load_config()
    except Exception:
        return {}
    cfg = loaded.get("stall_retry") if isinstance(loaded, Mapping) else None
    return cfg if isinstance(cfg, Mapping) else {}


def get_stall_retry_model(agent: Any | None = None) -> str:
    """Return the configured retry model, with env taking precedence."""
    env_model = os.environ.get("HERMES_STALL_RETRY_MODEL", "").strip()
    if env_model:
        return env_model
    cfg_model = _stall_retry_config(agent).get("model")
    return str(cfg_model or "").strip()


def get_stall_retry_max_chars(agent: Any | None = None) -> int:
    env_value = os.environ.get("HERMES_STALL_RETRY_MAX_CHARS")
    if env_value is not None:
        return _as_positive_int(env_value, 400)
    return _as_positive_int(_stall_retry_config(agent).get("max_chars"), 400)


def get_stall_retry_max_per_turn(agent: Any | None = None) -> int:
    env_value = os.environ.get("HERMES_STALL_RETRY_MAX_PER_TURN")
    if env_value is not None:
        try:
            return max(0, int(env_value))
        except ValueError:
            return 5
    cfg_value = _stall_retry_config(agent).get("max_per_turn")
    try:
        return max(0, int(cfg_value))
    except (TypeError, ValueError):
        return 5


def get_stall_retry_nudge_enabled(agent: Any | None = None) -> bool:
    env_value = os.environ.get("HERMES_STALL_RETRY_NUDGE")
    if env_value is not None:
        return _as_bool(env_value, True)
    return _as_bool(_stall_retry_config(agent).get("nudge"), True)


def get_stall_retry_telemetry_enabled(agent: Any | None = None) -> bool:
    env_value = os.environ.get("HERMES_STALL_RETRY_TELEMETRY")
    if env_value is not None:
        return _as_bool(env_value, True)
    return _as_bool(_stall_retry_config(agent).get("telemetry"), True)


def _has_natural_response_ending(content: str) -> bool:
    stripped = (content or "").rstrip()
    if not stripped:
        return False
    if stripped.endswith("```"):
        return True
    last = stripped[-1]
    if last in _NATURAL_END_CHARS:
        return True
    return ord(last) >= 0x1F300


def _ends_with_action_promise(content: str) -> bool:
    """True when the visible tail promises immediate work but stops there.

    The generic stall heuristic is intentionally length-capped because long
    prose is often a real answer. Explicit tail promises are different: a long
    diagnostic can still end with "Let me check that:" and no tool call, which
    is the exact dflash premature-stop shape this module exists to recover.
    """
    tail = (content or "").strip()[-_ACTION_TAIL_CHARS:]
    if not tail:
        return False
    if not _ACTION_RE.search(tail):
        return False
    return tail.rstrip().endswith(":")


def _safe_preview(value: Any, max_chars: int = 240) -> str:
    text = value if isinstance(value, str) else str(value or "")
    text = re.sub(r"^<think>.*?</think>\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _stall_retry_log_path(agent: Any | None = None) -> Path:
    cfg_path = _stall_retry_config(agent).get("telemetry_path")
    if cfg_path:
        return Path(str(cfg_path)).expanduser()
    try:
        from hermes_constants import get_hermes_home

        home = Path(get_hermes_home())
    except Exception:
        home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
    return home / "logs" / "stall-retry.ndjson"


def record_stall_retry_event(agent: Any, event: str, **fields: Any) -> None:
    """Record local, bounded stall-retry telemetry."""
    entry: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event": str(event),
        "session_id": str(getattr(agent, "session_id", "") or ""),
        "model": str(getattr(agent, "model", "") or ""),
        "provider": str(getattr(agent, "provider", "") or ""),
    }
    content = fields.pop("content", None)
    if content is not None:
        text = content if isinstance(content, str) else str(content)
        entry["content_chars"] = len(text)
        entry["content_preview"] = _safe_preview(text)
    entry.update({str(k): _jsonable(v) for k, v in fields.items()})

    events = getattr(agent, "_stall_retry_events", None)
    if not isinstance(events, list):
        events = []
        try:
            setattr(agent, "_stall_retry_events", events)
        except Exception:
            pass
    events.append(entry)

    if not get_stall_retry_telemetry_enabled(agent):
        return
    try:
        path = _stall_retry_log_path(agent)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        return


def stall_retry_summary(agent: Any) -> dict[str, Any] | None:
    events = getattr(agent, "_stall_retry_events", None)
    if not isinstance(events, list) or not events:
        return None
    counts = {
        "detected": 0,
        "attempted": 0,
        "recovered": 0,
        "failed": 0,
        "limit_exhausted": 0,
        "exceptions": 0,
    }
    for item in events:
        kind = item.get("event") if isinstance(item, Mapping) else None
        if kind == "detected":
            counts["detected"] += 1
        elif kind == "attempt":
            counts["attempted"] += 1
        elif kind == "recovered":
            counts["recovered"] += 1
        elif kind in {"failed_no_tool_call", "api_none", "skipped_same_model"}:
            counts["failed"] += 1
        elif kind == "limit_exhausted":
            counts["limit_exhausted"] += 1
        elif kind == "exception":
            counts["exceptions"] += 1
    summary: dict[str, Any] = dict(counts)
    summary["events"] = len(events)
    if get_stall_retry_telemetry_enabled(agent):
        summary["log_path"] = str(_stall_retry_log_path(agent))
    return summary


def _retry_messages_with_nudge(
    agent: Any,
    api_messages: list[dict[str, Any]],
    stalled_content: str,
) -> list[dict[str, Any]]:
    if not get_stall_retry_nudge_enabled(agent):
        return api_messages
    retry_messages = [msg.copy() if isinstance(msg, dict) else msg for msg in api_messages]
    visible = (stalled_content or "").strip()
    if visible:
        retry_messages.append({"role": "assistant", "content": visible})
    retry_messages.append({"role": "user", "content": _STALL_RETRY_NUDGE})
    return retry_messages


def looks_like_stall(content: str, finish_reason: str, has_tool_calls: bool,
                     max_chars: int) -> bool:
    """True when a no-tool-call turn looks like a premature agentic stall
    (announced an action, didn't call a tool) rather than a real final answer."""
    if has_tool_calls:
        return False
    if finish_reason not in ("stop", "length"):
        return False
    c = (content or "").strip()
    # Strip a leading <think>...</think> block if present; judge the visible tail.
    c = re.sub(r"^<think>.*?</think>\s*", "", c, flags=re.IGNORECASE | re.DOTALL).strip()
    if not c:
        # Truly empty responses have their own recovery path in the
        # conversation loop. Do not let stall retry preempt that machinery.
        return False
    if _COMPLETION_RE.search(c):
        return False  # model said it's done => respect it
    if _ends_with_action_promise(c):
        return True
    if len(c) > max_chars:
        return False  # long => almost certainly a real answer
    if _ACTION_RE.search(c):
        return True   # announced an action, no tool call => stall
    # Short prose that doesn't declare completion and isn't an obvious answer:
    # a trailing colon strongly implies "about to do something".
    if c.endswith(":"):
        return True
    # dflash can also stop after a tool result with ordinary-looking prose that
    # is simply cut off mid-sentence (for example after a CLI interrupt resumes
    # the turn). In an agentic tool loop, a short no-tool stop that declares no
    # completion and lacks a natural ending is safer to retry than to persist as
    # a final assistant message.
    if len(c) >= _MIN_INCOMPLETE_FINAL_CHARS and not _has_natural_response_ending(c):
        return True
    return False


def retry_on_stall(
    agent,
    api_messages,
    finish_reason,
    stalled_content: str = "",
    retry_index: int | None = None,
):
    """If the just-finished no-tool-call turn looks like a stall and a retry
    lane is configured, re-issue the turn against that lane (same provider /
    client / endpoint — only the model name changes). A retry-only nudge is
    appended by default so the fallback model is told to continue with a tool
    call instead of repeating the action preamble.

    Returns the normalized assistant_message from the retry IF it produced tool
    calls (caller should adopt it + its finish_reason='tool_calls'), else None.
    Never raises into the caller — any failure returns None so the caller can
    fail closed without storing the stalled assistant message.
    """
    retry_model = get_stall_retry_model(agent)
    if not retry_model:
        return None

    try:
        retry_messages = _retry_messages_with_nudge(agent, api_messages, stalled_content)
        # Build kwargs exactly as the normal turn would, then override only the
        # model name. Safe when the retry lane is served by the SAME provider/
        # endpoint as agent.model (e.g. taro serves both dflash and the Q6 lane),
        # so no client rebuild is needed.
        api_kwargs = agent._build_api_kwargs(retry_messages)
        orig_model = api_kwargs.get("model")
        if retry_model == orig_model:
            record_stall_retry_event(
                agent,
                "skipped_same_model",
                retry_model=retry_model,
                finish_reason=finish_reason,
                retry_index=retry_index,
            )
            return None  # nothing to gain retrying the same model
        api_kwargs = dict(api_kwargs)
        api_kwargs["model"] = retry_model
        # Force non-streaming for the retry (simpler, we only inspect the result).
        api_kwargs.pop("stream", None)
        api_kwargs["stream"] = False

        try:
            agent._vprint(
                f"{getattr(agent, 'log_prefix', '')}↻ stall detected "
                f"(no tool call) — retrying turn on '{retry_model}'",
                force=True,
            )
        except Exception:
            pass

        record_stall_retry_event(
            agent,
            "attempt",
            retry_model=retry_model,
            original_model=orig_model,
            finish_reason=finish_reason,
            retry_index=retry_index,
            nudge=get_stall_retry_nudge_enabled(agent),
            content=stalled_content,
        )
        response = agent._interruptible_api_call(api_kwargs)
        if response is None:
            record_stall_retry_event(
                agent,
                "api_none",
                retry_model=retry_model,
                retry_index=retry_index,
            )
            return None
        transport = agent._get_transport()
        normalize_kwargs = {}
        if getattr(agent, "api_mode", None) == "anthropic_messages":
            normalize_kwargs["strip_tool_prefix"] = getattr(agent, "_is_anthropic_oauth", False)
        normalized = transport.normalize_response(response, **normalize_kwargs)
        if getattr(normalized, "tool_calls", None):
            record_stall_retry_event(
                agent,
                "recovered",
                retry_model=retry_model,
                retry_index=retry_index,
                tool_call_count=len(getattr(normalized, "tool_calls", []) or []),
                content=getattr(normalized, "content", "") or "",
            )
            return normalized
        record_stall_retry_event(
            agent,
            "failed_no_tool_call",
            retry_model=retry_model,
            retry_index=retry_index,
            content=getattr(normalized, "content", "") or "",
        )
        return None
    except Exception as exc:
        record_stall_retry_event(
            agent,
            "exception",
            retry_model=retry_model,
            retry_index=retry_index,
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        # Any error => silently fall back to the original response.
        return None
