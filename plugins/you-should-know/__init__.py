"""you-should-know — surface the important things the user might have missed.

Inspired by the Claude Code "You should Know" mod, which spins off a sideagent
to observe Claude's output. This port accumulates a session's assistant output
(plus compact tool-result lines, so a tool error the agent glossed over stays
visible) in a bounded per-session buffer, then runs one bounded observer pass
per digest epoch through the host-owned plugin LLM (``ctx.llm``) and appends any
findings to the turn's user-visible output via ``transform_llm_output``.

Delivery deliberately uses the pre-persistence transform seam: only the current
turn's not-yet-written text is touched, so prompt caching, strict role
alternation, and the no-synthetic-user-message invariant all hold. A digest at
true session end is NOT attempted — by ``on_session_finalize`` the final
assistant row is already persisted and hook results reach no user-visible
surface (and the per-turn ``on_session_end`` hook fires after that turn's
transform for the same reason). See README.md.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_AUX_TASK_KEY = "you_should_know_observer"

# Marker opening the appended digest; post_llm_call strips from here on so the
# observer never re-reads its own output (no feedback loop).
_DIGEST_MARKER = "\n\n---\n🔎 **You should know**"

_DEFAULT_MAX_CHARS = 24000       # observer input + buffer bound
_DEFAULT_MIN_OBSERVE_CHARS = 1500  # trivial sessions never trigger a call
_DEFAULT_TOOL_LINE_CHARS = 400  # per tool-result line kept in the buffer
_MAX_SESSIONS = 256             # ceiling on live per-session buffers
_MAX_DIGEST_ITEMS = 8

_SEVERITY_ICONS = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}

_OBSERVER_INSTRUCTIONS = """\
You are reviewing the recent output of an AI assistant session. Identify the
important things the USER might have missed — things the assistant said or did
that deserve the user's attention but were probably overlooked.

Look for:
- Swallowed errors: tool calls that failed or returned errors/warnings which
  the assistant glossed over, downplayed, or never mentioned.
- Unanswered questions: questions the user asked that never got an answer.
- Risky actions taken: file deletions/overwrites, external sends, permission or
  config changes, anything costing money, anything hard to undo.
- Pending follow-ups: things the assistant said it would do, or that clearly
  still need doing before the task is complete.
- Contradictions: the assistant contradicting itself or its earlier statements.

Report ONLY items a reasonable user would want flagged. Skip routine narration,
successful steps with no caveats, and trivia. If nothing is worth flagging,
return an empty items list. Keep each title under 12 words and each detail
under 40 words."""

_DIGEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
                    "title": {"type": "string"},
                    "detail": {"type": "string"},
                },
                "required": ["severity", "title", "detail"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

# The ctx handed to register(); hooks read settings/llm through it per call so
# config edits apply without a restart. Set by register(); tests may replace it.
_CTX: Optional[Any] = None


class _SessionBuffer:
    """Bounded per-session accumulation of assistant text + tool-result lines."""

    def __init__(self) -> None:
        self._chunks: List[str] = []
        self.observed_chars = 0  # prefix length already covered by an observer pass
        self.updated_at = time.time()

    def append(self, text: str, cap: int) -> None:
        if text:
            self._chunks.append(text)
        self.updated_at = time.time()
        total = sum(len(c) for c in self._chunks)
        # Drop oldest chunks first; the observed prefix goes before unobserved
        # text, so accounting stays exact in the common case.
        while self._chunks and total > cap:
            oldest = self._chunks.pop(0)
            total -= len(oldest)
            self.observed_chars = max(0, self.observed_chars - len(oldest))

    def text(self) -> str:
        return "".join(self._chunks)

    def unobserved(self) -> str:
        return self.text()[self.observed_chars:]


_STATE_LOCK = threading.Lock()
# (hermes_home_key, session_id) -> _SessionBuffer. Never a single global slot:
# one process serves many profiles, so state is keyed per home AND session.
_BUFFERS: Dict[Tuple[str, str], _SessionBuffer] = {}
# (home, session_id, turn_id) present while an observer pass covered this turn;
# post_llm_call advances observed_chars past the appended text on sight.
_TURN_OBSERVED: Dict[Tuple[str, str, str], bool] = {}


def _home_key() -> str:
    from hermes_constants import hermes_home_key

    return hermes_home_key()


def _buffer_key(session_id: str) -> Tuple[str, str]:
    return (_home_key(), session_id or "")


def _get_buffer(session_id: str) -> _SessionBuffer:
    key = _buffer_key(session_id)
    with _STATE_LOCK:
        buf = _BUFFERS.get(key)
        if buf is None:
            buf = _SessionBuffer()
            if len(_BUFFERS) >= _MAX_SESSIONS:
                oldest = min(_BUFFERS, key=lambda k: _BUFFERS[k].updated_at)
                del _BUFFERS[oldest]
            _BUFFERS[key] = buf
        buf.updated_at = time.time()
        return buf


def _mark_turn_observed_locked(home: str, session_id: str, turn_id: str) -> None:
    """Record that an observer pass covered this turn. Call with _STATE_LOCK
    held. Bounded: a turn whose post_llm_call never fires (persist_disabled)
    must not leak the entry."""
    if len(_TURN_OBSERVED) >= 1024:
        _TURN_OBSERVED.pop(next(iter(_TURN_OBSERVED)))
    _TURN_OBSERVED[(home, session_id, turn_id)] = True


def _setting(name: str, default: Any) -> Any:
    if _CTX is None:
        return default
    try:
        return _CTX.get_config(name, default)
    except Exception:
        return default


def _enabled() -> bool:
    return bool(_setting("enabled", True))


def _int_setting(name: str, default: int) -> int:
    try:
        value = int(_setting(name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _fit_head_tail(text: str, max_chars: int) -> str:
    """Bound observer input, keeping head+tail so both the task setup and the
    latest (most likely missed) output stay visible."""
    if len(text) <= max_chars:
        return text
    head_len, tail_len = max_chars * 2 // 3, max_chars // 3
    return text[:head_len] + "\n\n[... %d chars omitted ...]\n\n" % (len(text) - head_len - tail_len) + text[-tail_len:]


def _compact_tool_line(tool_name: str, result: Any, max_chars: int) -> str:
    if isinstance(result, str):
        snippet = result.strip().replace("\n", " ")
    else:
        snippet = repr(result)
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "…"
    return f"[tool:{tool_name or '?'}] {snippet}"


def on_post_llm_call(*, session_id: str = "", turn_id: str = "",
                     assistant_response: Any = None, **_: Any) -> None:
    """Accumulate this turn's assistant text (minus any digest we appended)."""
    if not _enabled():
        return
    text = assistant_response if isinstance(assistant_response, str) else ""
    clean, had_digest = _strip_digest(text)
    buf = _get_buffer(session_id)
    with _STATE_LOCK:
        buf.append(clean, _int_setting("max_chars", _DEFAULT_MAX_CHARS))
        key = (_home_key(), session_id or "", turn_id or "")
        if had_digest or _TURN_OBSERVED.pop(key, False):
            # Everything appended so far was covered by this turn's observer
            # pass (the pass ran on prior-unobserved text + this turn's text).
            buf.observed_chars = max(buf.observed_chars, len(buf.text()))


def on_post_tool_call(*, session_id: str = "", tool_name: str = "",
                      result: Any = None, **_: Any) -> None:
    """Accumulate one compact line per tool result, so a tool error the agent
    glossed over is still in front of the observer."""
    if not _enabled():
        return
    line = _compact_tool_line(tool_name, result, _int_setting("tool_line_chars", _DEFAULT_TOOL_LINE_CHARS))
    if not line.split("] ", 1)[-1].strip():
        return  # empty result: no signal, skip the noise line
    buf = _get_buffer(session_id)
    with _STATE_LOCK:
        buf.append(line, _int_setting("max_chars", _DEFAULT_MAX_CHARS))


def on_transform_llm_output(*, response_text: str = "", session_id: str = "",
                            turn_id: str = "", **_: Any) -> Optional[str]:
    """Maybe run the observer and append its digest to this turn's output.

    Runs BEFORE the assistant row is first persisted, so the returned text is
    what the user sees and what SQLite replays — the sanctioned seam. Returns
    None (no change) unless there is enough unobserved output AND the observer
    found something worth flagging.
    """
    if not _enabled() or not response_text or _CTX is None:
        return None
    buf = _get_buffer(session_id)
    with _STATE_LOCK:
        snapshot = buf.text()
        pending = snapshot[buf.observed_chars:]
    candidate = (pending + "\n" + response_text).strip() if pending else response_text
    if len(candidate) < _int_setting("min_observe_chars", _DEFAULT_MIN_OBSERVE_CHARS):
        return None  # trivial session/epoch: skip the call entirely
    items = _run_observer(_fit_head_tail(candidate, _int_setting("max_chars", _DEFAULT_MAX_CHARS)))
    if items is None:
        return None  # observer failed or was denied: try again next turn
    with _STATE_LOCK:
        # Advance past everything the observer saw, whether or not it found
        # anything: each epoch is reviewed once (bounded cost, no re-litigation).
        # max() keeps a concurrent append's tail unobserved.
        buf.observed_chars = max(buf.observed_chars, len(snapshot))
        _mark_turn_observed_locked(_home_key(), session_id or "", turn_id or "")
    if not items:
        return None
    return response_text + _render_digest(items)


def _strip_digest(text: str) -> Tuple[str, bool]:
    idx = text.find(_DIGEST_MARKER)
    return (text[:idx], True) if idx >= 0 else (text, False)


def _sanitize_items(parsed: Any) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if not isinstance(parsed, dict):
        return items
    raw = parsed.get("items")
    if not isinstance(raw, list):
        return items
    for entry in raw[:_MAX_DIGEST_ITEMS]:
        if not isinstance(entry, dict):
            continue
        severity = str(entry.get("severity", "info")).strip().lower()
        title = str(entry.get("title", "")).strip()[:120]
        detail = str(entry.get("detail", "")).strip()[:300]
        if not title or not detail:
            continue
        items.append({
            "severity": severity if severity in _SEVERITY_ICONS else "info",
            "title": title,
            "detail": detail,
        })
    return items


def _render_digest(items: List[Dict[str, str]]) -> str:
    lines = [_DIGEST_MARKER + " — things from this session you might have missed:"]
    for item in items:
        lines.append(f"- {_SEVERITY_ICONS[item['severity']]} **{item['title']}**: {item['detail']}")
    return "\n".join(lines)


def _run_observer(transcript: str) -> Optional[List[Dict[str, str]]]:
    """One bounded structured observer pass. Fail-open AND fail-closed: any
    transport/model failure returns None (no digest), and a denied model
    override is never silently downgraded."""
    from agent.plugin_llm import PluginLlmTrustError

    ctx = _CTX
    if ctx is None:
        return None
    model_override = str(_setting("model", "") or "").strip() or None
    try:
        result = ctx.llm.complete_structured(
            instructions=_OBSERVER_INSTRUCTIONS,
            input=[{"type": "text", "text": transcript}],
            json_schema=_DIGEST_SCHEMA,
            schema_name="you_should_know_digest",
            system_prompt="You are a precise, terse reviewer. Return only the JSON object.",
            purpose="you-should-know observer digest",
            task=_AUX_TASK_KEY,
            model=model_override,
            temperature=0.2,
            max_tokens=600,
            timeout=60,
        )
    except PluginLlmTrustError as exc:
        # Fail closed: a configured model override without the trust flag must
        # not silently fall back to another model.
        logger.warning("you-should-know: model override denied (%s); skipping observer pass", exc)
        return None
    except Exception as exc:
        logger.warning("you-should-know: observer pass failed: %s", exc)
        return None
    if result.content_type != "json":
        logger.warning("you-should-know: observer returned non-JSON output; skipping digest")
        return None
    return _sanitize_items(result.parsed)


def register(ctx) -> None:
    global _CTX
    _CTX = ctx
    # Own auxiliary-task slot so users can pin a cheap model/route under
    # ``auxiliary.you_should_know_observer`` in config.yaml without trust-flag
    # friction; the explicit ``settings.model`` override still needs
    # ``llm.allow_model_override`` (fail-closed).
    ctx.register_auxiliary_task(
        _AUX_TASK_KEY,
        display_name="You Should Know observer",
        description="One bounded pass per digest epoch scanning session output "
                    "for things the user might have missed.",
    )
    for name, fn in (
        ("transform_llm_output", on_transform_llm_output),
        ("post_llm_call", on_post_llm_call),
        ("post_tool_call", on_post_tool_call),
    ):
        ctx.register_hook(name, fn)
