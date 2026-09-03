"""User-facing cost visibility: status footer, threshold warnings, /new handoff.

Self-contained by design. This module is carried as a LOCAL PATCH on a pinned
branch (see ``LOCAL_PATCHES.md`` at the repo root), so the whole feature lives
here and the call sites elsewhere are deliberately tiny — the cherry-pick onto
the next pin should be one commit, not a scatter of edits.

Three surfaces, all config-gated under the ``cost_visibility`` section of
``config.yaml`` (user-owned, so a code-replacing upgrade cannot reset them):

1. ``render_footer()`` — one line appended to every reply:
       ctx 42% · turn $0.31 · session $4.10
2. ``check_warnings()`` — session-cost and context-percentage warnings, each
   fired exactly once per threshold crossing and reset by /new.
3. ``build_handoff_note()`` / ``store_handoff()`` / ``consume_handoff()`` —
   a <=300 word summary written at /new and injected into the next session.

Design notes that are load-bearing:

* **Pricing is not reimplemented.** ``agent/usage_pricing.py`` already owns the
  price tables, provider routing and cache-token semantics; the agent loop
  already folds each call's cost into ``agent.session_estimated_cost_usd``.
  We read that number and difference it. Adding a second price table here
  would guarantee the footer and the billing analytics disagree.
* **The ledger is on disk, keyed by durable session id.** The gateway evicts
  cached agents (and restarts), which zeroes ``session_estimated_cost_usd``
  on a fresh agent object for a conversation that is still very much alive.
  A purely in-memory counter would silently reset the "session $" figure —
  precisely the blind spot this feature exists to close. The ledger detects
  a counter that went backwards and treats the new value as a delta rather
  than losing the accumulated total.
* **Nothing here mutates the system prompt or past context.** The handoff is
  injected as a prefix on the next *user* message, never as a system-prompt
  edit, so per-conversation prompt caching is preserved (AGENTS.md: "Prompt
  caching is sacred").
* **Every public entry point is failure-isolated by its caller** and returns a
  falsy value rather than raising: cost telemetry must never break a reply.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "CostVisibilityConfig",
    "load_cost_visibility_config",
    "format_footer_line",
    "render_footer",
    "check_warnings",
    "build_handoff_note",
    "store_handoff",
    "consume_handoff",
    "reset_session_state",
    "selfcheck_line",
    "log_selfcheck",
]

# Config section name in config.yaml.
CONFIG_SECTION = "cost_visibility"

# Ledger bound: keep the newest N sessions so the file cannot grow forever.
_MAX_LEDGER_ENTRIES = 300

_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "footer": True,
    "warnings": True,
    "handoff": True,
    "cost_warn_usd": 25.0,
    "ctx_warn_pct": 80,
    "handoff_max_words": 300,
    "include_cli": False,
}

_lock = threading.RLock()


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CostVisibilityConfig:
    enabled: bool = True
    footer: bool = True
    warnings: bool = True
    handoff: bool = True
    cost_warn_usd: float = 25.0
    ctx_warn_pct: float = 80.0
    handoff_max_words: int = 300
    include_cli: bool = False

    def as_log_fields(self) -> str:
        return (
            f"enabled={self.enabled} footer={self.footer} "
            f"warnings={self.warnings} handoff={self.handoff} "
            f"cost_warn_usd={self.cost_warn_usd} "
            f"ctx_warn_pct={self.ctx_warn_pct} "
            f"handoff_max_words={self.handoff_max_words} "
            f"include_cli={self.include_cli}"
        )


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _coerce_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out >= 0 else default


def _coerce_int(value: Any, default: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return out if out > 0 else default


def load_cost_visibility_config(
    config: Optional[Dict[str, Any]] = None,
) -> CostVisibilityConfig:
    """Resolve the ``cost_visibility`` config section.

    Reads the persisted user config when *config* is not supplied so the
    gateway, the CLI and the tests all land on the same values. Unknown or
    malformed values fall back to the shipped default rather than raising —
    a typo in config.yaml must not take the gateway down.
    """
    section: Dict[str, Any] = {}
    try:
        if config is None:
            from hermes_cli.config import load_config as _load_config

            config = _load_config() or {}
        if isinstance(config, dict):
            raw = config.get(CONFIG_SECTION)
            if isinstance(raw, dict):
                section = raw
    except Exception:  # pragma: no cover - defensive
        section = {}

    return CostVisibilityConfig(
        enabled=_coerce_bool(section.get("enabled"), _DEFAULTS["enabled"]),
        footer=_coerce_bool(section.get("footer"), _DEFAULTS["footer"]),
        warnings=_coerce_bool(section.get("warnings"), _DEFAULTS["warnings"]),
        handoff=_coerce_bool(section.get("handoff"), _DEFAULTS["handoff"]),
        cost_warn_usd=_coerce_float(
            section.get("cost_warn_usd"), _DEFAULTS["cost_warn_usd"]
        ),
        ctx_warn_pct=_coerce_float(
            section.get("ctx_warn_pct"), _DEFAULTS["ctx_warn_pct"]
        ),
        handoff_max_words=_coerce_int(
            section.get("handoff_max_words"), _DEFAULTS["handoff_max_words"]
        ),
        include_cli=_coerce_bool(
            section.get("include_cli"), _DEFAULTS["include_cli"]
        ),
    )


def surface_enabled(agent: Any, config: CostVisibilityConfig) -> bool:
    """Whether the reply surface driving *agent* should carry the footer.

    The footer targets messaging surfaces (Telegram et al.), where there is no
    other cost read-out and the $191 incident was invisible. The CLI and TUI
    already render live cost in the status bar, so adding a footer there just
    duplicates it — opt in with ``cost_visibility.include_cli: true``.
    """
    if config.include_cli:
        return True
    platform = str(getattr(agent, "platform", "") or "cli").strip().lower()
    return platform not in {"", "cli", "local"}


# ─────────────────────────────────────────────────────────────────────────────
# Ledger — durable per-session accumulation
# ─────────────────────────────────────────────────────────────────────────────


def _ledger_path() -> str:
    from hermes_constants import get_hermes_home

    base = os.path.join(str(get_hermes_home()), "cost_visibility")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "ledger.json")


def _read_ledger() -> Dict[str, Any]:
    try:
        with open(_ledger_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        # A corrupt ledger must not break replies; start clean.
        logger.debug("cost_visibility: unreadable ledger, starting fresh", exc_info=True)
        return {}


def _write_ledger(data: Dict[str, Any]) -> None:
    # Bound the file: keep the newest entries by last-touch ordinal.
    if len(data) > _MAX_LEDGER_ENTRIES:
        ordered = sorted(
            data.items(), key=lambda kv: kv[1].get("seq", 0) if isinstance(kv[1], dict) else 0
        )
        data = dict(ordered[-_MAX_LEDGER_ENTRIES:])
    path = _ledger_path()
    try:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception:
        logger.debug("cost_visibility: ledger write failed", exc_info=True)


def _blank_entry() -> Dict[str, Any]:
    return {
        "session_cost_usd": 0.0,
        "last_agent_cost": 0.0,
        "turn_cost_usd": 0.0,
        "warned_cost": False,
        "warned_ctx": False,
        "seq": 0,
    }


def reset_session_state(session_id: str) -> None:
    """Drop all accumulated state for *session_id* (called on /new).

    This is what makes the warnings "reset on /new": both latch flags and the
    running total disappear with the entry.
    """
    if not session_id:
        return
    with _lock:
        data = _read_ledger()
        if session_id in data:
            data.pop(session_id, None)
            _write_ledger(data)


def _observe_cost(session_id: str, agent_cost: float) -> Tuple[float, float]:
    """Fold the agent's cumulative cost into the durable ledger.

    Returns ``(turn_cost, session_cost)``.

    ``agent.session_estimated_cost_usd`` counts from the moment the *agent
    object* was constructed, not from the start of the conversation. The
    gateway evicts and rebuilds agents (cache pressure, /model, restart), so
    the raw counter periodically drops back toward zero mid-conversation.
    Differencing naively against the previous observation would yield a
    negative turn cost and stall the session total; treating a decrease as
    "new agent, this value is itself the delta" keeps the running total
    monotonic across those rebuilds.
    """
    with _lock:
        data = _read_ledger()
        entry = data.get(session_id)
        if not isinstance(entry, dict):
            entry = _blank_entry()

        prev_agent = float(entry.get("last_agent_cost", 0.0) or 0.0)
        if agent_cost >= prev_agent:
            delta = agent_cost - prev_agent
        else:
            # Counter went backwards → the agent object was rebuilt.
            delta = max(0.0, agent_cost)

        session_cost = float(entry.get("session_cost_usd", 0.0) or 0.0) + delta
        entry["session_cost_usd"] = session_cost
        entry["last_agent_cost"] = agent_cost
        entry["turn_cost_usd"] = delta
        entry["seq"] = int(entry.get("seq", 0) or 0) + 1

        data[session_id] = entry
        _write_ledger(data)
        return delta, session_cost


def _peek(session_id: str) -> Dict[str, Any]:
    with _lock:
        entry = _read_ledger().get(session_id)
        return entry if isinstance(entry, dict) else _blank_entry()


# ─────────────────────────────────────────────────────────────────────────────
# Measurement helpers
# ─────────────────────────────────────────────────────────────────────────────


def _agent_cost(agent: Any) -> float:
    try:
        return max(0.0, float(getattr(agent, "session_estimated_cost_usd", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def context_usage(agent: Any) -> Tuple[int, int]:
    """Return ``(used_tokens, window_tokens)``; either may be 0 when unknown.

    ``last_prompt_tokens`` is the provider-exact prompt size of the most recent
    request. It parks at a ``-1`` sentinel immediately after a compression
    (see ``agent/conversation_compression.py``), which is reported as unknown
    rather than as a bogus percentage.
    """
    comp = getattr(agent, "context_compressor", None)
    if comp is None:
        return 0, 0
    try:
        used = int(getattr(comp, "last_prompt_tokens", 0) or 0)
    except (TypeError, ValueError):
        used = 0
    try:
        window = int(getattr(comp, "context_length", 0) or 0)
    except (TypeError, ValueError):
        window = 0
    if used < 0:
        used = 0
    if window < 0:
        window = 0
    return used, window


def context_pct(agent: Any) -> Optional[float]:
    used, window = context_usage(agent)
    if used <= 0 or window <= 0:
        return None
    return min(100.0, (used / window) * 100.0)


def _money(amount: float) -> str:
    """Render a USD amount for the footer.

    Sub-cent amounts render at 4dp so a cheap turn never displays as a
    dishonest ``$0.00`` (the same concern ``usage_pricing.format_cost_label``
    handles for the CLI cost labels).
    """
    try:
        value = float(amount)
    except (TypeError, ValueError):
        value = 0.0
    if value <= 0:
        return "$0.00"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:,.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Status footer
# ─────────────────────────────────────────────────────────────────────────────


def format_footer_line(
    ctx_pct: Optional[float], turn_usd: float, session_usd: float
) -> str:
    """Build the footer string. Pure — this is the unit under test."""
    ctx_part = "ctx —" if ctx_pct is None else f"ctx {int(round(ctx_pct))}%"
    return f"{ctx_part} · turn {_money(turn_usd)} · session {_money(session_usd)}"


def render_footer(
    agent: Any,
    session_id: str,
    config: Optional[CostVisibilityConfig] = None,
    *,
    observe: bool = True,
) -> str:
    """Return the footer line for this turn, or ``""`` when disabled.

    ``observe=True`` advances the ledger (once per turn). ``check_warnings``
    then reads the already-advanced totals, so a turn is only counted once.
    """
    cfg = config or load_cost_visibility_config()
    if not cfg.enabled or not cfg.footer:
        return ""
    sid = session_id or getattr(agent, "session_id", "") or ""
    if not sid:
        return ""

    if observe:
        turn_usd, session_usd = _observe_cost(sid, _agent_cost(agent))
    else:
        entry = _peek(sid)
        turn_usd = float(entry.get("turn_cost_usd", 0.0) or 0.0)
        session_usd = float(entry.get("session_cost_usd", 0.0) or 0.0)

    return format_footer_line(context_pct(agent), turn_usd, session_usd)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Threshold warnings
# ─────────────────────────────────────────────────────────────────────────────


def check_warnings(
    agent: Any,
    session_id: str,
    config: Optional[CostVisibilityConfig] = None,
) -> List[str]:
    """Return any warnings that should fire now.

    Each warning latches: the flag is persisted in the ledger, so a threshold
    fires on the crossing turn and never again for the life of the session.
    ``reset_session_state`` (called by /new) clears the latches.
    """
    cfg = config or load_cost_visibility_config()
    if not cfg.enabled or not cfg.warnings:
        return []
    sid = session_id or getattr(agent, "session_id", "") or ""
    if not sid:
        return []

    pct = context_pct(agent)
    out: List[str] = []

    with _lock:
        data = _read_ledger()
        entry = data.get(sid)
        if not isinstance(entry, dict):
            entry = _blank_entry()
        session_usd = float(entry.get("session_cost_usd", 0.0) or 0.0)
        dirty = False

        if (
            cfg.cost_warn_usd > 0
            and session_usd >= cfg.cost_warn_usd
            and not entry.get("warned_cost")
        ):
            ctx_txt = "unknown" if pct is None else f"{int(round(pct))}%"
            out.append(
                f"Session at {_money(session_usd)}. Context {ctx_txt}. "
                "Consider /new if the remaining work doesn't need this history."
            )
            entry["warned_cost"] = True
            dirty = True

        if (
            cfg.ctx_warn_pct > 0
            and pct is not None
            and pct >= cfg.ctx_warn_pct
            and not entry.get("warned_ctx")
        ):
            out.append(
                f"Context at {int(round(pct))}% — compaction will start soon. "
                "/new now if you want a clean handoff."
            )
            entry["warned_ctx"] = True
            dirty = True

        if dirty:
            data[sid] = entry
            _write_ledger(data)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. /new handoff note
# ─────────────────────────────────────────────────────────────────────────────

_HANDOFF_HEADER = "[handoff from previous session]"

# Tools whose arguments name a file the session touched.
_FILE_TOOLS = {
    "write_file": ("path",),
    "read_file": ("path",),
    "patch": ("path",),
    "search_files": ("path",),
    "skill_manage": ("file_path",),
}


def _iter_tool_calls(messages: List[dict]):
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            name = fn.get("name") or call.get("name") or ""
            raw_args = fn.get("arguments")
            args: Dict[str, Any] = {}
            if isinstance(raw_args, dict):
                args = raw_args
            elif isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args)
                    if isinstance(parsed, dict):
                        args = parsed
                except Exception:
                    args = {}
            yield str(name), args


def _files_touched(messages: List[dict], limit: int = 8) -> List[str]:
    seen: List[str] = []
    for name, args in _iter_tool_calls(messages):
        for key in _FILE_TOOLS.get(name, ()):
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                short = val.strip()
                if short not in seen:
                    seen.append(short)
    return seen[:limit]


def _last_user_message(messages: List[dict]) -> str:
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    if block["text"].strip():
                        return block["text"].strip()
    return ""


def _open_todos(agent: Any) -> List[str]:
    """Best-effort read of the live todo list as 'open items'."""
    items: List[str] = []
    try:
        todos = getattr(agent, "_todo_items", None) or getattr(agent, "todos", None)
        if isinstance(todos, list):
            for entry in todos:
                if not isinstance(entry, dict):
                    continue
                status = str(entry.get("status", "")).lower()
                if status in {"completed", "cancelled"}:
                    continue
                content = str(entry.get("content", "")).strip()
                if content:
                    items.append(content)
    except Exception:
        return []
    return items[:6]


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:.") + " …"


def _condense(text: str, max_chars: int) -> str:
    flat = re.sub(r"\s+", " ", (text or "").strip())
    if len(flat) <= max_chars:
        return flat
    return flat[: max_chars - 1].rstrip() + "…"


def build_handoff_note(
    agent: Any,
    messages: Optional[List[dict]] = None,
    config: Optional[CostVisibilityConfig] = None,
) -> str:
    """Compose the <=N word handoff summary.

    Deliberately derived from conversation structure rather than an LLM call:
    /new must stay instant and free, and a summarizer that fails (or costs
    money) inside a cost-visibility feature would be self-defeating.
    """
    cfg = config or load_cost_visibility_config()
    msgs = messages if messages is not None else (getattr(agent, "messages", None) or [])

    last_req = _condense(_last_user_message(msgs), 320)
    files = _files_touched(msgs)
    todos = _open_todos(agent)

    tool_names: List[str] = []
    for name, _args in _iter_tool_calls(msgs):
        if name and name not in tool_names:
            tool_names.append(name)

    turns = sum(
        1 for m in msgs if isinstance(m, dict) and m.get("role") == "user"
    )

    entry = _peek(getattr(agent, "session_id", "") or "")
    spend = float(entry.get("session_cost_usd", 0.0) or 0.0)

    lines: List[str] = [_HANDOFF_HEADER]
    summary_bits = [f"{turns} user turn(s)"]
    if spend > 0:
        summary_bits.append(f"~{_money(spend)} spent")
    if tool_names:
        summary_bits.append("tools: " + ", ".join(tool_names[:6]))
    lines.append("Context: " + "; ".join(summary_bits) + ".")

    if last_req:
        lines.append(f"Last user request: {last_req}")
    if files:
        lines.append("Files touched: " + ", ".join(files))
    if todos:
        lines.append("Open items: " + "; ".join(todos))

    lines.append(
        "This is a summary of a prior session that was cleared with /new. "
        "Treat it as background only; ask before assuming it is still current."
    )

    note = "\n".join(lines)
    return _truncate_words(note, cfg.handoff_max_words)


def _handoff_path(session_key: str) -> str:
    from hermes_constants import get_hermes_home

    base = os.path.join(str(get_hermes_home()), "cost_visibility", "handoff")
    os.makedirs(base, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_key or "default")[:180]
    return os.path.join(base, f"{safe}.json")


def store_handoff(session_key: str, note: str) -> bool:
    """Persist *note* for *session_key*.

    Written to disk, not memory: the gateway may restart between the /new and
    the user's next message, and the requirement is that the handoff survives
    that gap.
    """
    if not note or not session_key:
        return False
    try:
        path = _handoff_path(session_key)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"note": note}, fh)
        os.replace(tmp, path)
        return True
    except Exception:
        logger.debug("cost_visibility: handoff write failed", exc_info=True)
        return False


def consume_handoff(session_key: str) -> str:
    """Return and delete the stored handoff note (one-shot)."""
    if not session_key:
        return ""
    try:
        path = _handoff_path(session_key)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        try:
            os.unlink(path)
        except OSError:
            pass
        note = data.get("note") if isinstance(data, dict) else ""
        return note if isinstance(note, str) else ""
    except FileNotFoundError:
        return ""
    except Exception:
        logger.debug("cost_visibility: handoff read failed", exc_info=True)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Startup self-check
# ─────────────────────────────────────────────────────────────────────────────


def selfcheck_line(config: Optional[CostVisibilityConfig] = None) -> str:
    cfg = config or load_cost_visibility_config()
    return f"cost_visibility loaded — {cfg.as_log_fields()}"


def log_selfcheck(target_logger: Optional[logging.Logger] = None) -> str:
    line = selfcheck_line()
    (target_logger or logger).info(line)
    return line
