"""Shared engine for the /review command — every surface calls this.

/review spawns an independent, full-privilege background subagent (the same async rail as
``delegate_task(background=true)``) to review whatever the recent conversation presented;
its result re-enters the spawning session as a normal async-delegation completion.
Model routing: ``auxiliary.review`` when configured, else the parent agent's credentials,
passed as ``credentials_cfg`` to ``delegate_task`` so native-SDK providers, api_mode
detection and credential pools behave identically to ``delegation.provider`` pins.
Surfaces (CLI/gateway ``/review``, TUI/Desktop) snapshot, call :func:`start_review`, print the note.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# How many recent chat messages (user + assistant turns) the reviewer gets.
DEFAULT_CONTEXT_MESSAGES = 10

# Per-message excerpt cap: generous (a PR summary/diff excerpt is exactly what the
# reviewer needs) but bounded against a pathological turn.
_MESSAGE_CHAR_CAP = 12_000

_REVIEW_GOAL = (
    "Act as an independent senior reviewer. Thoroughly review the work presented in the conversation excerpt "
    "provided in your context: investigate any code, pull request, branch, commit, documentation, design, or other "
    "artifact it references (open the PR, read the diff, run the code or tests where feasible) rather than judging "
    "from the excerpt alone. Produce a full, structured review: what the work does, whether it is correct and "
    "complete, concrete defects or risks found (with file/line references where possible), what was verified vs. "
    "only read, and a clear final verdict with recommended next steps."
)


def _message_text(message: Dict[str, Any]) -> str:
    """Display text of a message; multimodal parts are joined, non-text parts noted."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [str(part.get("text") or "") if part.get("type") == "text" else f"[{part.get('type', 'attachment')}]"
                 for part in content if isinstance(part, dict)]
        return "\n".join(p for p in parts if p)
    return ""


def snapshot_recent_messages(messages: List[Dict[str, Any]], limit: int = DEFAULT_CONTEXT_MESSAGES) -> List[Dict[str, str]]:
    """Last ``limit`` user/assistant messages with text as {role, text}, oldest first (system, tool and
    pure tool-call stubs excluded)."""
    out: List[Dict[str, str]] = []
    for message in reversed(list(messages or [])):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        text = _message_text(message).strip() if role in ("user", "assistant") else ""
        if not text:
            continue
        if len(text) > _MESSAGE_CHAR_CAP:
            text = text[:_MESSAGE_CHAR_CAP] + "\n[... truncated ...]"
        out.append({"role": role, "text": text})
        if len(out) >= limit:
            break
    out.reverse()
    return out


def collect_parent_loaded_skills(parent_agent, messages: List[Dict[str, Any]], limit: int = 8) -> List[str]:
    """Skills the parent was operating under: launch-preloaded (marker in ``ephemeral_system_prompt``)
    first, then ``skill_view`` loads from history, deduped, capped at ``limit`` (a reviewer told to load 30
    skills would burn its budget before working)."""
    names: List[str] = []
    prompt = str(getattr(parent_agent, "ephemeral_system_prompt", "") or "")
    candidates = [m.group(1) for m in re.finditer(r'with the "([^"]+)" skill\s+preloaded', prompt)]
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            fn = tool_call.get("function") or {} if isinstance(tool_call, dict) else {}
            if fn.get("name") != "skill_view":
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                continue
            # Only whole-skill loads seed the reviewer; a reference-file read is a detail
            # of the parent's task covered by loading the SKILL.md.
            if isinstance(args, dict) and not args.get("file_path"):
                candidates.append(str(args.get("name") or ""))
    for name in candidates:
        cleaned = name.strip()
        if cleaned and cleaned not in names:
            names.append(cleaned)
    return names[:limit]


def build_review_task(snapshot: List[Dict[str, str]], user_prompt: str = "", loaded_skills: Optional[List[str]] = None) -> tuple:
    """Compose the reviewer subagent's (goal, context) pair."""
    lines = [
        "You were spawned by the /review command. The following is an excerpt of the most recent conversation "
        "between the user and their primary agent. It is your starting evidence — the work to "
        "review is referenced in it.",
        "",
        "--- Recent conversation (oldest first) ---",
    ]
    for message in snapshot:
        lines += [f"[{'USER' if message['role'] == 'user' else 'PRIMARY AGENT'}]", message["text"], ""]
    lines.append("--- End of conversation excerpt ---")
    if loaded_skills:
        skill_list = ", ".join(loaded_skills)
        lines += [
            "",
            "The primary agent was operating under these loaded skills: "
            f"{skill_list}. Before reviewing, load each with "
            "skill_view(name=...) and treat their conventions, invariants, "
            "and review standards as binding for your assessment — the work "
            "was produced under them and must be judged against them.",
        ]
    if user_prompt.strip():
        lines += ["", "Additional review instructions from the user:", user_prompt.strip()]
    lines += [
        "",
        "Your review is delivered back into that conversation, addressed to "
        "the primary agent and its user. Be direct and specific; do not "
        "soften findings.",
    ]
    return _REVIEW_GOAL, "\n".join(lines)


def _load_review_credentials_cfg() -> Optional[Dict[str, Any]]:
    """``auxiliary.review`` as a delegation-credentials dict, or None when unconfigured (provider auto/empty
    and no model/base_url) so the reviewer inherits the parent's credentials."""
    try:
        from hermes_cli.config import load_config_readonly
        review = (load_config_readonly().get("auxiliary") or {}).get("review") or {}
    except Exception:
        return None
    if not isinstance(review, dict):
        return None

    cfg = {k: str(review.get(k) or "").strip() for k in ("provider", "model", "base_url", "api_key", "api_mode")}
    if cfg["provider"].lower() == "auto":
        cfg["provider"] = ""
    if not (cfg["provider"] or cfg["model"] or cfg["base_url"]):
        return None
    return cfg


def start_review(parent_agent, messages: List[Dict[str, Any]], user_prompt: str = "") -> Dict[str, Any]:
    """Dispatch the reviewer subagent; returns the parsed ``delegate_task`` dict (``status: "dispatched"`` +
    ``delegation_id``, or the synchronous result on channels without async completions). Raises ValueError
    when there is nothing to review or the dispatch is rejected/errored."""
    if parent_agent is None:
        raise ValueError("No active agent — send a message first.")
    snapshot = snapshot_recent_messages(messages)
    if not snapshot:
        raise ValueError("Nothing to review yet — the conversation is empty.")
    goal, context = build_review_task(snapshot, user_prompt, collect_parent_loaded_skills(parent_agent, messages))
    credentials_cfg = _load_review_credentials_cfg()

    from tools.delegate_tool import delegate_task
    raw = delegate_task(goal=goal, context=context, background=True, parent_agent=parent_agent, credentials_cfg=credentials_cfg)
    try:
        result = json.loads(raw)
    except Exception:
        result = None
    if isinstance(result, dict) and result.get("error"):
        raise ValueError(str(result["error"]))
    if not isinstance(result, dict):
        raise ValueError(f"Review dispatch failed: {raw!r}")
    result.setdefault("review_model", (credentials_cfg or {}).get("model") or "")
    return result


def format_dispatch_note(result: Dict[str, Any], user_prompt: str = "") -> str:
    """Human-facing one-liner for a successful dispatch. Shared by surfaces."""
    model = str(result.get("review_model") or "").strip()
    model_note = f" on {model}" if model else ""
    focus_note = f" (focus: {user_prompt.strip()})" if user_prompt.strip() else ""
    if result.get("status") == "dispatched":
        return (
            f"⚖ Review subagent dispatched{model_note}{focus_note} — it is "
            f"investigating the last {DEFAULT_CONTEXT_MESSAGES} messages in "
            f"the background and its full review will re-enter this conversation when it finishes."
        )
    # Synchronous fallback (channels that cannot route async completions).
    return (
        f"⚖ Review completed synchronously{model_note}{focus_note} — "
        f"results:\n{json.dumps(result.get('results', result), ensure_ascii=False)[:4000]}"
    )
