"""Shared engine for the /review command — every surface calls this.

/review spawns an independent, full-privilege background subagent (the same
async delegation rail as ``delegate_task(background=true)``) whose job is to
thoroughly review whatever the recent conversation presented: a PR, a diff,
code, documentation, or any other work product. The reviewer's result
re-enters the spawning session as a normal async-delegation completion, so
the primary agent sees the review and can act on it.

Model routing: the reviewer runs on ``auxiliary.review`` (provider/model/
base_url/api_key/api_mode in config.yaml) when configured; otherwise it
inherits the parent agent's credentials — main-model-first, same convention
as every other auxiliary task. Resolution reuses the delegation credential
resolver (``tools.delegate_tool._resolve_delegation_credentials``) via the
internal ``credentials_cfg`` parameter of ``delegate_task`` so native-SDK
providers, api_mode detection, and credential pools all behave identically
to ``delegation.provider`` pins.

Surfaces (CLI ``/review``, gateway ``/review``, TUI/Desktop live dispatch)
are thin adapters: snapshot the conversation, call :func:`start_review`,
print the dispatch note.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# How many recent chat messages (user + assistant turns) the reviewer gets.
DEFAULT_CONTEXT_MESSAGES = 10

# Per-message excerpt cap. Generous — a PR summary or diff excerpt the primary
# agent just printed is exactly what the reviewer needs — but bounded so a
# pathological turn can't blow up the child's opening context.
_MESSAGE_CHAR_CAP = 12_000


def _message_text(message: Dict[str, Any]) -> str:
    """Extract display text from a conversation message dict.

    Handles both plain-string content and OpenAI-style multimodal content
    lists (text parts joined; non-text parts noted).
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text") or ""))
                else:
                    parts.append(f"[{part.get('type', 'attachment')}]")
        return "\n".join(p for p in parts if p)
    return ""


def snapshot_recent_messages(
    messages: List[Dict[str, Any]],
    limit: int = DEFAULT_CONTEXT_MESSAGES,
) -> List[Dict[str, str]]:
    """Return the last ``limit`` user/assistant messages as {role, text} dicts.

    System messages and tool results are excluded — the chat turns are what
    the user and their primary agent actually said (the PR link, the summary,
    the diff excerpt). Empty-text messages (pure tool-call assistant stubs)
    are skipped.
    """
    out: List[Dict[str, str]] = []
    for message in reversed(list(messages or [])):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role not in ("user", "assistant"):
            continue
        text = _message_text(message).strip()
        if not text:
            continue
        if len(text) > _MESSAGE_CHAR_CAP:
            text = text[:_MESSAGE_CHAR_CAP] + "\n[... truncated ...]"
        out.append({"role": role, "text": text})
        if len(out) >= limit:
            break
    out.reverse()
    return out


def collect_parent_loaded_skills(
    parent_agent,
    messages: List[Dict[str, Any]],
    limit: int = 8,
) -> List[str]:
    """Names of skills the parent agent was operating under.

    Two sources, both surface-independent:

    * Launch-preloaded skills (``hermes -s``, kanban lanes, TUI skills env):
      their activation notes are embedded in the parent's
      ``ephemeral_system_prompt`` with a stable marker
      (see ``agent.skill_commands.build_preloaded_skills_prompt``).
    * Mid-session loads: ``skill_view`` tool calls in the parent's
      conversation history (assistant ``tool_calls`` entries).

    Order: preloaded first, then history loads, deduped, capped at ``limit``
    (a reviewer told to load 30 skills would burn its budget before working).
    """
    names: List[str] = []
    seen: set = set()

    def _add(name: str) -> None:
        cleaned = (name or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            names.append(cleaned)

    prompt = str(getattr(parent_agent, "ephemeral_system_prompt", "") or "")
    for match in re.finditer(r'with the "([^"]+)" skill\s+preloaded', prompt):
        _add(match.group(1))

    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            fn = tool_call.get("function") or {}
            if fn.get("name") != "skill_view":
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                continue
            # Only whole-skill loads seed the reviewer; a reference-file read
            # (file_path=...) is a detail of the parent's task, and the
            # reviewer loading the main SKILL.md covers it.
            if isinstance(args, dict) and not args.get("file_path"):
                _add(str(args.get("name") or ""))

    return names[:limit]


def load_workspace_context(parent_agent) -> str:
    """Project context files (AGENTS.md / CLAUDE.md / .cursorrules ...) from
    the parent's workspace.

    Reuses the SAME discovery/priority/cap logic the main agent's system
    prompt uses (``agent.prompt_builder.build_context_files_prompt``), pointed
    at the parent's resolved workspace directory. Subagents are built with
    ``skip_context_files=True``, so without this the reviewer would judge repo
    work without the repo's own conventions unless it thought to go read them.

    SOUL.md is skipped — identity belongs to the parent, not the reviewer.
    Returns "" when no workspace can be resolved or no context files exist.
    """
    try:
        from tools.delegate_tool import _resolve_workspace_hint

        cwd = _resolve_workspace_hint(parent_agent)
    except Exception:
        cwd = None
    if not cwd:
        return ""
    try:
        from agent.prompt_builder import build_context_files_prompt

        return build_context_files_prompt(cwd=str(cwd), skip_soul=True) or ""
    except Exception:
        logger.debug("review: workspace context load failed", exc_info=True)
        return ""


def build_review_task(
    snapshot: List[Dict[str, str]],
    user_prompt: str = "",
    loaded_skills: Optional[List[str]] = None,
    workspace_context: str = "",
) -> tuple:
    """Compose the reviewer subagent's (goal, context) pair."""
    goal = (
        "Act as an independent senior reviewer. Thoroughly review the work "
        "presented in the conversation excerpt provided in your context: "
        "investigate any code, pull request, branch, commit, documentation, "
        "design, or other artifact it references (open the PR, read the "
        "diff, run the code or tests where feasible) rather than judging "
        "from the excerpt alone. Produce a full, structured review: what "
        "the work does, whether it is correct and complete, concrete "
        "defects or risks found (with file/line references where possible), "
        "what was verified vs. only read, and a clear final verdict with "
        "recommended next steps."
    )

    lines = [
        "You were spawned by the /review command. The following is an "
        "excerpt of the most recent conversation between the user and "
        "their primary agent. It is your starting evidence — the work to "
        "review is referenced in it.",
        "",
        "--- Recent conversation (oldest first) ---",
    ]
    for message in snapshot:
        label = "USER" if message["role"] == "user" else "PRIMARY AGENT"
        lines.append(f"[{label}]")
        lines.append(message["text"])
        lines.append("")
    lines.append("--- End of conversation excerpt ---")
    if loaded_skills:
        skill_list = ", ".join(loaded_skills)
        lines.append("")
        lines.append(
            "The primary agent was operating under these loaded skills: "
            f"{skill_list}. Before reviewing, load each with "
            "skill_view(name=...) and treat their conventions, invariants, "
            "and review standards as binding for your assessment — the work "
            "was produced under them and must be judged against them."
        )
    if workspace_context.strip():
        lines.append("")
        lines.append(
            "The workspace's project context files are reproduced below. "
            "They are binding for your review — judge the work against "
            "these conventions and invariants."
        )
        lines.append("")
        lines.append(workspace_context.strip())
    if user_prompt.strip():
        lines.append("")
        lines.append("Additional review instructions from the user:")
        lines.append(user_prompt.strip())
    lines.append("")
    lines.append(
        "Your review is delivered back into that conversation, addressed to "
        "the primary agent and its user. Be direct and specific; do not "
        "soften findings."
    )
    return goal, "\n".join(lines)


def _load_review_credentials_cfg() -> Optional[Dict[str, Any]]:
    """Read ``auxiliary.review`` into a delegation-credentials-shaped dict.

    Returns None when the user configured nothing (provider=auto/empty and no
    model/base_url), which makes the reviewer inherit the parent agent's
    credentials — the main-model-first default.
    """
    try:
        from hermes_cli.config import load_config_readonly

        full = load_config_readonly()
        aux = full.get("auxiliary") or {}
        review = aux.get("review") or {}
        if not isinstance(review, dict):
            return None
    except Exception:
        return None

    provider = str(review.get("provider") or "").strip()
    if provider.lower() == "auto":
        provider = ""
    model = str(review.get("model") or "").strip()
    base_url = str(review.get("base_url") or "").strip()
    if not (provider or model or base_url):
        return None
    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": str(review.get("api_key") or "").strip(),
        "api_mode": str(review.get("api_mode") or "").strip(),
    }


def start_review(
    parent_agent,
    messages: List[Dict[str, Any]],
    user_prompt: str = "",
) -> Dict[str, Any]:
    """Dispatch the reviewer subagent in the background.

    Returns the parsed ``delegate_task`` dispatch dict (``status:
    "dispatched"`` with a ``delegation_id`` on success, or the synchronous
    result dict on channels that cannot route async completions).

    Raises ValueError when there is nothing to review or the dispatch is
    rejected/errored.
    """
    if parent_agent is None:
        raise ValueError("No active agent — send a message first.")

    snapshot = snapshot_recent_messages(messages)
    if not snapshot:
        raise ValueError("Nothing to review yet — the conversation is empty.")

    loaded_skills = collect_parent_loaded_skills(parent_agent, messages)
    workspace_context = load_workspace_context(parent_agent)
    goal, context = build_review_task(
        snapshot, user_prompt, loaded_skills, workspace_context
    )
    credentials_cfg = _load_review_credentials_cfg()

    from tools.delegate_tool import delegate_task

    raw = delegate_task(
        goal=goal,
        context=context,
        background=True,
        parent_agent=parent_agent,
        credentials_cfg=credentials_cfg,
    )
    try:
        result = json.loads(raw)
    except Exception:
        raise ValueError(f"Review dispatch failed: {raw!r}")
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
            f"the background and its full review will re-enter this "
            f"conversation when it finishes."
        )
    # Synchronous fallback (channels that cannot route async completions).
    return (
        f"⚖ Review completed synchronously{model_note}{focus_note} — "
        f"results:\n{json.dumps(result.get('results', result), ensure_ascii=False)[:4000]}"
    )
