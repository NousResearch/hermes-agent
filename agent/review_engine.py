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

# Every round: lead with the tally, name what was reviewed, keep nits from burying blockers.
_OUTPUT_RULES = (
    "Review round {round}. Open with one tally line (\"No blocking issues.\" or \"N blocking issues.\"), then "
    "state the round number and the commit SHA (or other version) you reviewed. Report every blocking finding; "
    "list at most 5 nits, then \"plus N similar\" for the rest."
)

# Round 2+: converge on the earlier findings and the new diff instead of re-reviewing from scratch.
_REREVIEW_RULES = (
    "This is a re-review: earlier rounds' results are quoted below, newest last. Re-review rules:\n"
    "- They apply only when the earlier rounds reviewed the same target (PR, branch or repository) as the work now "
    "presented. If the target changed, ignore the earlier results, review from scratch, and report it as round 1.\n"
    "- First re-check every earlier finding against the current state and mark it fixed, still open, or withdrawn "
    "(withdraw one you now judge wrong, and say why). A still-open blocker stays blocking.\n"
    "- Look for new findings only in what changed since the last reviewed version (the new diff since that "
    "round's SHA). Do not re-raise points an earlier round settled or accepted.\n"
    "- A new finding in code unchanged since an earlier round needs one line on why it was missed and why it "
    "matters now; without that it is a nit at most. Add no new nits on unchanged code."
)

# A finished review re-enters the conversation as an async-delegation completion (user role): the
# header, the briefing this module wrote, then the result (single- or batch-shaped, see
# tools/process_registry_notifications.py). Earlier rounds are quoted from there; only a successful
# result counts as a round.
_COMPLETION_HEADER = "[ASYNC DELEGATION"
# The briefing's last line; the completion preamble follows the echoed briefing with optional
# ``Toolsets:`` and the ``Role:`` line. The result is searched only after the LAST such boundary,
# so a delegation completion quoted inside the conversation excerpt never stands in for it.
_BRIEFING_END = ("Your review is delivered back into that conversation, addressed to the primary agent and its user. "
                 "Be direct and specific; do not soften findings.")
_BRIEFING_BOUNDARY_RE = re.compile(re.escape(_BRIEFING_END) + r"\n(?:Toolsets: [^\n]*\n)?Role: [^\n]*\n")
_RESULT_START_RE = re.compile(r"^(?:Status: (?:completed|success)\b.*\n--- RESULT ---|--- ✓ TASK 1/1\b.*\) ---)\n",
                              re.MULTILINE)
_RESULT_END_RE = re.compile(r"^(?:Full live transcript \(|\[INTERNAL NOTIFICATION)", re.MULTILINE)
_PRIOR_ROUNDS_QUOTED = 2
_PRIOR_RESULT_CHAR_CAP = 6_000


def _message_text(message: dict[str, Any]) -> str:
    """Display text of a message; multimodal parts are joined, non-text parts noted."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [str(part.get("text") or "") if part.get("type") == "text" else f"[{part.get('type', 'attachment')}]"
                 for part in content if isinstance(part, dict)]
        return "\n".join(p for p in parts if p)
    return ""


def snapshot_recent_messages(messages: list[dict[str, Any]], limit: int = DEFAULT_CONTEXT_MESSAGES) -> list[dict[str, str]]:
    """Last ``limit`` user/assistant messages with text as {role, text}, oldest first (system, tool and
    pure tool-call stubs excluded)."""
    out: list[dict[str, str]] = []
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


def _is_review_completion(message: Any) -> bool:
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    text = _message_text(message)
    return text.startswith(_COMPLETION_HEADER) and _REVIEW_GOAL in text


def prior_review_results(messages: list[dict[str, Any]]) -> list[str]:
    """Results of the /review rounds that already finished in this conversation, oldest first."""
    results: list[str] = []
    for message in messages or []:
        if not _is_review_completion(message):
            continue
        text = _message_text(message)
        boundaries = list(_BRIEFING_BOUNDARY_RE.finditer(text))
        start = _RESULT_START_RE.search(text, boundaries[-1].end()) if boundaries else None
        if start is None:
            continue  # failed or interrupted: not a round
        body = text[start.end():]
        end = _RESULT_END_RE.search(body)
        result = (body[:end.start()] if end else body).strip()
        if result:
            results.append(result)
    return results


def collect_parent_loaded_skills(parent_agent, messages: list[dict[str, Any]], limit: int = 8) -> list[str]:
    """Skills the parent was operating under: launch-preloaded (marker in ``ephemeral_system_prompt``)
    first, then ``skill_view`` loads from history, deduped, capped at ``limit`` (a reviewer told to load 30
    skills would burn its budget before working)."""
    names: list[str] = []
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


def build_review_task(snapshot: list[dict[str, str]], user_prompt: str = "", loaded_skills: Optional[list[str]] = None,
                      prior_results: Optional[list[str]] = None) -> tuple:
    """Compose a viewer-friendly goal and the complete reviewer briefing; ``prior_results`` (oldest
    first) makes this a re-review round."""
    prior_results = prior_results or []
    review_round = len(prior_results) + 1
    focus = " ".join(user_prompt.split())
    # Every round keeps the documented worker label; the round number lives in the briefing.
    goal = f"Review: {focus}" if focus else "Review recent work"
    if len(goal) > 80:
        goal = goal[:79].rstrip() + "…"
    # The goal is also the live worker label; keep the full instructions in context.
    lines = [
        _REVIEW_GOAL,
        "",
        _OUTPUT_RULES.format(round=review_round),
        "",
        "You were spawned by the /review command. The following is an excerpt of the most recent conversation "
        "between the user and their primary agent. It is your starting evidence — the work to "
        "review is referenced in it.",
        "",
        "--- Recent conversation (oldest first) ---",
    ]
    for message in snapshot:
        lines += [f"[{'USER' if message['role'] == 'user' else 'PRIMARY AGENT'}]", message["text"], ""]
    lines.append("--- End of conversation excerpt ---")
    if prior_results:
        lines += ["", _REREVIEW_RULES]
        first = review_round - min(len(prior_results), _PRIOR_ROUNDS_QUOTED)
        for offset, result in enumerate(prior_results[-_PRIOR_ROUNDS_QUOTED:]):
            if len(result) > _PRIOR_RESULT_CHAR_CAP:
                result = result[:_PRIOR_RESULT_CHAR_CAP] + "\n[... truncated ...]"
            lines += ["", f"--- Review round {first + offset} result ---", result]
        lines.append("--- End of earlier review results ---")
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
    lines += ["", _BRIEFING_END]
    return goal, "\n".join(lines)


def _load_review_credentials_cfg() -> Optional[dict[str, Any]]:
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


def start_review(parent_agent, messages: list[dict[str, Any]], user_prompt: str = "") -> dict[str, Any]:
    """Dispatch the reviewer subagent; returns the parsed ``delegate_task`` dict (``status: "dispatched"`` +
    ``delegation_id``, or the synchronous result on channels without async completions). Raises ValueError
    when there is nothing to review or the dispatch is rejected/errored."""
    if parent_agent is None:
        raise ValueError("No active agent — send a message first.")
    # Earlier review results are quoted whole in their own section; left in the excerpt, each would
    # repeat its briefing and lose its findings to the per-message cap.
    snapshot = snapshot_recent_messages([m for m in messages or [] if not _is_review_completion(m)])
    if not snapshot:
        raise ValueError("Nothing to review yet — the conversation is empty.")
    goal, context = build_review_task(snapshot, user_prompt, collect_parent_loaded_skills(parent_agent, messages),
                                      prior_review_results(messages))
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


def format_dispatch_note(result: dict[str, Any], user_prompt: str = "") -> str:
    """Human-facing one-liner for a successful dispatch. Shared by surfaces."""
    if result.get("status") == "dispatched":
        return "Review started. Results will return here."
    model = str(result.get("review_model") or "").strip()
    model_note = f" on {model}" if model else ""
    focus_note = f" (focus: {user_prompt.strip()})" if user_prompt.strip() else ""
    # Synchronous fallback (channels that cannot route async completions).
    return (
        f"⚖ Review completed synchronously{model_note}{focus_note} — "
        f"results:\n{json.dumps(result.get('results', result), ensure_ascii=False)[:4000]}"
    )
