"""User-turn intent detector (S-0518-01 direction B).

Auxiliary LLM-driven classifier that runs BEFORE Coach's turn begins. Reads
the user's most recent inbound message and decides whether it's an
artifact-deliverable request that should route through a sub-agent
(Type E in `docs/research/2026-05-21-subagent-appearance-taxonomy.md`).

When the user is asking for a saveable, structured artifact (cheat sheet,
draft text, list, framework, summary, comparison) — i.e. work that should
become a backend artifact owned by Scout / Analyst / Publicist rather than
Coach prose — this returns a hint that `gateway/session.py` injects into
the Coach system prompt as a `<detected-intent>` block. Coach then has an
unambiguous server-provided cue to enqueue + announce instead of inlining
the answer.

Same pattern as the existing `pending-announcements` injection that
handles Confirm leg (Type D): server pre-computes the decision, Coach LLM
only follows the guidance. This compresses Coach's freedom on the one
remaining type that requires Coach-side semantic judgment.

**Failures are silent.** Auxiliary call timeout, LLM not configured, parse
failure → return empty result; Coach proceeds normally without the hint.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# Only consider messages above this length — short turns ("ok", "yes",
# "thanks", "what") are not artifact requests.
_MIN_USER_MSG_LEN = 20

# Hard timeout — this runs synchronously on every user turn before Coach
# starts. Auxiliary must be fast or skipped.
_DETECT_TIMEOUT_S = 8.0


_DETECT_PROMPT = """\
You are a routing classifier for a career-coaching agent. A user just sent
a message. Decide whether the user is asking for a *saveable, structured
artifact* that should be authored by a backend sub-agent (Scout / Analyst /
Publicist), NOT inlined into Coach's conversational reply.

The sub-agents:
- **Scout** — market scanning, role discovery, company / event lookups
- **Analyst** — data analysis, interpretation, comparisons, frameworks,
  hiring-pattern synthesis, cheat sheets of facts/figures
- **Publicist** — draft text the user will send/use: cover letters,
  resumes, outreach messages, follow-ups, bios

Route to a sub-agent when the user is asking for something they would
**want to save / reference / send later** — a deliverable artifact, not a
conversational answer.

Examples that ROUTE:
- "put together a one-pager I can keep in my notes" → analyst
- "write the LinkedIn message to my old manager" → publicist
- "which Series A health-tech companies are hiring product folks?" → scout
- "break down the comp gap between these two offers" → analyst
- "give me a 30-second elevator pitch for the Stripe panel" → publicist

Examples that DO NOT route (Coach handles inline):
- "do you think the title bump is worth the pay cut?" → decision-prompting
- "i can't tell if i'm overthinking this" → emotional
- "yep" / "go" / "that works" → confirmation
- "what's the typical interview loop at growth-stage startups?" → conceptual
- "how's it going?" → conversational

Return STRICT JSON, no prose, no markdown fence:

{
  "route_to_subagent": <true|false>,
  "sub_agent": "<scout|analyst|publicist|null>",
  "id_slug": "<lowercase-hyphenated short slug describing this action, 3-6 words, no leading 'coach-commit-' prefix. Example: 'draft-stripe-panel-pitch'. Null if route_to_subagent=false>",
  "suggested_action": "<one-line verb+object describing the artifact. Null if route_to_subagent=false>",
  "suggested_announcement": "<one sentence, third-person, sub-agent as subject. Null if route_to_subagent=false>",
  "confidence": "<high|medium|low>",
  "reasoning": "<one short sentence>"
}

User's message:
\"\"\"
{user_message}
\"\"\"
"""


def _parse_response(raw: str) -> dict[str, Any] | None:
    """Tolerant JSON parse — strips markdown fence, returns None on failure."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        s = "\n".join(lines[1:-1]) if len(lines) >= 2 else s
        if s.startswith("json\n"):
            s = s[5:]
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def detect_turn_intent(user_message: str) -> dict[str, Any]:
    """Classify the user's turn for artifact-deliverable routing.

    Returns a dict that is always populated with a uniform schema so
    callers can log shape consistently:

      {
        "checked": bool,             # auxiliary call attempted
        "skipped": str|None,         # reason for skip if not checked
        "route_to_subagent": bool,
        "sub_agent": str|None,
        "suggested_action": str|None,
        "suggested_announcement": str|None,
        "confidence": str|None,
        "reasoning": str|None,
      }
    """
    out: dict[str, Any] = {
        "checked": False,
        "skipped": None,
        "route_to_subagent": False,
        "sub_agent": None,
        "id_slug": None,
        "suggested_action": None,
        "suggested_announcement": None,
        "confidence": None,
        "reasoning": None,
    }

    if not user_message or len(user_message) < _MIN_USER_MSG_LEN:
        out["skipped"] = "msg_too_short"
        return out

    try:
        from agent.auxiliary_client import call_llm  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        out["skipped"] = f"client_import_failed:{type(e).__name__}"
        return out

    prompt = _DETECT_PROMPT.replace("{user_message}", user_message)
    try:
        response = call_llm(
            task="compression",
            messages=[
                {"role": "system", "content": "You return only strict JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.0,
            timeout=_DETECT_TIMEOUT_S,
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        out["skipped"] = f"aux_call_failed:{type(e).__name__}"
        return out

    parsed = _parse_response(raw)
    if parsed is None:
        out["skipped"] = "aux_parse_failed"
        return out

    out["checked"] = True
    out["route_to_subagent"] = bool(parsed.get("route_to_subagent"))
    out["sub_agent"] = parsed.get("sub_agent") or None
    out["id_slug"] = _sanitize_slug(parsed.get("id_slug"))
    out["suggested_action"] = parsed.get("suggested_action") or None
    out["suggested_announcement"] = parsed.get("suggested_announcement") or None
    out["confidence"] = parsed.get("confidence") or None
    out["reasoning"] = parsed.get("reasoning") or None
    return out


def _sanitize_slug(raw: Any) -> str | None:
    """Coerce the LLM-provided slug into a safe lowercase-hyphenated form.

    Returns None on anything unusable so the caller can skip injection
    rather than ship a malformed id to Coach. Strips any 'coach-commit-'
    prefix in case the LLM included it despite the prompt saying not to.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    if s.startswith("coach-commit-"):
        s = s[len("coach-commit-"):]
    # Keep only [a-z0-9-], collapse runs of dashes, trim edges.
    out_chars: list[str] = []
    for ch in s:
        if ch.isalnum() or ch == "-":
            out_chars.append(ch)
        elif ch in (" ", "_"):
            out_chars.append("-")
    cleaned = "".join(out_chars)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    cleaned = cleaned.strip("-")
    if not cleaned or len(cleaned) > 60:
        return None
    return cleaned


def render_injection_block(detection: dict[str, Any]) -> str | None:
    """Render the FALLBACK system-prompt injection block.

    Used when the detector decided to route but the server did NOT
    auto-execute (low / medium confidence or executor failure). Coach is
    asked to perform the calls itself. Returns None when no injection is
    needed.

    For the auto-executed path (high confidence) use
    `render_already_executed_block` instead.
    """
    if not detection.get("checked"):
        return None
    if not detection.get("route_to_subagent"):
        return None
    sub_agent = detection.get("sub_agent")
    action = detection.get("suggested_action")
    announcement = detection.get("suggested_announcement")
    id_slug = detection.get("id_slug")
    if not (sub_agent and action and announcement and id_slug):
        return None
    full_id = f"coach-commit-{id_slug}"

    lines = [
        "",
        "**Detected user intent — artifact deliverable** (auxiliary "
        "classifier determined this turn asks for a saveable artifact "
        "that should be authored by a sub-agent, not inlined into your "
        "reply). Follow this routing unless the user message is clearly "
        "something else:",
        f"  - Call `enqueue_action(id=\"{full_id}\", "
        f"action=\"{action}\", sub_agent=\"{sub_agent}\")` to record "
        "the action.",
        f"  - Call `announce_subagent(sub_agent=\"{sub_agent}\", "
        f"text=\"{announcement}\")` so the user sees the team member "
        "taking the work.",
        "  - Your Coach-voice reply: brief emotional ack + correct-out "
        "ONLY. Do NOT inline the artifact content (no bullet lists of "
        "the cheat-sheet body, no draft text in your reply) — the "
        "sub-agent will deliver it as a separate artifact.",
    ]
    return "\n".join(lines)


def render_already_executed_block(
    sub_agent: str,
    action: str,
    full_id: str,
) -> str:
    """Render the system-prompt block for the server-auto-executed path.

    When the server already called enqueue_action + announce_subagent
    BEFORE Coach's turn started (S-0518-01 direction C), Coach sees this
    block. It tells Coach:
      1. The work is already in `action_queue` and the Slack push went
         out under the sub-agent prefix.
      2. Coach MUST NOT re-call enqueue_action or announce_subagent for
         this turn — that would duplicate state and produce a second
         Slack message.
      3. Coach's job this turn is the Coach-voice reply (framing,
         emotion, follow-up question) only.

    This is the architecture-level enforcement: side effects committed
    before LLM sees the turn, leaving Coach with one job that doesn't
    require it to choose between tools.
    """
    return "\n".join([
        "",
        "**Sub-agent action already executed** "
        "(server pre-executed the Type-E routing for this turn — backend "
        "state and the user-visible Slack push are already done):",
        f"  - `enqueue_action(id=\"{full_id}\", action=\"{action}\", "
        f"sub_agent=\"{sub_agent}\")` — committed to action_queue.",
        f"  - `announce_subagent(sub_agent=\"{sub_agent}\", ...)` — "
        "pushed to the user's Slack DM under the sub-agent prefix.",
        "",
        "**Do NOT call either tool again this turn** — both side effects "
        "are committed; re-calling duplicates state and posts a second "
        "Slack message. Your job this turn is the Coach-voice reply only: "
        "brief emotional ack + framing + optional correct-out / "
        "follow-up question. Do NOT inline the artifact content (no "
        "cheat-sheet body, no draft text); the sub-agent will deliver "
        "the artifact separately.",
    ])


def execute_via_helper(
    user_id: str,
    detection: dict[str, Any],
    *,
    helper_path: str | None = None,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Run the Artemis helper script that calls MCP server handlers.

    Returns:
      {"ok": True, "enqueue_result": ..., "announce_result": ...}  on success
      {"ok": False, "stage": "...", "error": "..."}                on failure

    The helper lives at $HERMES_HOME/scripts/execute-detected-action.py
    (deployed by setup.sh). Failures are not raised — caller inspects
    the dict and decides whether to fall back to the prompt-only path.
    """
    import os
    import subprocess

    fail = {"ok": False, "stage": "helper", "error": ""}

    if not detection.get("route_to_subagent"):
        fail["error"] = "detection has route_to_subagent=False"
        return fail
    sub_agent = detection.get("sub_agent")
    action = detection.get("suggested_action")
    announcement = detection.get("suggested_announcement")
    id_slug = detection.get("id_slug")
    if not (sub_agent and action and announcement and id_slug):
        fail["error"] = "detection missing required slots"
        return fail

    if helper_path is None:
        from pathlib import Path
        hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        helper_path = str(Path(hermes_home) / "scripts" / "execute-detected-action.py")
    if not os.path.exists(helper_path):
        fail["error"] = f"helper not found: {helper_path}"
        return fail

    payload = json.dumps({
        "user_id": user_id,
        "sub_agent": sub_agent,
        "id_slug": id_slug,
        "action": action,
        "announcement": announcement,
    })

    try:
        proc = subprocess.run(
            ["python3", helper_path],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        fail["error"] = f"helper timed out after {timeout_s}s"
        return fail
    except OSError as e:
        fail["error"] = f"helper exec failed: {e}"
        return fail

    raw = (proc.stdout or "").strip()
    if not raw:
        fail["error"] = f"helper produced no stdout (rc={proc.returncode}, stderr={proc.stderr[:200]!r})"
        return fail
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        fail["error"] = f"helper returned non-JSON: {raw[:200]!r}"
        return fail
    if not isinstance(result, dict):
        fail["error"] = f"helper returned non-dict: {raw[:200]!r}"
        return fail
    return result


def log_result(chat_id: str, detection: dict[str, Any]) -> None:
    """Single structured log line so accuracy is reviewable offline."""
    fields = (
        f"chat={chat_id or 'unknown'}",
        f"checked={detection.get('checked')}",
        f"skipped={detection.get('skipped')}",
        f"route={detection.get('route_to_subagent')}",
        f"sub_agent={detection.get('sub_agent')}",
        f"id_slug={detection.get('id_slug')!r}",
        f"confidence={detection.get('confidence')}",
        f"action={detection.get('suggested_action')!r}",
        f"announcement={detection.get('suggested_announcement')!r}",
        f"reasoning={detection.get('reasoning')!r}",
    )
    logger.info("turn-intent: %s", " ".join(fields))
