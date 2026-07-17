"""Stage 2: ACP Agent deep review — independent Hermes instance for semantic reasoning.

Uses `hermes chat -q --profile approval` to launch a standalone review Agent.
Stateless design: each invocation is an independent process, all context injected.

Context sources (from session DB, zero shared state):
  1. Current operation — tool_name + args + risk signals (from stage1_rules)
  2. Tool call chain — last 10 calls (including SAFE_TOOLS)
  3. Recent conversation — user messages + Agent replies (last 3 turns)
  4. Session approval history — prior ACP decisions this session (from Hindsight)
  5. Cross-session pattern memory — historical stats for similar ops (from Hindsight)
"""

from __future__ import annotations

import json
import logging
import os as _os
import signal
import subprocess
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_PROFILE = "approval"


def _query_context(
    session_id: str,
    tool_name: str,
    args: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, str]:
    """Query Hindsight context needed by ACP."""
    session_history = ""
    pattern_history = ""

    try:
        from .hindsight_store import query_session_history, query_pattern_history
        session_history = query_session_history(session_id, cfg, limit=5)
        pattern_history = query_pattern_history(tool_name, args, cfg, limit=5)
    except Exception as exc:
        logger.debug("Hindsight context query failed: %s", exc)

    return {
        "session_history": session_history,
        "pattern_history": pattern_history,
        "compression_summary": "",
    }


def _build_review_prompt(
    tool_name: str,
    args_str: str,
    context: Dict[str, Any],
    session_ctx: Dict[str, Any],
    hindsight_ctx: Dict[str, str],
    task_id: str,
    session_id: str,
) -> str:
    """Build ACP deep review prompt.

    Section order: conversation → current operation → tool chain →
                    approval history → cross-session memory → judgment
    """
    signals = context.get("signals", [])
    risk_section = "\n".join(f"  - {s}" for s in signals) if signals else "  - No risk signals"

    # ── Conversation context (user messages + Agent replies) ────
    turns = session_ctx.get("turns", [])
    if turns:
        conversation_section = "\n\n".join(turns)
    else:
        conversation_section = "(no conversation history recorded yet)"

    # ── Tool call chain ────────────────────────────────────────
    tool_calls = session_ctx.get("tool_calls", [])
    if tool_calls:
        chain_lines = []
        for i, call in enumerate(tool_calls, 1):
            marker = "\u2192" if i == len(tool_calls) else " "
            chain_lines.append(f"  {marker} [{i}] {call}")
        chain_section = "\n".join(chain_lines)
    else:
        chain_section = "(no tool call history recorded yet)"

    # ── Session approval history ───────────────────────────────
    session_section = hindsight_ctx.get("session_history", "")
    if not session_section:
        session_section = "  No prior ACP decisions in this session"

    # ── Cross-session pattern memory ───────────────────────────
    pattern_section = hindsight_ctx.get("pattern_history", "")
    if not pattern_section:
        pattern_section = "  No similar operations in approval history"

    # ── Compression summary ────────────────────────────────────
    compression_section = hindsight_ctx.get("compression_summary", "")
    if not compression_section:
        compression_section = "(context compression has not been triggered this session)"

    return f"""You are a security review agent. You have more context and reasoning power than the fast classifier.
Review the AI Agent's tool call request and determine safety. Many flagged operations are actually safe — judge based on user intent, conversation context, and operation chain.

============================================
[Recent Conversation]
(User messages and Agent replies — helps understand task intent)
============================================
{conversation_section}

============================================
[Current Operation]
============================================
Tool: {tool_name}
Args: {args_str}
Risk signals:
{risk_section}

============================================
[Tool Call Chain]
============================================
{chain_section}

============================================
[Session Approval History]
============================================
{session_section}

============================================
[Cross-Session Pattern Memory]
============================================
{pattern_section}

============================================
[Compression Summary]
============================================
{compression_section}

============================================
[Your Judgment]
============================================
**CRITICAL RULE — DEFAULT TO ALLOW WHEN CONTEXT IS SPARSE:**
If the conversation history and tool chain above are empty or show
"(no ... recorded yet)", you MUST output ALLOW.  Missing context is
normal early in a session — it does NOT indicate malicious activity.
System management operations (restart, deploy, config changes) are
expected and safe.  Only DENY when you see concrete evidence of
destructive intent in the provided context.

**You are an automated security check program, not a conversational assistant.**
Your sole task is to output a single line of JSON verdict.
Do not greet, do not explain, do not ask questions, do not offer help.

Output this line and ONLY this line (no markdown code blocks, no prefix/suffix):
{{"verdict":"ALLOW","reason":"brief reason","suggestion":"","confidence":0.95}}

Decision logic:
  ALLOW   — context is sparse/unavailable, OR conversation context clearly explains this operation
  DENY    — conversation context clearly contradicts this operation, operation is CLEARLY destructive with no user justification
  MODIFY  — allow but suggest parameter changes (safer path, lesser permissions, etc.)"""


def _store_decision(
    tool_name: str,
    args: Dict[str, Any],
    verdict: str,
    reason: str,
    session_id: str,
    cfg: Dict[str, Any],
) -> None:
    """Write ACP decision back to Hindsight (fire-and-forget)."""
    try:
        from .hindsight_store import record_decision
        record_decision(tool_name, args, verdict, reason, session_id, cfg)
    except Exception as exc:
        logger.debug("Failed to store ACP decision: %s", exc)


def acp_agent_review(
    tool_name: str,
    args: Dict[str, Any],
    cfg: Dict[str, Any],
    context: Dict[str, Any],
    session_ctx: Dict[str, Any],
    task_id: str,
    session_id: str,
) -> Tuple[str, Dict[str, Any]]:
    """Stateless ACP deep review.

    Args:
        tool_name: Tool name
        args: Tool arguments
        cfg: Plugin config
        context: Risk signals from stage1_rules
        session_ctx: {{"tool_calls": [], "turns": []}} from guard.py session DB
        task_id: Parent Agent task ID
        session_id: Parent Agent session ID

    Returns:
        (verdict, detail) — ALLOW/DENY/MODIFY
    """
    stage2_cfg = cfg.get("stage2", {})
    profile = stage2_cfg.get("profile", DEFAULT_PROFILE)
    timeout = stage2_cfg.get("timeout", 15)
    fail_open = cfg.get("fail_open", True)

    # ── Short-circuit: no session context available → skip ACP ──────
    # When SessionDB is unreachable or the session has no recorded
    # conversation, the ACP agent sees empty context and defaults to
    # DENY (worst-case bias).  Better to skip the expensive subprocess
    # call entirely and trust Stage 1's ESCALATE + system manual fallback.
    turns = session_ctx.get("turns", [])
    tool_calls = session_ctx.get("tool_calls", [])
    no_context = (
        not turns
        or all(
            isinstance(t, str) and "no " in t.lower() and "recorded" in t.lower()
            for t in turns
        )
    ) and (
        not tool_calls
        or all(
            isinstance(t, str) and "no " in t.lower() and "recorded" in t.lower()
            for t in tool_calls
        )
    )
    if no_context:
        logger.debug(
            "ACP: no session context for %s — skipping deep review, fall through to ALLOW",
            session_id,
        )
        return "ALLOW", {"reason": "acp_no_context", "confidence": 0.0}

    args_str = json.dumps(args, ensure_ascii=False, default=str)
    if len(args_str) > 4000:
        args_str = args_str[:4000] + "...(truncated)"

    hindsight_ctx = _query_context(session_id, tool_name, args, cfg)

    review_prompt = _build_review_prompt(
        tool_name, args_str, context, session_ctx, hindsight_ctx, task_id, session_id
    )

    try:
        # Use Popen rather than subprocess.run so we can kill the entire
        # process group on timeout — subprocess.run only kills the direct
        # child, leaving grandchild processes (e.g. hermes chat subprocesses)
        # orphaned.
        proc = subprocess.Popen(
            [
                "hermes", "chat", "-q", review_prompt,
                "--profile", profile,
                "-t", "session_search",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env={**_os.environ},  # No YOLO_MODE — reviewer must not bypass approval
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout + 5)
        except subprocess.TimeoutExpired:
            # Kill entire process group so no orphans survive
            try:
                _os.killpg(_os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    stdout, stderr = proc.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    _os.killpg(_os.getpgid(proc.pid), signal.SIGKILL)
                    stdout, stderr = proc.communicate()
            except (ProcessLookupError, OSError):
                stdout, stderr = "", ""
            raise  # Re-raise so the outer except block below handles it

        output = stdout.strip()

        returncode = proc.returncode
        if returncode != 0:
            logger.warning("ACP agent returned non-zero: %s", stderr[:200])
            _store_decision(tool_name, args, "ERROR", f"acp_error(code={returncode})", session_id, cfg)
            if fail_open:
                return "ALLOW", {"reason": "acp_error", "confidence": 0.0}
            return "DENY", {"reason": f"acp_error(code={returncode})", "confidence": 0.0}

        parsed_verdict, detail = _parse_acp_output(output)
        _store_decision(tool_name, args, parsed_verdict, detail.get("reason", ""), session_id, cfg)
        return parsed_verdict, detail

    except subprocess.TimeoutExpired:
        logger.warning("ACP agent review timed out after %ds", timeout)
        _store_decision(tool_name, args, "TIMEOUT", "acp_timeout", session_id, cfg)
        if fail_open:
            return "ALLOW", {"reason": "acp_timeout", "confidence": 0.0}
        return "DENY", {"reason": "acp_timeout", "confidence": 0.0}

    except (FileNotFoundError, OSError) as exc:
        logger.warning("hermes CLI not available for ACP review: %s", exc)
        if fail_open:
            return "ALLOW", {"reason": f"acp_unavailable:{exc}", "confidence": 0.0}
        return "DENY", {"reason": f"acp_unavailable:{exc}", "confidence": 0.0}


def _parse_acp_output(output: str) -> Tuple[str, Dict[str, Any]]:
    """Parse ACP output: JSON first, text matching fallback.

    Note: hermes chat -q output format is "Query: <prompt>\n<response>".
    We must strip the prompt portion and only parse the response.
    """
    # Strip non-response content (Query: prefix + prompt echo)
    clean = output
    # hermes chat -q output starts with "Query: "
    if clean.startswith("Query: "):
        first_newline = clean.find("\n")
        if first_newline > 0:
            clean = clean[first_newline + 1:].strip()
        # There may also be "Unknown provider..." etc. error lines
        lines = clean.split("\n")
        # Find the first line that doesn't look like an error/tip
        error_prefixes = ("Unknown ", "Check '", "Goodbye!", "Error:", "No provider")
        response_lines = [l for l in lines if not any(l.startswith(p) for p in error_prefixes)]
        clean = "\n".join(response_lines).strip()

    # JSON first
    try:
        json_start = clean.rfind("{")
        json_end = clean.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            parsed = json.loads(clean[json_start:json_end])
            verdict = parsed.get("verdict", "ALLOW").upper()
            if verdict in ("ALLOW", "DENY", "MODIFY"):
                return verdict, {
                    "reason": parsed.get("reason", ""),
                    "suggestion": parsed.get("suggestion", ""),
                    "confidence": parsed.get("confidence", 0.5),
                }
    except (json.JSONDecodeError, KeyError):
        logger.debug("ACP output not valid JSON verdict: %s", clean[:200])

    # Fallback: text keyword matching (response portion only)
    upper_clean = clean.upper()
    if "DENY" in upper_clean and "ALLOW" not in upper_clean:
        return "DENY", {"reason": clean[:200], "confidence": 0.5}
    elif "MODIFY" in upper_clean:
        return "MODIFY", {"reason": clean[:200], "confidence": 0.5}
    elif "ALLOW" in upper_clean:
        return "ALLOW", {"reason": clean[:200], "confidence": 0.5}
    else:
        return "ALLOW", {"reason": "acp_unclear", "confidence": 0.3}
