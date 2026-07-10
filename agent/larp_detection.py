"""Post-turn LARP guard: catch "claimed an action but didn't call a tool".

Policy-only (mirrors :mod:`agent.verification_stop`): it inspects the just-finished
turn and, when the model asserts it performed an action but made **no** matching
(substantive) tool call, returns a corrective nudge so the conversation loop
re-prompts instead of finalizing.

Three-way contract:
  (a) action-claim + ZERO substantive tool calls this turn -> TRUE LARP -> re-prompt
  (b) action-claim + a substantive tool that FAILED         -> honest narration of a
      broken tool -> pass through (the error is already in context; do not punish)
  (c) action-claim + a successful substantive tool          -> pass through

Disabled by default (opt-in); see ``DEFAULT_CONFIG["larp_detection"]``.
Tier-2 (an LLM judge for outcome-specific claims) is a further opt-in and fails
open (never re-prompts on error).
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

import logging

logger = logging.getLogger(__name__)

_FALSEY = {"0", "false", "no", "off", ""}

# Tool-name tokens whose calls do NOT count as "substantive" work. EMPTY by
# default for the lowest false-positive rate: any real tool call this turn means
# the model "did something", so a claim is not flagged. Users can add tokens via
# ``larp_detection.exempt_toolsets`` to make detection stricter (e.g. so a turn
# that only wrote to memory/todo still counts as a no-op for substantive claims).
_DEFAULT_EXEMPT: set[str] = set()

# Past-tense completion of a substantive action.
_ACTION_VERBS = (
    "updated|created|saved|wrote|written|ingested|added|removed|deleted|ran|"
    "executed|searched|fetched|downloaded|installed|configured|committed|pushed|"
    "sent|applied|fixed|implemented|generated|stored|recorded|registered|modified|"
    "edited|patched|built|deployed|uploaded|inserted|populated|completed"
)

_CLAIM_PATTERNS = [
    # "I have updated ...", "I've saved ...", "I updated ..."
    re.compile(
        r"\bI(?:\s+have|'ve)?\s+(?:just\s+|now\s+|already\s+|successfully\s+)?(?:"
        + _ACTION_VERBS
        + r")\b",
        re.IGNORECASE,
    ),
    # bare completion status
    re.compile(
        r"\b(?:all\s+(?:steps|tasks|items)\s+(?:are\s+)?(?:complete|completed|done)|"
        r"task\s+(?:is\s+)?(?:complete|completed|done)|completed\s+successfully|"
        r"successfully\s+completed)\b",
        re.IGNORECASE,
    ),
]

# Present-progressive / imperative action verbs used to ANNOUNCE (not report)
# work. Local models routinely end a turn with these instead of calling the
# tool: "I am proceeding with X now.", "Executing now.", "Proceeding with
# dispatch...". (Deliberately excludes status words like "waiting"/"processing".)
_ACTION_GERUNDS = (
    "proceeding|executing|dispatching|initiating|starting|running|creating|"
    "fixing|correcting|continuing|beginning|generating|building|deploying|"
    "uploading|downloading|fetching|searching|updating|writing|saving|"
    "ingesting|installing|committing|pushing|sending|applying|implementing|"
    "rewriting|recreating|moving|kicking\s+off"
)

# "narrate then stop": the message END announces intent instead of doing it.
# Covers "I'll X" / "I will X" / "I am going to X" AND the present-progressive
# "I am (now) proceeding/dispatching..." / "I'm executing..." — the dominant
# real-world form the earlier future-only pattern missed. Requires a gerund
# after "I am" so states ("I am unable/ready/done/sorry") don't match.
_NARRATE_THEN_STOP = re.compile(
    r"\bI(?:'ll|\s+will)\s+\w+"
    r"|\bI(?:'m|\s+am)\s+(?:now\s+|currently\s+)?(?:going\s+to\s+\w+|\w+ing\b)",
    re.IGNORECASE,
)

# Bare terminal action announcement: a sentence STARTING with an action gerund
# and ENDING the message with "now"/"immediately"/"…" (no trailing question).
# Catches "Executing now.", "Starting Batch 1 now.", "Proceeding with dispatch…".
_TERMINAL_ACTION = re.compile(
    r"(?:^|[.!\n]\s*)(?:" + _ACTION_GERUNDS + r")\b[^?\n]*?"
    r"(?:\bnow\b|\bimmediately\b|\.\.\.|…)[.!…\"'\s]*$",
    re.IGNORECASE,
)

# Modal/conditional words right before a verb that make it NOT a completion claim.
_MODAL_PREFIX = re.compile(r"\b(can|could|should|would|might|may|need to|try to|plan to)\s*$", re.IGNORECASE)


def _section(config: Optional[dict]) -> dict:
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}
    sec = config.get("larp_detection") if isinstance(config, dict) else None
    return sec if isinstance(sec, dict) else {}


def _flag(sec: dict, key: str, default: bool) -> bool:
    val = sec.get(key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() not in _FALSEY
    return bool(val)


def larp_detection_enabled(config: Optional[dict] = None, agent: Any = None) -> bool:
    env = os.environ.get("HERMES_LARP_DETECTION")
    if env is not None:
        return env.strip().lower() not in _FALSEY
    sec = _section(config)
    if _flag(sec, "enabled", False):
        return True
    # Opt-in high-risk window: LARPing spikes right after context compaction —
    # the summary reads as completed-action prose with the tool calls stripped,
    # and the model imitates it. When post_compaction_window > 0, run the guard
    # for that many turns after each compaction even if otherwise disabled.
    # Default 0 -> no behavior change.
    window = int(sec.get("post_compaction_window", 0) or 0)
    if window > 0 and agent is not None:
        tsc = getattr(agent, "_turns_since_compaction", None)
        if isinstance(tsc, int) and 0 <= tsc <= window:
            return True
    return False


def _exempt_tokens(config: Optional[dict]) -> set[str]:
    tokens = set(_DEFAULT_EXEMPT)
    raw = _section(config).get("exempt_toolsets")
    if isinstance(raw, (list, tuple, set)):
        tokens |= {str(t).strip().lower() for t in raw if str(t).strip()}
    return tokens


def _is_substantive(name: str, exempt: set[str]) -> bool:
    n = (name or "").strip().lower()
    if not n:
        return False
    return not any(tok in n for tok in exempt)


def _first_claim(text: str) -> Optional[str]:
    for pat in _CLAIM_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        prefix = text[max(0, m.start() - 16) : m.start()]
        if _MODAL_PREFIX.search(prefix):
            continue
        return text[m.start() : m.start() + 140].strip()
    # Intent-announcement (narrate-then-stop / bare terminal action) counts only
    # at the END of the message — and NOT when the message ends by asking the
    # user ("Want me to X now?" / "... now?"), which is correct stop-to-confirm
    # behavior, not a LARP.
    if text.rstrip().endswith("?"):
        return None
    tail = text[-200:]
    m = _NARRATE_THEN_STOP.search(tail)
    if m:
        return tail[m.start() : m.start() + 140].strip()
    m = _TERMINAL_ACTION.search(tail)
    if m:
        return tail[m.start() : m.start() + 140].strip()
    return None


def _looks_specific(claim: str) -> bool:
    return bool(re.search(r"\d", claim)) or bool(
        re.search(r"\b(found|contains?|returned|listed|retrieved)\b", claim, re.IGNORECASE)
    )


def _last_user_index(messages: list) -> int:
    synthetic = ("_verification_stop_synthetic", "_larp_reprompt_synthetic", "_empty_recovery_synthetic")
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "user" and not any(m.get(k) for k in synthetic):
            return i
    return -1


def _turn_tool_activity(messages: list, exempt: set[str]) -> tuple[bool, bool, bool]:
    """Return (made_substantive_call, any_success, any_fail) since the last user msg."""
    try:
        from agent.display import _detect_tool_failure
    except Exception:
        _detect_tool_failure = None  # type: ignore[assignment]

    start = _last_user_index(messages)
    made = any_success = any_fail = False
    id_to_name: dict[str, str] = {}
    for m in messages[start + 1 :]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                name = fn.get("name") or (tc.get("name") if isinstance(tc, dict) else "") or ""
                if _is_substantive(name, exempt):
                    made = True
                tcid = tc.get("id") if isinstance(tc, dict) else None
                if tcid:
                    id_to_name[tcid] = name
        elif role == "tool":
            name = m.get("name") or m.get("tool_name") or id_to_name.get(m.get("tool_call_id"), "")
            if not _is_substantive(name, exempt):
                continue
            is_err = False
            if _detect_tool_failure is not None:
                try:
                    is_err, _ = _detect_tool_failure(name, m.get("content"))
                except Exception:
                    is_err = False
            if is_err:
                any_fail = True
            else:
                any_success = True
    return made, any_success, any_fail


def _judge_ungrounded(messages: list, final_response: str, claim: str) -> bool:
    """Tier-2: one cheap aux-LLM check. Returns True only on a confident UNGROUNDED
    verdict; fails open (False) on any error."""
    try:
        from agent.auxiliary_client import call_llm

        start = _last_user_index(messages)
        tool_lines = []
        for m in messages[start + 1 :]:
            if isinstance(m, dict) and m.get("role") == "tool":
                c = m.get("content")
                tool_lines.append(f"- {m.get('name') or m.get('tool_name')}: {str(c)[:300]}")
        tool_summary = "\n".join(tool_lines[:20]) or "(no tool results this turn)"
        sys = (
            "You audit whether an assistant's claims of completed actions are GROUNDED "
            "in the tool results from this turn. Reply with exactly one word: GROUNDED "
            "or UNGROUNDED."
        )
        usr = (
            f"Assistant final message:\n{final_response[:1500]}\n\n"
            f"Tool results this turn:\n{tool_summary[:2000]}\n\n"
            f"Specifically check this claim: {claim[:200]}"
        )
        resp = call_llm(
            task="larp_detection",
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            max_tokens=8,
            temperature=0.0,
            timeout=20.0,
        )
        verdict = (resp.choices[0].message.content or "").strip().upper()
        return verdict.startswith("UNGROUND")
    except Exception:
        logger.debug("LARP judge failed (fail-open)", exc_info=True)
        return False


def _nudge(claim: str, *, specific: bool) -> str:
    extra = (
        " Your claim references a specific result that the tool outputs do not support."
        if specific
        else ""
    )
    return (
        "[System: In your previous message you indicated you completed an action "
        f'("{claim[:120]}") but no corresponding tool call was made this turn.{extra} '
        "Either perform the action now using the appropriate tool, or clearly state that "
        "you did not/cannot do it and why. Do not report actions as done unless a tool "
        "call actually performed them.]"
    )


def build_larp_nudge(
    *,
    messages: list,
    final_response: str,
    agent: Any = None,
    config: Optional[dict] = None,
    attempts: int = 0,
) -> Optional[str]:
    """Return a corrective re-prompt when the turn LARPed, else None."""
    sec = _section(config)
    if attempts >= int(sec.get("max_reprompts", 2) or 2):
        return None
    text = (final_response or "").strip()
    if not text:
        return None
    claim = _first_claim(text)
    if not claim:
        return None

    exempt = _exempt_tokens(config)
    made, any_success, _any_fail = _turn_tool_activity(messages, exempt)

    if made:
        # (b)/(c): real tool activity backs (or honestly fails) the turn -> pass,
        # unless the opt-in judge says an outcome-specific claim is ungrounded.
        if _flag(sec, "judge_tier_enabled", False) and any_success and _looks_specific(claim):
            if _judge_ungrounded(messages, text, claim):
                return _nudge(claim, specific=True)
        return None

    # (a): an action claim with ZERO substantive tool calls this turn = LARP.
    return _nudge(claim, specific=False)


__all__ = ["larp_detection_enabled", "build_larp_nudge"]
