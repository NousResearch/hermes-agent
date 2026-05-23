"""Onboarding-complete signal detector (S-0518-01 Type A).

Runs AFTER Coach's reply is composed but BEFORE it is sent to Slack.
Decides whether this Coach reply ends Coach's onboarding intake (Coach has
gathered enough context to "brief the team") and the user's sub-agents
should now post their self-introductions — Type A in
`docs/research/2026-05-21-subagent-appearance-taxonomy.md`.

Pattern follows turn_intent_detector + execute_via_helper: auxiliary
classifier LLM (cheap, narrow prompt) decides, then helper subprocess
commits the side effect (3 announce_subagent pushes) before any
subsequent Coach turn can see them.

The detector also generates the three self-intro texts in the same call,
context-aware from the user's profile + Coach reply (so Jordan's intros
can mention Northwestern alumni access, Maya's stay generic).

**Failures are silent.** Auxiliary timeout, parse failure → return empty;
gateway/run.py skips the dispatch and Coach reply proceeds normally.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# Skip below this Coach-reply length — onboarding-complete signals appear
# in mid-length replies that name "team" or "briefing" or similar; very
# short replies (acks, single-word) can't carry the signal.
_MIN_REPLY_LEN_FOR_CHECK = 60

# Hard timeout. This runs synchronously between Coach reply ready and
# Slack send, so it must be fast.
_DETECT_TIMEOUT_S = 10.0


_DETECT_PROMPT = """\
You are a classifier for a career-coaching agent.

The agent (Coach) runs a team of three named sub-agents:
- **Scout** — market scanning, role discovery, company / event lookups
- **Analyst** — data analysis, interpretation, profile reads
- **Publicist** — drafts the materials the user sends (resumes, cover letters, outreach)

The first time Coach decides it has gathered enough about a new user to
"brief the team", the three sub-agents post short first-person
self-introductions in the Slack DM. We need to detect that moment.

You will receive (a) Coach's most recent reply text and (b) the user's
current profile. Your job: decide whether Coach's reply marks the
"team-briefing" handoff that should trigger the sub-agent self-intros now.

Signals that indicate YES:
- Coach explicitly says it's "briefing the team", "putting the team on
  it", "team is spinning up", "team has enough to work with", "you'll
  hear from them", or similar future-team-engagement phrasing.
- Coach references "first briefing" / "tomorrow morning" / "in the
  morning" as the next backend touchpoint.
- The reply concludes a profile-gathering exchange (Coach has just
  reflected the user's goal + background back to them) and pivots to
  team engagement.

Signals that indicate NO:
- This is a returning user (profile already had values before this
  conversation started — assume true if profile has non-empty `goal`
  AND `background` AND `sub_agent_intros_pushed` is true).
- Coach reply is mid-sharpening (asking the user another question to
  refine the profile).
- Coach is acknowledging a piece of new information without referencing
  team engagement.
- Coach is wrapping up a regular conversation turn unrelated to
  onboarding (e.g. confirming a Slack action, ending an emotional
  conversation, post-briefing follow-up).

If YES, also generate the three self-intro texts. They should:
- Be first-person (sub-agent speaking about itself).
- Briefly describe the sub-agent's work scope.
- Reference one concrete next step the sub-agent is taking right now
  (e.g. "already scanning", "reading your resume", "standing by").
- When the user profile contains specific affordances (e.g. school
  affiliation, alumni database access, specific industry), bake them
  into the relevant sub-agent's intro. Otherwise keep generic.
- Each intro is 1-3 sentences, plain text, no emoji or name prefix —
  the server adds those.

Return STRICT JSON, no prose, no markdown fence:

{
  "trigger": <true|false>,
  "confidence": "<high|medium|low>",
  "reasoning": "<one short sentence>",
  "intros": [
    {"sub_agent": "scout",     "text": "<scout's first-person self-intro>"},
    {"sub_agent": "analyst",   "text": "<analyst's first-person self-intro>"},
    {"sub_agent": "publicist", "text": "<publicist's first-person self-intro>"}
  ]
}

When trigger is false, return an empty `intros` array.

User's profile (JSON):

{user_profile}

Coach's reply:

\"\"\"
{coach_reply}
\"\"\"
"""


def _parse_response(raw: str) -> dict[str, Any] | None:
    """Tolerant JSON parse. None on failure."""
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


_VALID_SUB_AGENTS = {"scout", "analyst", "publicist"}


def _normalize_intros(raw: Any) -> list[dict[str, str]]:
    """Validate intro list. Returns [] if any required slot missing or
    if any sub_agent is duplicated / not in registry. Ordering must be
    scout / analyst / publicist (Maya + Jordan simulation order)."""
    if not isinstance(raw, list) or len(raw) != 3:
        return []
    seen: set[str] = set()
    by_sub: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            return []
        sa = item.get("sub_agent")
        tx = item.get("text")
        if sa not in _VALID_SUB_AGENTS or sa in seen:
            return []
        if not isinstance(tx, str) or not tx.strip():
            return []
        seen.add(sa)
        by_sub[sa] = tx.strip()
    if seen != _VALID_SUB_AGENTS:
        return []
    # Force canonical order regardless of input order.
    return [
        {"sub_agent": "scout", "text": by_sub["scout"]},
        {"sub_agent": "analyst", "text": by_sub["analyst"]},
        {"sub_agent": "publicist", "text": by_sub["publicist"]},
    ]


def is_onboarding_pushed(user_id: str) -> bool:
    """Check the persistent onboarding-flag file. Source of truth for the
    Type A idempotency latch — separate from profile.json so Coach's
    `save_user_profile` overwrites can't wipe it."""
    if not user_id:
        return False
    import os
    from pathlib import Path
    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    flag = Path(hermes_home) / "artemis" / user_id / "onboarding_pushed.flag"
    return flag.exists()


def detect_onboarding_complete(
    coach_reply: str,
    user_profile: dict | None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Classify whether Coach's reply marks the onboarding handoff.

    Args:
      coach_reply: Coach's outgoing reply text (the one about to land on
        Slack).
      user_profile: The user's profile dict (from get_user_profile or
        gateway's session prep). May be None / empty for cold-start.

    Returns:
      {
        "checked": bool,         # auxiliary call attempted + parsed
        "skipped": str|None,     # skip reason if not checked
        "trigger": bool,         # true = dispatch self-intros now
        "confidence": str|None,
        "reasoning": str|None,
        "intros": list[dict],    # 3 items if trigger; [] otherwise
      }
    """
    out: dict[str, Any] = {
        "checked": False,
        "skipped": None,
        "trigger": False,
        "confidence": None,
        "reasoning": None,
        "intros": [],
    }

    if not isinstance(coach_reply, str) or len(coach_reply) < _MIN_REPLY_LEN_FOR_CHECK:
        out["skipped"] = "reply_too_short"
        return out

    # Short-circuit: if the persistent onboarding flag is set, we already
    # did this once for the user; don't fire again. The flag lives at
    # `<hermes_home>/artemis/<user_id>/onboarding_pushed.flag` and is
    # written by `scripts/dispatch-team-self-intros.py` after the three
    # `*X here.*` pushes succeed. The profile-field fallback (legacy) is
    # kept for tests that pre-seed the profile rather than the file.
    if user_id and is_onboarding_pushed(user_id):
        out["skipped"] = "intros_already_pushed"
        return out
    if isinstance(user_profile, dict) and user_profile.get("sub_agent_intros_pushed"):
        out["skipped"] = "intros_already_pushed"
        return out

    try:
        from agent.auxiliary_client import call_llm  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        out["skipped"] = f"client_import_failed:{type(e).__name__}"
        return out

    profile_json = json.dumps(user_profile or {}, indent=2)
    prompt = _DETECT_PROMPT.replace(
        "{user_profile}", profile_json
    ).replace("{coach_reply}", coach_reply)

    try:
        response = call_llm(
            task="compression",
            messages=[
                {"role": "system", "content": "You return only strict JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.2,
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

    trigger = bool(parsed.get("trigger"))
    intros = _normalize_intros(parsed.get("intros"))
    # If the LLM said trigger but failed to produce valid intros, demote
    # to no-trigger (don't dispatch malformed pushes).
    if trigger and not intros:
        trigger = False

    out["checked"] = True
    out["trigger"] = trigger
    out["confidence"] = parsed.get("confidence") or None
    out["reasoning"] = parsed.get("reasoning") or None
    out["intros"] = intros if trigger else []
    return out


def execute_via_helper(
    user_id: str,
    intros: list[dict[str, str]],
    *,
    helper_path: str | None = None,
    timeout_s: float = 15.0,
    delay_seconds: float = 0.0,
    fire_and_forget: bool = False,
) -> dict[str, Any]:
    """Run the Artemis helper script that pushes the 3 self-intros and
    sets the onboarding-pushed flag.

    Args:
      delay_seconds: Helper sleeps this long before posting the first
        intro. Used so Coach's reply lands on Slack before the team's
        self-introductions start (Coach speaks first, then the team).
      fire_and_forget: When true, spawn the helper as a detached
        background process and return immediately with
        {"ok": True, "mode": "fire_and_forget"}; the caller does not
        observe success/failure. When false (default), wait for the
        helper to finish and parse its JSON output.

    Returns:
      Sync mode: {"ok": True, "pushed": <int>} on success
                 {"ok": False, "stage": "...", "error": "..."} on failure
      Fire-and-forget mode: {"ok": True, "mode": "fire_and_forget"}

    Failures are not raised — caller logs and proceeds.
    """
    import os
    import subprocess

    fail: dict[str, Any] = {"ok": False, "stage": "helper", "error": ""}

    if not intros or len(intros) == 0:
        fail["error"] = "no intros supplied"
        return fail

    if helper_path is None:
        from pathlib import Path
        hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        helper_path = str(Path(hermes_home) / "scripts" / "dispatch-team-self-intros.py")
    if not os.path.exists(helper_path):
        fail["error"] = f"helper not found: {helper_path}"
        return fail

    payload = json.dumps({
        "user_id": user_id,
        "intros": intros,
        "delay_seconds": delay_seconds,
    })

    # Same venv resolution as turn_intent_detector — Artemis MCP server
    # needs the Hermes venv python (mcp SDK).
    from pathlib import Path
    hermes_repo = os.environ.get("HERMES_REPO") or str(Path.home() / "hermes-agent")
    venv_python = str(Path(hermes_repo) / "venv" / "bin" / "python")
    if not Path(venv_python).exists():
        import sys as _sys
        venv_python = _sys.executable

    # Propagate thread_ts so the 3 self-intros land in the same Slack
    # thread Coach's reply lands in (matches Type F lead-in pattern).
    _subprocess_env = os.environ.copy()
    try:
        from tools.session_context import get_thread_ts as _ctx_thread_ts
        _tts = _ctx_thread_ts()
    except Exception:
        _tts = None
    if _tts:
        _subprocess_env["HERMES_SESSION_THREAD_TS"] = _tts

    if fire_and_forget:
        # Detached background process: dispatcher will sleep
        # delay_seconds then post intros. Gateway returns immediately
        # so Coach's reply can land on Slack first.
        try:
            proc_bg = subprocess.Popen(
                [venv_python, helper_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=_subprocess_env,
                start_new_session=True,
            )
            if proc_bg.stdin is not None:
                proc_bg.stdin.write(payload.encode("utf-8"))
                proc_bg.stdin.close()
        except OSError as e:
            fail["error"] = f"helper spawn failed: {e}"
            return fail
        return {"ok": True, "mode": "fire_and_forget"}

    try:
        proc = subprocess.run(
            [venv_python, helper_path],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=_subprocess_env,
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
    """Single structured log line for offline review."""
    intros = detection.get("intros") or []
    fields = (
        f"chat={chat_id or 'unknown'}",
        f"checked={detection.get('checked')}",
        f"skipped={detection.get('skipped')}",
        f"trigger={detection.get('trigger')}",
        f"confidence={detection.get('confidence')}",
        f"n_intros={len(intros)}",
        f"reasoning={detection.get('reasoning')!r}",
    )
    logger.info("onboarding-complete: %s", " ".join(fields))
