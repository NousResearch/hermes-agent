"""Autopilot's anti-sycophancy judge — drives the real Hermes Council.

This wires the user's separate ``hermes_council`` package into the agent through
its **Hermes-native backend** (``COUNCIL_PROVIDER=hermes``) so every persona
deliberates *in-process* on the user's own configured provider/model — no
external CLI, model-agnostic, works on weak models. The Council is the
independent reviewer that replaces the human at autopilot decision points:

    * :func:`judge_completion` — "is the GOAL verifiably done, or must the agent
      keep working?"  (the goal-chasing quality gate)
    * :func:`choose_answer`   — "what is the most-recommended answer?"  (used by
      the clarify auto-answer seam)

Design contract (mirrors the user's engine-enforced philosophy):
    * The judge is an INDEPENDENT pass, never the main model grading itself.
    * Completion requires the Council to *fail to refute* a completion claim —
      the Skeptic's whole job is to find why it is NOT done, so a lazy/sycophantic
      "done" cannot pass.
    * The Council is OPTIONAL: if it cannot be imported or reached, the gate
      degrades to a single independent ``auxiliary_client`` reviewer pass so
      autopilot never hard-crashes, and finally FAILS OPEN (stop) rather than
      looping blindly.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Resolved once per process.
_COUNCIL_READY: Optional[bool] = None
_COUNCIL_SRC: Optional[str] = None

# Bound prompt sizes so the gate stays affordable on low-context models.
_MAX_GOAL = 1500
_MAX_FINAL = 3000
_MAX_WORK = 3000


def _candidate_council_srcs() -> list[Path]:
    out: list[Path] = []
    for env in ("COUNCIL_SRC", "AUTOPILOT_COUNCIL_SRC"):
        v = os.environ.get(env, "").strip()
        if v:
            out.append(Path(v).expanduser())
    # Common local layouts: <hermes_parent>/council/src and ~/.hermes/council/src.
    try:
        here = Path(__file__).resolve()
        # .../<checkout>/agent/autopilot/council_gate.py -> parents[3] == <parent of checkout>
        parent_of_checkout = here.parents[3]
        out.append(parent_of_checkout / "council" / "src")
    except Exception:
        pass
    out.append(Path.home() / ".hermes" / "council" / "src")
    return out


def ensure_council_importable(council_model: str = "") -> bool:
    """Locate the Council package, add it to ``sys.path``, select the hermes lane.

    Returns True if ``hermes_council`` is importable afterwards. Idempotent.
    """
    global _COUNCIL_READY, _COUNCIL_SRC
    if _COUNCIL_READY is not None:
        return _COUNCIL_READY

    # Default the Council backend to the Hermes-native lane unless the operator
    # has explicitly pointed it somewhere else (e.g. a CLI provider).
    os.environ.setdefault("COUNCIL_PROVIDER", "hermes")
    if council_model:
        os.environ.setdefault("COUNCIL_HERMES_MODEL", council_model)

    try:
        import hermes_council.deliberation  # noqa: F401  (already on path)
        _COUNCIL_READY = True
        return True
    except Exception:
        pass

    for src in _candidate_council_srcs():
        try:
            libs = src / "libs"
            if (libs / "hermes_council" / "deliberation.py").exists():
                if str(libs) not in sys.path:
                    sys.path.insert(0, str(libs))
                os.environ.setdefault("COUNCIL_SRC", str(src))
                import hermes_council.deliberation  # noqa: F401
                _COUNCIL_SRC = str(src)
                _COUNCIL_READY = True
                logger.info("autopilot: Hermes Council loaded from %s", src)
                return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("autopilot: council candidate %s failed: %s", src, exc)
            continue

    _COUNCIL_READY = False
    logger.info("autopilot: Hermes Council not available; using auxiliary reviewer fallback")
    return False


def _trunc(text: Any, limit: int) -> str:
    s = "" if text is None else str(text)
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n…[truncated {len(s) - limit} chars]"


@dataclass
class CompletionVerdict:
    """Result of the goal-completion quality gate."""

    complete: bool
    directive: str = ""          # the next-action directive when not complete
    confidence: float = 0.0
    verdict: str = ""            # raw council verdict: allow|deny|conditional
    source: str = "council"      # council | aux | fallback
    summary: str = ""            # short human-readable rationale
    raw: dict[str, Any] = field(default_factory=dict)


_COMPLETION_FRAME = (
    "You are deciding whether an autonomous coding agent may STOP now because its "
    "GOAL is fully and verifiably COMPLETE. Apply these verdict semantics strictly:\n"
    "- 'allow'       = STOP: the goal is genuinely, verifiably complete; no substantive work remains.\n"
    "- 'deny'        = DO NOT STOP: substantive required work is missing, wrong, or only promised.\n"
    "- 'conditional' = NOT YET: only specific verification or checks remain before it can be called done.\n"
    "Judge by evidence in the agent's ACTUAL result, not by how confident it sounds. "
    "A polished claim of completion with no verification is NOT complete. Promises of "
    "future work ('I will…', 'next I'd…') are NOT completion."
)


def _completion_question(goal: str, work_summary: str, final_response: str) -> str:
    return (
        f"{_COMPLETION_FRAME}\n\n"
        f"GOAL:\n{_trunc(goal, _MAX_GOAL)}\n\n"
        f"AGENT'S LATEST RESULT (what it would deliver as final):\n"
        f"{_trunc(final_response, _MAX_FINAL)}\n\n"
        f"WORK CONTEXT (recent steps):\n{_trunc(work_summary, _MAX_WORK)}"
    )


def _compose_directive(arbiter: dict[str, Any], deliberations: list[dict[str, Any]]) -> str:
    """Turn a 'not complete' council result into a concrete next-step directive."""
    bits: list[str] = []
    wrong = str(arbiter.get("most_likely_wrong_point", "") or "").strip()
    if wrong:
        bits.append(f"Gap found by independent review: {wrong}")
    checks = [str(c).strip() for c in (arbiter.get("required_checks") or []) if str(c).strip()]
    fastest = str(arbiter.get("fastest_uncertainty_reducing_check", "") or "").strip()
    if checks:
        bits.append("Do these next: " + "; ".join(checks[:4]) + ".")
    elif fastest:
        bits.append(f"Do this next: {fastest}.")
    else:
        # Fall back to the sharpest critic's key point.
        for d in deliberations:
            kp = d.get("key_points") or []
            claim = str(d.get("claim", "") or "").strip()
            if kp:
                bits.append("Address: " + "; ".join(str(k) for k in kp[:3]) + ".")
                break
            if claim:
                bits.append(f"Address: {claim}")
                break
    safe = str(arbiter.get("safest_reversible_path", "") or "").strip()
    if safe:
        bits.append(f"Safest path: {safe}")
    if not bits:
        bits.append("The goal is not yet verifiably complete; identify and finish the remaining work.")
    return " ".join(bits)


def _council_run(question: str, *, mode: str, max_tokens: int) -> dict[str, Any]:
    """Seam: run the real Council (kept separate so tests can stub it)."""
    from hermes_council.deliberation import run_council

    return run_council(question, mode=mode, evidence_search=False, max_tokens=max_tokens)


def judge_completion(
    goal: str,
    work_summary: str,
    final_response: str,
    *,
    mode: str = "fast",
    council_model: str = "",
    max_tokens: int = 1200,
) -> CompletionVerdict:
    """Decide whether the goal is verifiably complete (the goal-chasing gate).

    Uses the real Hermes Council when available; otherwise a single independent
    auxiliary reviewer pass. Never raises — on total failure it FAILS OPEN
    (``complete=True``) so a broken judge cannot trap the user in a loop.
    """
    question = _completion_question(goal, work_summary, final_response)

    if ensure_council_importable(council_model):
        try:
            res = _council_run(question, mode=mode, max_tokens=max_tokens)
            verdict = str(res.get("verdict", "")).strip().lower()
            confidence = float(res.get("confidence", 0.0) or 0.0)
            arbiter = res.get("arbiter", {}) or {}
            deliberations = res.get("deliberations", []) or []
            complete = verdict == "allow"
            directive = "" if complete else _compose_directive(arbiter, deliberations)
            syco = res.get("sycophancy", {}) or {}
            summary = (
                f"council verdict={verdict or '?'} confidence={confidence:.2f} "
                f"panel={res.get('meta', {}).get('panel', '?')} "
                f"sycophancy={syco.get('overall', 0.0)}"
            )
            return CompletionVerdict(
                complete=complete,
                directive=directive,
                confidence=confidence,
                verdict=verdict,
                source="council",
                summary=summary,
                raw=res,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("autopilot: council judge failed (%s); falling back to aux reviewer", exc)

    return _aux_completion(goal, work_summary, final_response, council_model=council_model)


# --------------------------------------------------------------------------- #
# Fallback: a single independent auxiliary-model reviewer pass.                #
# Still NOT the main model grading itself — call_llm resolves an aux backend.  #
# --------------------------------------------------------------------------- #

_AUX_SYSTEM = (
    "You are an adversarial completion reviewer for an autonomous coding agent. "
    "You are NOT the agent and you do not trust it. Decide whether its GOAL is "
    "fully and verifiably complete. Reject lazy or sycophantic 'done' claims and "
    "promises of future work. Respond with ONLY one JSON object: "
    '{"complete": true|false, "confidence": 0..1, "next_action": "single concrete '
    'next step if not complete, else empty", "reason": "one sentence"}.'
)


def _aux_call(messages: list[dict[str, Any]], *, model: str, max_tokens: int, timeout: float) -> str:
    """Seam: single auxiliary-model call (kept separate so tests can stub it)."""
    from agent.auxiliary_client import call_llm

    kwargs: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens, "timeout": timeout}
    if model:
        kwargs["model"] = model
    resp = call_llm(**kwargs)
    return resp.choices[0].message.content or ""


def _aux_completion(
    goal: str, work_summary: str, final_response: str, *, council_model: str = ""
) -> CompletionVerdict:
    try:
        user = (
            f"GOAL:\n{_trunc(goal, _MAX_GOAL)}\n\n"
            f"AGENT'S LATEST RESULT:\n{_trunc(final_response, _MAX_FINAL)}\n\n"
            f"WORK CONTEXT:\n{_trunc(work_summary, _MAX_WORK)}"
        )
        content = _aux_call(
            [
                {"role": "system", "content": _AUX_SYSTEM},
                {"role": "user", "content": user},
            ],
            model=council_model,
            max_tokens=600,
            timeout=90,
        )
        data = _extract_json(content) or {}
        complete = bool(data.get("complete", False))
        confidence = float(data.get("confidence", 0.0) or 0.0)
        next_action = str(data.get("next_action", "") or "").strip()
        reason = str(data.get("reason", "") or "").strip()
        directive = "" if complete else (next_action or "Finish the remaining work toward the goal.")
        return CompletionVerdict(
            complete=complete,
            directive=directive,
            confidence=confidence,
            verdict="allow" if complete else "deny",
            source="aux",
            summary=f"aux reviewer complete={complete} ({reason[:80]})",
            raw=data,
        )
    except Exception as exc:  # noqa: BLE001
        # Total judge failure: fail OPEN (stop) so we never loop blindly.
        logger.warning("autopilot: aux reviewer failed (%s); failing open (stop)", exc)
        return CompletionVerdict(
            complete=True,
            directive="",
            confidence=0.0,
            verdict="allow",
            source="fallback",
            summary=f"judge unavailable ({exc}); delivered result",
        )


def _council_decision(options: list[str], decision_context: str) -> dict[str, Any]:
    """Seam: council multi-option decision (separate so tests can stub it)."""
    from hermes_council.deliberation import decision as _decision

    return _decision(options, decision_context=decision_context)


def _match_option(text: str, options: list[str]) -> str:
    """Return the single option that clearly appears in ``text``, else ''."""
    low = (text or "").lower()
    hits = [o for o in options if o and o.lower() in low]
    return hits[0] if len(hits) == 1 else ""


_PICK_SYSTEM = (
    "You stand in for an absent user and must choose the MOST DEFENSIBLE answer "
    "to a question an autonomous agent asked — not the most agreeable or easiest. "
    "Judge by evidence and consequences. Respond with ONLY one JSON object: "
    '{"choice": "the exact chosen option text, or your concise answer if open-ended", '
    '"rationale": "one sentence"}.'
)


def _aux_pick(question: str, options: list[str], *, context: str = "", council_model: str = "") -> str:
    """Independent reviewer that picks the most-defensible answer."""
    opt_block = ("\nOPTIONS (choose exactly one, return its exact text):\n- " + "\n- ".join(options)) if options else ""
    ctx_block = f"\nIndependent review notes:\n{_trunc(context, 1200)}" if context else ""
    user = f"QUESTION:\n{_trunc(question, _MAX_GOAL)}{opt_block}{ctx_block}"
    try:
        content = _aux_call(
            [{"role": "system", "content": _PICK_SYSTEM}, {"role": "user", "content": user}],
            model=council_model, max_tokens=400, timeout=90,
        )
        data = _extract_json(content) or {}
        choice = str(data.get("choice", "") or "").strip()
        if options:
            exact = _match_option(choice, options) or _match_option(question + " " + choice, options)
            return exact or choice or options[0]
        return choice
    except Exception as exc:  # noqa: BLE001
        logger.warning("autopilot: aux pick failed (%s)", exc)
        return options[0] if options else ""


def choose_answer(
    question: str,
    options: Optional[list[str]] = None,
    *,
    council_model: str = "",
    max_tokens: int = 1200,
) -> str:
    """Most-recommended answer to a clarify question — the human-surrogate.

    Multi-option questions go through the Council's adversarial ``decision`` so
    the pick is anti-sycophantic; open-ended questions get the Council's safest
    recommended path. Falls back to a single independent reviewer pass, then to
    the first option, so a clarify call in autopilot always resolves.
    """
    opts = [str(o).strip() for o in (options or []) if str(o).strip()]

    if ensure_council_importable(council_model):
        try:
            if opts:
                res = _council_decision(opts, question)
            else:
                res = _council_run(
                    f"What is the single most defensible answer to this question, and why?\n{question}",
                    mode="fast", max_tokens=max_tokens,
                )
            arb = res.get("arbiter", {}) or {}
            notes = " ".join(
                str(arb.get(k, "") or "")
                for k in ("safest_reversible_path", "what_evidence_supports", "most_likely_wrong_point")
            )
            if opts:
                direct = _match_option(notes, opts)
                if direct:
                    return direct
                return _aux_pick(question, opts, context=notes, council_model=council_model)
            ans = str(arb.get("safest_reversible_path", "") or "").strip()
            if ans:
                return ans
        except Exception as exc:  # noqa: BLE001
            logger.warning("autopilot: council choose_answer failed (%s); aux fallback", exc)

    return _aux_pick(question, opts, council_model=council_model)


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """Minimal JSON-object extraction (fenced or inline)."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s
        s = s.split("\n", 1)[-1] if "\n" in s else s
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = s.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
        start = s.find("{", start + 1)
    return None
