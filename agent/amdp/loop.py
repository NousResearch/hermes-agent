"""AMDP episode loop, integrated into the agent runtime.

Model calls go through ``agent.auxiliary_client.call_llm`` (resolved via MoA's
``_slot_runtime``). Believed state comes from a pluggable state feed (see
``state.py``): the universal ``gateway`` feed by default, with the
proprioception plugin as optional enrichment when installed. AMDP has no hard
dependency on any monitoring plugin.

``maybe_amdp_context`` is the single entry point the conversation loop calls.
It is fail-closed: it returns ``""`` (inject nothing, proceed normally) on a
disabled config, a turn that doesn't clear the gate, blind/stale state, or ANY
exception. It never raises into the turn.

Misconfiguration is fail-LOUD, not fail-silent: an ``enabled: true`` block with
an empty planner/reviewer is resolved once (cached) and logged at ERROR, and
AMDP stays disabled — it does NOT silently no-op every turn while looking fine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agent.amdp import prompts, schemas, scoring
from agent.amdp.config import AmdpConfig, AmdpConfigError, resolve_amdp_config

logger = logging.getLogger(__name__)

_MAX_REVIEW_WORKERS = 8
_AUDIT_LOCK = threading.Lock()

# Per-call timeout for the current episode, set from cfg.call_timeout_s at the
# top of maybe_amdp_context. A module global (not a _call kwarg) so the test
# suite's mock _call signature stays stable; concurrent episodes use a similar
# value so the benign race is harmless.
_active_call_timeout: float = 90.0
# Active state-feed mode for the current episode, set from cfg.state_feed. A
# module global (not an _intake kwarg) so the test suite's mock _intake
# signature stays stable — same pattern as _active_call_timeout.
_active_state_feed: str = "auto"

# Startup-resolved config cache. Resolved ONCE (first turn on the hook path),
# then reused, so the disabled/absent path costs a cheap identity check instead
# of a fresh load_config() + deepcopy every turn. Config changes need a restart
# (consistent with the ConfigSentinel/golden workflow).
_UNSET = object()
_cached_cfg: Any = _UNSET
_cached_raw: dict[str, Any] = {}

# Per-user-turn plan cache. The conversation loop calls the hook on EVERY
# tool-loop iteration; without this, AMDP re-plans (a full ~40s episode) on each
# one. We plan ONCE per user turn and re-inject the cached result on subsequent
# iterations with zero model calls — same idea as MoA's user_turn fan-out. Keyed
# by a hash of the original user prompt + the conversation prefix up to the last
# user message (stable across a turn's tool iterations, changes on a new turn).
# Bounded; oldest entry dropped. A cached value of "" (a gate-skip or refusal) is
# distinct from a cache miss (None from .get).
_PLAN_CACHE: "OrderedDict[str, str]" = OrderedDict()
_PLAN_CACHE_MAX = 64
_PLAN_CACHE_LOCK = threading.Lock()


def _turn_signature(user_prompt: Any, api_messages: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    h.update(("PROMPT\x00" + str(user_prompt)).encode("utf-8", "replace"))
    last_user = -1
    for i, m in enumerate(api_messages):
        if isinstance(m, dict) and m.get("role") == "user":
            last_user = i
    prefix = api_messages[: last_user + 1] if last_user >= 0 else api_messages
    for m in prefix:
        if isinstance(m, dict):
            c = m.get("content")
            c = c if isinstance(c, str) else json.dumps(c, default=str, ensure_ascii=False)
            h.update(f"\x00{m.get('role')}\x00{c}".encode("utf-8", "replace"))
    return h.hexdigest()


def _cache_get(sig: str) -> str | None:
    with _PLAN_CACHE_LOCK:
        return _PLAN_CACHE.get(sig)


def _cache_put(sig: str, val: str) -> None:
    with _PLAN_CACHE_LOCK:
        _PLAN_CACHE[sig] = val
        _PLAN_CACHE.move_to_end(sig)
        while len(_PLAN_CACHE) > _PLAN_CACHE_MAX:
            _PLAN_CACHE.popitem(last=False)


# --------------------------------------------------------------------------- #
# Model plumbing (real infra)
# --------------------------------------------------------------------------- #
def _extract_text(response: Any) -> str:
    """Assistant text with the gpt-oss reasoning_content fallback.

    Handles both a plain dict message (some transports / tests) and the OpenAI
    SDK ``ChatCompletionMessage`` (a pydantic model), where provider-extension
    fields like ``reasoning_content`` arrive in ``message.model_extra`` rather
    than as a declared attribute. Missing that path silently blinded the
    reviewer on reasoning-only models."""
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return ""
    if isinstance(message, dict):
        for field in ("content", "reasoning_content", "reasoning"):
            val = message.get(field)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""
    extra = getattr(message, "model_extra", None)
    if not isinstance(extra, dict):
        extra = {}
    for field in ("content", "reasoning_content", "reasoning"):
        val = getattr(message, field, None)
        if not (isinstance(val, str) and val.strip()):
            val = extra.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _call(
    slot: dict[str, str],
    messages: list[dict[str, Any]],
    *,
    temperature: float | None,
    max_tokens: int | None,
    json_mode: bool = False,
) -> tuple[str, str]:
    """One model call through call_llm. Returns (text, error). Never raises.
    Bounded by the per-episode ``_active_call_timeout`` so a flapping local
    endpoint cannot stall the turn indefinitely."""
    try:
        from agent.auxiliary_client import call_llm
        from agent.moa_loop import _slot_runtime

        runtime = _slot_runtime(slot)
        extra_body = {"response_format": {"type": "json_object"}} if json_mode else None
        response = call_llm(
            task="amdp",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
            timeout=_active_call_timeout,
            **runtime,
        )
        return _extract_text(response), ""
    except Exception as exc:  # fail-closed
        logger.warning("AMDP model call failed for %s: %s", slot, exc)
        return "", f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# State intake (via a pluggable state feed — see state.py)
# --------------------------------------------------------------------------- #
def _intake(config: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
    """Return {brief, sensors_down, staleness_s, ...} from the configured state
    feed. Never raises. The source is pluggable (``gateway`` universal default /
    ``proprioception`` enrichment / ``auto``); enrichment being unavailable
    degrades gracefully — only a missing gateway status (``gateway-status`` in
    sensors_down) or excessive staleness blinds the planner."""
    from agent.amdp import state

    return state.get_believed_state(config or {}, timeout_s=timeout_s, mode=_active_state_feed)


def _should_refuse(state: dict[str, Any], *, staleness_max_s: float) -> tuple[bool, str]:
    # Refuse only when truly blind (no gateway status) or the state is stale —
    # NOT when only the optional dashboard is unavailable.
    if "gateway-status" in state.get("sensors_down", []):
        return True, "gateway status unavailable (no usable state)"
    if state.get("staleness_s", 0) > staleness_max_s:
        return True, f"state staleness {state['staleness_s']:.0f}s exceeds max {staleness_max_s:.0f}s"
    return False, ""


# --------------------------------------------------------------------------- #
# Gate — is this turn dispatch-worthy? (cheap heuristic, no model call)
# --------------------------------------------------------------------------- #
# Deliberately excludes bare "then"/"first" — they fire on ordinary conversation
# ("first, thanks; then...") and would tax a chat turn with a 30-60s planning
# episode. Keep multi-word / task-shaped markers only.
_MULTISTEP_HINTS = re.compile(
    r"\b(migrate|refactor|pipeline|deploy|and then|after that|step\s*\d|phase\s*\d|"
    r"each of|all of|for every|for each|end[- ]to[- ]end|multi[- ]step|"
    r"exec\w*|implement|build\s*out|carry\s*out|roll\s*out|wire\s*up|scaffold)\b",
    re.IGNORECASE,
)

_BACKGROUND_INTENT_HINTS = re.compile(
    r"(?:\[/learn\]|update (?:the )?skill library|memory maintenance|"
    r"scheduled (?:cron|job|summary|digest)|nightly (?:journal|summary|backup)|"
    r"routine health check|background (?:curator|review|maintenance))",
    re.IGNORECASE,
)

_HIGH_RISK_HINTS = re.compile(
    r"\b(delete|remove|purge|drop|overwrite|restore|restart|kill|terminate|"
    r"production|client data|billing|payment|credential|secret|irreversible)\b",
    re.IGNORECASE,
)


def _is_background_intent(user_prompt: Any) -> bool:
    text = user_prompt if isinstance(user_prompt, str) else ""
    return bool(_BACKGROUND_INTENT_HINTS.search(text))


def _review_reason(coas: list[dict[str, Any]], intent: str) -> str:
    if len(coas) > 1:
        return "multiple candidate plans"
    for coa in coas:
        if any(bool(d.get("irreversible")) for d in coa.get("dispatches", [])):
            return "irreversible action"
    if _HIGH_RISK_HINTS.search(intent or ""):
        return "high-risk intent"
    return ""


def _estimate_steps(user_prompt: Any, api_messages: list[dict[str, Any]]) -> int:
    """Very cheap dispatch-worthiness estimate. Don't spend a model call deciding
    whether to spend model calls.

    Terse execution orders ("execu phase 1", "implement it") are as multi-step as
    verbose ones ("migrate the pipeline") — the hint set now covers execution verbs
    and phase markers, not just planning phrasing. And a turn arriving mid-build
    (lots of prior tool activity) is dispatch-worthy on its own: a short "continue"
    after 8+ tool calls is real work, not chat."""
    text = user_prompt if isinstance(user_prompt, str) else ""
    score = 0
    score += len(_MULTISTEP_HINTS.findall(text))
    score += text.count("\n- ") + text.count("\n1.") + text.count("\n2.")  # list items
    if len(text) > 400:
        score += 1
    tool_msgs = sum(1 for m in api_messages if isinstance(m, dict) and m.get("role") == "tool")
    if tool_msgs >= 3:
        score += 1
    if tool_msgs >= 8:  # a build/investigation is well underway — terse orders count
        score += 1
    return score


# --------------------------------------------------------------------------- #
# Planner (ported from the proven prototype)
# --------------------------------------------------------------------------- #
def _generate_coas(cfg: AmdpConfig, intent: str, state_brief: str, errors: list[str]) -> list[dict[str, Any]]:
    msgs = prompts.commander_prompt(intent, state_brief, cfg.n_coas)
    text, err = _call(cfg.planner, msgs, temperature=0.4, max_tokens=3500, json_mode=True)
    coas: list[dict[str, Any]] | None = None
    if text:
        try:
            coas = schemas.coerce_coas(schemas.extract_json(text))
        except ValueError as exc:
            errors.append(f"COA parse failed ({exc}); retrying")
    else:
        errors.append(f"commander failed: {err or 'empty'}")

    if coas is None:
        # Repair WITHOUT json_mode (prompt-only): if the endpoint rejected
        # response_format, or emitted prose, extract_json can still recover.
        repair = msgs + [
            {"role": "assistant", "content": (text or "")[:2000]},
            {"role": "user", "content": "Return ONLY the JSON object with a top-level 'coas' array. No prose, no fences."},
        ]
        text2, err2 = _call(cfg.planner, repair, temperature=0.1, max_tokens=3500, json_mode=False)
        if not text2:
            errors.append(f"commander repair failed: {err2 or 'empty'}")
            return []
        try:
            coas = schemas.coerce_coas(schemas.extract_json(text2))
        except ValueError as exc:
            errors.append(f"COA parse failed after repair: {exc}")
            return []

    # Force unique coa_ids regardless of what the model labeled them, so two
    # COAs sharing an id can't collapse in _decide's review-by-id map and pin
    # the wrong review to a plan.
    for i, coa in enumerate(coas):
        coa["coa_id"] = chr(ord("A") + i) if i < 26 else f"C{i}"
    return coas


def _review_one(cfg: AmdpConfig, intent: str, state_brief: str, coa: dict[str, Any]) -> dict[str, Any]:
    msgs = prompts.review_prompt(intent, state_brief, coa)
    text, _err = _call(cfg.reviewer, msgs, temperature=0.3, max_tokens=cfg.reviewer_max_tokens)
    if text:
        try:
            return schemas.coerce_review(schemas.extract_json(text), coa["coa_id"])
        except ValueError:
            pass
    repair = msgs + [
        {"role": "assistant", "content": (text or "")[:1500]},
        {"role": "user", "content": "Return ONLY the JSON verdict object. No prose."},
    ]
    text2, _err2 = _call(cfg.reviewer, repair, temperature=0.1, max_tokens=cfg.reviewer_max_tokens)
    if text2:
        try:
            return schemas.coerce_review(schemas.extract_json(text2), coa["coa_id"])
        except ValueError:
            pass
    return {
        "coa_id": coa["coa_id"], "alignment_1to10": 0.0,
        "risks": [{"desc": "reviewer unavailable — COA unvetted", "severity_1to5": 5}],
        "unstated_assumptions": [], "fragility_0to1": 1.0, "_review_failed": True,
    }


def _review_all(cfg: AmdpConfig, intent: str, state_brief: str, coas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any] | None] = [None] * len(coas)
    workers = min(_MAX_REVIEW_WORKERS, len(coas)) or 1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_review_one, cfg, intent, state_brief, c): i for i, c in enumerate(coas)}
        for fut, i in futs.items():
            results[i] = fut.result()
    return [r for r in results if r is not None]


def _decide(cfg: AmdpConfig, coas: list[dict[str, Any]], reviews: list[dict[str, Any]], staleness_norm: float):
    review_by_id = {r["coa_id"]: r for r in reviews}
    scored = []
    for coa in coas:
        review = review_by_id.get(coa["coa_id"]) or {"coa_id": coa["coa_id"], "alignment_1to10": 0.0, "risks": [], "fragility_0to1": 1.0}
        by_profile = {
            name: {"score": round(scoring.score(review, staleness_norm=staleness_norm, profile=name).score, 3)}
            for name in scoring.PROFILES
        }
        scored.append({"coa": coa, "review": review, "scores": by_profile})
    prof = cfg.decision_profile if cfg.decision_profile in scoring.PROFILES else scoring.DEFAULT_PROFILE
    best = max(scored, key=lambda s: (s["scores"][prof]["score"], -len(s["coa"]["dispatches"])))
    return best, scored


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
def _audit_path(cfg: AmdpConfig) -> str:
    home = os.environ.get("HERMES_HOME") or os.path.join(os.environ.get("LOCALAPPDATA", ""), "hermes")
    return os.path.join(home, cfg.audit_log)


def _append_audit(cfg: AmdpConfig, record: dict[str, Any]) -> None:
    try:
        path = _audit_path(cfg)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        # Serialize writes across the gateway's concurrent sessions so large
        # multi-KB records can't interleave into a corrupt JSONL line.
        with _AUDIT_LOCK:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line)
    except Exception as exc:  # audit must never break a turn
        logger.debug("AMDP audit write failed: %s", exc)


# --------------------------------------------------------------------------- #
# Plan rendering + entry point
# --------------------------------------------------------------------------- #
def _render_plan(best: dict[str, Any], cfg: AmdpConfig, unvetted: bool = False) -> str:
    coa, review = best["coa"], best["review"]
    lines = [
        "[AMDP plan — private guidance for the agent loop. You may follow it, adapt it, "
        "or finish normally; you own tool calling and turn termination.]",
    ]
    if unvetted:
        lines.append(
            "WARNING: war-gaming unavailable — the reviewer could not be reached, so this "
            "plan was NOT risk-reviewed. Treat it as an unvetted draft and be extra careful."
        )
    lines.append(f"Chosen course of action ({coa['coa_id']}): {coa['summary']}")
    lines.append("Dispatches:")
    for i, d in enumerate(coa["dispatches"], 1):
        flag = " [IRREVERSIBLE — confirm before acting]" if d.get("irreversible") else ""
        crit = f" success: {'; '.join(d['success_criteria'])}" if d.get("success_criteria") else ""
        lines.append(f"  {i}. [{d['kind']}]{flag} {d['task']}.{crit}")
    if coa.get("assumptions"):
        lines.append("Assumptions: " + "; ".join(coa["assumptions"][:5]))
    if review.get("risks"):
        top = sorted(review["risks"], key=lambda r: r.get("severity_1to5", 0), reverse=True)[:3]
        lines.append("Watch for: " + "; ".join(f"{r['desc']} (sev {r['severity_1to5']})" for r in top))
    return "\n".join(lines)


def _get_cached() -> tuple[Any, dict[str, Any]]:
    """Resolve the active AMDP config ONCE (hook path), fail-loud on misconfig,
    and cache so the disabled/absent path is a cheap check thereafter."""
    global _cached_cfg, _cached_raw
    if _cached_cfg is not _UNSET:
        return _cached_cfg, _cached_raw
    raw: dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config

        raw = load_config() or {}
    except Exception as exc:
        logger.warning("AMDP config load failed; planning disabled: %s", exc)
        _cached_cfg, _cached_raw = None, {}
        return _cached_cfg, _cached_raw
    try:
        _cached_cfg = resolve_amdp_config(raw)
    except AmdpConfigError as exc:
        logger.error("AMDP MISCONFIGURED — planning disabled (fix config and restart): %s", exc)
        _cached_cfg = None
    _cached_raw = raw
    if _cached_cfg is not None:
        logger.info("AMDP enabled: planner=%s reviewer=%s profile=%s",
                    _cached_cfg.planner, _cached_cfg.reviewer, _cached_cfg.decision_profile)
    return _cached_cfg, _cached_raw


def reset_cache_for_tests() -> None:
    global _cached_cfg, _cached_raw
    _cached_cfg, _cached_raw = _UNSET, {}
    with _PLAN_CACHE_LOCK:
        _PLAN_CACHE.clear()


def maybe_amdp_context(
    user_prompt: str,
    api_messages: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> str:
    """The single entry point. Returns an injectable plan block, or "" to inject
    nothing. Plans at most ONCE per user turn — subsequent tool-loop iterations
    of the same turn return the cached result (zero model calls). Fail-closed
    against crashes, fail-LOUD against misconfiguration."""
    try:
        # Resolve config. An explicit config (tests / callers) resolves fresh and
        # surfaces AmdpConfigError LOUDLY; None uses the cached startup resolve.
        if config is not None:
            try:
                cfg = resolve_amdp_config(config)
            except AmdpConfigError as exc:
                logger.error("AMDP MISCONFIGURED — planning disabled: %s", exc)
                return ""
            raw_config: dict[str, Any] = config
        else:
            cfg, raw_config = _get_cached()
        if cfg is None:
            return ""

        # Once-per-user-turn: reuse the cached outcome across a turn's tool-loop
        # iterations. A hit returns the same plan (or "") with no model calls.
        sig = _turn_signature(user_prompt, api_messages)
        cached = _cache_get(sig)
        if cached is not None:
            return cached

        result = _plan_turn(cfg, raw_config, user_prompt, api_messages)
        _cache_put(sig, result)
        return result
    except Exception as exc:  # fail-closed: never break the turn
        logger.warning("AMDP planning failed, proceeding without a plan: %s", exc)
        return ""


def _plan_turn(
    cfg: AmdpConfig,
    raw_config: dict[str, Any],
    user_prompt: str,
    api_messages: list[dict[str, Any]],
) -> str:
    """Run one AMDP episode for a fresh user turn. Returns the plan block or "".
    Wrapped by maybe_amdp_context's cache + top-level fail-closed guard."""
    t0 = time.monotonic()
    global _active_call_timeout, _active_state_feed
    _active_call_timeout = cfg.call_timeout_s
    _active_state_feed = cfg.state_feed

    est = _estimate_steps(user_prompt, api_messages)
    if cfg.exclude_background and _is_background_intent(user_prompt):
        return ""
    if est < cfg.min_estimated_steps:
        return ""  # not dispatch-worthy; don't spend model calls

    state = _intake(raw_config or {}, timeout_s=cfg.intake_timeout_s)
    refuse, reason = _should_refuse(state, staleness_max_s=cfg.staleness_max_s)
    if refuse:
        _append_audit(cfg, {
            "ts": time.time(), "intent": str(user_prompt)[:2000], "refused": True,
            "refuse_reason": reason,
            "believed_state": {k: state.get(k) for k in ("verdict", "gateway_state", "sensors_down", "staleness_s", "system_count")},
        })
        logger.info("AMDP refused to plan: %s", reason)
        return ""

    errors: list[str] = []
    coas = _generate_coas(cfg, user_prompt, state["brief"], errors)
    if not coas:
        _append_audit(cfg, {"ts": time.time(), "intent": str(user_prompt)[:2000], "coas": 0, "errors": errors})
        return ""

    if time.monotonic() - t0 > cfg.episode_deadline_s:
        _append_audit(cfg, {"ts": time.time(), "intent": str(user_prompt)[:2000], "coas": len(coas),
                            "aborted": "episode deadline exceeded before war-gaming", "errors": errors})
        logger.info("AMDP aborted: episode deadline exceeded")
        return ""

    review_reason = _review_reason(coas, user_prompt)
    should_review = not cfg.review_only_on_risk_or_disagreement or bool(review_reason)
    reviews = _review_all(cfg, user_prompt, state["brief"], coas) if should_review else []
    staleness_norm = min(1.0, state["staleness_s"] / cfg.staleness_max_s) if cfg.staleness_max_s else 0.0
    best, scored = _decide(cfg, coas, reviews, staleness_norm)
    unvetted = bool(reviews) and all(r.get("_review_failed") for r in reviews)

    _append_audit(cfg, {
        "ts": time.time(), "intent": str(user_prompt)[:2000], "refused": False,
        "decision_profile": cfg.decision_profile, "estimated_steps": est, "unvetted": unvetted,
        "believed_state": {k: state.get(k) for k in ("verdict", "gateway_state", "sensors_down", "staleness_s", "system_count")},
        "coas": coas, "reviews": reviews,
        "scores": [{"coa_id": s["coa"]["coa_id"], "by_profile": s["scores"]} for s in scored],
        "chosen": best["coa"]["coa_id"], "errors": errors,
        "activation_reason": f"estimated_steps={est} >= {cfg.min_estimated_steps}",
        "review_invoked": should_review,
        "review_reason": review_reason or "routine single-plan execution",
        "elapsed_s": round(time.monotonic() - t0, 2),
    })
    return _render_plan(best, cfg, unvetted=unvetted)
