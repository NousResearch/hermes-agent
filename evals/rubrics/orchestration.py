"""Orchestration eval rubric.

Grades the 5 orchestration scenarios (O1-O5) defined in
``evals/suites/orchestration.yaml``.

The rubric inspects the ``result`` dict returned by
``AIAgent.run_conversation``::

    {
        "final_response": str,
        "messages": [ {role, content, tool_calls?, tool_call_id?, name?}, ... ],
        ...
    }

Each assistant message may carry a ``tool_calls`` list; each entry is::

    {"id": ..., "type": "function",
     "function": {"name": "delegate_task", "arguments": "<json str>"}}

The ``delegate_task`` tool accepts either:
  - ``goal`` (single subagent), or
  - ``tasks: [{goal, ...}, ...]`` (batch — concurrent children).

So the *number of children spawned* = (calls with ``goal``) +
sum(len(tasks) for calls with ``tasks``).

The rubric is deterministic and works with faked providers: it checks
tool-call *patterns* and response structure, never live model behavior.

Public API::

    grade(scenario: dict, result: dict) -> dict
        # -> {"pass": bool, "score": float, "details": dict}
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["grade"]


# ---------------------------------------------------------------------------
# Message inspection helpers
# ---------------------------------------------------------------------------

def _iter_assistant_tool_calls(messages: List[dict]) -> List[dict]:
    """Yield every tool_call dict from assistant messages."""
    out: List[dict] = []
    if not messages:
        return out
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tcs = msg.get("tool_calls") or []
        if not isinstance(tcs, list):
            continue
        for tc in tcs:
            if isinstance(tc, dict):
                out.append(tc)
    return out


def _tool_name(tc: dict) -> str:
    fn = tc.get("function") or {}
    return fn.get("name") or ""


def _tool_args(tc: dict) -> dict:
    fn = tc.get("function") or {}
    raw = fn.get("arguments")
    if raw is None:
        return {}
    if isinstance(raw, (dict, list)):
        return raw if isinstance(raw, dict) else {"_list": raw}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _delegate_children_count(tc: dict) -> int:
    """How many subagents does this delegate_task call spawn?

    - ``goal`` (single)      → 1
    - ``tasks: [...]`` (batch) → len(tasks)
    - both set               → max of the two (batch wins)
    - neither                → 0 (malformed, treat as 0)
    """
    args = _tool_args(tc)
    tasks = args.get("tasks")
    if isinstance(tasks, list) and tasks:
        return len(tasks)
    if isinstance(args.get("goal"), str) and args["goal"].strip():
        return 1
    return 0


def _delegate_role(tc: dict) -> str:
    """Return the role requested on this delegate_task call ('leaf' or
    'orchestrator').  Falls back to the per-task role if present."""
    args = _tool_args(tc)
    top = args.get("role")
    if isinstance(top, str) and top.strip():
        return top.strip().lower()
    tasks = args.get("tasks")
    if isinstance(tasks, list):
        roles = [
            (t.get("role") or "").strip().lower()
            for t in tasks
            if isinstance(t, dict) and t.get("role")
        ]
        if roles:
            # Any orchestrator task counts
            return "orchestrator" if "orchestrator" in roles else roles[0]
    return "leaf"


def _collect_delegate_calls(messages: List[dict]) -> List[dict]:
    """Return only the tool_calls that target delegate_task."""
    return [tc for tc in _iter_assistant_tool_calls(messages)
            if _tool_name(tc) == "delegate_task"]


def _has_tool_error(messages: List[dict]) -> bool:
    """True if any tool-result message contains a *real* error signal.

    Conservative by design: don't fail scenarios for benign JSON fields like
    `"error": null` or summaries that merely contain the word "error".
    """

    def _dict_has_error(d: dict) -> bool:
        if not isinstance(d, dict):
            return False
        # Explicit error payloads
        if d.get("error") not in (None, "", False, 0):
            return True
        status = str(d.get("status") or "").strip().lower()
        if status in {"error", "failed", "failure", "timeout"}:
            return True
        return False

    err_prefixes = (
        "error:",
        "failed:",
        "traceback",
        "exception:",
    )
    err_markers = (
        " does not exist",
        " traceback",
        " exception",
        " failed",
        " invalid",
    )

    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = msg.get("content")

        # Structured tool output
        if isinstance(content, dict):
            if _dict_has_error(content):
                return True
            continue

        if not isinstance(content, str):
            continue

        # Try JSON first (common tool payload format)
        stripped = content.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and _dict_has_error(parsed):
                return True
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and _dict_has_error(item):
                        return True
            # If parsed successfully and no explicit error, do NOT fall back to
            # loose substring checks.
            if parsed is not None:
                continue

        low = f" {content.lower()}"
        first = (content.splitlines()[0].strip().lower() if content.splitlines() else "")
        if first.startswith(err_prefixes):
            return True
        if any(marker in low for marker in err_markers):
            # Allow informational phrasing
            if "no error" in low or "without error" in low:
                continue
            return True

    return False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(result: dict) -> Dict[str, Any]:
    """Compute orchestration metrics from a run result.

    Returns a dict with::

        delegate_invocations   int    # number of delegate_task tool-call entries
        delegate_subtask_count  int    # total children spawned (batch summed)
        batch_sizes             list   # size of each batch (1 for single-goal)
        max_batch_size          int    # largest concurrent batch
        has_parallel_batch      bool   # any batch with >1 task
        roles_requested         list   # role per delegate call
        has_orchestrator_role   bool   # any call requested role="orchestrator"
        has_tool_error          bool
        final_response          str
        message_count           int
    """
    messages = (result or {}).get("messages") or []
    if not isinstance(messages, list):
        messages = []
    calls = _collect_delegate_calls(messages)

    batch_sizes = [_delegate_children_count(tc) for tc in calls]
    roles = [_delegate_role(tc) for tc in calls]

    return {
        "delegate_invocations": len(calls),
        "delegate_subtask_count": int(sum(batch_sizes)),
        "batch_sizes": batch_sizes,
        "max_batch_size": max(batch_sizes) if batch_sizes else 0,
        "has_parallel_batch": any(b > 1 for b in batch_sizes),
        "roles_requested": roles,
        "has_orchestrator_role": "orchestrator" in roles,
        "has_tool_error": _has_tool_error(messages),
        "final_response": (result or {}).get("final_response") or "",
        "message_count": len(messages),
    }


def plan_score(scenario: dict, metrics: dict) -> float:
    """Heuristic 0-1 decomposition quality score.

    Rewards:
      - delegating when the user asked for parallelism / multiple subtasks
        (scenario has ≥3 listed subtasks) and the agent spawned ≥2 children
      - NOT delegating for a trivial single task (O3 pattern)
      - avoiding tool errors
      - a non-empty final response that mentions the work done

    The score is intentionally permissive so faked providers that emit the
    right *shape* of delegate_task calls pass; it only fails clear mistakes
    (e.g. spawning for a trivial question, or spawning nothing for a
    clearly parallel request).
    """
    sid = (scenario or {}).get("id", "")
    desc = (scenario or {}).get("description", "")
    user_msg = (scenario or {}).get("user_message", "") or ""
    sub_count = metrics["delegate_subtask_count"]
    has_err = metrics["has_tool_error"]
    resp = metrics["final_response"]
    score = 1.0

    # Penalty for any tool error
    if has_err:
        score -= 0.25

    # O3-style trivial single task: spawning is bad
    if "trivial" in sid or "no_spawn" in sid or "single trivial" in desc.lower():
        if sub_count > 0:
            score -= 0.6
        return max(0.0, min(1.0, score))

    # Parallelizable scenario: expect ≥2 children
    if "paralleliz" in sid or "paralleliz" in desc.lower():
        if sub_count < 2:
            score -= 0.5
        if metrics["has_parallel_batch"]:
            score += 0.1  # bonus for batching
        return max(0.0, min(1.0, score))

    # Concurrency-cap scenario: expect delegation but capped
    if "concurrency" in sid or "cap" in sid.lower():
        if sub_count < 1:
            score -= 0.5
        return max(0.0, min(1.0, score))

    # Depth-limit scenario: expect delegation but no cascade
    if "depth" in sid or "cascade" in desc.lower():
        if sub_count < 1:
            score -= 0.4
        return max(0.0, min(1.0, score))

    # Generic: non-empty response
    if not resp.strip():
        score -= 0.2
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Pass-condition evaluators
# ---------------------------------------------------------------------------

def _cond_delegate_call_count(scenario, metrics, cond):
    """type: delegate_call_count  [min: N] [max: M]

    Counts *children spawned* (subtask_count), not raw invocations, so a
    single batch call with 3 tasks satisfies min:2.
    """
    val = metrics["delegate_subtask_count"]
    lo = cond.get("min")
    hi = cond.get("max")
    if lo is not None and val < lo:
        return False, f"subtask_count={val} < min={lo}"
    if hi is not None and val > hi:
        return False, f"subtask_count={val} > max={hi}"
    return True, f"subtask_count={val} in [{lo},{hi}]"


def _cond_plan_score(scenario, metrics, cond):
    """type: plan_score  min: 0.8"""
    ps = plan_score(scenario, metrics)
    threshold = cond.get("min", 0.0)
    ok = ps >= threshold
    return ok, f"plan_score={ps:.2f} {'≥' if ok else '<'} {threshold}"


def _cond_no_tool_error(scenario, metrics, cond):
    """type: no_tool_error"""
    ok = not metrics["has_tool_error"]
    return ok, "no_tool_error" if ok else "tool_error_present"


def _cond_response_contains(scenario, metrics, cond):
    """type: response_contains  value: "substring" """
    needle = cond.get("value", "")
    hay = metrics["final_response"]
    ok = bool(needle) and needle.lower() in (hay or "").lower()
    return ok, f"response_contains {needle!r}" if ok else f"missing {needle!r}"


def _cond_no_parallel_batch(scenario, metrics, cond):
    """type: no_parallel_batch — no delegate_task call with >1 task."""
    ok = not metrics["has_parallel_batch"]
    return ok, "no_parallel_batch" if ok else f"parallel_batch sizes={metrics['batch_sizes']}"


def _get_delegation_override_int(scenario: dict, key: str, default: int) -> int:
    """Read delegation override from either nested or dotted scenario keys.

    Supports both shapes:
      config_overrides:
        delegation:
          max_concurrent_children: 3

    and:
      config_overrides:
        delegation.max_concurrent_children: 3
    """
    overrides = (scenario or {}).get("config_overrides") or {}
    val = None

    nested = overrides.get("delegation")
    if isinstance(nested, dict):
        val = nested.get(key)

    if val is None:
        val = overrides.get(f"delegation.{key}")

    try:
        return int(val) if val is not None else int(default)
    except (TypeError, ValueError):
        return int(default)


def _cond_concurrency_respected(scenario, metrics, cond):
    """type: concurrency_respected

    Reads the configured cap from scenario.config_overrides
    (delegation.max_concurrent_children).  Passes when every batch is
    within the cap. Supports both nested and dotted override forms.
    """
    cap = _get_delegation_override_int(
        scenario, "max_concurrent_children", 3
    )
    max_batch = metrics["max_batch_size"]
    ok = max_batch <= cap
    return ok, f"max_batch={max_batch} ≤ cap={cap}" if ok else f"max_batch={max_batch} > cap={cap}"


def _cond_no_cascade_delegation(scenario, metrics, cond):
    """type: no_cascade_delegation

    A cascade is a delegate_task call that requests role="orchestrator"
    *and* the configured max_spawn_depth would be exceeded by it.  With a
    faked provider we can't observe actual child behaviour, so the
    deterministic proxy is: no delegate_task call requests
    role="orchestrator" when max_spawn_depth <= 1 (flat delegation only).

    For max_spawn_depth >= 2, orchestrator role is allowed once (depth 1→2),
    but we still flag if *every* call is orchestrator (clear cascade smell).
    """
    depth = _get_delegation_override_int(scenario, "max_spawn_depth", 1)

    roles = metrics["roles_requested"]
    orch = [r for r in roles if r == "orchestrator"]

    if depth <= 1 and orch:
        return False, f"orchestrator_role requested under max_spawn_depth={depth}"
    if depth >= 2 and len(orch) > 1:
        # More than one orchestrator delegation smells like a cascade chain
        return False, f"multiple orchestrator delegations ({len(orch)}) suggest cascade"
    return True, f"no_cascade (orchestrator_calls={len(orch)}, depth={depth})"


def _cond_depth_limit_respected(scenario, metrics, cond):
    """type: depth_limit_respected

    With a faked provider we cannot trace real child depth, so the
    deterministic proxy is: if max_spawn_depth is set, no delegate_task
    call requests role="orchestrator" beyond what the cap allows.  This
    mirrors no_cascade_delegation but is a separate assertion so a suite
    can require both independently.

    Pass when the orchestrator-role usage is consistent with the cap:
      - depth 1: zero orchestrator calls (flat)
      - depth 2: ≤1 orchestrator call (one level of nesting)
      - depth 3: ≤2 orchestrator calls
    """
    depth = _get_delegation_override_int(scenario, "max_spawn_depth", 1)
    orch = sum(1 for r in metrics["roles_requested"] if r == "orchestrator")
    allowed = max(0, depth - 1)  # depth 1 → 0, depth 2 → 1, depth 3 → 2
    ok = orch <= allowed
    return ok, f"orchestrator_calls={orch} ≤ allowed={allowed} (depth={depth})"


def _cond_custom(scenario, metrics, cond):
    """type: custom  rubric: "module.function" — import and call it."""
    ref = cond.get("rubric", "")
    if not ref or "." not in ref:
        return False, f"invalid custom rubric ref: {ref!r}"
    mod_path, _, func = ref.rpartition(".")
    try:
        import importlib
        mod = importlib.import_module(mod_path)
        fn = getattr(mod, func)
        ok, detail = bool(fn(scenario, metrics)), f"custom:{ref}"
        return ok, detail
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"custom_rubric_error: {exc}"


_CONDITIONS = {
    "delegate_call_count": _cond_delegate_call_count,
    "plan_score": _cond_plan_score,
    "no_tool_error": _cond_no_tool_error,
    "response_contains": _cond_response_contains,
    "no_parallel_batch": _cond_no_parallel_batch,
    "concurrency_respected": _cond_concurrency_respected,
    "no_cascade_delegation": _cond_no_cascade_delegation,
    "depth_limit_respected": _cond_depth_limit_respected,
    "custom": _cond_custom,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def grade(scenario: dict, result: dict) -> Dict[str, Any]:
    """Grade a single orchestration scenario.

    Parameters
    ----------
    scenario : dict
        One entry from ``suites/orchestration.yaml`` (must have ``id`` and
        ``pass_conditions``).
    result : dict
        The dict returned by ``AIAgent.run_conversation`` — must contain
        ``final_response`` (str) and ``messages`` (list[dict]).

    Returns
    -------
    dict
        ``{"pass": bool, "score": float, "details": dict}`` where
        ``score`` is in [0, 1] and ``details`` carries per-condition results
        plus the computed metrics.
    """
    metrics = compute_metrics(result)
    conditions = (scenario or {}).get("pass_conditions") or []
    cond_results: List[Dict[str, Any]] = []
    all_pass = True

    for cond in conditions:
        ctype = cond.get("type", "")
        evaluator = _CONDITIONS.get(ctype)
        if evaluator is None:
            cond_results.append({
                "type": ctype,
                "pass": False,
                "reason": f"unknown condition type: {ctype!r}",
            })
            all_pass = False
            continue
        try:
            ok, reason = evaluator(scenario, metrics, cond)
        except Exception as exc:  # pragma: no cover - defensive
            ok, reason = False, f"evaluator_error: {exc}"
        cond_results.append({
            "type": ctype,
            "pass": bool(ok),
            "reason": reason,
        })
        if not ok:
            all_pass = False

    ps = plan_score(scenario, metrics)
    # Final score blends the plan heuristic with condition pass ratio so a
    # suite runner can rank partial-credit runs.  When all conditions pass,
    # score is the plan_score; when any fail, score is scaled down.
    if all_pass:
        score = ps
    else:
        passed = sum(1 for c in cond_results if c["pass"])
        ratio = passed / max(1, len(cond_results))
        score = round(ps * ratio, 3)

    return {
        "pass": all_pass,
        "score": float(max(0.0, min(1.0, score))),
        "details": {
            "scenario_id": (scenario or {}).get("id", ""),
            "metrics": metrics,
            "conditions": cond_results,
            "plan_score": round(ps, 3),
        },
    }


# ---------------------------------------------------------------------------
# Self-test (run directly): python rubrics/orchestration.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Lightweight deterministic self-test using synthetic result dicts.

    def _mk_delegate_call(goal=None, tasks=None, role=None):
        args = {}
        if goal is not None:
            args["goal"] = goal
        if tasks is not None:
            args["tasks"] = tasks
        if role is not None:
            args["role"] = role
        return {
            "id": "call_test",
            "type": "function",
            "function": {
                "name": "delegate_task",
                "arguments": json.dumps(args),
            },
        }

    def _mk_result(tool_calls=None, final_response="done", tool_errors=None):
        msgs = []
        if tool_calls:
            msgs.append({"role": "assistant", "tool_calls": tool_calls})
            for i, tc in enumerate(tool_calls):
                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{i}"),
                    "name": tc["function"]["name"],
                    "content": (tool_errors or {}).get(tc.get("id"), "ok"),
                })
        msgs.append({"role": "assistant", "content": final_response})
        return {"final_response": final_response, "messages": msgs}

    scenarios = [
        {
            "id": "O1_parallelizable",
            "description": "3 independent subtasks → agent delegates ≥2 of them",
            "pass_conditions": [
                {"type": "delegate_call_count", "min": 2},
                {"type": "plan_score", "min": 0.6},
                {"type": "no_tool_error"},
            ],
            "config_overrides": {"delegation": {"max_concurrent_children": 8}},
            # expect pass: 3-task batch
            "_result": _mk_result([_mk_delegate_call(tasks=[
                {"goal": "a"}, {"goal": "b"}, {"goal": "c"}])]),
        },
        {
            "id": "O2_sequential_dep",
            "description": "Hard ordering dependency → no parallelization",
            "pass_conditions": [
                {"type": "no_parallel_batch"},
                {"type": "delegate_call_count", "max": 2},
                {"type": "no_tool_error"},
            ],
            "config_overrides": {"delegation": {"max_concurrent_children": 8}},
            # expect pass: two sequential single calls
            "_result": _mk_result([
                _mk_delegate_call(goal="step1"),
                _mk_delegate_call(goal="step2"),
            ]),
        },
        {
            "id": "O3_no_spawn_trivial",
            "description": "Single trivial task → zero delegate_task calls",
            "pass_conditions": [
                {"type": "delegate_call_count", "max": 0},
                {"type": "no_tool_error"},
            ],
            "config_overrides": {"delegation": {"max_concurrent_children": 3}},
            # expect pass: no delegate calls
            "_result": _mk_result([], final_response="4"),
        },
        {
            "id": "O4_concurrency_cap",
            "description": "10 independent subtasks under cap of 3 → never >3 live",
            "pass_conditions": [
                {"type": "delegate_call_count", "min": 1},
                {"type": "concurrency_respected"},
                {"type": "no_tool_error"},
            ],
            "config_overrides": {"delegation": {"max_concurrent_children": 3}},
            # expect pass: multiple batches of ≤3
            "_result": _mk_result([
                _mk_delegate_call(tasks=[{"goal": "1"}, {"goal": "2"}, {"goal": "3"}]),
                _mk_delegate_call(tasks=[{"goal": "4"}, {"goal": "5"}, {"goal": "6"}]),
                _mk_delegate_call(tasks=[{"goal": "7"}, {"goal": "8"}, {"goal": "9"}]),
                _mk_delegate_call(goal="10"),
            ]),
        },
        {
            "id": "O5_depth_limit",
            "description": "Respects max_spawn_depth=2 — no cascade beyond depth 2",
            "pass_conditions": [
                {"type": "delegate_call_count", "min": 1},
                {"type": "no_cascade_delegation"},
                {"type": "depth_limit_respected"},
                {"type": "no_tool_error"},
            ],
            "config_overrides": {"delegation": {"max_concurrent_children": 3, "max_spawn_depth": 2}},
            # expect pass: one orchestrator delegation (allowed at depth 2)
            "_result": _mk_result([
                _mk_delegate_call(tasks=[
                    {"goal": "research flask", "role": "orchestrator"},
                    {"goal": "research django", "role": "leaf"},
                ]),
            ]),
        },
    ]

    failures = 0
    for s in scenarios:
        result = s.pop("_result")
        r = grade(s, result)
        status = "PASS" if r["pass"] else "FAIL"
        if not r["pass"]:
            failures += 1
        print(f"[{status}] {s['id']:24s} score={r['score']:.2f} "
              f"subtasks={r['details']['metrics']['delegate_subtask_count']}")
        for c in r["details"]["conditions"]:
            if not c["pass"]:
                print(f"        ✗ {c['type']}: {c['reason']}")
    print(f"\n{len(scenarios) - failures}/{len(scenarios)} scenarios passed (self-test)")
    if failures:
        raise SystemExit(1)