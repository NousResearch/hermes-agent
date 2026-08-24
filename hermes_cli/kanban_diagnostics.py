"""Kanban diagnostics — structured, actionable distress signals for tasks.

A ``Diagnostic`` carries a **kind** (canonical code the UI/tests match on), a
**severity**, title/detail text, and **actions** the dashboard renders as
buttons and the CLI as hints. Rules are stateless and read-only over
(task, events, runs, optional graph); callers compute on demand. Only
operator-fixable signals (not a one-off provider 502); every diagnostic has a
recovery action and auto-clears when the failure mode resolves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Optional
import json
import time


# Least → most urgent; sorted outputs put critical first.
SEVERITY_ORDER = ("warning", "error", "critical")


def severity_at_or_above(severity: Optional[str], threshold: Optional[str]) -> bool:
    """Return True when ``severity`` meets or exceeds ``threshold``."""
    if threshold is None:
        return True
    if severity not in SEVERITY_ORDER or threshold not in SEVERITY_ORDER:
        return False
    return SEVERITY_ORDER.index(severity) >= SEVERITY_ORDER.index(threshold)


@dataclass
class DiagnosticAction:
    """A recovery action. ``kind`` drives rendering: ``reclaim``/``reassign``
    POST to /tasks/:id/*; ``unblock`` PATCHes status to ready; ``cli_hint``
    shows ``payload.command``; ``open_docs`` links ``payload.url``; ``comment``
    nudges the operator. ``suggested=True`` = recommended first step."""

    kind: str
    label: str
    payload: dict = field(default_factory=dict)
    suggested: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Diagnostic:
    """One active distress signal on a task."""

    kind: str
    severity: str  # "warning" | "error" | "critical"
    title: str
    detail: str
    actions: list[DiagnosticAction] = field(default_factory=list)
    first_seen_at: int = 0
    last_seen_at: int = 0
    count: int = 1
    run_id: Optional[int] = None  # None = task-wide
    data: dict = field(default_factory=dict)  # structured payload for the UI

    def to_dict(self) -> dict:
        return asdict(self)


# --- Rule helpers ---

def _task_field(task, name, default=None):
    """Read a field from a sqlite3.Row, a kanban_db.Task dataclass, or a dict."""
    if task is None:
        return default
    try:
        if hasattr(task, "keys") and name in task.keys():
            return task[name]
    except Exception:
        pass
    if isinstance(task, dict):
        return task.get(name, default)
    return getattr(task, name, default)


def _parse_payload(ev) -> dict:
    """Tolerate event.payload being either a dict or a JSON string."""
    p = _task_field(ev, "payload", None)
    if isinstance(p, dict):
        return p
    if isinstance(p, str):
        try:
            return json.loads(p) or {}
        except Exception:
            return {}
    return {}


def _event_kind(ev) -> str:
    return _task_field(ev, "kind", "") or ""


def _event_ts(ev) -> int:
    return int(_task_field(ev, "created_at", 0) or 0)


def _first_field(task, primary: str, legacy: str, default=None):
    """``task[primary]`` unless it is None, else ``task[legacy]`` (old DB rows)."""
    v = _task_field(task, primary, None)
    return v if v is not None else _task_field(task, legacy, default)


def _latest_event_ts(events: Iterable[Any], kinds: set[str]) -> int:
    """Max ``created_at`` over events whose kind is in ``kinds`` (0 if none)."""
    return max([0, *(_event_ts(ev) for ev in events if _event_kind(ev) in kinds)])


def _cli_hint(label: str, command: str, *, suggested: bool = False) -> DiagnosticAction:
    return DiagnosticAction(kind="cli_hint", label=label, payload={"command": command},
                            suggested=suggested)


def _log_hint_action(task_id: str) -> DiagnosticAction:
    cmd = f"hermes kanban log {task_id}"
    return _cli_hint(f"Check logs: {cmd}", cmd, suggested=True)


def _error_snippet(last_err) -> str:
    """First 500 chars of the error (with ellipsis), or "" when absent."""
    err_text = (last_err or "").strip() if last_err else ""
    return err_text[:500] + ("…" if len(err_text) > 500 else "") if err_text else ""


def _active_hallucination_events(events: Iterable[Any], kind: str) -> list[Any]:
    """Events of ``kind`` with no ``completed``/``edited`` event strictly after
    them. Requires id-sorted (arrival-order) input, which the DB provides."""
    active: list[Any] = []
    for ev in events:
        k = _event_kind(ev)
        if k in {"completed", "edited"}:
            active.clear()
        elif k == kind:
            active.append(ev)
    return active


def _unique_payload_ids(hits: list[Any], key: str) -> list[str]:
    """Ordered, de-duplicated ``payload[key]`` entries across ``hits``."""
    out: list[str] = []
    for ev in hits:
        for pid in _parse_payload(ev).get(key, []) or []:
            if pid not in out:
                out.append(pid)
    return out


def _generic_recovery_actions(task: Any, *, running: bool) -> list[DiagnosticAction]:
    """Baseline recovery primitives every diagnostic can fall back on."""
    out: list[DiagnosticAction] = []
    if running:
        out.append(DiagnosticAction(kind="reclaim", label="Reclaim task", payload={}))
    out.append(DiagnosticAction(
        kind="reassign", label="Reassign to different profile", payload={"reclaim_first": running},
    ))
    return out


def _is_running(task) -> bool:
    return _task_field(task, "status") == "running"


def _runs_newest_first(runs) -> list[Any]:
    # reversed(sorted()) not sorted(reverse=True): equal ids must keep the
    # last-listed run first.
    return list(reversed(sorted(runs, key=lambda r: _task_field(r, "id", 0))))


# --- Rule implementations ---

# Each rule: (task, events, runs, now_ts, config) -> list[Diagnostic].
# ``events``/``runs`` are kanban_db rows/dataclasses or same-shaped dicts.

RuleFn = Callable[[Any, list[Any], list[Any], int, dict], list[Diagnostic]]


def _aux_slot_explicit(slot: Any) -> bool:
    """True if the aux slot was user-configured: provider other than "auto",
    or any of model/base_url/api_key set (the default falls through to the
    main model)."""
    if not isinstance(slot, dict):
        return False
    provider = str(slot.get("provider") or "").strip().lower()
    if provider and provider != "auto":
        return True
    return any(str(slot.get(key) or "").strip() for key in ("model", "base_url", "api_key"))


def _main_model_visible(raw_config: Any) -> bool:
    """Best-effort "a main model is configured" from the raw config dict (the
    dashboard process may not share CLI runtime state). Unprovable => False,
    which errs toward NOT firing the diagnostic."""
    if not isinstance(raw_config, dict):
        return False
    model_cfg = raw_config.get("model")
    if isinstance(model_cfg, dict):
        provider = str(model_cfg.get("provider") or "").strip()
        model = str(
            model_cfg.get("default") or model_cfg.get("model") or model_cfg.get("name") or ""
        ).strip()
        return bool(provider and model)
    return bool(str(model_cfg or "").strip())


def triage_aux_status(config: Optional[dict]) -> Optional[dict]:
    """Report whether the triage aux paths look configured: ``{auto_decompose,
    decomposer_explicit, specifier_explicit, main_model_visible}``. ``None``
    when no config context is present (keeps low-level callers/tests silent)."""
    if not isinstance(config, dict):
        return None
    explicit = config.get("triage_aux_status")
    if isinstance(explicit, dict):
        return explicit

    aux = config.get("auxiliary")
    kanban_cfg = config.get("kanban") if isinstance(config.get("kanban"), dict) else {}
    # No auxiliary/kanban/model keys at all => a low-level caller passing {}.
    if not isinstance(aux, dict) and not kanban_cfg and "model" not in config:
        return None
    aux = aux if isinstance(aux, dict) else {}
    return {
        # ``auto_decompose`` defaults to True per kanban DEFAULT_CONFIG.
        "auto_decompose": bool(kanban_cfg["auto_decompose"]) if "auto_decompose" in kanban_cfg else True,
        "decomposer_explicit": _aux_slot_explicit(aux.get("kanban_decomposer")),
        "specifier_explicit": _aux_slot_explicit(aux.get("triage_specifier")),
        "main_model_visible": _main_model_visible(config),
    }


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


def _rule_hallucinated_cards(task, events, runs, now, cfg) -> list[Diagnostic]:
    """A worker's kanban_complete named created_cards that don't exist / weren't
    its own; the completion was blocked. Clears on a later completion/edit."""
    hits = _active_hallucination_events(events, "completion_blocked_hallucination")
    if not hits:
        return []
    actions = [DiagnosticAction(kind="comment", label="Add a comment explaining what to do",
                                suggested=False)]
    actions += _generic_recovery_actions(task, running=_is_running(task))
    return [Diagnostic(
        kind="hallucinated_cards", severity="error",
        title="Worker claimed cards that don't exist",
        detail="The completing worker declared created_cards that either didn't exist or weren't "
               "created by its profile. The completion was blocked and the task stayed in its prior "
               "state. Usually means the worker hallucinated ids instead of capturing return values "
               "from kanban_create.",
        actions=actions,
        first_seen_at=_event_ts(hits[0]), last_seen_at=_event_ts(hits[-1]), count=len(hits),
        data={"phantom_ids": _unique_payload_ids(hits, "phantom_cards")},
    )]


# (primary_slot, fallback_slot, primary_desc, detail_path) keyed by auto_decompose.
_TRIAGE_SLOTS = {
    True: (
        "auxiliary.kanban_decomposer", "auxiliary.triage_specifier", "decomposer",
        "Auto-decompose is on, so the dispatcher needs auxiliary.kanban_decomposer (with "
        "auxiliary.triage_specifier as a fallback for non-fan-out tasks).",
    ),
    False: (
        "auxiliary.triage_specifier", "auxiliary.kanban_decomposer", "specifier",
        "Auto-decompose is off, so triage tasks need "
        "`hermes kanban specify`, which uses auxiliary.triage_specifier.",
    ),
}


def _rule_triage_aux_unavailable(task, events, runs, now, cfg) -> list[Diagnostic]:
    """A triage task can't leave triage without a usable aux model. With
    auto-decompose on the primary slot is ``auxiliary.kanban_decomposer``
    (specifier as fallback); off, it is ``auxiliary.triage_specifier``. The
    default ``provider: auto`` falls back to the main model, so this fires only
    when the slot isn't explicit AND no main model is visible. Requires config
    context ({} keeps it silent)."""
    if _task_field(task, "status") != "triage":
        return []
    status = triage_aux_status(cfg)
    if status is None:
        return []

    auto_decompose = bool(status.get("auto_decompose"))
    main_visible = bool(status.get("main_model_visible"))
    decomposer_explicit = bool(status.get("decomposer_explicit"))
    specifier_explicit = bool(status.get("specifier_explicit"))
    primary_slot, fallback_slot, primary_desc, detail_path = _TRIAGE_SLOTS[auto_decompose]
    primary_explicit, fallback_explicit = (
        (decomposer_explicit, specifier_explicit) if auto_decompose
        else (specifier_explicit, decomposer_explicit)
    )
    if primary_explicit or main_visible:
        return []

    task_id = _task_field(task, "id") or "<task_id>"
    actions = [_cli_hint(
        f"Configure {primary_slot}", f"hermes config set {primary_slot}.provider auto", suggested=True,
    )]
    if not fallback_explicit and not main_visible:
        actions.append(_cli_hint(
            f"Or configure fallback {fallback_slot}", f"hermes config set {fallback_slot}.provider auto",
        ))
    if not auto_decompose:
        cmd = f"hermes kanban specify {task_id}"
        actions.append(_cli_hint(f"Specify manually: {cmd}", cmd))

    return [Diagnostic(
        kind="triage_aux_unavailable", severity="warning",
        title=f"Triage {primary_desc} has no usable model",
        detail=f"This task is still in triage and no working auxiliary model is visible to the "
               f"dispatcher. {detail_path} The default slot uses `provider: auto` which falls back to "
               f"the main model, but no main model is configured either. Configure the slot directly "
               f"or set a main model so the auto fallback can take over.",
        actions=actions,
        first_seen_at=now, last_seen_at=now, count=1,
        data={"task_id": task_id, "auto_decompose": auto_decompose,
              "primary_slot": primary_slot, "main_model_visible": main_visible},
    )]


def _rule_prose_phantom_refs(task, events, runs, now, cfg) -> list[Diagnostic]:
    """Advisory: the completion summary mentions ``t_<hex>`` ids that don't
    resolve. Warning only; clears on a later clean completion."""
    hits = _active_hallucination_events(events, "suspected_hallucinated_references")
    if not hits:
        return []
    return [Diagnostic(
        kind="prose_phantom_refs", severity="warning",
        title="Completion summary references unknown task ids",
        detail="The completion summary mentions task ids that don't resolve in this board's database. "
               "The completion itself succeeded, but downstream consumers parsing the summary may be "
               "pointed at cards that never existed.",
        actions=_generic_recovery_actions(task, running=_is_running(task)),
        first_seen_at=_event_ts(hits[0]), last_seen_at=_event_ts(hits[-1]), count=len(hits),
        data={"phantom_refs": _unique_payload_ids(hits, "phantom_refs")},
    )]


def _failure_threshold(cfg: dict) -> Any:
    """``failure_threshold`` with the legacy ``spawn_failure_threshold`` alias."""
    return cfg.get("failure_threshold", cfg.get("spawn_failure_threshold", 3))


_OUTCOME_LABELS = {"spawn_failed": "spawn", "timed_out": "timeout", "crashed": "crash"}


def _rule_repeated_failures(task, events, runs, now, cfg) -> list[Diagnostic]:
    """``consecutive_failures`` >= cfg["failure_threshold"] (legacy key
    ``spawn_failure_threshold``), regardless of failure mode — the kernel keeps
    retrying and the operator must intervene. Runtime callers derive the
    threshold from ``kanban.failure_limit`` so it doesn't lag the breaker.

    Exempt: done/archived (a manual done ends no run, so the streak is history)
    and running (a retry in flight must not read as a current failure; re-fires
    if it fails too)."""
    if _task_field(task, "status") in ("done", "archived", "running"):
        return []
    threshold = _positive_int(_failure_threshold(cfg), 3)
    failure_limit = _positive_int(cfg.get("failure_limit"), threshold)
    failures = _first_field(task, "consecutive_failures", "spawn_failures", 0)
    if failures is None or failures < threshold:
        return []
    last_err = _first_field(task, "last_failure_error", "last_spawn_error")
    assignee = _task_field(task, "assignee")

    # Most recent failure outcome makes the title/action specific.
    most_recent_outcome = next(
        (oc for oc in (_task_field(r, "outcome") for r in _runs_newest_first(runs))
         if oc in {"spawn_failed", "timed_out", "crashed"}),
        None,
    )

    actions: list[DiagnosticAction] = []
    if most_recent_outcome == "spawn_failed" and assignee and assignee != "default":
        # Spawn is failing specifically — profile setup issue.
        doctor, auth = f"hermes -p {assignee} doctor", f"hermes -p {assignee} auth"
        actions.append(_cli_hint(f"Verify profile: {doctor}", doctor, suggested=True))
        actions.append(_cli_hint(f"Fix profile auth: {auth}", auth))
    elif most_recent_outcome in {"timed_out", "crashed"}:
        # Worker got off the ground but died: logs diagnose, reclaim/reassign recover.
        task_id = _task_field(task, "id")
        if task_id:
            actions.append(_log_hint_action(task_id))
    actions.extend(_generic_recovery_actions(task, running=_is_running(task)))

    severity = "critical" if failures >= threshold * 2 else "error"
    err_snippet = _error_snippet(last_err)
    outcome_label = _OUTCOME_LABELS.get(most_recent_outcome or "", "failure")
    if err_snippet:
        title = f"Agent {outcome_label} x{failures}: {err_snippet.splitlines()[0][:160]}"
        detail = (
            f"This task has failed {failures} times in a row (most recent: {outcome_label}). Full "
            f"last error:\n\n{err_snippet}\n\nThe dispatcher circuit breaker is configured for "
            f"{failure_limit} consecutive non-success attempts. Fix the root cause and reclaim or "
            f"unblock the task to retry."
        )
    else:
        title = f"Agent {outcome_label} x{failures} (no error recorded)"
        detail = (
            f"This task has failed {failures} times in a row (most recent: {outcome_label}) but no "
            f"error text was captured. Check the suggested command or the worker log."
        )
    return [Diagnostic(
        kind="repeated_failures", severity=severity,
        title=title, detail=detail, actions=actions,
        first_seen_at=now, last_seen_at=now, count=failures,
        data={
            "consecutive_failures": failures,
            "most_recent_outcome": most_recent_outcome,
            "last_error": last_err,
            "failure_threshold": threshold,
            "failure_limit": failure_limit,
        },
    )]


def _rule_repeated_crashes(task, events, runs, now, cfg) -> list[Diagnostic]:
    """Trailing run outcomes show >= cfg["crash_threshold"] (default 2)
    consecutive ``crashed`` with no ``completed``/``reclaimed`` between. Fires
    earlier than ``repeated_failures`` for a crash-specific heads-up and
    suppresses itself when the unified rule is about to fire.

    Exempt: done/archived (a manual done appends no completed run, so the
    streak would be permanent) and running (an in-flight run has no outcome
    and wouldn't break the scan)."""
    if _task_field(task, "status") in ("done", "archived", "running"):
        return []
    # Unified rule will catch this — let it handle to avoid double fire.
    if (_task_field(task, "consecutive_failures", 0) or 0) >= int(_failure_threshold(cfg)):
        return []

    threshold = int(cfg.get("crash_threshold", 2))
    # Count trailing consecutive 'crashed' outcomes; a success (or manual
    # reclaim) breaks the streak, other outcomes neither count nor break it.
    consecutive = 0
    last_err = None
    for r in _runs_newest_first(runs):
        outcome = _task_field(r, "outcome")
        if outcome == "crashed":
            consecutive += 1
            if last_err is None:
                last_err = _task_field(r, "error")
        elif outcome in {"completed", "reclaimed"}:
            break
    if consecutive < threshold:
        return []
    task_id = _task_field(task, "id")
    actions: list[DiagnosticAction] = []
    if task_id:
        actions.append(_log_hint_action(task_id))
    actions.extend(_generic_recovery_actions(task, running=_is_running(task)))
    severity = "critical" if consecutive >= threshold * 2 else "error"
    # Error up-front so operators see WHAT broke without opening the logs.
    err_snippet = _error_snippet(last_err)
    if err_snippet:
        title = f"Agent crashed {consecutive}x: {err_snippet.splitlines()[0][:160]}"
        detail = (
            f"The last {consecutive} runs ended with outcome=crashed. "
            f"Full last error:\n\n{err_snippet}"
        )
    else:
        title = f"Agent crashed {consecutive}x (no error recorded)"
        detail = (
            f"The last {consecutive} runs ended with outcome=crashed but "
            f"no error text was captured. Check the worker log for more."
        )
    return [Diagnostic(
        kind="repeated_crashes", severity=severity,
        title=title, detail=detail, actions=actions,
        first_seen_at=now, last_seen_at=now, count=consecutive,
        data={"consecutive_crashes": consecutive, "last_error": last_err},
    )]


def _rule_review_dependency_deadlock(task, events, runs, now, cfg) -> list[Diagnostic]:
    """Legacy review handoff starving children: the implementation is
    sticky-blocked with a ``review-required:`` reason while todo children wait
    for it to be terminal. Graph-aware; deliberately mutates nothing."""
    if _task_field(task, "status") != "blocked":
        return []
    latest_block = next((ev for ev in reversed(list(events)) if _event_kind(ev) == "blocked"), None)
    if latest_block is None:
        return []
    reason = str(_parse_payload(latest_block).get("reason") or "").strip()
    if not reason.lower().startswith("review-required:"):
        return []

    graph = cfg.get("_graph")
    if not isinstance(graph, dict):
        return []
    waiting_children = [
        child for child in (graph.get("children") or [])
        if isinstance(child, dict) and child.get("status") == "todo"
    ]
    if not waiting_children:
        return []

    task_id = str(_task_field(task, "id") or "")
    child_ids = [str(child.get("id")) for child in waiting_children if child.get("id")]
    actions: list[DiagnosticAction] = []
    if task_id:
        actions.append(_cli_hint(
            "Complete the finished implementation phase", f"hermes kanban complete {task_id}",
            suggested=True,
        ))
    if task_id and child_ids:
        actions.append(_cli_hint(
            "Or unlink the incorrectly gated reviewer", f"hermes kanban unlink {task_id} {child_ids[0]}",
        ))

    blocked_at = _event_ts(latest_block) or now
    return [Diagnostic(
        kind="review_dependency_deadlock", severity="error",
        title=f"Review handoff blocks {len(child_ids)} dependent task(s)",
        detail="This implementation is sticky-blocked for review while its downstream task(s) require "
               "the implementation to be done or archived before they can run. Complete the finished "
               "phase, unlink the incorrect dependency, or migrate this workflow to the first-class "
               "review lifecycle.",
        actions=actions,
        first_seen_at=blocked_at, last_seen_at=blocked_at, count=len(child_ids),
        data={"blocked_parent_id": task_id, "waiting_child_ids": child_ids, "block_reason": reason},
    )]


def _rule_stuck_in_blocked(task, events, runs, now, cfg) -> list[Diagnostic]:
    """Blocked for >= cfg["blocked_stale_hours"] (default 24) with no comment
    or unblock since the last ``blocked`` event."""
    hours = float(cfg.get("blocked_stale_hours", 24))
    if _task_field(task, "status") != "blocked":
        return []
    last_blocked_ts = _latest_event_ts(events, {"blocked"})
    if last_blocked_ts == 0:
        return []
    age_hours = (now - last_blocked_ts) / 3600.0
    if age_hours < hours:
        return []
    # Any comment / unblock after the block breaks the "stale" signal.
    if any(_event_kind(ev) in {"commented", "unblocked"} and _event_ts(ev) > last_blocked_ts
           for ev in events):
        return []
    return [Diagnostic(
        kind="stuck_in_blocked", severity="warning",
        title=f"Task has been blocked for {int(age_hours)}h",
        detail=f"This task transitioned to blocked {int(age_hours)}h ago and has had no comments or "
               f"unblock attempts since. Blocked tasks are waiting for human input — check the block "
               f"reason and either unblock with feedback or answer with a comment.",
        actions=[DiagnosticAction(kind="comment", label="Add a comment / unblock the task",
                                  suggested=True)],
        first_seen_at=last_blocked_ts, last_seen_at=last_blocked_ts, count=1,
        data={"blocked_at": last_blocked_ts, "age_hours": round(age_hours, 1)},
    )]


def _rule_block_unblock_cycling(task, events, runs, now, cfg) -> list[Diagnostic]:
    """>= cfg["block_cycle_threshold"] (default 3) blocked-after-unblocked
    cycles within cfg["block_cycle_window_seconds"] (default 24h). Complements
    ``_rule_stuck_in_blocked``, whose timer any unblock resets, so fast cyclers
    are invisible to it.

    ``_rule_stuck_in_blocked`` resets its timer on any ``commented`` / ``unblocked`` event, so a task that
    cycles every few minutes is invisible to it regardless of how many times it cycles (#29747 gap 1). This
    rule complements that one by counting block→unblock cycles in a sliding window.
    """
    threshold = _positive_int(cfg.get("block_cycle_threshold"), 3)
    window_seconds = float(cfg.get("block_cycle_window_seconds", 24 * 3600))
    cycle_cutoff = now - window_seconds

    # Walk in id (arrival) order — created_at alone can't order events that
    # share a second. A blocked event after >= 1 unblocked since the last
    # counted cycle is a new cycle.
    cycles = 0
    seen_unblock_since_last_cycle = False
    initial_blocked_ts = 0
    last_cycle_blocked_ts = 0
    for ev in events:
        ts = _event_ts(ev)
        if ts < cycle_cutoff:
            continue
        kind = _event_kind(ev)
        if kind == "blocked":
            if initial_blocked_ts == 0:
                initial_blocked_ts = ts
            if seen_unblock_since_last_cycle:
                cycles += 1
                last_cycle_blocked_ts = ts
                seen_unblock_since_last_cycle = False
        elif kind == "unblocked":
            seen_unblock_since_last_cycle = True

    if cycles < threshold:
        return []

    task_id = _task_field(task, "id")
    actions: list[DiagnosticAction] = []
    if task_id:
        cmd = f"hermes kanban events {task_id}"
        actions.append(_cli_hint(f"Check block reasons: {cmd}", cmd, suggested=True))
    return [Diagnostic(
        kind="block_unblock_cycling", severity="warning",
        title=f"Task block→unblock cycled {cycles}x in {int(window_seconds/3600)}h",
        detail=f"This task has been blocked {cycles} times after being unblocked, suggesting the "
               f"unblock is not addressing the root cause and the worker keeps hitting the same wall. "
               f"Review the block reasons in the event history; a different intervention (reassign, "
               f"change scope, archive) may be needed.",
        actions=actions,
        first_seen_at=int(initial_blocked_ts) if initial_blocked_ts else int(now),
        last_seen_at=int(last_cycle_blocked_ts) if last_cycle_blocked_ts else int(now),
        count=cycles,
        data={"cycles": cycles, "window_seconds": int(window_seconds)},
    )]


def _rule_stranded_in_ready(task, events, runs, now, cfg) -> list[Diagnostic]:
    """Assigned, unclaimed, ``ready`` for >= cfg["stranded_threshold_seconds"]
    (default 30 min). Deliberately age-based and identity-agnostic so it
    catches typo'd assignees, deleted profiles, and down external worker
    pools alike without a registry to curate. Unassigned tasks are excluded —
    the dispatcher's ``skipped_unassigned`` already covers them."""
    threshold_seconds = float(cfg.get("stranded_threshold_seconds", 30 * 60))
    if _task_field(task, "status") != "ready":
        return []
    # A live claim means it's being worked on even without progress yet.
    if _task_field(task, "claim_lock"):
        return []
    assignee = _task_field(task, "assignee") or ""
    if not assignee.strip():
        return []

    # Most recent event that put the task into ready; with none (old task /
    # truncated events) fall back to created_at — over-flagging an ancient
    # task beats missing a stranded one.
    last_ready_ts = _latest_event_ts(events, {"created", "promoted", "reclaimed", "unblocked"})
    if last_ready_ts == 0:
        last_ready_ts = int(_task_field(task, "created_at", default=0) or 0)
    if last_ready_ts == 0:
        return []

    age_seconds = now - last_ready_ts
    if age_seconds < threshold_seconds:
        return []

    age_str = f"{age_seconds / 3600:.1f}h" if age_seconds >= 3600 else f"{int(age_seconds / 60)}m"
    # Escalate with age: <2x threshold warning, 2x-6x error, >6x critical.
    if age_seconds >= threshold_seconds * 6:
        severity = "critical"
    elif age_seconds >= threshold_seconds * 2:
        severity = "error"
    else:
        severity = "warning"

    actions = [
        DiagnosticAction(kind="reassign", label="Reassign to a different worker",
                         payload={"current_assignee": assignee}),
        _cli_hint("Check dispatcher status", "hermes kanban diagnostics"),
    ]
    return [Diagnostic(
        kind="stranded_in_ready", severity=severity,
        title=f"Ready for {age_str} with no worker",
        detail=f"This task has been ready for {age_str} but nothing has claimed it. Common causes: "
               f"assignee {assignee!r} is misspelled, the profile was deleted, or the external worker "
               f"pool for this lane is down. Confirm the assignee is correct and that a worker is "
               f"actually polling for it.",
        actions=actions,
        first_seen_at=last_ready_ts, last_seen_at=last_ready_ts, count=1,
        data={"ready_since": last_ready_ts, "age_seconds": int(age_seconds),
              "assignee": assignee, "threshold_seconds": int(threshold_seconds)},
    )]


# Order matters: earlier rules render first on severity ties.
_RULES: list[RuleFn] = [
    _rule_hallucinated_cards,
    _rule_triage_aux_unavailable,
    _rule_prose_phantom_refs,
    _rule_repeated_failures,
    _rule_repeated_crashes,
    _rule_review_dependency_deadlock,
    _rule_stuck_in_blocked,
    _rule_block_unblock_cycling,
    _rule_stranded_in_ready,
]


DEFAULT_CONFIG = {
    # Match the dispatcher default (kanban.failure_limit) so repeated-failure
    # diagnostics do not lag behind the default auto-block threshold.
    "failure_threshold": 2,
    # Legacy alias accepted at read time by _rule_repeated_failures.
    "spawn_failure_threshold": 2,
    "crash_threshold": 2,
    "blocked_stale_hours": 24,
    # Below 30 min the signal is dominated by tasks about to be claimed on
    # the next dispatcher tick.
    "stranded_threshold_seconds": 30 * 60,
}


def _has_explicit_threshold(cfg: dict) -> bool:
    return "failure_threshold" in cfg or "spawn_failure_threshold" in cfg


def config_from_kanban_config(kanban_cfg: Optional[dict]) -> dict:
    """Diagnostics config from the ``kanban`` section. ``kanban.diagnostics.
    failure_threshold`` is an explicit override; otherwise the threshold is
    ``kanban.failure_limit`` so diagnostics match the dispatcher's breaker."""
    kanban_cfg = kanban_cfg or {}
    diag_cfg = dict(kanban_cfg.get("diagnostics") or {})
    diag_cfg.setdefault(
        "failure_limit", kanban_cfg.get("failure_limit", DEFAULT_CONFIG["failure_threshold"]),
    )
    if not _has_explicit_threshold(diag_cfg):
        diag_cfg["failure_threshold"] = diag_cfg["failure_limit"]
    return diag_cfg


def config_from_runtime_config(raw_config: Optional[dict]) -> dict:
    """Diagnostics config from the full runtime config: folds ``kanban`` through
    ``config_from_kanban_config`` and carries ``kanban``/``auxiliary``/``model``
    through for the triage-aware rules."""
    raw_config = raw_config or {}
    if not isinstance(raw_config, dict):
        return {}
    cfg: dict = {}
    kanban_cfg = raw_config.get("kanban")
    if isinstance(kanban_cfg, dict):
        cfg.update(config_from_kanban_config(kanban_cfg))
        cfg["kanban"] = kanban_cfg
    for key in ("auxiliary", "model"):
        value = raw_config.get(key)
        if value is not None:
            cfg[key] = value
    return cfg


def compute_task_diagnostics(
    task,
    events: list,
    runs: list,
    *,
    now: Optional[int] = None,
    config: Optional[dict] = None,
    graph: Optional[dict] = None,
) -> list[Diagnostic]:
    """Run every rule for one task; critical first, then error, warning; ties
    broken by most-recent ``last_seen_at``."""
    now_ts = int(now if now is not None else time.time())
    config = config or {}
    cfg = {**DEFAULT_CONFIG, **config}
    if graph is not None:
        cfg["_graph"] = graph
    if not _has_explicit_threshold(config) and "failure_limit" in config:
        cfg["failure_threshold"] = _positive_int(
            config.get("failure_limit"), DEFAULT_CONFIG["failure_threshold"],
        )
    out: list[Diagnostic] = []
    for rule in _RULES:
        try:
            out.extend(rule(task, events, runs, now_ts, cfg))
        except Exception:
            # A broken rule must never 500 a whole /board request.
            continue
    severity_idx = {s: i for i, s in enumerate(SEVERITY_ORDER)}
    out.sort(key=lambda d: (-severity_idx.get(d.severity, -1), -(d.last_seen_at or 0)))
    return out


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

DIAGNOSTIC_KINDS = (
    "hallucinated_cards",
    "triage_aux_unavailable",
    "prose_phantom_refs",
    "repeated_failures",
    "repeated_crashes",
    "review_dependency_deadlock",
    "stuck_in_blocked",
    "block_unblock_cycling",
    "stranded_in_ready",
)
# ---- END PLUGIN-COMPAT ----
