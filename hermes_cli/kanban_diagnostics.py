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


def _latest_gave_up_is_terminal_provider(events: Iterable[Any]) -> bool:
    """True when the most recent breaker trip was a terminal provider error (credential
    revoked, model gone) and nothing has resumed the task since."""
    for ev in reversed(list(events)):
        kind = _event_kind(ev)
        if kind == "gave_up":
            return bool(_parse_payload(ev).get("terminal_provider"))
        if kind in {"unblocked", "promoted", "completed", "claimed"}:
            return False
    return False


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
    # A terminal provider error (credential revoked, model gone) blocks the card after ONE
    # attempt, below any threshold; it still needs an operator, so diagnose it now.
    terminal_trip = _latest_gave_up_is_terminal_provider(events)
    if not terminal_trip and (failures is None or failures < threshold):
        return []
    failures = failures or 0
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
    if terminal_trip:
        title = "Provider rejected this profile's credential or model — blocked after one attempt"
        detail = (
            f"The worker's provider call failed with an error a retry cannot fix (revoked or invalid "
            f"API key, model not found), so the dispatcher blocked the task instead of spending the "
            f"{failure_limit}-attempt retry budget on it. Full last error:\n\n{err_snippet}\n\n"
            f"Fix the assignee profile's provider credentials/model, then unblock the task."
        )
    elif err_snippet:
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


def _rule_running_with_open_parents(task, events, runs, now, cfg) -> list[Diagnostic]:
    """A ``running`` card with a direct parent that is not ``done``/``archived``:
    the dependency gate is not holding it (the parent reopened mid-run, or the
    edge predates the running-child refusal) and ``kanban_complete`` will be
    refused until the parents finish. Graph-aware; mutates nothing."""
    if _task_field(task, "status") != "running":
        return []
    graph = cfg.get("_graph")
    if not isinstance(graph, dict):
        return []
    open_parents = [
        parent for parent in (graph.get("parents") or [])
        if isinstance(parent, dict) and parent.get("id")
        and parent.get("status") not in ("done", "archived")
    ]
    if not open_parents:
        return []
    task_id = str(_task_field(task, "id") or "")
    parent_ids = [str(parent["id"]) for parent in open_parents]
    seen_at = int(_task_field(task, "started_at", default=0) or 0) or now
    return [Diagnostic(
        kind="running_with_open_parents", severity="warning",
        title=f"Running while {len(parent_ids)} parent(s) are not done",
        detail="This card is running concurrently with a parent it declares a dependency on, so the "
               "parent's work is not serialised ahead of it and completion will be refused until every "
               "parent is done or archived. Finish the parent, or unlink the edge if it was never meant "
               "to gate this run.",
        actions=[_cli_hint("Unlink the parent that should not gate this run",
                           f"hermes kanban unlink {parent_ids[0]} {task_id}")],
        first_seen_at=seen_at, last_seen_at=now, count=len(parent_ids),
        data={"open_parents": [{"id": p["id"], "status": p.get("status")} for p in open_parents]},
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


# --- Stranded-in-ready routing: board-state causes and their owners ---
#
# The rule below can only see (task, events, runs, cfg) — it has no connection
# and must stay that way. "Why is this card not running?" is a question about the
# BOARD, so the facts come from a caller that has one
# (:func:`board_facts_for_ready_lane`) and ride in through
# ``cfg["_board_facts"]``, the same seam the graph rules use for ``_graph``.
# Without them the diagnostic degrades to its old generic text — never to a guess.
#
# Cause -> owner is a fixed table: the mapping is not inferred at runtime, so the
# same board state always routes to the same lane.
STRANDED_CAUSE_ASSIGNEE_UNKNOWN = "assignee_unknown"
STRANDED_CAUSE_FLEET_AT_CAPACITY = "fleet_at_capacity"
STRANDED_CAUSE_LANE_QUEUE_AHEAD = "lane_queue_ahead"
STRANDED_CAUSE_LANE_NO_WORKER = "lane_no_worker"

#: ``default`` is the ops head — the hub every cross-lane need routes through, and
#: the owner of host/dispatch-capacity/worker-availability questions.
DEFAULT_REPAIR_OWNER = "default"

#: Cause -> owner lane, or the constant ``LANE_AUTHORITY`` for "the lane that owns
#: this card's own queue" (its design authority: a queue deeper than the spawn
#: budget is a queue-shaping decision, not a host fault).
LANE_AUTHORITY = "<lane-authority>"
STRANDED_CAUSE_OWNER = {
    STRANDED_CAUSE_ASSIGNEE_UNKNOWN: DEFAULT_REPAIR_OWNER,
    STRANDED_CAUSE_FLEET_AT_CAPACITY: DEFAULT_REPAIR_OWNER,
    STRANDED_CAUSE_LANE_NO_WORKER: DEFAULT_REPAIR_OWNER,
    STRANDED_CAUSE_LANE_QUEUE_AHEAD: LANE_AUTHORITY,
}

#: Deterministic remedy text per cause, carried into the repair card.
STRANDED_CAUSE_REMEDY = {
    STRANDED_CAUSE_ASSIGNEE_UNKNOWN:
        "create the missing profile, or reassign the card to a profile that exists",
    STRANDED_CAUSE_FLEET_AT_CAPACITY:
        "raise kanban.max_in_progress, or drain the in-flight fleet that is holding the slots",
    STRANDED_CAUSE_LANE_QUEUE_AHEAD:
        "re-prioritise the card (a NEGATIVE priority front-runs a same-priority age group) "
        "or split the lane's queue",
    STRANDED_CAUSE_LANE_NO_WORKER:
        "find why the lane has no worker despite a free slot (spawn failure, missing profile "
        "in kanban.dispatch_profiles, or a down external pool)",
}

#: Role suffix -> the design authority that grooms that lane's queue.
_LANE_ROLE_SUFFIXES = ("coder", "worker", "sme", "reviewer", "stl")

#: Prefix for the auto-filed repair cards' ``idempotency_key``. Doubles as the
#: one-generation guard: a card that carries this prefix is never routed again, so
#: a stuck repair card cannot spawn repair cards of its own.
REPAIR_IDEMPOTENCY_PREFIX = "kanban-stranded-repair:"


def _profile_exists(name: str) -> Optional[bool]:
    """``True``/``False`` when this host can answer, ``None`` when it cannot.

    Lazy import: diagnostics are also computed in dashboard/plugin contexts where
    the profiles module may not be importable, and an unknown answer must never
    be reported as a missing profile.
    """
    if not name:
        return False
    try:
        from hermes_cli.profiles import profile_exists
    except Exception:
        return None
    try:
        return bool(profile_exists(name))
    except Exception:
        return None


def lane_design_authority(assignee: str) -> str:
    """The lane that owns *assignee*'s queue shape — e.g. ``demo-coder`` -> ``demo-stl``.

    Falls back to :data:`DEFAULT_REPAIR_OWNER` when the assignee does not follow
    the ``<lane>-<role>`` convention or the derived profile does not exist.
    """
    name = (assignee or "").strip()
    if not name:
        return DEFAULT_REPAIR_OWNER
    head, _, role = name.rpartition("-")
    candidate = name if role == "stl" else (f"{head}-stl" if head and role in _LANE_ROLE_SUFFIXES else "")
    if not candidate:
        return DEFAULT_REPAIR_OWNER
    existing = _profile_exists(candidate)
    if existing is False and candidate != DEFAULT_REPAIR_OWNER:
        return DEFAULT_REPAIR_OWNER
    return candidate


def stranded_cause(facts: Optional[dict], assignee: str) -> str:
    """Classify, from board facts alone, why a ready card has no worker.

    Ordered and total: every input yields exactly one cause, so the same board
    state always produces the same owner and the same repair card.
    """
    if not facts:
        return ""
    if facts.get("profile_exists") is False:
        return STRANDED_CAUSE_ASSIGNEE_UNKNOWN
    cap = facts.get("fleet_cap")
    if isinstance(cap, int) and cap > 0 and int(facts.get("fleet_running") or 0) >= cap:
        return STRANDED_CAUSE_FLEET_AT_CAPACITY
    if int(facts.get("queue_ahead") or 0) > 0:
        return STRANDED_CAUSE_LANE_QUEUE_AHEAD
    return STRANDED_CAUSE_LANE_NO_WORKER


def stranded_cause_detail(cause: str, facts: Optional[dict], assignee: str) -> str:
    """One deterministic sentence naming the board state that explains the cause."""
    facts = facts or {}
    queue_ahead = int(facts.get("queue_ahead") or 0)
    lane_running = int(facts.get("lane_running") or 0)
    fleet_running = int(facts.get("fleet_running") or 0)
    cap = facts.get("fleet_cap")
    if cause == STRANDED_CAUSE_ASSIGNEE_UNKNOWN:
        return f"assignee {assignee!r} has no profile on this host, so no worker can be spawned for it"
    if cause == STRANDED_CAUSE_FLEET_AT_CAPACITY:
        return (f"the fleet is at its concurrency cap ({fleet_running}/{cap} running), so this "
                f"card's lane cannot get a spawn slot")
    if cause == STRANDED_CAUSE_LANE_QUEUE_AHEAD:
        return (f"{queue_ahead} ready card(s) for {assignee!r} rank ahead of it in the dispatch "
                f"order, and the lane has {lane_running} running")
    if cause == STRANDED_CAUSE_LANE_NO_WORKER:
        return (f"nothing ranks ahead of it, the fleet has a free slot "
                f"({fleet_running}/{cap if cap else 'unset'} running), and {assignee!r} has no "
                f"running card — the spawn did not happen")
    return ""


def stranded_owner(cause: str, assignee: str) -> str:
    """The lane that owns the fix for *cause* (fixed table, never inferred)."""
    owner = STRANDED_CAUSE_OWNER.get(cause, DEFAULT_REPAIR_OWNER)
    if owner == LANE_AUTHORITY:
        return lane_design_authority(assignee)
    return owner


def _fleet_cap(cfg: Optional[dict]) -> Optional[int]:
    kanban_cfg = (cfg or {}).get("kanban")
    if isinstance(kanban_cfg, dict):
        cap = kanban_cfg.get("max_in_progress")
        if isinstance(cap, int) and cap > 0:
            return cap
    return None


def board_facts_for_ready_lane(conn, *, config: Optional[dict] = None) -> dict[str, dict]:
    """``{task_id: facts}`` for the ready lane, read in dispatch order.

    Dispatch order is the dispatcher's own (``priority DESC, created_at ASC``, see
    ``kanban_db_dispatch._lane_rows``): a card is only "not running" relative to
    the cards that beat it to the queue, so the order is part of the fact set.
    Unassigned or claimed rows have no lane question to answer and are omitted.
    """
    rows = conn.execute(
        "SELECT id, assignee, status, claim_lock, idempotency_key FROM tasks "
        "WHERE status = 'ready' AND assignee IS NOT NULL AND assignee != '' "
        "AND claim_lock IS NULL ORDER BY priority DESC, created_at ASC"
    ).fetchall()
    running_by_assignee: dict[str, int] = {}
    for row in conn.execute(
        "SELECT assignee, COUNT(*) AS n FROM tasks WHERE status = 'running' "
        "AND assignee IS NOT NULL GROUP BY assignee"
    ):
        running_by_assignee[row["assignee"]] = int(row["n"])
    fleet_running = sum(running_by_assignee.values())
    cap = _fleet_cap(config)
    ahead: dict[str, int] = {}
    facts: dict[str, dict] = {}
    for row in rows:
        assignee = row["assignee"]
        facts[row["id"]] = {
            "assignee": assignee,
            "queue_ahead": ahead.get(assignee, 0),
            "lane_running": running_by_assignee.get(assignee, 0),
            "fleet_running": fleet_running,
            "fleet_cap": cap,
            "profile_exists": _profile_exists(assignee),
        }
        ahead[assignee] = ahead.get(assignee, 0) + 1
    return facts


def _route_action(task_id: str, assignee: str, facts: dict) -> Optional[DiagnosticAction]:
    """The routing action for a stranded card: owner + cause, from board state."""
    cause = stranded_cause(facts, assignee)
    if not cause:
        return None
    owner = stranded_owner(cause, assignee)
    why = stranded_cause_detail(cause, facts, assignee)
    return DiagnosticAction(
        kind="route",
        label=f"Route repair to {owner} ({cause})",
        payload={
            "owner": owner,
            "cause": cause,
            "task_id": task_id,
            "assignee": assignee,
            "reason": why,
            "remedy": STRANDED_CAUSE_REMEDY.get(cause, ""),
        },
        suggested=True,
    )


def _rule_stranded_in_ready(task, events, runs, now, cfg) -> list[Diagnostic]:
    """Assigned, unclaimed, ``ready`` for >= cfg["stranded_threshold_seconds"]
    (default 30 min). Deliberately age-based and identity-agnostic so it
    catches typo'd assignees, deleted profiles, and down external worker
    pools alike without a registry to curate. Unassigned tasks are excluded —
    the dispatcher's ``skipped_unassigned`` already covers them.

    When the caller supplies ``cfg["_board_facts"]`` (see
    :func:`board_facts_for_ready_lane`) the diagnostic also names WHY the card is
    not running — queue position, lane occupancy, fleet capacity — and carries a
    ``route`` action naming the owning lane. Without those facts it reports only
    what it can see, never a guessed cause.
    """
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

    facts = (cfg.get("_board_facts") or {}).get(_task_field(task, "id")) or {}
    route = _route_action(_task_field(task, "id"), assignee, facts) if facts else None
    if route is not None:
        cause = route.payload["cause"]
        why = route.payload["reason"]
        detail = (f"This task has been ready for {age_str} but nothing has claimed it. Why, from "
                  f"board state: {why}. Owner of the fix: {route.payload['owner']} — "
                  f"{route.payload['remedy']}.")
        data = {
            "ready_since": last_ready_ts, "age_seconds": int(age_seconds),
            "assignee": assignee, "threshold_seconds": int(threshold_seconds),
            "cause": cause, "owner": route.payload["owner"], "cause_detail": why,
            "queue_ahead": facts.get("queue_ahead"), "lane_running": facts.get("lane_running"),
            "fleet_running": facts.get("fleet_running"), "fleet_cap": facts.get("fleet_cap"),
            "profile_exists": facts.get("profile_exists"),
        }
    else:
        detail = (f"This task has been ready for {age_str} but nothing has claimed it. Common "
                  f"causes: assignee {assignee!r} is misspelled, the profile was deleted, or the "
                  f"external worker pool for this lane is down. Confirm the assignee is correct and "
                  f"that a worker is actually polling for it.")
        data = {"ready_since": last_ready_ts, "age_seconds": int(age_seconds),
                "assignee": assignee, "threshold_seconds": int(threshold_seconds)}
    actions = [a for a in (
        route,
        DiagnosticAction(kind="reassign", label="Reassign to a different worker",
                         payload={"current_assignee": assignee}),
        _cli_hint("Check dispatcher status", "hermes kanban diagnostics"),
    ) if a is not None]
    return [Diagnostic(
        kind="stranded_in_ready", severity=severity,
        title=f"Ready for {age_str} with no worker",
        detail=detail,
        actions=actions,
        first_seen_at=last_ready_ts, last_seen_at=last_ready_ts, count=1,
        data=data,
    )]


# Order matters: earlier rules render first on severity ties.
_RULES: list[RuleFn] = [
    _rule_hallucinated_cards,
    _rule_triage_aux_unavailable,
    _rule_prose_phantom_refs,
    _rule_repeated_failures,
    _rule_repeated_crashes,
    _rule_review_dependency_deadlock,
    _rule_running_with_open_parents,
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
    board_facts: Optional[dict] = None,
) -> list[Diagnostic]:
    """Run every rule for one task; critical first, then error, warning; ties
    broken by most-recent ``last_seen_at``.

    ``board_facts`` is the caller's read of the board for this task (see
    :func:`board_facts_for_ready_lane`) — the rules cannot open a connection, so
    board-level questions like "why is this ready card not running?" arrive as
    data. Omitted, every rule behaves exactly as before.
    """
    now_ts = int(now if now is not None else time.time())
    config = config or {}
    cfg = {**DEFAULT_CONFIG, **config}
    if graph is not None:
        cfg["_graph"] = graph
    if board_facts:
        cfg["_board_facts"] = board_facts
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


def repair_idempotency_key(task_id: str, cause: str) -> str:
    """The one key that makes routing idempotent: one open repair card per
    (stranded card, cause). A second pass — or a second dispatcher — returns the
    same card instead of filing a twin."""
    return f"{REPAIR_IDEMPOTENCY_PREFIX}{task_id}:{cause}"


def _existing_repair_card(conn, key: str) -> Optional[str]:
    row = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' "
        "ORDER BY created_at DESC LIMIT 1",
        (key,),
    ).fetchone()
    return row["id"] if row else None


def bulk_repair_key(cause: str) -> str:
    """Key for a collapsed repair card: one per cause for the whole board."""
    return f"{REPAIR_IDEMPOTENCY_PREFIX}bulk:{cause}"


def _stranded_repair_body(row, diagnostic: Diagnostic, route: dict, key: str) -> str:
    """The repair card's body — the diagnosis, verbatim and deterministic."""
    age_seconds = float(diagnostic.data.get("age_seconds") or 0)
    age = f"{age_seconds / 3600:.1f}h" if age_seconds >= 3600 else f"{int(age_seconds / 60)}m"
    facts = diagnostic.data
    return (
        f"Auto-filed by the kanban dispatcher's stranded-card router — no human in the loop.\n\n"
        f"Stranded card: {row['id']} — {row['title']}\n"
        f"Assignee: {row['assignee']}\n"
        f"Ready for: {age}\n"
        f"Severity: {diagnostic.severity}\n\n"
        f"Why it is not running (board state, not inference): {route['reason']}\n\n"
        f"Cause: {route['cause']}\n"
        f"Board facts: queue_ahead={facts.get('queue_ahead')} "
        f"lane_running={facts.get('lane_running')} "
        f"fleet_running={facts.get('fleet_running')}/{facts.get('fleet_cap')} "
        f"profile_exists={facts.get('profile_exists')}\n\n"
        f"Required fix: {route['remedy']}\n\n"
        f"Routing is idempotent: idempotency key {key} guarantees exactly one open repair "
        f"card per (stranded card, cause). A repair card is never itself routed — if this card "
        f"strands too, the board's `stranded_in_ready` diagnostic reports it and no further card "
        f"is filed.\n"
    )


def _age_label(seconds) -> str:
    """Age as the diagnostics render it: ``2.1h`` / ``45m``."""
    value = float(seconds or 0)
    return f"{value / 3600:.1f}h" if value >= 3600 else f"{int(value / 60)}m"


def _bulk_repair_body(cause: str, entries: list[dict], key: str) -> str:
    """Body for a collapsed repair card: one card standing for a stalled board."""
    shown = entries[:20]
    lines = "\n".join(
        f"- {e['task_id']}  {e['assignee']}  ready {e['age']}  {e['title'][:70]}" for e in shown
    )
    more = f"\n... and {len(entries) - len(shown)} more." if len(entries) > len(shown) else ""
    facts = entries[0]["facts"]
    return (
        f"Auto-filed by the kanban dispatcher's stranded-card router — no human in the loop.\n\n"
        f"{len(entries)} ready cards are stranded with cause {cause}. Past "
        f"{len(shown) if len(shown) < len(entries) else len(entries)} cards of one cause this "
        f"card stands for the population instead of one card per stuck card.\n\n"
        f"Why they are not running (board state, not inference): {entries[0]['reason']}\n\n"
        f"Board facts (oldest card): queue_ahead={facts.get('queue_ahead')} "
        f"lane_running={facts.get('lane_running')} "
        f"fleet_running={facts.get('fleet_running')}/{facts.get('fleet_cap')}\n\n"
        f"Required fix: {STRANDED_CAUSE_REMEDY.get(cause, '')}\n\n"
        f"Stranded cards:\n{lines}{more}\n\n"
        f"Routing is idempotent: idempotency key {key} guarantees exactly one open repair "
        f"card per (board, cause).\n"
    )


def route_stranded_cards(
    conn,
    *,
    board: Optional[str] = None,
    now: Optional[int] = None,
    config: Optional[dict] = None,
    min_severity: str = "error",
    limit: int = 2,
    collapse_over: int = 3,
    dry_run: bool = False,
) -> list[dict]:
    """Turn the ``stranded_in_ready`` detection into a control: file the repair.

    Detection without a control is a log line. For every ready card that has
    waited longer than the stranded threshold at ``min_severity`` or above, this
    files exactly ONE repair card on the same board, routed to the lane that owns
    the cause (fixed table — see ``STRANDED_CAUSE_OWNER``), carrying the board
    facts that explain it. Idempotent per (card, cause), capped at ``limit`` new
    cards per pass, and one generation deep: a repair card is never itself routed.

    Past ``collapse_over`` cards of one cause the whole population gets ONE card
    (idempotency key ``bulk:<cause>``) instead of a card each: a board-wide stall
    is one systemic condition, and a card per stuck card would bury the board it
    is meant to report on.

    Returns one entry per stranded card considered:
    ``{"task_id", "cause", "owner", "severity", "repair_card", "collapsed",
    "outcome"}`` with ``outcome`` in ``filed`` / ``existing`` / ``dry_run`` /
    ``capped`` / ``skipped_repair_card``.
    """
    from hermes_cli import kanban_db as kb

    now_ts = int(now if now is not None else time.time())
    cfg = dict(config or {})
    threshold_seconds = float(
        cfg.get("stranded_threshold_seconds", DEFAULT_CONFIG["stranded_threshold_seconds"])
    )
    # Sound pre-filter: ready_since >= created_at, so a card created inside the
    # threshold window cannot be stranded yet. Keeps the per-tick scan bounded.
    rows = conn.execute(
        "SELECT id, title, assignee, created_at, idempotency_key FROM tasks "
        "WHERE status = 'ready' AND assignee IS NOT NULL AND assignee != '' "
        "AND claim_lock IS NULL AND created_at <= ? "
        "ORDER BY priority DESC, created_at ASC",
        (now_ts - int(threshold_seconds),),
    ).fetchall()
    if not rows:
        return []
    facts_by_task = board_facts_for_ready_lane(conn, config=cfg)
    out: list[dict] = []
    candidates: list[dict] = []
    for row in rows:
        task_id = row["id"]
        seen = row["idempotency_key"]
        if isinstance(seen, str) and seen.startswith(REPAIR_IDEMPOTENCY_PREFIX):
            out.append({"task_id": task_id, "cause": "", "owner": "", "severity": "",
                        "repair_card": None, "collapsed": False,
                        "outcome": "skipped_repair_card"})
            continue
        # The pre-filter query is deliberately narrow; the rules read a whole task
        # row, so load the canonical one instead of handing them a partial Row.
        task = kb.get_task(conn, task_id)
        if task is None:
            continue
        diags = compute_task_diagnostics(
            task, kb.list_events(conn, task_id), kb.list_runs(conn, task_id),
            now=now_ts, config=cfg, board_facts={task_id: facts_by_task.get(task_id, {})},
        )
        for diagnostic in diags:
            if diagnostic.kind != "stranded_in_ready":
                continue
            if not severity_at_or_above(diagnostic.severity, min_severity):
                continue
            route = next((a.payload for a in diagnostic.actions if a.kind == "route"), None)
            if not route:
                continue
            candidates.append({"task_id": task_id, "owner": route["owner"], "cause": route["cause"],
                               "severity": diagnostic.severity, "row": row, "diagnostic": diagnostic,
                               "route": route})
            break

    # Collapse a board-wide stall: past `collapse_over` cards of one cause, ONE
    # card stands for the population instead of a card per stranded card. A
    # dispatch stall is systemic — hundreds of per-card repair cards are noise.
    by_cause: dict[str, list[dict]] = {}
    for cand in candidates:
        by_cause.setdefault(cand["cause"], []).append(cand)
    filed = 0
    for cause, group in by_cause.items():
        collapsed = len(group) > max(int(collapse_over), 1)
        key = bulk_repair_key(cause) if collapsed else repair_idempotency_key(
            group[0]["task_id"], cause)
        headline = {"cause": cause, "owner": group[0]["owner"], "severity": group[0]["severity"],
                    "collapsed": collapsed}
        existing = _existing_repair_card(conn, key)
        if existing:
            out.append({**headline, "task_id": group[0]["task_id"],
                        "repair_card": existing, "outcome": "existing"})
            continue
        if dry_run:
            out.append({**headline, "task_id": group[0]["task_id"],
                        "repair_card": None, "outcome": "dry_run"})
            continue
        if filed >= max(int(limit), 0):
            for cand in group:
                out.append({**headline, "task_id": cand["task_id"],
                            "repair_card": None, "outcome": "capped"})
            continue
        if collapsed:
            entries = [{"task_id": cand["task_id"], "assignee": cand["row"]["assignee"],
                        "title": cand["row"]["title"],
                        "age": _age_label(cand["diagnostic"].data.get("age_seconds")),
                        "reason": cand["route"]["reason"],
                        "facts": cand["diagnostic"].data} for cand in group]
            title = f"Dispatch starvation: {len(group)} cards stranded ({cause})"
            body = _bulk_repair_body(cause, entries, key)
        else:
            title = f"Dispatch starvation: {group[0]['task_id']} is {cause}"
            body = _stranded_repair_body(group[0]["row"], group[0]["diagnostic"],
                                        group[0]["route"], key)
        repair_card = kb.create_task(
            conn,
            title=title,
            body=body,
            assignee=group[0]["owner"],
            idempotency_key=key,
            creator_task_id=group[0]["task_id"],
            board=board,
        )
        # Audit the routing on the stranded card itself: one event per individually
        # routed card, one for a collapsed population (N events per tick is the
        # write amplification this collapse exists to avoid).
        for cand in (group[:1] if collapsed else group):
            with kb.write_txn(conn):
                kb._append_event(
                    conn, cand["task_id"], "stranded_routed",
                    {"repair_card": repair_card, "cause": cause, "owner": group[0]["owner"],
                     "severity": cand["severity"], "collapsed": collapsed,
                     "cards": len(group) if collapsed else 1},
                )
        filed += 1
        out.append({**headline, "task_id": group[0]["task_id"],
                    "repair_card": repair_card, "outcome": "filed"})
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
