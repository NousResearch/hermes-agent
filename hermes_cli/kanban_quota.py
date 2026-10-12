"""Durable, profile-local provider quota gates for Kanban claims.

Credentials never enter the circuit keys or durable evidence. Provider labels identify configured
credential surfaces; model changes alone cannot evade a spending limit.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS provider_circuits (
    scope TEXT NOT NULL, backend TEXT NOT NULL,
    retry_at REAL, attempts INTEGER NOT NULL DEFAULT 1,
    probe_task TEXT, probe_run_id INTEGER, PRIMARY KEY(scope, backend)
);
CREATE TABLE IF NOT EXISTS task_provider_waits (
    task_id TEXT PRIMARY KEY, scope TEXT NOT NULL, backend TEXT NOT NULL,
    resume_status TEXT NOT NULL, hard INTEGER NOT NULL DEFAULT 1, route_backend TEXT NOT NULL, automatic INTEGER NOT NULL DEFAULT 1
);
"""


def failure_evidence(agent, classified, error) -> dict:
    """Carry classifier proof and authoritative timing through the worker exit."""
    from agent.error_classifier import FailoverReason, _extract_error_body, _rate_limit_reset_seconds
    reason = classified.reason
    hard = (reason == FailoverReason.billing and not classified.billing_unverified) or (
        reason == FailoverReason.rate_limit and classified.error_context.get("quota_exhausted") is True)
    response = getattr(error, "response", None)
    delay = _rate_limit_reset_seconds("", _extract_error_body(error), getattr(response, "headers", None))
    reset = classified.error_context.get("reset_at")
    if delay is not None:
        reset = time.time() + delay
    return {"hard_quota": bool(hard), "provider": str(agent.provider or "").strip().lower(),
            "retry_at": reset, "reason": reason.value, "endpoint": _endpoint(getattr(agent, "base_url", ""))}


def _endpoint(base_url):
    from hermes_cli.route_identity import normalize_route_base_url
    return hashlib.sha256(normalize_route_base_url(base_url).encode()).hexdigest() if base_url else ""


def route(row) -> dict:
    from hermes_cli.profiles import resolve_profile_env
    from hermes_cli.kanban_db_dispatch import _worker_profile_scope
    from hermes_cli.config import load_config, get_compatible_custom_providers
    from hermes_constants import get_hermes_home
    from hermes_cli.providers import resolve_user_provider, resolve_custom_provider, get_provider, custom_provider_slug
    assignee = row["assignee"] or "default"
    provider = row["provider_override"] or "unconfigured"
    endpoint = ""
    try:
        home = resolve_profile_env(assignee)
    except FileNotFoundError:
        home = str(Path(get_hermes_home()) / "profiles" / assignee)
    else:
        with _worker_profile_scope(home):
            cfg = load_config()
            model = cfg.get("model") or {}
            provider = row["provider_override"] or (model.get("provider") if isinstance(model, dict) else None) or "auto"
            custom = None if provider == "custom" and isinstance(model, dict) and model.get("base_url") else (resolve_user_provider(provider, cfg.get("providers") or {}) or resolve_custom_provider(provider, get_compatible_custom_providers(cfg)))
            from hermes_cli.auth import AuthError, resolve_provider
            if custom:
                provider, endpoint = custom_provider_slug(custom.id), custom.base_url
            else:
                try:
                    provider = resolve_provider(provider)
                except AuthError:
                    pass
                if isinstance(model, dict) and (not row["provider_override"] or row["provider_override"] == model.get("provider")):
                    endpoint = model.get("base_url") or ""
                definition = get_provider(provider, allow_network=False)
                if not endpoint and definition:
                    endpoint = definition.base_url or ""
    endpoint = _endpoint(endpoint)
    backend = hashlib.sha256((str(provider).strip().lower() + "\0" + endpoint).encode()).hexdigest()
    return {"scope": str(Path(home).resolve()), "backend": backend,
            "provider": str(provider).strip().lower(), "endpoint": endpoint}


def identity(row) -> tuple[str, str]:
    resolved = route(row)
    return resolved["scope"], resolved["backend"]


def _failed_route(row, metadata, evidence):
    pinned = metadata.get("quota_route") or route(row)
    actual = evidence.get("provider") or ""
    endpoint = evidence.get("endpoint") or ""
    matches = actual == pinned["provider"] or (actual == "custom" and pinned["provider"].startswith("custom:"))
    if not actual or (matches and (not endpoint or not pinned["endpoint"] or endpoint == pinned["endpoint"])):
        return pinned
    # A configured fallback failed: guard that route without benching the healthy primary.
    backend = hashlib.sha256((actual + "\0" + endpoint).encode()).hexdigest()
    return {"scope": pinned["scope"], "backend": backend, "provider": actual, "endpoint": endpoint}


def _finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError):
        return None


def record_worker_result(result: dict) -> None:
    """Persist a run-scoped witness before the exit trailer (restart safe)."""
    from agent.delegation_context import is_dispatcher_owned_worker_context
    if not is_dispatcher_owned_worker_context() or not os.environ.get("HERMES_KANBAN_RUN_ID"):
        return
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    import contextlib
    run_id = int(os.environ["HERMES_KANBAN_RUN_ID"])
    task_id = os.environ["HERMES_KANBAN_TASK"]
    evidence = result.get("provider_failure") or {"hard_quota": False}
    with contextlib.closing(kbc.connect()) as conn, kb.write_txn(conn):
        row = conn.execute("SELECT * FROM tasks WHERE id=? AND status='running' AND current_run_id=? AND claim_lock=?",
                           (task_id, run_id, os.environ.get("HERMES_KANBAN_CLAIM_LOCK"))).fetchone()
        if row is None:
            return
        metadata = kb._json_dict(conn.execute("SELECT metadata FROM task_runs WHERE id=?", (run_id,)).fetchone()[0])
        already_armed = metadata.get("quota_armed")
        metadata["has_tool_results"] = any(isinstance(message, dict) and message.get("role") == "tool" for message in result.get("messages") or [])
        metadata["provider_failure"] = evidence
        metadata["quota_armed"] = bool(evidence.get("hard_quota"))
        conn.execute("UPDATE task_runs SET metadata=? WHERE id=?", (kb._json_or_null(metadata), run_id))
        if evidence.get("hard_quota") and not already_armed:
            arm(conn, row, evidence, metadata)


def arm(conn, row, evidence, metadata):
    failed = _failed_route(row, metadata, evidence)
    scope, backend = failed["scope"], failed["backend"]
    pinned = metadata.get("quota_route") or route(row)
    reset = _finite(evidence.get("retry_at")) if failed == pinned else None
    existing = conn.execute("SELECT * FROM provider_circuits WHERE scope=? AND backend=?", (scope, backend)).fetchone()
    if existing is None:
        conn.execute("INSERT INTO provider_circuits(scope,backend,retry_at) VALUES(?,?,?)", (scope, backend, reset))
    elif (existing["probe_run_id"] is not None
          and existing["probe_run_id"] == row["current_run_id"]
          and existing["probe_task"] == row["id"]):
        conn.execute("UPDATE provider_circuits SET attempts=attempts+1,retry_at=NULL,probe_task=NULL,probe_run_id=NULL WHERE scope=? AND backend=?", (scope, backend))
    # Other workers already in flight belong to the initial failure wave; they
    # must neither extend its reset nor spend its single recovery probe.


def park(conn, row, scope, backend, resume, *, route_backend=None, automatic=True):
    from hermes_cli import kanban_db as kb
    if conn.execute("SELECT 1 FROM task_provider_waits WHERE task_id=?", (row["id"],)).fetchone():
        return
    route_backend = route_backend or identity(row)[1]
    conn.execute("INSERT INTO task_provider_waits(task_id,scope,backend,resume_status,route_backend,automatic) VALUES(?,?,?,?,?,?)",
                 (row["id"], scope, backend, resume, route_backend, int(automatic)))
    conn.execute("UPDATE tasks SET status='blocked', last_failure_error=? WHERE id=?",
                 ("Provider spending quota exhausted; restore quota and explicitly unblock, or switch provider.", row["id"]))
    kb._append_event(conn, row["id"], "gave_up", {"sticky": True, "provider_quota": True,
                     "retry_status": resume, "error": "Provider spending quota exhausted; task parked."})


def guard_claim(conn, task_id, resume, *, reserve=True, pinned=None) -> bool:
    """Caller holds BEGIN IMMEDIATE: gate check and claim share one transaction."""
    if reserve:
        adopt_legacy_limits(conn)
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if row is None or row["status"] != resume or row["claim_lock"] is not None:
        return False
    # Transient retries are task scoped and cannot bypass timing via direct claims.
    latest = conn.execute("SELECT metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    from hermes_cli import kanban_db as kb
    meta = kb._json_dict(latest[0]) if latest else {}
    pinned = pinned or route(row)
    scope, backend = pinned["scope"], pinned["backend"]
    until = _finite(meta.get("retry_at"))
    if until is not None and until > time.time() and meta.get("retry_backend", backend) == backend and meta.get("retry_scope", scope) == scope:
        return False
    circuit = conn.execute("SELECT * FROM provider_circuits WHERE scope=? AND backend=?", (scope, backend)).fetchone()
    if circuit is None:
        return True
    if _probe_succeeded(conn, circuit):
        if reserve:
            conn.execute("DELETE FROM provider_circuits WHERE scope=? AND backend=?", (scope, backend))
        return True
    until = circuit["retry_at"]
    if (until is not None and until <= time.time()
            and circuit["probe_task"] is None and circuit["probe_run_id"] is None):
        if reserve:
            conn.execute("UPDATE provider_circuits SET retry_at=NULL, probe_task=? WHERE scope=? AND backend=?", (task_id, scope, backend))
        return True
    if reserve:
        park(conn, row, scope, backend, resume)
    return False


def bind_probe_run(conn, task_id, run_id, pinned):
    """Bind the reservation before the claim/open-run transaction can commit."""
    conn.execute(
        "UPDATE provider_circuits SET probe_run_id=? "
        "WHERE scope=? AND backend=? AND probe_task=? AND probe_run_id IS NULL",
        (run_id, pinned["scope"], pinned["backend"], task_id),
    )


def _probe_succeeded(conn, circuit):
    from hermes_cli import kanban_db as kb
    probe = circuit["probe_task"]
    if not probe or circuit["probe_run_id"] is None:
        return False
    run = conn.execute(
        "SELECT metadata FROM task_runs WHERE id=? AND task_id=? AND ended_at IS NOT NULL "
        "AND outcome='completed' AND status='done'", (circuit["probe_run_id"], probe)
    ).fetchone()
    pinned = kb._json_dict(run["metadata"]).get("quota_route") if run else None
    return bool(pinned and (pinned.get("scope"), pinned.get("backend"))
                == (circuit["scope"], circuit["backend"]))


def recover_waits(conn):
    """Only quota-owned waits may resume after an explicit route change or reset."""
    from hermes_cli import kanban_db as kb
    for circuit in conn.execute("SELECT * FROM provider_circuits WHERE probe_task IS NOT NULL").fetchall():
        if _probe_succeeded(conn, circuit):
            conn.execute("DELETE FROM provider_circuits WHERE scope=? AND backend=?", (circuit["scope"], circuit["backend"]))
    for wait in conn.execute("SELECT * FROM task_provider_waits").fetchall():
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (wait["task_id"],)).fetchone()
        if row is None or row["status"] != "blocked":
            continue
        scope, backend = identity(row)
        if not wait["automatic"]:
            continue
        circuit = conn.execute("SELECT retry_at,probe_task,probe_run_id FROM provider_circuits WHERE scope=? AND backend=?", (wait["scope"], wait["backend"])).fetchone()
        if (scope, backend) == (wait["scope"], wait["route_backend"]):
            if not wait["hard"]:
                continue
            if circuit is not None and (
                circuit["probe_task"] is not None or circuit["probe_run_id"] is not None
                or circuit["retry_at"] is None or circuit["retry_at"] > time.time()
            ):
                continue
        if not wait["hard"]:
            clear_wait(conn, row["id"])
        conn.execute("UPDATE tasks SET status=?, consecutive_failures=0, last_failure_error=NULL WHERE id=?", (wait["resume_status"], row["id"]))
        conn.execute("DELETE FROM task_provider_waits WHERE task_id=?", (row["id"],))
        kb._append_event(conn, row["id"], "unblocked", {"status": wait["resume_status"], "provider_recovery": True})


def clear_wait(conn, task_id):
    from hermes_cli import kanban_db as kb
    latest = conn.execute("SELECT id,metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    if latest:
        metadata = kb._json_dict(latest["metadata"])
        metadata.pop("retry_at", None)
        conn.execute("UPDATE task_runs SET metadata=? WHERE id=?", (kb._json_or_null(metadata), latest["id"]))
    wait = conn.execute("SELECT * FROM task_provider_waits WHERE task_id=?", (task_id,)).fetchone()
    if wait:
        # A switched task must never clear the old provider's circuit for siblings.
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if wait["hard"] and identity(row) == (wait["scope"], wait["route_backend"]):
            conn.execute("DELETE FROM provider_circuits WHERE scope=? AND backend=?", (wait["scope"], wait["backend"]))
        conn.execute("DELETE FROM task_provider_waits WHERE task_id=?", (task_id,))


def account_exit(conn, row, dead, failure_limit):
    """Book hard quota once; spend the ordinary transient failure budget."""
    from hermes_cli import kanban_db as kb
    latest = conn.execute("SELECT metadata FROM task_runs WHERE id=?", (row["current_run_id"],)).fetchone()
    metadata = kb._json_dict(latest[0]) if latest else {}
    evidence = metadata.get("provider_failure")
    if evidence is None:
        # Older workers left only their provider's rendered error in the log.
        from agent.error_classifier import classify_api_error
        import re
        text = dead.event_payload.get("worker_output", "")
        match = re.search(r"(?:error code|status(?: code)?)[ :='\"]+(40[23]|429)", text, re.I)
        error = RuntimeError(text)
        if match:
            error.status_code = int(match[1])
        if "personal-team-blocked:spending-limit" in text and match:
            error.body = {"error": {"code": "personal-team-blocked:spending-limit"}}
        _, backend = identity(row)
        provider = row["provider_override"] or ""
        if not provider:
            from hermes_cli.profiles import resolve_profile_env
            from hermes_cli.kanban_db_dispatch import _worker_profile_scope
            from hermes_cli.config import load_config
            try:
                with _worker_profile_scope(resolve_profile_env(row["assignee"] or "default")):
                    model = load_config().get("model") or {}
                provider = model.get("provider", "") if isinstance(model, dict) else ""
            except FileNotFoundError:
                provider = ""
        verdict = classify_api_error(error, provider=provider)
        from types import SimpleNamespace
        evidence = failure_evidence(SimpleNamespace(provider=provider), verdict, error)
        if not match:
            evidence["hard_quota"] = False
    dead.event_payload.update(metadata)
    dead.event_payload["provider_failure"] = evidence
    dead.event_payload["limit_accounted"] = True
    resume = dead.event_payload["retry_status"]
    if evidence.get("hard_quota"):
        if not metadata.get("quota_armed"):
            arm(conn, row, evidence, metadata)
        failed = _failed_route(row, metadata, evidence)
        pinned = metadata.get("quota_route") or route(row)
        park(conn, row, failed["scope"], failed["backend"], resume, route_backend=pinned["backend"], automatic=not metadata.get("has_tool_results"))
        dead.event_payload.update({"provider_quota": True, "retry_status": "blocked"})
        return
    failures = int(row["consecutive_failures"] or 0) + 1
    limit = row["max_retries"] if row["max_retries"] is not None else failure_limit
    delay = min(60 * 2 ** min(failures - 1, 6), 3600)
    until = max(time.time() + delay, _finite(evidence.get("retry_at")) or 0)
    dead.event_payload["retry_at"] = until
    pinned = metadata.get("quota_route") or route(row)
    dead.event_payload["retry_backend"] = pinned["backend"]
    dead.event_payload["retry_scope"] = pinned["scope"]
    dead.event_payload["failures"] = failures
    conn.execute("UPDATE tasks SET consecutive_failures=?, last_failure_error=? WHERE id=?",
                 (failures, "Transient provider failure; bounded retry budget.", row["id"]))
    if failures >= max(1, int(limit)) or metadata.get("has_tool_results"):
        scope, backend = identity(row)
        conn.execute("INSERT OR REPLACE INTO task_provider_waits(task_id,scope,backend,resume_status,hard,route_backend,automatic) VALUES(?,?,?,?,0,?,?)",
                     (row["id"], scope, backend, resume, backend, int(not metadata.get("has_tool_results"))))
        conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (row["id"],))
        kb._append_event(conn, row["id"], "gave_up", {"sticky": True, "failures": failures,
                         "retry_status": resume, "error": "Provider retry budget exhausted; explicitly unblock after recovery."})
        dead.event_payload["retry_status"] = "blocked"


def adopt_legacy_limits(conn, *, failure_limit=None, board=None):
    """Adopt ended pre-upgrade limits before ANY claim can launch a sibling."""
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatcher
    from types import SimpleNamespace
    rows = conn.execute("SELECT t.*, r.id AS legacy_run_id FROM tasks t JOIN task_runs r ON r.id="
                        "(SELECT MAX(id) FROM task_runs WHERE task_id=t.id) "
                        "WHERE t.status IN ('ready','review') AND r.outcome='rate_limited' "
                        "AND COALESCE(r.metadata,'') NOT LIKE '%\"limit_accounted\": true%'").fetchall()
    for row in rows:
        # The last worker log is a migration witness only; never inspect older
        # attempts once the new accounting marker has been committed.
        dead = SimpleNamespace(event_payload={"retry_status": row["status"],
                     "worker_output": dispatcher._worker_final_output(row["id"], board=board)})
        data = dict(row)
        data["current_run_id"] = row["legacy_run_id"]
        account_exit(conn, data, dead, dispatcher.DEFAULT_FAILURE_LIMIT if failure_limit is None else failure_limit)
        conn.execute("UPDATE task_runs SET metadata=? WHERE id=?", (kb._json_or_null(dead.event_payload), row["legacy_run_id"]))
