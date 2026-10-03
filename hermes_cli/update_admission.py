"""Fresh fail-closed pre-update admission snapshot (#127611).

A supported machine-readable read-only observation for unattended updaters,
exposed as the ``admission`` section of ``hermes update --plan --json``.

What it is
-----------
A read-only observation/handshake: for each relevant runtime/work class it
reports an explicit ``idle`` | ``busy`` | ``unknown`` outcome plus
computed-at/freshness metadata and a producer-defined bounded admission
boundary. ``admissible`` is true only when every covered class is proven
``idle`` on a fresh observation.

What it is not
--------------
This snapshot alone does not block later arrivals. Until admission is
actually granted by the producer-defined boundary, ongoing work is left
alone and normal work acceptance is preserved. Re-query immediately before
any mutation; the snapshot grants no lease. This observation is distinct
from a safe updater handoff or drain operation, and it never uses drain
as the read-only probe.

Fail-closed rules
-----------------
- ``unknown`` for every unreadable, malformed, corrupt, or unavailable
  component. Never convert it to zero/empty.
- Never rewrite or quarantine state while answering (no drain, no repair,
  no ledger quarantine, no status-file writes, no ``cleanup_stale=True``).
- Persisted ``gateway_state.json`` alone is stale-suspect between lifecycle
  edges: a live gateway with no live socket answer and a non-fresh file is
  ``unknown``, never ``idle``.
- Scoped to this install's known profiles/runtimes (``_profile_homes`` plus
  the spawn-ledger rows for this install).

Covered work classes (keys of ``work``)
---------------------------------------
- ``foreground_turns``: gateway chat turns (``_running_agents``).
- ``cron_jobs``: scheduler in-flight jobs, including restart-safe external
  worker pids (``get_running_job_details``) and the durable executions
  ledger's live ``claimed``/``running`` rows.
- ``api_runs``: API-server agent work (handler count + executor worker
  count).
- ``deferred_workers``: executor workers that outlived their turn.
- ``background_delegations``: live async-delegation units.
- ``background_processes``: terminal background processes + pending
  completion watchers.
- ``backend_agent_work``: serve/dashboard backend in-flight work. A live
  backend is ``unknown`` from the CLI (its turns are not observable
  out-of-process); no live backend is ``idle``.
- ``external_workers``: restart-safe worker completion / durable
  claimed-running rows with a live owner + pending process watchers.

Each entry is ``{"state", "count", "detail"}`` where ``count`` is an int
when proven and ``None`` when ``unknown`` (never 0 on error).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

ADMISSION_SCHEMA_VERSION = 1
# Producer-defined bounded window: the observation is admissible only while
# fresh. The updater must re-query immediately before any mutation.
ADMISSION_VALID_FOR_S = 60.0
# Runtime-status heartbeat staleness (mirrors gateway.status TTL: 2x the 60 s
# housekeeping interval). Older than this, a file claiming idle is suspect.
RUNTIME_STALE_TTL_S = 120
# Dashboard/Desktop client attachment window for the heartbeat marker.
DESKTOP_ATTACHED_WINDOW_S = 300.0

WORK_CLASSES = (
    "foreground_turns",
    "cron_jobs",
    "api_runs",
    "deferred_workers",
    "background_delegations",
    "background_processes",
    "backend_agent_work",
    "external_workers",
)

_VALID_STATES = frozenset({"idle", "busy", "unknown"})


def _utc_now_iso(when: Optional[float] = None) -> str:
    ts = time.time() if when is None else float(when)
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _idle(count: int = 0, detail: Optional[dict] = None) -> dict[str, Any]:
    return {"state": "idle", "count": int(count), "detail": detail or {}}


def _busy(count: int, detail: Optional[dict] = None) -> dict[str, Any]:
    return {"state": "busy", "count": int(count), "detail": detail or {}}


def _unknown(reason: str, detail: Optional[dict] = None) -> dict[str, Any]:
    merged = {"reason": str(reason)}
    merged.update(detail or {})
    return {"state": "unknown", "count": None, "detail": merged}


def _coerce_work_entry(raw: Any) -> dict[str, Any]:
    """Validate one work entry; anything malformed becomes ``unknown``."""
    try:
        if not isinstance(raw, dict):
            return _unknown("malformed_work_entry")
        state = raw.get("state")
        if state not in _VALID_STATES:
            return _unknown("malformed_work_state", {"seen": state})
        count = raw.get("count")
        detail = raw.get("detail")
        if not isinstance(detail, dict):
            detail = {"seen_detail": detail} if detail is not None else {}
        if state == "unknown":
            # Unknown never carries a numeric claim; preserve the reason.
            return {"state": "unknown", "count": None, "detail": detail}
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            return _unknown("malformed_work_count", {"seen": count})
        if state == "idle" and count != 0:
            return _unknown("idle_must_be_zero", {"seen": count})
        if state == "busy" and count <= 0:
            return _unknown("busy_must_be_positive", {"seen": count})
        return {"state": state, "count": count, "detail": detail}
    except Exception as exc:  # noqa: BLE001 - never raise from a read-only probe
        logger.debug("work entry coercion failed: %s", exc)
        return _unknown("work_coercion_failed")


def summarize_overall(work: dict[str, Any]) -> str:
    """``busy`` if any busy, else ``unknown`` if any unknown/missing, else ``idle``."""
    try:
        states = []
        for key in WORK_CLASSES:
            entry = work.get(key)
            if not isinstance(entry, dict) or entry.get("state") not in _VALID_STATES:
                return "unknown"
            states.append(entry["state"])
        if any(s == "busy" for s in states):
            return "busy"
        if any(s == "unknown" for s in states):
            return "unknown"
        return "idle"
    except Exception:  # noqa: BLE001
        return "unknown"


def admission_boundary(valid_for_s: float = ADMISSION_VALID_FOR_S) -> dict[str, Any]:
    """Producer-defined bounded admission information."""
    return {
        "type": "read-only-observation",
        "valid_for_s": float(valid_for_s),
        "does_not_block_arrivals": True,
        "requires_recheck_before_mutation": True,
        "mechanism": "re-query `hermes update --plan --json` immediately before any mutation; snapshot alone grants no lease",
        "note": (
            "Read-only admission observation/handshake, not a claim that later "
            "arrivals are impossible without the producer-defined boundary."
        ),
    }


def build_admission_document(
    runtimes: list[dict[str, Any]],
    work: dict[str, Any],
    *,
    computed_at: Optional[float] = None,
    valid_for_s: float = ADMISSION_VALID_FOR_S,
) -> dict[str, Any]:
    """Assemble the ``admission`` document (pure; no I/O). Never raises."""
    try:
        now = time.time() if computed_at is None else float(computed_at)
    except Exception:  # noqa: BLE001
        now = time.time()
    coerced: dict[str, Any] = {}
    for key in WORK_CLASSES:
        coerced[key] = _coerce_work_entry(work.get(key) if isinstance(work, dict) else None)
    overall = summarize_overall(coerced)
    valid_until = _utc_now_iso(now + float(valid_for_s))
    admissible = overall == "idle"
    reason = (
        "all work classes proven idle on a fresh observation"
        if admissible
        else f"overall={overall}: re-query before mutating; snapshot alone blocks nothing"
    )
    safe_runtimes = runtimes if isinstance(runtimes, list) else []
    return {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "computed_at": _utc_now_iso(now),
        "valid_until": valid_until,
        "valid_for_s": float(valid_for_s),
        "overall": overall,
        "admissible": admissible,
        "reason": reason,
        "admission_boundary": admission_boundary(valid_for_s),
        "runtimes": safe_runtimes,
        "work": coerced,
    }


# --- read-only probes (never write, never drain, never quarantine) --------------


def _query_socket_admission(home: Path, timeout: float = 2.0) -> Optional[dict[str, Any]]:
    """Live gateway ``admission`` verb for ``home``; None when unavailable.

    Any failure (no socket, timeout, malformed, ``ok: false``) is None so the
    caller falls back fail-closed to ``unknown``.
    """
    try:
        from gateway.control_socket import query_gateway_control

        result = query_gateway_control(home, "admission", timeout=timeout)
        return result if isinstance(result, dict) else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("socket admission probe failed for %s: %s", home, exc)
        return None


def _query_socket_status(home: Path, timeout: float = 2.0) -> Optional[dict[str, Any]]:
    """Live gateway ``status`` verb for ``home``; None when unavailable."""
    try:
        from gateway.control_socket import query_gateway_control

        result = query_gateway_control(home, "status", timeout=timeout)
        return result if isinstance(result, dict) else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("socket status probe failed for %s: %s", home, exc)
        return None


def _read_status_file(home: Path) -> tuple[Optional[dict[str, Any]], str]:
    """Read ``gateway_state.json`` read-only; (record|None, reason)."""
    try:
        from gateway.status import _read_json_file

        record = _read_json_file(Path(home) / "gateway_state.json")
        if record is None:
            return None, "absent_or_unreadable"
        if not isinstance(record, dict):
            return None, "malformed_not_object"
        return record, "ok"
    except Exception as exc:  # noqa: BLE001
        logger.debug("status-file read failed for %s: %s", home, exc)
        return None, "unreadable"


def _heartbeat_age_s(record: Optional[dict[str, Any]]) -> Optional[int]:
    """Whole seconds since ``updated_at``; None when missing/unparseable."""
    try:
        from gateway.status import runtime_status_heartbeat_age_s

        return runtime_status_heartbeat_age_s(record)
    except Exception:  # noqa: BLE001
        return None


def _live_gateway_pid(home: Path) -> tuple[Optional[int], str]:
    """Verified live gateway PID for ``home`` read-only; (pid|None, reason)."""
    try:
        from gateway.status import live_gateway_pid_for_home

        pid = live_gateway_pid_for_home(Path(home))
        return (pid, "live" if pid is not None else "not_running")
    except Exception as exc:  # noqa: BLE001
        logger.debug("liveness probe failed for %s: %s", home, exc)
        return None, "unknown_liveness"


def _read_ledger_readonly() -> Optional[list[dict]]:
    """Spawn-ledger rows read-only via ``_read_ledger`` (no quarantine).

    Returns None on corrupt/unreadable (caller reports ``unknown``); [] when
    the ledger file is absent/empty. Never moves the corrupt file aside.
    """
    try:
        from hermes_cli.process_identity import _ledger_path, _read_ledger

        return _read_ledger(_ledger_path())
    except Exception as exc:  # noqa: BLE001
        logger.debug("ledger read failed: %s", exc)
        return None


def _ledger_liveness(pid: Any, create_time: Any) -> Optional[bool]:
    """True/False when provable; None when psutil cannot say."""
    try:
        from hermes_cli.process_identity import _pid_alive_matches

        if not isinstance(pid, int) or pid <= 0:
            return False
        return _pid_alive_matches(pid, create_time)
    except Exception:  # noqa: BLE001
        return None


def _read_cron_executions_readonly(home: Path) -> Optional[list[dict[str, Any]]]:
    """Live ``claimed``/``running`` cron rows for ``home`` read-only.

    Opens the profile's ``cron/executions.db`` with ``mode=ro`` (no schema
    init, no dir creation). Returns None on unreadable/corrupt (caller
    reports ``unknown``); [] when the ledger is absent or has no live rows.
    Never rewrites rows (unlike recovery) and never creates the DB.
    """
    import sqlite3

    path = Path(home) / "cron" / "executions.db"
    try:
        if not path.is_file():
            return []
    except Exception:  # noqa: BLE001
        return None
    try:
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT job_id, status, pid, process_started_at, claimed_at "
                "FROM executions WHERE status IN ('claimed','running')"
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - missing table, locked, corrupt
        logger.debug("cron executions read failed for %s: %s", home, exc)
        return None
    live: list[dict[str, Any]] = []
    try:
        from gateway.status import _pid_exists, get_process_start_time, start_time_fingerprints_match
    except Exception:  # noqa: BLE001
        return None
    for row in rows:
        try:
            item = dict(row)
        except Exception:  # noqa: BLE001
            return None
        try:
            pid = int(item.get("pid") or 0)
        except (TypeError, ValueError):
            return None
        if pid <= 0 or not _pid_exists(pid):
            continue
        try:
            recorded = item.get("process_started_at")
            current = get_process_start_time(pid)
            if recorded is not None and current is not None:
                if not start_time_fingerprints_match(recorded, current):
                    continue  # recycled PID: not this owner
            # Unreadable fingerprint on either side: cannot prove death, keep it.
            live.append(
                {
                    "job_id": str(item.get("job_id") or ""),
                    "status": str(item.get("status") or ""),
                    "pid": pid,
                    "claimed_at": item.get("claimed_at"),
                }
            )
        except Exception:  # noqa: BLE001
            return None
    return live


def _desktop_heartbeat(home: Path) -> tuple[Optional[float], str]:
    """Seconds since a dashboard/Desktop client was last seen for ``home``.

    (age|None, reason). Missing marker is idle (nobody attached); unreadable
    is unknown (fail-closed); future mtimes clamp to 0.
    """
    import os

    marker = Path(home) / "state" / "dashboard_clients.heartbeat"
    try:
        if not marker.exists():
            # Fall back to the process-home marker only for the default home;
            # named profiles never borrow another home's attachment signal.
            return None, "no_marker"
        age = max(0.0, time.time() - os.stat(marker).st_mtime)
        return age, "ok"
    except OSError as exc:
        logger.debug("desktop heartbeat unreadable for %s: %s", home, exc)
        return None, "unreadable"


def _profile_homes() -> list[tuple[str, Path]]:
    try:
        from hermes_cli.update_receipt import _profile_homes as _homes

        homes = _homes()
        return [(str(n), Path(h)) for n, h in homes or []]
    except Exception as exc:  # noqa: BLE001
        logger.debug("profile enumeration failed: %s", exc)
        return []


def _runtime_entry(runtime: Any) -> dict[str, Any]:
    """Fresh per-runtime admission entry; never raises, never writes."""
    try:
        kind = str(getattr(runtime, "kind", "") or "")
        profile = str(getattr(runtime, "profile", "") or "")
        pid = getattr(runtime, "pid", None)
        supervisor = str(getattr(runtime, "supervisor", "") or "")
        restart_via = str(getattr(runtime, "restart_via", "") or "")
        code_sha = getattr(runtime, "code_sha", None)
        code_version = getattr(runtime, "code_version", None)
    except Exception:  # noqa: BLE001
        return {
            "kind": "unknown", "profile": "unknown", "pid": None,
            "supervisor": "unknown", "restart_via": "manual",
            "live": None, "state": "unknown",
            "source": "unknown", "heartbeat_age_s": None, "stale": None,
            "detail": {"reason": "malformed_runtime_record"},
        }
    # Resolve the home for this profile (scoped to known profiles only).
    home: Optional[Path] = None
    try:
        for name, path in _profile_homes():
            if name == profile:
                home = path
                break
    except Exception:  # noqa: BLE001
        home = None

    live: Optional[bool] = None
    live_pid: Optional[int] = None
    source = "unknown"
    heartbeat_age: Optional[int] = None
    stale: Optional[bool] = None
    state = "unknown"
    detail: dict[str, Any] = {}

    try:
        if kind == "gateway" and home is not None:
            live_pid, live_reason = _live_gateway_pid(home)
            if live_reason == "unknown_liveness":
                live = None
                detail["reason"] = "liveness_unreadable"
            else:
                live = live_pid is not None
                if live and live_pid != pid:
                    detail["live_pid"] = live_pid
                    detail["planned_pid"] = pid
            # Freshness from the status file (read-only).
            record, read_reason = _read_status_file(home)
            if record is None:
                heartbeat_age, stale = None, None
                detail["status_file"] = read_reason
            else:
                heartbeat_age = _heartbeat_age_s(record)
                stale = None if heartbeat_age is None else bool(heartbeat_age > RUNTIME_STALE_TTL_S)
                try:
                    active = int(record.get("active_agents", 0))
                except (TypeError, ValueError):
                    active = None
                detail["active_agents_file"] = record.get("active_agents")
                if heartbeat_age is None:
                    detail["heartbeat"] = "unparseable"
            # Live socket admission/status is fresher than the file.
            socket_adm = _query_socket_admission(home) if home is not None else None
            socket_status = _query_socket_status(home) if socket_adm is None and home is not None else None
            if socket_adm is not None:
                source = "control_socket"
                try:
                    works = socket_adm.get("work") if isinstance(socket_adm, dict) else None
                    if isinstance(works, dict):
                        states = [str(v.get("state")) for v in works.values() if isinstance(v, dict)]
                        if any(s == "busy" for s in states):
                            state = "busy"
                        elif any(s not in ("idle", "busy", "unknown") for s in states) or any(
                            s == "unknown" for s in states
                        ):
                            state = "unknown"
                            detail["reason"] = "socket_reports_unknown"
                        else:
                            state = "idle" if live else "idle"
                    else:
                        state = "unknown"
                        detail["reason"] = "malformed_socket_admission"
                except Exception:  # noqa: BLE001
                    state = "unknown"
                    detail["reason"] = "socket_admission_unreadable"
            elif socket_status is not None:
                source = "control_socket"
                try:
                    from gateway.status import parse_active_agents

                    active = parse_active_agents(socket_status.get("active_agents"))
                    running = bool(socket_status.get("gateway_running", True))
                    gstate = str(socket_status.get("gateway_state") or "")
                    detail["active_agents_live"] = active
                    detail["gateway_state_live"] = gstate or None
                    if not live:
                        state = "idle"
                    elif not running or gstate in ("stopped", "startup_failed"):
                        state = "idle"
                    elif active > 0 or gstate == "draining":
                        state = "busy"
                    else:
                        state = "idle"
                except Exception:  # noqa: BLE001
                    state = "unknown"
                    detail["reason"] = "socket_status_unreadable"
            else:
                source = "runtime_status" if record is not None else "none"
                if live is None:
                    state = "unknown"
                elif not live:
                    state = "idle"
                elif record is None:
                    state = "unknown"
                    detail.setdefault("reason", "no_fresh_source")
                else:
                    try:
                        from gateway.status import parse_active_agents

                        active = parse_active_agents(record.get("active_agents"))
                    except Exception:  # noqa: BLE001
                        state = "unknown"
                        detail["reason"] = "malformed_active_agents"
                    else:
                        if heartbeat_age is None or stale is None or stale:
                            state = "unknown"
                            detail.setdefault("reason", "stale_or_missing_heartbeat")
                        elif active > 0:
                            state = "busy"
                        else:
                            state = "idle"
        elif kind in ("serve", "dashboard") and isinstance(pid, int):
            source = "ledger"
            alive = _ledger_liveness(pid, None)
            # Ledger rows carry create_time; match it when the plan has one.
            try:
                create_time = (getattr(runtime, "detail", {}) or {}).get("create_time")
            except Exception:  # noqa: BLE001
                create_time = None
            if create_time is not None:
                alive = _ledger_liveness(pid, create_time)
            if alive is None:
                live = None
                state = "unknown"
                detail["reason"] = "backend_liveness_unprovable"
            else:
                live = bool(alive)
                if not live:
                    state = "idle"
                else:
                    # A live backend's in-flight work is not observable
                    # out-of-process: fail closed, never idle.
                    state = "unknown"
                    detail["reason"] = "live_backend_work_unobservable"
            heartbeat_age, stale = None, None
        else:
            live = None if pid is None else None
            state = "unknown"
            source = "unknown"
            detail["reason"] = "unsupported_runtime_kind"
    except Exception as exc:  # noqa: BLE001
        logger.debug("runtime entry failed: %s", exc)
        state = "unknown"
        detail["reason"] = "runtime_probe_failed"

    return {
        "kind": kind or "unknown", "profile": profile or "unknown", "pid": pid,
        "supervisor": supervisor or "unknown", "restart_via": restart_via or "manual",
        "code_sha": code_sha, "code_version": code_version,
        "live": live, "state": state if state in _VALID_STATES else "unknown",
        "source": source, "heartbeat_age_s": heartbeat_age, "stale": stale,
        "detail": detail,
    }


def _collect_gateway_work(
    homes: list[tuple[str, Path]],
) -> dict[str, dict[str, Any]]:
    """Gateway-owned work from live socket admissions + file aggregates.

    Prefers the fresh ``admission`` verb; falls back to the live ``status``
    verb; falls back to the status file only when fresh. Anything else is
    ``unknown`` (a live gateway with no fresh source is never ``idle``).
    """
    socket_works: dict[str, dict] = {}
    socket_statuses: dict[str, dict] = {}
    file_actives: dict[str, tuple[Optional[int], Optional[int]]] = {}
    live_map: dict[str, Optional[bool]] = {}
    for _profile, home in homes:
        adm = _query_socket_admission(home)
        if isinstance(adm, dict) and isinstance(adm.get("work"), dict):
            socket_works[_profile] = adm["work"]
            live_map[_profile] = True
            continue
        st = _query_socket_status(home)
        if isinstance(st, dict):
            socket_statuses[_profile] = st
        pid, reason = _live_gateway_pid(home)
        live_map[_profile] = None if reason == "unknown_liveness" else (pid is not None)
        record, _ = _read_status_file(home)
        age = _heartbeat_age_s(record)
        try:
            from gateway.status import parse_active_agents

            active = parse_active_agents(record.get("active_agents")) if isinstance(record, dict) else None
        except Exception:  # noqa: BLE001
            active = None
        file_actives[_profile] = (active, age)

    any_live = any(v is True for v in live_map.values())
    any_unknown_live = any(v is None for v in live_map.values())
    no_gateway = not any_live and not any_unknown_live

    def _merged_state(key: str) -> dict[str, Any]:
        # Fresh socket admission wins per class.
        for work in socket_works.values():
            entry = work.get(key)
            if isinstance(entry, dict) and entry.get("state") in _VALID_STATES:
                if entry.get("state") == "busy":
                    return _busy(
                        entry.get("count") if isinstance(entry.get("count"), int) else 1,
                        {"source": "control_socket", **(entry.get("detail") or {})},
                    )
        # If every socket admission that names this class says idle, keep
        # looking: another home without the verb may still hold work.
        socket_states = [
            str(w.get(key, {}).get("state"))
            for w in socket_works.values()
            if isinstance(w.get(key), dict)
        ]
        all_socket_idle = bool(socket_works) and socket_states and all(s == "idle" for s in socket_states)
        # Live aggregate from socket status or fresh files.
        agg_busy = False
        agg_proven_zero = True
        agg_unknown = False
        for profile, _home in homes:
            if profile in socket_works:
                continue  # covered by the verb above
            st = socket_statuses.get(profile)
            if isinstance(st, dict):
                try:
                    from gateway.status import parse_active_agents

                    if parse_active_agents(st.get("active_agents")) > 0:
                        agg_busy = True
                    continue  # live answer: no file fallback needed
                except Exception:  # noqa: BLE001
                    agg_unknown = True
                    agg_proven_zero = False
                    continue
            live = live_map.get(profile)
            active, age = file_actives.get(profile, (None, None))
            if live is None:
                agg_unknown = True
                agg_proven_zero = False
            elif live:
                if active is None or age is None or age > RUNTIME_STALE_TTL_S:
                    agg_unknown = True
                    agg_proven_zero = False
                elif active > 0:
                    agg_busy = True
                    agg_proven_zero = False
                # else: this home proves zero; keep checking others
            # not live: contributes zero, no uncertainty
        if agg_busy:
            # Aggregate proves work, but without the verb we cannot isolate
            # the class: fail closed to unknown for the class (the overall
            # still blocks via the runtime entries + reason).
            return _unknown(f"{key}_aggregate_busy_unattributed", {"source": "aggregate"})
        if no_gateway and not socket_works and not socket_statuses:
            # No live gateway anywhere: gateway-owned work cannot exist.
            # Durable cron/external rows are resolved by their own probes.
            if key in ("foreground_turns", "api_runs", "deferred_workers",
                       "background_delegations", "background_processes"):
                return _idle(0, {"source": "no_live_gateway"})
        if all_socket_idle and agg_proven_zero and not agg_unknown:
            return _idle(0, {"source": "control_socket" if socket_works else "fresh_status"})
        return _unknown(f"{key}_unprovable", {"source": "fail_closed"})

    work: dict[str, dict[str, Any]] = {}
    for key in ("foreground_turns", "api_runs", "deferred_workers",
                "background_delegations", "background_processes"):
        work[key] = _merged_state(key)

    # Cron + external workers combine socket state with the durable ledger.
    work["cron_jobs"] = _cron_work(homes, socket_works, live_map, file_actives, socket_statuses)
    work["external_workers"] = _external_work(homes, socket_works)
    return work


def _cron_work(
    homes: list[tuple[str, Path]],
    socket_works: dict[str, dict],
    live_map: dict[str, Optional[bool]],
    file_actives: dict[str, tuple[Optional[int], Optional[int]]],
    socket_statuses: dict[str, dict],
) -> dict[str, Any]:
    # Socket verb first (includes restart-safe worker pids).
    for work in socket_works.values():
        entry = work.get("cron_jobs")
        if isinstance(entry, dict) and entry.get("state") == "busy":
            count = entry.get("count")
            return _busy(count if isinstance(count, int) else 1,
                         {"source": "control_socket", **(entry.get("detail") or {})})
    # Durable ledger: live claimed/running rows are busy (restart-safe
    # workers included via their pids).
    try:
        unprovable = False
        live_rows: list[dict] = []
        for _profile, home in homes:
            rows = _read_cron_executions_readonly(home)
            if rows is None:
                unprovable = True
            else:
                live_rows.extend(rows)
        if live_rows:
            return _busy(len(live_rows), {
                "source": "durable_executions",
                "jobs": [
                    {"job_id": r.get("job_id"), "worker_pid": r.get("pid"),
                     "restart_safe_worker": True}
                    for r in live_rows[:20]
                ],
            })
        # No durable rows: idle only when no live gateway could hold
        # in-memory cron work unobservable from here.
        any_live = any(v is True for v in live_map.values())
        any_unknown = any(v is None for v in live_map.values())
        # Socket-status/file aggregates already searched by the caller for
        # busy; here only decide idle vs unknown for the quiet case.
        if unprovable:
            return _unknown("cron_ledger_unreadable", {"source": "fail_closed"})
        if any_live or any_unknown:
            # A live gateway without the admission verb may run cron work
            # invisible to the ledger yet (claim before row commit).
            idle_via_socket = bool(socket_works) and all(
                str(w.get("cron_jobs", {}).get("state")) == "idle"
                for w in socket_works.values()
                if isinstance(w.get("cron_jobs"), dict)
            )
            if idle_via_socket and not any_unknown:
                return _idle(0, {"source": "control_socket"})
            return _unknown("cron_inmemory_unobservable", {"source": "fail_closed"})
        return _idle(0, {"source": "no_live_gateway_no_rows"})
    except Exception as exc:  # noqa: BLE001
        logger.debug("cron work probe failed: %s", exc)
        return _unknown("cron_probe_failed")


def _external_work(
    homes: list[tuple[str, Path]], socket_works: dict[str, dict]
) -> dict[str, Any]:
    for work in socket_works.values():
        entry = work.get("external_workers")
        if isinstance(entry, dict) and entry.get("state") == "busy":
            count = entry.get("count")
            return _busy(count if isinstance(count, int) else 1,
                         {"source": "control_socket", **(entry.get("detail") or {})})
    # Durable claimed/running rows with a live owner + pending process
    # watchers are the CLI-visible completion state.
    try:
        live_rows: list[dict] = []
        unreadable = False
        for _profile, home in homes:
            rows = _read_cron_executions_readonly(home)
            if rows is None:
                unreadable = True
            else:
                live_rows.extend(rows)
        if live_rows:
            return _busy(len(live_rows), {
                "source": "durable_executions",
                "pending_completion": len(live_rows),
            })
        if unreadable:
            return _unknown("external_completion_unreadable", {"source": "fail_closed"})
        # Pending process-completion watchers live in the gateway process;
        # without the admission verb they are unobservable when live.
        any_socket_idle = bool(socket_works) and all(
            str(w.get("external_workers", {}).get("state")) == "idle"
            for w in socket_works.values()
            if isinstance(w.get("external_workers"), dict)
        )
        if any_socket_idle:
            return _idle(0, {"source": "control_socket"})
        # Fall back to gateway liveness: unknown when a gateway may hold
        # watchers, idle only with no live gateway at all.
        any_live = False
        any_unknown = False
        for _profile, home in homes:
            _pid, reason = _live_gateway_pid(home)
            if reason == "unknown_liveness":
                any_unknown = True
            elif _pid is not None:
                any_live = True
        if any_live or any_unknown:
            return _unknown("external_watchers_unobservable", {"source": "fail_closed"})
        return _idle(0, {"source": "no_live_gateway"})
    except Exception as exc:  # noqa: BLE001
        logger.debug("external work probe failed: %s", exc)
        return _unknown("external_probe_failed")


def _backend_work(plan_runtimes: list) -> dict[str, Any]:
    """Serve/dashboard in-flight work (CLI cannot observe turns inside them)."""
    try:
        entries = _read_ledger_readonly()
        if entries is None:
            return _unknown("spawn_ledger_corrupt_or_unreadable")
        live_backends: list[dict] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("purpose") not in ("serve", "dashboard"):
                continue
            pid = entry.get("pid")
            alive = _ledger_liveness(pid, entry.get("create_time"))
            if alive is None:
                return _unknown("backend_liveness_unprovable", {"pid": pid})
            if alive:
                live_backends.append(entry)
        # Desktop attachment signal (read-only mtime; never writes).
        desktop: dict[str, Any] = {"attached": False, "heartbeat_age_s": None}
        try:
            for _profile, home in _profile_homes():
                age, reason = _desktop_heartbeat(home)
                if reason == "unreadable":
                    return _unknown("desktop_heartbeat_unreadable")
                if age is not None and age <= DESKTOP_ATTACHED_WINDOW_S:
                    desktop = {"attached": True, "heartbeat_age_s": round(age, 1)}
                    break
        except Exception:  # noqa: BLE001
            return _unknown("desktop_probe_failed")
        if live_backends:
            return _unknown("live_backend_work_unobservable", {
                "live_backends": len(live_backends),
                "desktop": desktop,
            })
        if desktop.get("attached"):
            # A Desktop client is attached with no ledger backend found:
            # fail closed — work may arrive at any moment.
            return _unknown("desktop_attached_no_backend_row", {"desktop": desktop})
        return _idle(0, {"live_backends": 0, "desktop": desktop})
    except Exception as exc:  # noqa: BLE001
        logger.debug("backend work probe failed: %s", exc)
        return _unknown("backend_probe_failed")


def collect_admission_snapshot(plan: Any = None) -> dict[str, Any]:
    """Collect the fresh admission snapshot (read-only; never raises).

    Never writes, quarantines, drains, or repairs: every collector degrades
    to ``unknown`` independently.
    """
    computed_at = time.time()
    try:
        if plan is None:
            from hermes_cli.update_inventory import collect_runtime_inventory

            plan = collect_runtime_inventory()
        runtimes_in = list(getattr(plan, "runtimes", []) or [])
    except Exception as exc:  # noqa: BLE001
        logger.debug("inventory for admission failed: %s", exc)
        runtimes_in = []
    try:
        runtime_entries = [_runtime_entry(r) for r in runtimes_in]
    except Exception:  # noqa: BLE001
        runtime_entries = [_runtime_entry(None)]
    try:
        homes = _profile_homes()
    except Exception:  # noqa: BLE001
        homes = []
    try:
        gateway_work = _collect_gateway_work(homes)
    except Exception as exc:  # noqa: BLE001
        logger.debug("gateway work collection failed: %s", exc)
        gateway_work = {k: _unknown("gateway_collection_failed") for k in WORK_CLASSES}
    try:
        backend = _backend_work(runtimes_in)
    except Exception:  # noqa: BLE001
        backend = _unknown("backend_collection_failed")
    work: dict[str, Any] = dict(gateway_work)
    work["backend_agent_work"] = _coerce_work_entry(backend)
    # Ensure every covered class is present (missing -> unknown, never idle).
    for key in WORK_CLASSES:
        if key not in work:
            work[key] = _unknown("work_class_missing")
        else:
            work[key] = _coerce_work_entry(work[key])
    # A live runtime whose own entry is busy forces the matching class busy
    # when the class would otherwise claim idle on a stale aggregate.
    try:
        for entry in runtime_entries:
            if entry.get("state") == "busy" and entry.get("kind") == "gateway":
                for key in ("foreground_turns", "api_runs"):
                    if work.get(key, {}).get("state") == "idle":
                        work[key] = _unknown("runtime_busy_unattributed", {
                            "profile": entry.get("profile"),
                        })
    except Exception:  # noqa: BLE001
        pass
    return build_admission_document(runtime_entries, work, computed_at=computed_at)


def print_admission_summary(admission: dict[str, Any]) -> None:
    """Human-readable admission lines for ``hermes update --plan``."""
    try:
        work = admission.get("work") if isinstance(admission, dict) else None
        overall = admission.get("overall") if isinstance(admission, dict) else "unknown"
        admissible = bool(admission.get("admissible")) if isinstance(admission, dict) else False
        print(f"  Admission: {overall} ({'admissible' if admissible else 'not admissible'})")
        if isinstance(work, dict):
            for key in WORK_CLASSES:
                entry = work.get(key) if isinstance(work.get(key), dict) else None
                state = entry.get("state") if entry else "unknown"
                count = entry.get("count") if entry else None
                suffix = "" if count is None else f" x{count}"
                print(f"    • {key}: {state}{suffix}")
        boundary = (admission.get("admission_boundary") or {}) if isinstance(admission, dict) else {}
        print(f"    Valid for {boundary.get('valid_for_s', ADMISSION_VALID_FOR_S):.0f}s; "
              f"re-query before mutating (snapshot blocks nothing).")
    except Exception as exc:  # noqa: BLE001
        logger.debug("admission summary failed: %s", exc)


def plan_and_admission_payload(plan: Any, admission: dict[str, Any]) -> dict[str, Any]:
    """Combined ``--plan --json`` document (JSON-serializable)."""
    try:
        plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else {}
    except Exception:  # noqa: BLE001
        plan_dict = {}
    try:
        json.dumps(admission)
        adm = admission
    except Exception:  # noqa: BLE001
        adm = build_admission_document([], {})
    return {"plan": plan_dict, "admission": adm}
