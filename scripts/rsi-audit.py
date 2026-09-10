#!/usr/bin/env python3
"""Audit fleet chats and failed durable runs since the previous RSI tick.

Writes ~/.hermes/rsi/audit/latest.json. Does not stamp last_tick.json
(RSI stamps that after the tick).
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HOME = Path.home()
STORE = HOME / ".hermes" / "rsi"
LAST = STORE / "last_tick.json"
JOBS = HOME / ".hermes" / "cron" / "jobs.json"
EXEC_DB = HOME / ".hermes" / "cron" / "executions.db"
KANBAN = HOME / ".hermes" / "kanban.db"
PROFILES_DIR = HOME / ".hermes" / "profiles"
CONTRACT = STORE / "contract.yaml"

DONE_RE = re.compile(r"\b(done|completed|all (?:set|good)|nothing to (?:do|report))\b", re.I)
TASK_ID_RE = re.compile(r"\bt_[a-z0-9]+\b", re.I)
CRON_SESSION_RE = re.compile(r"^cron_([^_]+)_")

CRON_TO_PROFILE = {
    "x-ops": "x",
    "x-daily-drafts": "x",
    "yuki-digest": "yuki",
    "product-inbox": "product",
    "gh-pr-scout": "reviewer",
    "eng-completion": "product",
    "rsi-loop": "rsi",
}

FAIL_END = {
    "error",
    "timeout",
    "killed",
    "max_turns",
    "budget",
    "exception",
    "failed",
    "cron_incomplete_no_output",
    "orphaned_compression",
    "session_persistence_failed",
}
FAIL_END_PREFIXES = (
    "error_",
    "max_iterations_reached",
    "repeated_outer_errors",
)
NONTERMINAL_BOUNDARY_ENDS = {
    "adopted_by_profile",
    "branched",
    "compression",
    "resumed_other",
    "session_reset",
    "session_switch",
    "superseded_by_repair",
}
FAIL_STATUSES = {
    "blocked_config",
    "error",
    "failed",
    "gave_up",
    "interrupted",
    "killed",
    "spawn_failed",
    "timeout",
    "timed_out",
}
SUCCESS_STATUSES = {"completed", "done", "no_change", "ok", "skipped", "success"}
NONTERMINAL_STATUSES = {"claimed", "pending", "queued", "running", "scheduled"}
EXECUTION_STATUS_TOOLS = {"execute_code", "process", "terminal"}


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def fleet() -> list[str]:
    """Every installed profile, contract-listed or not.

    The interview contract requires a structured audit slice for every
    installed profile: a missing slice is a mandatory validation failure for
    that profile. Installed = the default instance plus every directory under
    ``~/.hermes/profiles/``; contract names without an install are skipped.
    Contract-only ordering is preserved first, then any unlisted installs
    alphabetically, so output stays deterministic.
    """
    installed = ["default"]
    if PROFILES_DIR.is_dir():
        installed.extend(
            sorted(
                entry.name
                for entry in PROFILES_DIR.iterdir()
                if entry.is_dir() and not entry.name.startswith(".")
            )
        )
    contracted = [str(n) for n in (load_yaml(CONTRACT).get("fleet") or [])]
    ordered = [n for n in contracted if n in installed]
    ordered.extend(n for n in installed if n not in contracted)
    return ordered or installed


def since_unix() -> float:
    if LAST.exists():
        try:
            return float(json.loads(LAST.read_text(encoding="utf-8")).get("unix") or 0)
        except Exception:
            pass
    if JOBS.exists():
        try:
            for job in json.loads(JOBS.read_text(encoding="utf-8")).get("jobs") or []:
                if job.get("name") == "rsi-loop" and job.get("last_run_at"):
                    return datetime.fromisoformat(str(job["last_run_at"])).timestamp()
        except Exception:
            pass
    return (datetime.now(timezone.utc) - timedelta(hours=4)).timestamp()


def db_path(profile: str) -> Path | None:
    if profile == "default":
        path = HOME / ".hermes" / "state.db"
    else:
        path = HOME / ".hermes" / "profiles" / profile / "state.db"
    return path if path.exists() else None


def _json_object(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _status_failure(value: Any) -> bool:
    return str(value or "").strip().lower() in FAIL_STATUSES


def _tool_error_reason(
    tool_name: str,
    content: str,
    effect_disposition: Any = None,
) -> str | None:
    """Return a structural tool-error marker, never a lexical content match."""
    if str(effect_disposition or "").lower() == "unknown":
        return "effect_disposition=unknown"

    payload = _json_object(content)
    if payload is None:
        # Hermes' exception wrapper is a stable transport envelope. Arbitrary
        # non-JSON tool prose is data and must not be interpreted lexically.
        if content.lstrip().startswith("Error executing tool "):
            return "exception"
        return None

    if payload.get("success") is False or payload.get("ok") is False:
        return "reported_failure"

    if tool_name in EXECUTION_STATUS_TOOLS:
        exit_code = payload.get("exit_code")
        if isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0:
            return f"exit_code={exit_code}"
        if _status_failure(payload.get("status")):
            return f"status={str(payload['status']).lower()}"
        if _status_failure(payload.get("state")):
            return f"state={str(payload['state']).lower()}"
        if _status_failure(payload.get("outcome")):
            return f"outcome={str(payload['outcome']).lower()}"

    # ``tool_error()`` is Hermes' canonical generic failure envelope. A
    # declared successful result wins over incidental error-shaped domain data.
    if payload.get("success") is not True and payload.get("ok") is not True and payload.get("error"):
        return "error"

    return None


def _tool_call_parts(raw: Any) -> tuple[str, dict]:
    call = raw if isinstance(raw, dict) else {}
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(function.get("name") or call.get("name") or "")
    arguments = function.get("arguments", call.get("arguments"))
    parsed = _json_object(arguments) or (arguments if isinstance(arguments, dict) else {})
    return name, parsed


def _message_evidence(con: sqlite3.Connection, session_id: str, since: float) -> tuple[list[str], bool, list[str]]:
    rows = con.execute(
        """
        SELECT role, content, tool_calls, tool_name, effect_disposition
        FROM messages
        WHERE session_id = ? AND timestamp >= ?
          AND role IN ('assistant', 'tool', 'user')
        ORDER BY id ASC
        """,
        (session_id, since),
    ).fetchall()

    hits: list[str] = []
    task_ids: list[str] = []
    has_final_response = False
    for role, content, tool_calls, tool_name, effect_disposition in rows:
        text = str(content or "")
        if role == "user":
            has_final_response = False
            task_ids.extend(match.group(0).lower() for match in TASK_ID_RE.finditer(text))
            continue
        if role == "tool":
            has_final_response = False
            reason = _tool_error_reason(
                str(tool_name or "unknown"),
                text,
                effect_disposition,
            )
            if reason:
                hits.append(f"tool:{tool_name or 'unknown'}:{reason}")
            payload = _json_object(text)
            if (
                str(tool_name or "") == "clarify"
                and payload is not None
                and payload.get("timed_out") is True
            ):
                hits.append("lifecycle:needs_input")
            if (
                str(tool_name or "") == "kanban_block"
                and payload is not None
                and str(payload.get("block_kind") or "").lower() == "needs_input"
            ):
                hits.append("lifecycle:needs_input")
            continue

        calls: list[Any] = []
        if tool_calls:
            try:
                loaded = json.loads(tool_calls) if isinstance(tool_calls, str) else tool_calls
                calls = loaded if isinstance(loaded, list) else []
            except (json.JSONDecodeError, TypeError):
                calls = []
        if calls:
            has_final_response = False
        elif text.strip():
            has_final_response = True
        for call in calls:
            name, arguments = _tool_call_parts(call)
            if name == "kanban_block" and str(arguments.get("kind") or "").lower() == "needs_input":
                hits.append("lifecycle:needs_input")

    return list(dict.fromkeys(hits)), has_final_response, list(dict.fromkeys(task_ids))


def _kanban_paths() -> list[Path]:
    paths = [KANBAN]
    boards = HOME / ".hermes" / "kanban" / "boards"
    if boards.is_dir():
        paths.extend(sorted(boards.glob("*/kanban.db")))
    return list(dict.fromkeys(path for path in paths if path.exists()))


def _kanban_evidence(
    session_id: str,
    task_ids: list[str],
    started_at: float,
) -> tuple[list[str], bool]:
    for path in _kanban_paths():
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            if task_ids:
                placeholders = ",".join("?" for _ in task_ids)
                tasks = con.execute(
                    f"""SELECT id, status, current_run_id, block_kind
                        FROM tasks WHERE id IN ({placeholders})""",
                    tuple(task_ids),
                ).fetchall()
            else:
                tasks = con.execute(
                    """SELECT id, status, current_run_id, block_kind
                       FROM tasks WHERE session_id = ?""",
                    (session_id,),
                ).fetchall()
            candidates = []
            for task in tasks:
                runs = con.execute(
                    """SELECT id, status, outcome, error, started_at, ended_at
                       FROM task_runs WHERE task_id = ? ORDER BY id DESC""",
                    (task[0],),
                ).fetchall()
                candidates.extend(
                    (
                        abs(float(run[4]) - float(started_at or 0)),
                        task,
                        run,
                    )
                    for run in runs
                    if run[4] is not None
                )
            con.close()
        except (sqlite3.Error, OSError):
            continue

        distance, task, run = min(
            candidates,
            default=(float("inf"), None, None),
            key=lambda item: item[0],
        )
        if distance > 300:
            run = None
            task = tasks[0] if len(tasks) == 1 and not task_ids else None
        if task is None:
            continue

        _, task_status, current_run_id, block_kind = task
        hits: list[str] = []
        status = str(task_status or "").lower()
        kind = str(block_kind or "").lower()
        run_is_current = bool(run and current_run_id == run[0])
        if run_is_current and status == "blocked" and kind == "needs_input":
            hits.append("lifecycle:needs_input")
        elif run_is_current and _status_failure(status):
            hits.append(f"kanban:status={status}")

        durable_success = False
        if run:
            _, run_status, outcome, error, _, _ = run
            run_status_l = str(run_status or "").lower()
            outcome_l = str(outcome or "").lower()
            if _status_failure(outcome_l):
                hits.append(f"kanban:outcome={outcome_l}")
            elif _status_failure(run_status_l):
                hits.append(f"kanban:status={run_status_l}")
            elif error:
                hits.append("kanban:error")
            durable_success = (
                not hits
                and (outcome_l in SUCCESS_STATUSES or run_status_l in SUCCESS_STATUSES)
            )
        return list(dict.fromkeys(hits)), durable_success
    return [], False


def _parse_iso_timestamp(value: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _cron_evidence(session_id: str, started_at: float) -> tuple[list[str], bool]:
    match = CRON_SESSION_RE.match(session_id)
    if not match or not EXEC_DB.exists():
        return [], False
    job_id = match.group(1)
    try:
        con = sqlite3.connect(f"file:{EXEC_DB}?mode=ro", uri=True)
        rows = con.execute(
            """SELECT status, error, started_at, finished_at
               FROM executions WHERE job_id = ? ORDER BY started_at DESC LIMIT 20""",
            (job_id,),
        ).fetchall()
        con.close()
    except (sqlite3.Error, OSError):
        return [], False

    candidates = []
    for row in rows:
        row_started = _parse_iso_timestamp(row[2])
        if row_started is not None:
            candidates.append((abs(row_started - float(started_at or 0)), row))
    if not candidates:
        return [], False
    distance, row = min(candidates, key=lambda item: item[0])
    if distance > 300:
        return [], False

    status, error, _, finished_at = row
    status_l = str(status or "").lower()
    if _status_failure(status_l) or error:
        return [f"cron:status={status_l or 'error'}"], False
    return [], status_l in SUCCESS_STATUSES and bool(finished_at)


def _end_reason_failed(end_reason: Any) -> bool:
    reason = str(end_reason or "").lower()
    return reason in FAIL_END or reason.startswith(FAIL_END_PREFIXES)


def claimed_ok(con: sqlite3.Connection, session_id: str, since: float) -> bool:
    rows = con.execute(
        """
        SELECT content FROM messages
        WHERE session_id = ? AND timestamp >= ? AND role = 'assistant'
        ORDER BY id DESC LIMIT 3
        """,
        (session_id, since),
    ).fetchall()
    return any(DONE_RE.search(row[0] or "") for row in rows)


def scan_sessions(profile: str, since: float) -> list[dict]:
    found: list[dict] = []
    paths: list[tuple[str, Path]] = []
    own = db_path(profile)
    if own:
        paths.append(("own", own))
    default = db_path("default")
    if default and default not in [path for _, path in paths]:
        paths.append(("default", default))

    for origin, path in paths:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        if origin == "default":
            rows = con.execute(
                """
                SELECT id, source, title, end_reason, last_activity_description,
                       last_activity_at, started_at, ended_at
                FROM sessions
                WHERE last_activity_at >= ?
                  AND (profile_name = ? OR (profile_name IS NULL AND source IN ('cron','kanban')))
                ORDER BY last_activity_at DESC
                """,
                (since, profile),
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT id, source, title, end_reason, last_activity_description,
                       last_activity_at, started_at, ended_at
                FROM sessions
                WHERE last_activity_at >= ?
                ORDER BY last_activity_at DESC
                """,
                (since,),
            ).fetchall()

        for sid, source, title, end_reason, last_desc, last_at, started, ended_at in rows:
            if origin == "default" and source == "cron":
                title_l = (title or "").lower()
                mapped = next(
                    (
                        owner
                        for job, owner in CRON_TO_PROFILE.items()
                        if job in title_l or job.replace("-", " ") in title_l
                    ),
                    None,
                )
                if mapped and mapped != profile:
                    continue
                if mapped is None and profile != "default":
                    continue

            fail_hits, has_final_response, task_ids = _message_evidence(con, sid, since)
            title_task_ids = [match.group(0).lower() for match in TASK_ID_RE.finditer(title or "")]
            task_ids = list(dict.fromkeys([*task_ids, *title_task_ids]))
            durable_success = False
            if source == "kanban":
                durable_hits, durable_success = _kanban_evidence(
                    sid,
                    task_ids,
                    float(started or 0),
                )
                fail_hits.extend(durable_hits)
            elif source == "cron":
                durable_hits, durable_success = _cron_evidence(sid, float(started or 0))
                fail_hits.extend(durable_hits)

            end_reason_l = str(end_reason or "").lower()
            if _end_reason_failed(end_reason):
                fail_hits.append(f"session:end_reason={end_reason_l}")
            if (
                ended_at is not None
                and not has_final_response
                and not durable_success
                and end_reason_l not in NONTERMINAL_BOUNDARY_ENDS
            ):
                fail_hits.append("session:missing_final_response")

            fail_hits = list(dict.fromkeys(fail_hits))
            found.append(
                {
                    "id": sid,
                    "source": source,
                    "title": title or "",
                    "end_reason": end_reason,
                    "last_activity": (last_desc or "")[:120],
                    "ts": int(last_at or started or 0),
                    "failed": bool(fail_hits),
                    "fail_hits": fail_hits,
                    "claimed_ok": claimed_ok(con, sid, since),
                    "db": origin,
                }
            )
        con.close()
    found.sort(key=lambda item: item.get("ts") or 0, reverse=True)
    return found


def cron_failures(since: float) -> list[dict]:
    if not EXEC_DB.exists():
        return []
    con = sqlite3.connect(f"file:{EXEC_DB}?mode=ro", uri=True)
    names = {}
    if JOBS.exists():
        try:
            for job in json.loads(JOBS.read_text(encoding="utf-8")).get("jobs") or []:
                names[job.get("id")] = job.get("name")
        except Exception:
            pass
    since_iso = datetime.fromtimestamp(since, timezone.utc).isoformat()
    rows = con.execute(
        """
        SELECT id, job_id, status, error, started_at, finished_at
        FROM executions
        WHERE (finished_at IS NOT NULL AND finished_at >= ?)
           OR (started_at IS NOT NULL AND started_at >= ?)
        ORDER BY finished_at DESC
        """,
        (since_iso, since_iso),
    ).fetchall()
    out = []
    for execution_id, job_id, status, error, started, finished in rows:
        status_l = (status or "").lower()
        if status_l in NONTERMINAL_STATUSES and not error:
            continue
        if status_l in SUCCESS_STATUSES and not error:
            continue
        name = names.get(job_id) or job_id
        out.append(
            {
                "execution_id": execution_id,
                "job_id": job_id,
                "name": name,
                "profile": CRON_TO_PROFILE.get(str(name), "default"),
                "status": status,
                "error": (error or "")[:300],
                "started_at": started,
                "finished_at": finished,
            }
        )
    con.close()
    return out


def kanban_failures(since: float) -> list[dict]:
    out: list[dict] = []
    since_i = int(since)
    for path in _kanban_paths():
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        runs = con.execute(
            """
            SELECT id, task_id, profile, status, outcome, substr(error,1,300),
                   substr(summary,1,200), started_at, ended_at
            FROM task_runs
            WHERE COALESCE(ended_at, started_at, 0) >= ?
            ORDER BY COALESCE(ended_at, started_at) DESC
            """,
            (since_i,),
        ).fetchall()
        for run_id, task_id, profile, status, outcome, error, summary, started, ended in runs:
            outcome_l = (outcome or "").lower()
            status_l = (status or "").lower()
            failed = _status_failure(outcome_l) or _status_failure(status_l) or bool(error)
            if not failed:
                continue
            row = con.execute("SELECT title FROM tasks WHERE id = ?", (task_id,)).fetchone()
            out.append(
                {
                    "run_id": run_id,
                    "task_id": task_id,
                    "title": (row[0] if row else "") or "",
                    "profile": profile,
                    "status": status,
                    "outcome": outcome,
                    "error": error or "",
                    "summary": summary or "",
                    "started_at": started,
                    "ended_at": ended,
                    "board": str(path),
                }
            )
        tasks = con.execute(
            """
            SELECT id, title, assignee, status, consecutive_failures,
                   last_failure_error, block_kind
            FROM tasks
            WHERE consecutive_failures > 0 OR last_failure_error IS NOT NULL
               OR (status = 'blocked' AND block_kind = 'needs_input')
            """
        ).fetchall()
        for task_id, title, assignee, status, failures, error, block_kind in tasks:
            if any(item.get("task_id") == task_id for item in out):
                continue
            outcome = "needs_input" if block_kind == "needs_input" else "consecutive_failures"
            out.append(
                {
                    "run_id": None,
                    "task_id": task_id,
                    "title": title or "",
                    "profile": assignee,
                    "status": status,
                    "outcome": outcome,
                    "error": (error or "")[:300],
                    "summary": f"consecutive_failures={failures or 0}",
                    "started_at": None,
                    "ended_at": None,
                    "board": str(path),
                }
            )
        con.close()
    return out


def stamp() -> None:
    now = time.time()
    LAST.write_text(
        json.dumps(
            {
                "unix": int(now),
                "iso": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--stamp":
        stamp()
        print(LAST.read_text(encoding="utf-8"))
        return
    since = since_unix()
    profiles = fleet()
    cron_f = cron_failures(since)
    kanban_f = kanban_failures(since)
    by_profile = {}
    for name in profiles:
        sessions = scan_sessions(name, since)
        by_profile[name] = {
            "sessions": sessions,
            "session_failures": [session for session in sessions if session.get("failed")],
            "omission_risks": [
                session
                for session in sessions
                if session.get("failed") and session.get("claimed_ok")
            ],
            "cron_failures": [failure for failure in cron_f if failure.get("profile") == name],
            "kanban_failures": [failure for failure in kanban_f if failure.get("profile") == name],
        }
    payload = {
        "since_unix": int(since),
        "since_iso": datetime.fromtimestamp(since, timezone.utc).isoformat(),
        "generated_unix": int(time.time()),
        "profiles": by_profile,
        "cron_failures": cron_f,
        "kanban_failures": kanban_f,
    }
    outdir = STORE / "audit"
    outdir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True)
    (outdir / "latest.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
