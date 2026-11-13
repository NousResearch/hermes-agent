"""Durable subagent records in ``state.db`` (``resumable_subagents``): what ``delegate_task(action='continue')``
rebuilds a child from after a restart or crash, how a stopped child may resume, and which worktrees the startup
sweep must keep for it.

One row per subagent_id, written by the run itself (``record_start`` when its workspace is seeded, ``checkpoint`` on
each heartbeat while its terminal cwd moves, ``record_stop`` after its cleanup ran) and by boot recovery for a run
whose owner process died (``recover_abandoned``). The row's ``pinned`` flag is the worktree pin: set for a child that
did not complete and was not stopped by a person, for ``delegation.resumable_ttl_hours`` (newest
``delegation.max_resumable`` per profile; the oldest beyond the cap loses its pin, the row stays). Automatic
resumption is ``delegate_tool_auto_resume`` (gated on ``delegation.auto_resume``), which reads ``resume_mode`` /
``next_attempt_at`` and marks a record ``superseded`` when its parent moved on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger("hermes_agent.tools.delegate_tool")  # log-record parity with the origin module

AUTO_AT, AUTO_ON_BOOT, ON_FIX = "AUTO_AT", "AUTO_ON_BOOT", "ON_FIX"
REPAIR_THEN_CONTINUE, HANDOFF = "REPAIR_THEN_CONTINUE", "HANDOFF"
GATEWAY_SHUTDOWN = "gateway_shutdown"
# A record ``action='continue'`` can rebuild from (``claimed`` while a continue is starting, ``running`` live).
_FINISHED = ("resumable", "completed", "stopped", "superseded")
_DEFAULT_TTL_HOURS, _DEFAULT_MAX_RESUMABLE = 48.0, 32
_EXPIRED_RETENTION_S = 7 * 24 * 3600.0

# DESIGN.md per-reason table (agent/error_classifier.py::FailoverReason): mode, and the default delay before an
# AUTO_AT attempt when the failure carried no reset time.
_MODE_BY_REASON: dict[str, tuple[str, Optional[float]]] = {
    "rate_limit": (AUTO_AT, 15 * 60.0), "upstream_rate_limit": (AUTO_AT, 15 * 60.0),
    "timeout": (AUTO_AT, 5 * 60.0), "overloaded": (AUTO_AT, 5 * 60.0), "server_error": (AUTO_AT, 5 * 60.0),
    "session_busy": (AUTO_AT, 30.0), "interpreter_shutdown": (AUTO_ON_BOOT, None),
    "content_policy_blocked": (HANDOFF, None),
    **dict.fromkeys(("context_overflow", "payload_too_large", "image_too_large", "image_corrupt",
                     "thinking_signature", "invalid_encrypted_content"), (REPAIR_THEN_CONTINUE, None)),
}


def _settings() -> tuple[float, int]:
    from hermes_agent.tools.delegate_tool_config import _load_config
    cfg = _load_config()
    try:
        ttl = max(0.0, float(cfg.get("resumable_ttl_hours", _DEFAULT_TTL_HOURS)))
        cap = max(0, int(cfg.get("max_resumable", _DEFAULT_MAX_RESUMABLE)))
    except (TypeError, ValueError):
        ttl, cap = _DEFAULT_TTL_HOURS, _DEFAULT_MAX_RESUMABLE
    return ttl, cap


def classify(entry: dict[str, Any], stop_cause: Optional[str]) -> tuple[str, Optional[str], Optional[float]]:
    """``(state, resume_mode, next_attempt_at)`` for a stopped child's result entry. ``state`` is ``resumable``
    (pinned), ``completed`` or ``stopped`` (a person or the parent stopped it: no pin)."""
    status, exit_reason = entry.get("status"), entry.get("exit_reason")
    if status == "completed" and exit_reason == "completed":
        return "completed", None, None
    if status == "interrupted":
        return ("resumable", AUTO_ON_BOOT, None) if stop_cause == GATEWAY_SHUTDOWN else ("stopped", None, None)
    if status == "unknown":
        return "resumable", AUTO_ON_BOOT, None
    reason = str(entry.get("failure_reason") or "")
    resets_at = entry.get("failure_resets_at")
    if isinstance(resets_at, (int, float)) and not isinstance(resets_at, bool) and reason in (
            "billing", "rate_limit", "upstream_rate_limit"):
        return "resumable", AUTO_AT, float(resets_at)
    mode, delay = _MODE_BY_REASON.get(reason, (ON_FIX, None))
    return "resumable", mode, (time.time() + delay if delay else None)


# ── storage ────────────────────────────────────────────────────────────────
def _db_path_for(parent_agent: Any = None) -> Optional[Path]:
    """The profile store the parent's children write their transcripts to (else this scope's ``state.db``);
    None in a managed worker, which owns no ledger."""
    from agent.runtime_session_store import is_worker_process
    if is_worker_process():
        return None
    path = getattr(getattr(parent_agent, "_session_db", None), "db_path", None)
    if isinstance(path, (str, Path)):
        return Path(path)
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "state.db"


def _connect(path: Path):
    from hermes_cli.sqlite_util import transaction
    from hermes_agent.tools.async_delegation import open_state_db
    return transaction(open_state_db(path, "state.db (resumable_subagents)"))


def _write(path: Optional[Path], fn) -> Any:
    if path is None:
        return None
    try:
        with _connect(path) as conn:
            return fn(conn)
    except FileNotFoundError:  # its profile was removed: a late write must not recreate it
        logger.debug("resumable subagent record skipped: %s is gone", path.parent)
        return None
    except Exception:
        logger.warning("resumable subagent record write failed (%s)", path, exc_info=True)
        return None


def _row(conn, sid: str) -> Optional[dict[str, Any]]:
    """The record, with the columns a claim moves without rewriting the JSON (state, owner) taken from the row."""
    got = conn.execute("SELECT record_json, state, stop_cause, owner_pid, owner_started_at FROM resumable_subagents "
                       "WHERE subagent_id=?", (sid,)).fetchone()
    if got is None:
        return None
    record = json.loads(got[0] or "{}")
    return {**record, "state": got[1], "stop_cause": got[2], "owner_pid": got[3], "owner_started_at": got[4]}


def _upsert(conn, sid: str, record: dict[str, Any], *, state: str, pinned: bool, expires_at: Optional[float],
            stop_cause: Optional[str] = None) -> None:
    now = time.time()
    # ``auto_claim`` / ``auto_cancel`` mark only a claim no run has started yet: a written run record never carries them.
    record = {k: v for k, v in record.items() if k not in ("state", "stop_cause", "auto_claim", "auto_cancel")}
    conn.execute(
        """INSERT INTO resumable_subagents (subagent_id, state, pinned, stop_cause, owner_pid, owner_started_at,
               record_json, created_at, updated_at, expires_at) VALUES (?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(subagent_id) DO UPDATE SET state=excluded.state, pinned=excluded.pinned,
               stop_cause=excluded.stop_cause, owner_pid=excluded.owner_pid,
               owner_started_at=excluded.owner_started_at, record_json=excluded.record_json,
               updated_at=excluded.updated_at, expires_at=excluded.expires_at""",
        (sid, state, int(pinned), stop_cause, record.get("owner_pid"), record.get("owner_started_at"),
         json.dumps(record, ensure_ascii=False, default=str), now, now, expires_at))


def _apply_limits(conn) -> None:
    """TTL un-pins and retires a record; beyond the cap the oldest pins drop (the row stays continuable)."""
    ttl, cap = _settings()
    now = time.time()
    conn.execute("UPDATE resumable_subagents SET pinned=0, state='expired' WHERE expires_at IS NOT NULL "
                 "AND expires_at <= ? AND state IN ('resumable','completed','stopped','superseded')", (now,))
    from hermes_agent.tools.delegate_tool_auto_resume import OWED_SQL  # kept until the scheduler delivers its one give-up
    conn.execute(f"DELETE FROM resumable_subagents WHERE state='expired' AND updated_at < ? AND NOT COALESCE(({OWED_SQL}), 0)",
                 (now - _EXPIRED_RETENTION_S - ttl * 3600,))
    conn.execute("""UPDATE resumable_subagents SET pinned=0 WHERE pinned=1 AND subagent_id NOT IN (
                        SELECT subagent_id FROM resumable_subagents WHERE pinned=1
                        ORDER BY updated_at DESC LIMIT ?)""", (cap,))


# ── worktrees ──────────────────────────────────────────────────────────────
def _git_lines(args: list[str], cwd: str) -> list[str]:
    from hermes_cli._subprocess_compat import noninteractive_repo_git_env
    env = noninteractive_repo_git_env(cwd)
    if env is None:
        return []
    try:
        res = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=10, stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    return res.stdout.splitlines() if res.returncode == 0 else []


def _worktrees_of(cwd: Optional[str]) -> list[str]:
    """Every worktree of the repository *cwd* is in (``git worktree list``), or ``[]``."""
    if not cwd or not os.path.isdir(cwd):
        return []
    return [os.path.realpath(line[9:]) for line in _git_lines(["worktree", "list", "--porcelain"], cwd)
            if line.startswith("worktree ")]


def _toplevel(cwd: Optional[str]) -> Optional[str]:
    if not cwd or not os.path.isdir(cwd):
        return None
    lines = _git_lines(["rev-parse", "--show-toplevel"], cwd)
    return os.path.realpath(lines[0]) if lines else None


def _new_worktrees(cwd: Optional[str], baseline: Iterable[str]) -> list[str]:
    known = set(baseline)
    return [p for p in _worktrees_of(cwd) if p not in known] if known else []


def _pin_paths(record: dict[str, Any]) -> list[str]:
    """Where the child was when it STOPPED: its last terminal cwd's worktree, its isolation worktree, and every
    worktree a repository it worked in gained since it first entered it (a ``git worktree add`` it ran without
    cd-ing into it), wherever its cwd went afterwards. Each repository is compared with its own baseline."""
    cwd = record.get("cwd")
    paths = [p for p in (_toplevel(cwd), (record.get("worktree") or {}).get("path")) if p]
    paths += _new_worktrees(record.get("start_cwd") or cwd, record.get("worktrees_at_start") or ())
    for main, baseline in (record.get("repo_baselines") or {}).items():
        paths += _new_worktrees(main, baseline)
    return sorted({os.path.realpath(p) for p in paths})


_LIVE_PIN_TTL_S = 30.0
_live_pin_cache: dict[str, tuple[float, list[str]]] = {}
_live_pin_lock = threading.Lock()


def _live_pins(record: dict[str, Any], *, memo: bool = True) -> list[str]:
    """Pins of a ``running``/``claimed`` record: where the child is NOW, plus where its previous run stopped (a
    continue's stored ``pin_paths``); a cached set never hides the live location.

    The sweep asks again right before each tree it touches, so the git discovery behind the live part is memoized
    for a few seconds, keyed on every record field that drives it: a checkpointed move changes the key and is
    discovered at once, while the record itself is always read fresh. The only thing a cached answer can miss is a
    worktree created within the TTL, and a tree that new is never past the sweep's idle tier. Recovery and reclaim
    store their answer as the stopped record's pins, so they pass ``memo=False``."""
    if not memo:
        return sorted({*(record.get("pin_paths") or ()), *_pin_paths(record)})
    key = json.dumps([record.get(k) for k in ("cwd", "start_cwd", "worktrees_at_start", "repo_baselines")]
                     + [(record.get("worktree") or {}).get("path")], sort_keys=True, default=str)
    now = time.monotonic()
    with _live_pin_lock:
        hit = _live_pin_cache.get(key)
    if hit is None or now - hit[0] > _LIVE_PIN_TTL_S:
        hit = (now, _pin_paths(record))  # git runs outside the lock
        with _live_pin_lock:
            for stale in [k for k, (at, _p) in _live_pin_cache.items() if now - at > _LIVE_PIN_TTL_S]:
                del _live_pin_cache[stale]
            _live_pin_cache[key] = hit
    return sorted({*(record.get("pin_paths") or ()), *hit[1]})


def _note_repo(baselines: dict[str, list[str]], start_main: Optional[str], cwd: Optional[str]) -> None:
    """Record the baseline of the repository *cwd* is in when the child first enters it (the start repository's
    is ``worktrees_at_start``)."""
    trees = _worktrees_of(cwd)
    if trees and trees[0] != start_main and trees[0] not in baselines:  # the main worktree is always listed first
        baselines[trees[0]] = trees


# ── run lifecycle ──────────────────────────────────────────────────────────
def endpoint_identity(provider: Any, base_url: Any) -> tuple[Any, Optional[str]]:
    """``(provider, sha256 of the exact base_url)``: what a continue compares to refuse a changed backend. The URL
    itself is never stored (it can carry a key in its query or userinfo); the real one is re-resolved from the
    task's route at continue. Exact bytes, so any change to the URL still refuses."""
    return provider, (hashlib.sha256(str(base_url).encode("utf-8")).hexdigest() if base_url is not None else None)


def _owner_identity() -> dict[str, Any]:
    from hermes_agent.gateway.status import get_process_start_time
    started = get_process_start_time(os.getpid())
    return {"owner_pid": os.getpid(), "owner_started_at": started}


def _spec(child: Any) -> Optional[dict[str, Any]]:
    """The rebuild spec without live objects or secrets: the route is kept as the request that re-routes it."""
    build = getattr(child, "_delegate_resume", None)
    session_id = getattr(child, "session_id", None)
    if not isinstance(build, dict) or not isinstance(session_id, str) or not session_id:
        return None
    route = build.get("route")
    return {"goal": build.get("goal"), "context": build.get("context"), "toolsets": build.get("toolsets"),
            "endpoint": list(build.get("endpoint") or ()), "model": getattr(child, "model", None),
            "session_id": session_id, "route_request": getattr(route, "request", None)}


def record_start(run: Any) -> None:
    """A child began (or resumed) running: durable identity, rebuild spec, owner and starting workspace. A write
    that fails stays pending on the run (``_flush``) and is retried, whole, at the next checkpoint and at the stop."""
    child = run.child
    spec = _spec(child)
    sid = run.subagent_id
    if spec is None or not sid:
        return
    path = _db_path_for(run.parent_agent)
    cwd = _child_cwd(run)
    from hermes_agent.tools.delegate_tool_registry import _recent_subagents, _active_subagents_lock
    owners = {str(getattr(child, "_parent_session_id", "") or "") or str(getattr(run.parent_agent, "session_id", "")
                                                                         or "")}
    # The git probes run before the write transaction (seconds on a slow repo; no state.db lock held through them).
    earlier = _write(path, lambda conn: _row(conn, sid)) or {}
    start_cwd = earlier.get("start_cwd") or cwd
    start_trees = _worktrees_of(start_cwd)
    # ``lock`` serializes the start-record retries (heartbeat thread vs the stop path) with the terminal write, so a
    # retry still waiting on the busy timeout can never land after ``record_stop`` and flip the row back to running.
    # Who started this run: a parent's spawn, a parent's continue (no automatic resume after it), the scheduler.
    origin = getattr(child, "_delegate_run_origin", None)
    origin = origin if origin in ("continue", "auto") else "spawn"
    auto = {"auto_attempts": int(earlier.get("auto_attempts") or 0), "auto_off": earlier.get("auto_off")
            or ("the parent continued it itself" if origin == "continue" else None),
            # The automatic-resume deadline counts from the FIRST stop, across every resumed run.
            "auto_first_stopped_at": earlier.get("auto_first_stopped_at")}
    durable: dict[str, Any] = {"db": path, "cwd": cwd, "start_main": start_trees[0] if start_trees else None,
                               "baselines": dict(earlier.get("repo_baselines") or {}), "lock": threading.RLock(),
                               "auto": auto}
    _note_repo(durable["baselines"], durable["start_main"], cwd)

    def _do(conn):
        if origin == "auto":  # the parent's last word: a stop, /stop or replacement since the claim cancels it here
            from hermes_agent.tools.delegate_tool_auto_resume import admit_start
            if cancelled := admit_start(conn, sid):
                return cancelled
        prev = _row(conn, sid) or {}
        with _active_subagents_lock:
            owners.update((_recent_subagents.get(sid) or {}).get("ancestor_owner_session_ids") or ())
        owners.update(prev.get("owners") or ())
        owners.update(_ancestor_sessions(child))
        owners.discard("")
        record = {**prev, **spec, **_owner_identity(), "subagent_id": sid,
                  "delegation_id": getattr(child, "_delegation_id", None), "owners": sorted(owners),
                  # The async unit whose ledger row reports this run (``<call>-k`` with independent completions).
                  "unit_id": getattr(child, "_delegation_unit_id", None),
                  "start_cwd": start_cwd, "cwd": durable["cwd"], "worktree": prev.get("worktree")
                  or run.worktree_info, "outcome": "running", "resume_mode": None,
                  # Kept across continues: a tree the child made in an earlier run is still its own.
                  "worktrees_at_start": prev.get("worktrees_at_start") or start_trees,
                  # A promise belongs to one stop: this run's own stop (or boot recovery) makes it afresh.
                  "run_origin": origin, "run_started_at": time.time(), "auto_promised": False, **auto}
        record["repo_baselines"] = durable["baselines"]
        # A running record pins too (its paths are resolved at sweep time): a crash before boot recovery ran must
        # not hand its tree to the sweep. Bounded by the same TTL.
        _upsert(conn, sid, record, state="running", pinned=True, expires_at=time.time() + _settings()[0] * 3600)
        return True

    outcome = _write(path, _do)
    if isinstance(outcome, str):  # cancelled by the parent before it ran: no run record, the child does not start
        child._delegate_auto_cancelled = outcome
        child._delegate_claim = None  # the row is the parent's now: cleanup must not hand any claim back
        return
    if origin == "auto" and not outcome:
        # Fail closed: without its run record the parent's last word could not be checked, and a later retry would
        # check it while the child already runs. It does not start; cleanup hands the claim back to ``resumable``.
        child._delegate_auto_cancelled = "Not continued: its run record could not be written"
        return
    durable["pending"] = None if outcome else _do
    child._delegate_durable = durable


def _flush(durable: dict[str, Any]) -> bool:
    """Retry the run's pending start record in its own transaction; True once it is persisted. Cleared only after
    that transaction committed, so a later write failing never leaves "persisted" set over a rolled-back row."""
    with durable["lock"]:
        pending = durable.get("pending")
        if pending is not None and not durable.get("stopped") and _write(durable.get("db"), pending):
            durable["pending"] = pending = None
        return pending is None


def _ancestor_sessions(child: Any) -> set[str]:
    owners, cur = set(), child
    for _ in range(8):
        ref = getattr(cur, "_delegate_parent_ref", None)
        cur = ref() if callable(ref) else None
        if cur is None:
            break
        sid = getattr(cur, "session_id", None)
        if isinstance(sid, str) and sid:
            owners.add(sid)
    return owners


def _child_cwd(run: Any) -> Optional[str]:
    from hermes_agent.tools.delegate_tool_routes import child_profile_scope
    from hermes_agent.tools.terminal_tool import get_session_cwd
    with child_profile_scope(run.child):
        cwd = get_session_cwd(run.child_task_id)
    return os.path.realpath(cwd) if isinstance(cwd, str) and cwd else None


def checkpoint(run: Any) -> None:
    """Heartbeat: persist the child's terminal cwd when it moved, so a hard kill still knows where it was."""
    durable = getattr(run.child, "_delegate_durable", None)
    if not isinstance(durable, dict) or not run.subagent_id:
        return
    cwd = _child_cwd(run)
    if not cwd or cwd == durable.get("cwd"):
        _flush(durable)
        return
    durable["cwd"] = cwd
    baselines = durable["baselines"]
    _note_repo(baselines, durable["start_main"], cwd)

    if not _flush(durable):
        return  # the full start record goes first; this cwd rides on its next retry or on the stop record

    def _do(conn):
        conn.execute("UPDATE resumable_subagents SET record_json=json_set(record_json, '$.cwd', ?, "
                     "'$.repo_baselines', json(?)), updated_at=? WHERE subagent_id=? AND state='running'",
                     (cwd, json.dumps(baselines), time.time(), run.subagent_id))
    _write(durable.get("db"), _do)


def stop_cause_live(run: Any) -> Optional[str]:
    """``gateway_shutdown`` when the gateway's shutdown stopped this child (``note_gateway_shutdown`` marked it, or
    its own turn recorded the shutdown interrupt), else None."""
    if getattr(run.child, "_delegate_stop_cause", None) == GATEWAY_SHUTDOWN or GATEWAY_SHUTDOWN in str(
            getattr(run, "turn_exit_reason", "") or ""):
        return GATEWAY_SHUTDOWN
    return None


def keeps_worktree(run: Any, entry: dict[str, Any]) -> bool:
    """Whether the stopped child is resumable, so its isolation worktree must survive the stop: only with its
    record persisted (a pending one is retried first); without one nothing pins the tree, so it is finalized as
    before (archived when it holds work)."""
    durable = getattr(run.child, "_delegate_durable", None)
    if not run.subagent_id or not isinstance(durable, dict) or not _flush(durable):
        return False
    return classify(entry, stop_cause_live(run))[0] == "resumable"


def stamp_entry(run: Any, entry: dict[str, Any]) -> None:
    """The result entry says how this stopped child may resume (absent when it completed or a person stopped it)."""
    durable = getattr(run.child, "_delegate_durable", None)
    if not run.subagent_id or not isinstance(durable, dict):
        return
    durable["cwd"] = _child_cwd(run) or durable.get("cwd")  # before cleanup drops the child's cwd record
    if not _flush(durable):
        return  # no record to continue from: the entry must not offer a resume
    state, mode, next_at = classify(entry, stop_cause_live(run))
    if state == "resumable":
        entry["resume_mode"] = mode
        if getattr(run.child, "_delegation_unit_id", None) and not durable["auto"]["auto_off"]:
            from hermes_agent.tools.delegate_tool_auto_resume import promise
            from hermes_agent.tools.delegate_tool_auto_resume import gave_up, gave_up_line
            candidate = {**durable["auto"], "resume_mode": mode, "next_attempt_at": next_at}
            if auto := promise(candidate):
                entry["auto_resume"] = auto
                durable["promised"] = True  # only a stop that promised a resume may be resumed automatically
            elif reason := gave_up(candidate):  # this stop's own completion is the one terminal failure
                entry["auto_gave_up"] = reason
                entry["error"] = f"{entry.get('error') or entry.get('status')}; {gave_up_line(run.subagent_id, reason)}"
                durable["gave_up"] = reason


def record_stop(run: Any, entry: Optional[dict[str, Any]]) -> None:
    """The child's run ended and its resources are closed: its outcome, resume mode, last cwd and pins."""
    durable = getattr(run.child, "_delegate_durable", None)
    sid = run.subagent_id
    if not isinstance(durable, dict) or not sid:
        return
    entry = entry or {"status": "unknown"}
    cwd = durable.get("cwd") or _child_cwd(run)
    live_cause = stop_cause_live(run)
    # The segment it stopped in: with ``compression.in_place: false`` each rotation moved it to a new session.
    session_id = getattr(run.child, "session_id", None)

    def _do(conn):
        record = _row(conn, sid) or {}
        if isinstance(session_id, str) and session_id:
            record["session_id"] = session_id
        record["auto_promised"] = bool(durable.get("promised"))
        if durable.get("gave_up"):
            record["auto_gave_up"] = record["auto_note"] = durable["gave_up"]
        _finish(conn, sid, record, entry, live_cause or record.get("stop_cause"), cwd=cwd)

    with durable["lock"]:
        flushed = _flush(durable)
        durable["stopped"] = True  # no start retry may run after the stop, landed or not
        if flushed:  # never a partial row (no goal, route or owners) the parent cannot claim
            _write(durable.get("db"), _do)


def _finish(conn, sid: str, record: dict[str, Any], entry: dict[str, Any], cause: Optional[str], *,
            cwd: Optional[str], pins: Optional[list[str]] = None) -> None:
    """Write the stopped record. *pins*: its pin paths computed BEFORE the caller's write transaction opened (git
    probes take seconds; boot recovery and the claim reclaim must not hold the state.db write lock through them)."""
    ttl, _cap = _settings()
    state, mode, next_at = classify(entry, cause)
    record = {**record, "cwd": cwd or record.get("cwd"), "outcome": entry.get("status"), "resume_mode": mode,
              "next_attempt_at": next_at, "failure_reason": entry.get("failure_reason"),
              "failure_resets_at": entry.get("failure_resets_at"), "usage_limit": entry.get("usage_limit"),
              "stopped_at": time.time()}
    if state == "resumable" and not record.get("auto_first_stopped_at"):
        record["auto_first_stopped_at"] = record["stopped_at"]
    if isinstance(entry.get("worktree"), dict):
        record["worktree"] = {**(record.get("worktree") or {}), **entry["worktree"]}
    record["pin_paths"] = (_pin_paths(record) if pins is None else pins) if state == "resumable" else []
    _upsert(conn, sid, record, state=state, pinned=state == "resumable" and bool(record["pin_paths"]),
            expires_at=time.time() + ttl * 3600, stop_cause=cause)
    _apply_limits(conn)


def note_gateway_shutdown() -> int:
    """Gateway shutdown is about to stop this process's children: mark them first (in memory and on every profile's
    running records), so each reads as a restart (resumable) rather than a user stop however its run ends, a
    SIGKILL mid-unwind included. Returns how many records were marked."""
    from hermes_agent.tools.delegate_tool_registry import _active_subagents, _active_subagents_lock
    with _active_subagents_lock:
        live = [r.get("agent") for r in _active_subagents.values()]
    for agent in live:
        if agent is not None:
            agent._delegate_stop_cause = GATEWAY_SHUTDOWN
    marked = 0
    for db in _profile_dbs():
        marked += _write(db, lambda conn: conn.execute(
            "UPDATE resumable_subagents SET stop_cause=? WHERE state='running' AND owner_pid=?",
            (GATEWAY_SHUTDOWN, os.getpid())).rowcount) or 0
    return marked


def _abandoned_entry(record: dict[str, Any]) -> tuple[dict[str, Any], Optional[str]]:
    """``(entry, cause)`` for a run whose owner died: a gateway shutdown when that was marked, else unknown."""
    cause = record.get("stop_cause")
    return {"status": "interrupted" if cause == GATEWAY_SHUTDOWN else "unknown"}, cause


def plan_recovery(path: Optional[Path], alive) -> dict[str, tuple[Any, Any, list[str]]]:
    """Boot, BEFORE the recovery transaction: ``{subagent_id: (owner_pid, owner_started_at, pin_paths)}`` for every
    ``running``/``claimed`` record whose owner is gone. The git probes behind the pins run here, holding no lock."""
    rows = _write(path, lambda conn: conn.execute(
        "SELECT subagent_id, owner_pid, owner_started_at, record_json FROM resumable_subagents "
        "WHERE state IN ('running','claimed')").fetchall()) or []
    plan = {}
    for sid, pid, started, raw in rows:
        if not alive(pid, started):
            plan[sid] = (pid, started, _live_pins(json.loads(raw or "{}"), memo=False))
    return plan


def _begin_immediate(conn) -> None:
    """Take the write lock now, so the re-check that follows and the write after it are one transaction."""
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")


def _still_abandoned(conn, sid: str, pid: Any, started: Any) -> Optional[dict[str, Any]]:
    """The record, when it is still the same dead owner's ``running``/``claimed`` row (read inside the write lock)."""
    got = conn.execute("SELECT state, owner_pid, owner_started_at FROM resumable_subagents WHERE subagent_id=?",
                       (sid,)).fetchone()
    if got is None or got[0] not in ("running", "claimed") or (got[1], got[2]) != (pid, started):
        return None
    return _row(conn, sid)


def recover_abandoned(conn, plan: dict[str, tuple[Any, Any, list[str]]],
                      records: Optional[dict[str, dict[str, Any]]] = None) -> None:
    """Boot: each planned record (``plan_recovery``) still held by its dead owner becomes resumable (outcome
    ``unknown``, or a gateway shutdown when that was marked), pinned on where it last was; *records* collects each
    recovered record. A row that died after the plan was made is reclaimed by ``claim``."""
    if not plan:
        return
    _begin_immediate(conn)
    for sid, (pid, started, pins) in plan.items():
        record = _still_abandoned(conn, sid, pid, started)
        if record is None:
            continue
        entry, cause = _abandoned_entry(record)
        _finish(conn, sid, record, entry, cause, cwd=record.get("cwd"), pins=pins)
        if records is not None:
            records[sid] = record


def continuable_of(conn, unit_id: str) -> list[str]:
    """Every child of the async unit *unit_id* its parent can continue now: the ones boot recovery just finished,
    and the ones that recorded their own resumable stop before the owner exited (a gateway restart stops each
    child cleanly, while the unit's ledger row is left running)."""
    return [sid for (sid,) in conn.execute(
        "SELECT subagent_id FROM resumable_subagents WHERE state='resumable' AND COALESCE("
        "json_extract(record_json, '$.unit_id'), json_extract(record_json, '$.delegation_id'))=? "
        "ORDER BY created_at", (unit_id,))]


def _reclaim_dead_owner(path: Optional[Path], sid: str, record: dict[str, Any]) -> Optional[dict[str, Any]]:
    """A ``running``/``claimed`` row whose owner process died (after this process's boot recovery ran): finish it
    the way boot recovery would and return the finished record; None while its owner is alive."""
    from hermes_agent.tools.async_delegation import _owner_liveness
    alive = _owner_liveness()
    pid, started = record.get("owner_pid"), record.get("owner_started_at")
    if alive is None or alive(pid, started):
        return None
    pins = _live_pins(record, memo=False)

    def _do(conn):
        _begin_immediate(conn)
        current = _still_abandoned(conn, sid, pid, started)
        if current is None:
            return None
        entry, cause = _abandoned_entry(current)
        _finish(conn, sid, current, entry, cause, cwd=current.get("cwd"), pins=pins)
        return _row(conn, sid)
    return _write(path, _do)


# ── continue ───────────────────────────────────────────────────────────────
def claim(parent_agent: Any, sid: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Take a finished child's durable record for ``continue`` (``(record, None)``, or ``(None, error)``). The
    state flip to ``claimed`` is one guarded UPDATE, so two continues never run one conversation twice."""
    from hermes_agent.tools.delegate_tool_registry import _owns_subagent_record
    path = _db_path_for(parent_agent)
    record = _write(path, lambda conn: (_apply_limits(conn), _row(conn, sid))[1])
    if record is None:
        return None, None
    owned = {"owner_agent_session_id": None, "ancestor_owner_session_ids": record.get("owners") or ()}
    if not _owns_subagent_record(owned, parent_agent):
        return None, None
    if record.get("state") in ("running", "claimed"):
        record = _reclaim_dead_owner(path, sid, record) or record
    if record.get("state") not in _FINISHED:
        return None, (f"Subagent '{sid}' is still running or being continued. Try again shortly; if this persists, "
                      "spawn a new task.") if record.get("state") in ("running", "claimed") else None
    owner = _owner_identity()  # boot recovery must not reclaim a continue this live process is starting
    from hermes_agent.tools.delegate_tool_auto_resume import CANCEL_DEBT  # the parent took it over: no automatic report owed
    claimed = _write(path, lambda conn: conn.execute(
        f"UPDATE resumable_subagents SET state='claimed', owner_pid=?, owner_started_at=?, updated_at=?, "
        f"record_json=json_set(record_json, {CANCEL_DEBT}) WHERE subagent_id=? AND state=?",
        (owner["owner_pid"], owner["owner_started_at"], time.time(), sid, record["state"])).rowcount)
    return (record, None) if claimed == 1 else (None, f"Subagent '{sid}' is already being continued.")


def _hand_back(conn, sid: str, state: str, *, promised: Optional[bool] = None, owner_pid: Optional[int] = None) -> None:
    """Return a claimed record no run started to *state* (with *promised* as its debt when given), in one write
    transaction: a stop that landed during the claim (``$.auto_cancel``) wins over handing a promise back."""
    _begin_immediate(conn)
    mine, where = (" AND owner_pid=?", (sid, owner_pid)) if owner_pid is not None else ("", (sid,))
    got = conn.execute("SELECT json_extract(record_json, '$.auto_cancel.state'), json_extract(record_json, "
                       f"'$.auto_cancel.note') FROM resumable_subagents WHERE subagent_id=? AND state='claimed'{mine}",
                       where).fetchone()
    if got is None:
        return
    record_json, params, unpin = "record_json", [], ""
    if got[0]:  # the parent cancelled it during the claim: no automatic resume, no report
        promised = False
        if state == "resumable":
            state, record_json, params = got[0], "json_set(record_json, '$.auto_note', ?)", [got[1]]
            unpin = ", pinned=0" if state == "stopped" else ""
    if promised is not None:
        record_json, params = f"json_set({record_json}, '$.auto_promised', json(?))", [*params, json.dumps(promised)]
    conn.execute(f"UPDATE resumable_subagents SET record_json=json_remove({record_json}, '$.auto_claim', "
                 f"'$.auto_cancel'), state=?{unpin} WHERE subagent_id=? AND state='claimed'{mine}",
                 (*params, state, *where))


def release(parent_agent: Any, sid: str, record: dict[str, Any]) -> None:
    """Hand a claimed record back (the continue was refused before anything ran), with the debt it had."""
    _write(_db_path_for(parent_agent), lambda conn: _hand_back(conn, sid, record["state"],
                                                               promised=bool(record.get("auto_promised"))))


def hold_claim(child: Any, parent_agent: Any, sid: str, record: dict[str, Any]) -> None:
    """The continued child carries its claim until its run records itself (``record_start``), so a run that fails
    before that hands the row back (``release_unstarted``) instead of leaving it ``claimed`` by a live process."""
    child._delegate_claim = (_db_path_for(parent_agent), sid, record["state"])


def release_unstarted(child: Any) -> bool:
    """Hand back the claim of a continued child whose run ended before ``record_start`` (idempotent). True when
    the hand-back committed (or there was none); a failed one is queued for ``retry_hand_backs``."""
    claim = getattr(child, "_delegate_claim", None)
    durable = getattr(child, "_delegate_durable", None)
    if not isinstance(claim, tuple) or (isinstance(durable, dict) and durable.get("pending") is None):
        return True
    path, sid, state = claim
    # A stop that reached this registered child before it ran (an automatic run failing closed) is the parent's
    # decision: the row ends ``stopped``, never handed back for the scheduler to resume.
    stopped = (getattr(child, "_delegate_run_origin", None) == "auto"
               and getattr(child, "_delegate_stop_requested", False) is True)
    pending = (path, sid, "stopped" if stopped else state, False if stopped else None)
    committed = _hand_back_pending(pending)
    if not committed:  # the store is busy: the queue owns the claim now, and the scheduler's tick retries it
        with _PENDING_HAND_BACKS_LOCK:
            _PENDING_HAND_BACKS.append(pending)
    child._delegate_claim = None
    return committed


# Hand-backs whose write failed (``release_unstarted``): a row left ``claimed`` by this live process is invisible to
# the scheduler and to boot recovery, so a promised child would never report while the process lives.
_PENDING_HAND_BACKS: list[tuple[Optional[Path], str, str, Optional[bool]]] = []
_PENDING_HAND_BACKS_LOCK = threading.Lock()


def _hand_back_pending(pending: tuple[Optional[Path], str, str, Optional[bool]]) -> bool:
    """One hand-back attempt; True once it committed (a row no longer ``claimed`` by this process counts)."""
    path, sid, state, promised = pending
    return _write(path, lambda conn: (_hand_back(conn, sid, state, promised=promised, owner_pid=os.getpid()),
                                      True)[1]) is True


def retry_hand_backs() -> None:
    """Scheduler tick: retry each queued hand-back once, keeping only the ones that still failed."""
    with _PENDING_HAND_BACKS_LOCK:
        queued = list(_PENDING_HAND_BACKS)
        _PENDING_HAND_BACKS.clear()
    failed = [p for p in queued if not _hand_back_pending(p)]
    if failed:
        with _PENDING_HAND_BACKS_LOCK:
            _PENDING_HAND_BACKS.extend(failed)


def last_workspace(parent_agent: Any, sid: str) -> Optional[dict[str, Any]]:
    """``{cwd, worktree}`` the child stopped in, for a continue to start there instead of the parent's cwd."""
    record = _write(_db_path_for(parent_agent), lambda conn: _row(conn, sid)) or {}
    cwd = record.get("cwd")
    if not (isinstance(cwd, str) and os.path.isdir(cwd)):
        return None
    worktree = record.get("worktree") if isinstance(record.get("worktree"), dict) else None
    return {"cwd": cwd, "worktree": worktree if worktree and os.path.isdir(worktree.get("path") or "") else None}


# ── worktree sweep pin ─────────────────────────────────────────────────────
def _profile_dbs() -> list[Path]:
    from hermes_cli.profiles import profiles_to_serve
    homes = [home for _name, home in profiles_to_serve(multiplex=True, include_standalone=True, include_parked=True)]
    return [Path(home) / "state.db" for home in homes if (Path(home) / "state.db").is_file()]


def _settings_of(home: Path) -> tuple[float, int]:
    """*home*'s own ``delegation`` limits, read under its scope (the sweep runs under whichever profile started it)."""
    from agent.runtime_scope import bind_profile_scope
    try:
        with bind_profile_scope(home, hydrate_secrets=False):
            return _settings()
    except Exception:  # health: allow BLE001 -- an unreadable profile keeps the default cap, never loses its pins
        logger.debug("delegation settings of %s unreadable", home, exc_info=True)
        return _DEFAULT_TTL_HOURS, _DEFAULT_MAX_RESUMABLE


def _read_pins(db: Path, now: float) -> Iterable[tuple[str, dict[str, Any]]]:
    import sqlite3
    from hermes_cli.sqlite_safe_read import connect_tracked
    from hermes_agent.hermes_state_holders import read_only_db_uri
    _ttl, cap = _settings_of(db.parent)
    try:
        conn = connect_tracked(read_only_db_uri(db), tracking_path=db, uri=True, timeout=2.0)
    except sqlite3.Error:
        return []
    try:
        rows = conn.execute("SELECT subagent_id, record_json, expires_at, state FROM resumable_subagents "
                            "WHERE pinned=1 AND state IN ('resumable','superseded','running','claimed') "
                            "AND (expires_at IS NULL OR expires_at > ?) ORDER BY updated_at DESC LIMIT ?",
                            (now, cap)).fetchall()
    except sqlite3.Error:
        rows = []  # no table yet: nothing pinned
    finally:
        conn.close()
    out = []
    for sid, raw, exp, state in rows:
        record = {**json.loads(raw or "{}"), "expires_at": exp}
        if state in ("running", "claimed"):
            record["pin_paths"] = _live_pins(record)
        out.append((sid, record))
    return out


def pinned_worktrees(now: Optional[float] = None) -> dict[str, str]:
    """``{realpath: reason}`` for every worktree a live resumable record pins, across this install's profiles.
    The startup sweep's only view of the records."""
    now = now or time.time()
    pins: dict[str, str] = {}
    for db in _profile_dbs():
        for sid, record in _read_pins(db, now):
            until = time.strftime("%Y-%m-%d %H:%M", time.localtime(record["expires_at"])) if record.get(
                "expires_at") else "?"
            for path in record.get("pin_paths") or ():
                pins.setdefault(os.path.realpath(path), f"resumable subagent {sid} (pinned until {until})")
    return pins


def pin_reason(path: str, pins: dict[str, str]) -> Optional[str]:
    """Why the tree at *path* is pinned (a pin on the tree or any directory inside it), else None."""
    root = os.path.realpath(path)
    for pinned, reason in pins.items():
        if pinned == root or pinned.startswith(root + os.sep):
            return reason
    return None
