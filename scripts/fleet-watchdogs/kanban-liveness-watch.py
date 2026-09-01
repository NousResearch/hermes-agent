#!/usr/bin/env python3
"""Kanban liveness + stale-diagnosis watchdog (no_agent, silent unless wrong).

Solves two recurring fleet failures, both of which burned agent-hours this week:

  1. STALE DIAGNOSIS — a card is spawned to fix a bug that is already fixed on
     disk. The worker reads a stale report, re-diagnoses the old crash, and
     spawns a rework card for a defect that does not exist. This script reads
     every `blocked`/`todo`/`ready` card whose body or latest comment cites a
     code defect ("X does not exist", "no attribute", "no such table",
     "AttributeError", "missing") and verifies the cited symbol/file against the
     workspace tree. If the claimed-missing symbol resolves on disk, the card's
     premise is stale -> flag hard.

  2. LIVENESS DEADLOCK — live gateways exist but no worker is running and
     cards sit `ready`/`todo` for too long, OR the dependency graph has a card
     whose parents include an archived card (an archived parent can never reach
     "done", so the child is permanently stuck). Both are silent stalls.

Wire-up: `no_agent` cron, 1-minute interval (or 5m after the board stabilises).
Silent-on-healthy: empty stdout = no action. Non-empty stdout = delivered to the
operator / wakes the orchestrator to act.

Usage:
    kanban-liveness-watch.py                 # report only (cron default)
    kanban-liveness-watch.py --audit          # always print, for smoke-testing
"""

import json
import os
import re
import sqlite3
import sys

HERMES_HOME = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
if not os.path.isdir(os.path.join(HERMES_HOME, "profiles")):
    _up = os.path.dirname(os.path.dirname(HERMES_HOME))
    if os.path.isdir(os.path.join(_up, "profiles")):
        HERMES_HOME = _up

KANBAN_DB = os.path.join(HERMES_HOME, "kanban.db")
WORKSPACE = "/Users/werolloperator/Projects/Meeting notes transcription tool"

# How long a `ready`/`todo` card may sit with a live dispatcher and no run
# before we call it dead. 90s is tight enough to catch a stall early, loose
# enough to ignore a normal claim+spawn round-trip.
STALL_SECONDS = 90

# ---------------------------------------------------------------------------
# Auto-restart arms (2026-09-02, chains on fix-20260902-dispatcher-takeover).
#
# Alert-only watchdogs protect nothing on a 24/7 board. When a ready card has
# no dispatcher AND the dispatcher heartbeat is stale on TWO consecutive
# passes, we SIGUSR1 the dispatcher-owning (root) gateway to drain-and-restart
# it, then post ONE line describing the action and its evidence. Elsewhere the
# takeover card makes the ROOT gateway the dispatcher owner and writes its
# heartbeat here every tick. This file is the watchdog's copy of that
# contract, so we resolve the root gateway's pidfile and send SIGUSR1 to it.
#
# Fail-safe: if the heartbeat file is absent (takeover not yet deployed), we
# treat the situation as "heartbeat contract not live" and do NOT restart —
# we fall back to today's alert-only behaviour. Never guess.
# ---------------------------------------------------------------------------
RESTART_COOLDOWN_SECONDS = 30 * 60          # at most one restart per 30 min
RESTART_REQUIRED_PASSES = 2                 # confirmed stall, not a blip
DISPATCHER_HEARTBEAT = os.path.join(HERMES_HOME, "kanban", ".dispatcher.heartbeat")
DISPATCHER_STALE_SECONDS = 5 * 60           # takeover contract: >5 min = stale
ROOT_GATEWAY_PIDFILE = os.path.join(HERMES_HOME, "gateway.pid")
RESTART_STATE = os.path.join(HERMES_HOME, "state", "kanban-liveness-restart.json")
SIGUSR1 = 30                                # graceful drain-and-restart (cleanup-20260902 precedent)

NOW = __import__("time").time()

# How long a decision-shaped triage card may sit unaccepted before we flag it
# as a parked decision the PM forgot to accept/reject. 30m is loose enough to
# give the PM a real window, tight enough that a stranded decision surfaces
# instead of silently parking forever.
PARKED_DECISION_SECONDS = 30 * 60

# Defect-claim patterns -> the symbol/file to verify on disk. Group names are
# matched against the card text; the symbol is then grepped in the workspace.
CLAIM_PATTERNS = [
    # "db.write_conn() does not exist"  /  "no attribute 'write_conn'"
    (re.compile(r"\b(?:no attr(?:ibute)?\s*['\"]?([A-Za-z_][\w]*)|([A-Za-z_][\w.]*)\.( ?call)?s? (?:does not exist|has no attribute|is missing))\b", re.I), "symbol"),
    (re.compile(r"[\w.]*\.([A-Za-z_]\w*)\s*\(\).*(?:does not exist|missing|not defined|has no attribute)", re.I), "symbol"),
    (re.compile(r"(no such table|no such column|no such file)\b", re.I), "sql"),  # informational only
]


def _conn():
    c = sqlite3.connect(f"file:{KANBAN_DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def active_cards(con):
    """Cards that are not done/archived, with their title/body and last run ts."""
    return con.execute(
        "SELECT id, title, body, assignee, status, priority, created_at, "
        " COALESCE((SELECT MAX(created_at) FROM task_runs WHERE task_id=tasks.id),0) AS last_run_ts "
        "FROM tasks WHERE status IN ('blocked','todo','ready','running','review') "
        "ORDER BY priority DESC, created_at ASC"
    ).fetchall()


def symbols_on_disk(symbols, root=WORKSPACE):
    """Return {symbol: [paths]} for each symbol that resolves on disk."""
    hits = {}
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            # skip heavy/vendored dirs
            dirnames[:] = [d for d in dirnames if d not in
                           ("node_modules", ".venv", "__pycache__", ".git", "build", "dist",
                            ".venv-diarize", ".venv-frontend", "node_modules2")]
            if ".py" not in " ".join(filenames):
                continue
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                except OSError:
                    continue
                for s in symbols:
                    if re.search(rf"\bdef\s+{re.escape(s)}\b", text) or \
                       re.search(rf"\b{re.escape(s)}\s*=", text):
                        hits.setdefault(s, []).append(p)
    except OSError:
        pass
    return hits


def read_json_state(path):
    """Tolerantly read a JSON state file; return {} on any failure."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_json_state(path, obj):
    """Atomically write a JSON state file (tmp + replace)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


class RestartDecision:
    """Outcome of evaluating whether to restart the dispatcher-owning gateway."""

    def __init__(self, should_restart=False, reason="", evidence=None):
        self.should_restart = should_restart
        self.reason = reason
        self.evidence = evidence or {}

    @property
    def line(self):
        return "[kanban-liveness-watch] " + self.reason


def dispatcher_heartbeat_stale(now=NOW):
    """True iff the dispatcher heartbeat file exists AND is older than the
    stale window. Returns (bool, detail). Absent file => (False, explanation):
    the heartbeat contract is not live yet (takeover card not deployed), so we
    must NOT act on staleness — we fall back to alert-only."""
    try:
        mtime = os.path.getmtime(DISPATCHER_HEARTBEAT)
    except OSError:
        return False, "no dispatcher heartbeat file (#%s absent; takeover not live)" % os.path.basename(DISPATCHER_HEARTBEAT)
    age = now - mtime
    if age <= DISPATCHER_STALE_SECONDS:
        return False, "dispatcher heartbeat fresh (%ds old)" % int(age)
    return True, "dispatcher heartbeat stale (%ds > %ds window)" % (int(age), DISPATCHER_STALE_SECONDS)


def root_gateway_pid(now=NOW):
    """Resolve the dispatcher-owner (root) gateway pid from its pidfile.

    Never signal a pid whose pidfile is fresher than the stall window — a fresh
    pidfile means the gateway just restarted (possibly by us) and we must not
    bounce a process that is booting. Returns (pid|None, detail). Guards:
      * pidfile must exist, be a JSON object with a numeric ``pid``.
      * pidfile mtime must not be fresher than STALL_SECONDS (a just-started
        gateway would have a pidfile written seconds ago).
    """
    try:
        st = os.stat(ROOT_GATEWAY_PIDFILE)
    except OSError:
        return None, "no root gateway pidfile (%s)" % os.path.basename(ROOT_GATEWAY_PIDFILE)
    if (now - st.st_mtime) < STALL_SECONDS:
        return None, "root gateway pidfile fresh (%ds old); refusing to signal a booting gateway" % int(now - st.st_mtime)
    try:
        rec = read_json_state(ROOT_GATEWAY_PIDFILE)
        raw = rec.get("pid")
        pid = int(raw) if raw is not None and str(raw).strip().lstrip("-").isdigit() else None
    except Exception:
        raw = None
        pid = None
    if not pid or pid <= 0:
        return None, "invalid root gateway pid (%r)" % raw
    return pid, "root gateway pid=%s" % pid


def _in_cooldown(state, now=NOW):
    last = state.get("last_restart_at")
    if not last:
        return False
    return (now - last) < RESTART_COOLDOWN_SECONDS


def _pending_pass(state):
    """Return the accumulated confirmed-stall pass count from state (0 if none)."""
    return int(state.get("confirmed_stall_passes", 0) or 0)


def evaluate_restart(existing_state, stalled_ready, now=NOW):
    """Decide whether to restart the dispatcher-owning gateway this pass.

    Fires only when BOTH arms hold on RESTART_REQUIRED_PASSES consecutive
    passes:
      1. dispatcher heartbeat stale, AND
      2. there is at least one `ready` card that is a bona fide stall (has had
         no run for > STALL_SECONDS).

    The two-pass confirmation distinguishes a genuine stall from a transient
    blip, and the cooldown caps restarts at one per RESTART_COOLDOWN_SECONDS.

    Escalation instead of looping: if a restart already happened within the
    cooldown window (``restarted_recently``) and the board is STILL stalled-and-
    stale, do not loop a second restart — record an ESCALATE decision so the
    caller can wake Richie via photon iMessage (costcap-watch pattern).

    Returns (RestartDecision, new_state). new_state is ``existing_state``
    updated with the confirmed-stall pass counter and (on fire) last_restart_at.
    Pure: no signals are sent here — the caller sends SIGUSR1 and prints.
    """
    state = dict(existing_state or {})
    passes = _pending_pass(state)

    if not stalled_ready:
        state["confirmed_stall_passes"] = 0
        return RestartDecision(False, "no confirmed-ready stall; resetting arm counter"), state

    stale, _detail = dispatcher_heartbeat_stale(now)
    if not stale:
        state["confirmed_stall_passes"] = 0
        return RestartDecision(False, "no stale dispatcher heartbeat; resetting arm counter"), state

    passes += 1
    state["confirmed_stall_passes"] = passes
    if passes < RESTART_REQUIRED_PASSES:
        return RestartDecision(False, "confirmed stall pass %d/%d; awaiting second confirming pass" % (passes, RESTART_REQUIRED_PASSES)), state

    if _in_cooldown(state, now):
        # A restart happened within the cooldown window yet the board is still
        # stalled-and-stale. The restart did not revive dispatch: escalate to a
        # human (photon iMessage) instead of looping another restart.
        reason = ("dispatcher still stalled %ds after prior restart — ESCALATE to photon iMessage "
                  "(restart did not revive dispatch)") % int(now - state.get("last_restart_at", now))
        return RestartDecision(False, reason, {"escalate": True}), state

    pid, detail = root_gateway_pid(now)
    if pid is None:
        # We cannot resolve the owner gateway to signal. Keep the counter so we
        # keep trying once the pid appears, but do NOT claim a restart happened.
        return RestartDecision(False, "cannot resolve dispatcher-owner pid: %s" % detail), state

    state["confirmed_stall_passes"] = 0
    state["last_restart_at"] = now
    evidence = {"pid": pid, "pid_detail": detail, "stalled_ready": len(stalled_ready)}
    return RestartDecision(
        True,
        "dispatcher-owner gateway restart: SIGUSR1 to root gateway pid=%s after confirmed stall (%d ready card(s), heartbeat stale)" % (pid, len(stalled_ready)),
        evidence,
    ), state


def _send_restart(decision, problems):
    """Send SIGUSR1 to the dispatcher-owner gateway pid recorded in the decision
    and record ONE Slack line describing what it did and why (with evidence).

    SIGUSR1 is the gateway's graceful drain-and-restart signal (cleanup-20260902
    precedent, gateway/restart.py). If the signal fails (process gone), we log a
    problem but do NOT let it masquerade as a successful restart — the next pass
    re-evaluates. If the restart does not revive dispatch, the next stale pass
    after the cooldown would normally hold; here we instead escalate to photon
    iMessage (costcap-watch pattern) so Richie is woken rather than us looping.
    """
    pid = decision.evidence.get("pid")
    try:
        os.kill(pid, SIGUSR1)
    except ProcessLookupError:
        problems.append(
            "RESTART SIGNAL FAILED: dispatcher-owner pid %s no longer exists; will not claim a restart." % pid
        )
        return
    except Exception as exc:  # noqa: BLE001
        problems.append(
            "RESTART SIGNAL FAILED: could not SIGUSR1 pid %s: %s" % (pid, exc)
        )
        return
    problems.append(decision.line)


def run(audit=False):
    problems = []
    con = _conn()

    # --- check 1: stale diagnosis -------------------------------------------------
    cards = active_cards(con)
    for c in cards:
        text = (c["body"] or "") + "\n" + (c["title"] or "")
        claimed = set()
        for pat, kind in CLAIM_PATTERNS:
            for m in pat.finditer(text):
                for g in m.groups():
                    if g and (kind == "symbol" or kind == "sql"):
                        # only treat looked-up method-ish identifiers as verifiable symbols
                        if re.fullmatch(r"[A-Za-z_]\w*", g or ""):
                            claimed.add(g)
        if not claimed:
            continue
        hits = symbols_on_disk(claimed)
        for sym, paths in hits.items():
            # a card cites a missing symbol that exists on disk -> stale premise
            problems.append(
                f"STALE DIAGNOSIS: {c['id']} ({c['status']}, @{c['assignee']}) cites "
                f"'{sym}' as missing/absent, but it resolves on disk: {paths[0]}. "
                f"Verify against the LIVE tree before spawning any rework."
            )

    # --- check 2: liveness / deadlock --------------------------------------------
    # 2a. archived parent gating a live card (permanent stall)
    rows = con.execute(
        "SELECT l.child_id, l.parent_id, t.status AS child_status "
        "FROM task_links l JOIN tasks t ON t.id=l.child_id "
        "JOIN tasks p ON p.id=l.parent_id "
        "WHERE p.status='archived' AND t.status IN ('blocked','todo','ready','review')"
    ).fetchall()
    for r in rows:
        problems.append(
            f"ARCHIVED-PARENT DEADLOCK: {r['child_id']} ({r['child_status']}) is gated on "
            f"archived parent {r['parent_id']} which can never complete. Unlink or re-parent."
        )

    # 2b. ready card with a live worker-eligible assignee but no run for too long.
    #     A card is only "stranded" if it has been ready for STALL_SECONDS and has
    #     either never run OR its last run is older than STALL_SECONDS. This avoids
    #     flagging a card that was unblocked moments ago.
    stalled_ready = []
    for c in cards:
        if c["status"] != "ready":
            continue
        anchor = c["last_run_ts"] or c["created_at"]
        if (NOW - anchor) > STALL_SECONDS:
            stalled_ready.append(c["id"])
            problems.append(
                f"STRANDED READY: {c['id']} (@{c['assignee']}, pri {c['priority']}) has been "
                f"'ready' with no recent run for >{STALL_SECONDS}s. Dispatcher may be stalled or "
                f"assignee unknown."
            )

    # 2c. unverified self-completed PM decision.
    #     An auto-decomposer spawns a jobsy-assigned card titled "Decide the X
    #     design" / "Approve the Y approach". A run self-completes it, posting
    #     "PM decision locked" / "option (b) approved" / N acceptance criteria
    #     as if the real PM (Jobsy) made the call. Neither Jobsy nor the owner
    #     did. Downstream workers treat it as authoritative. This is an
    #     AUTHORITY failure, distinct from stale-diagnosis: the code claims may
    #     all be TRUE on disk (they were here) — the fraud is the decision
    #     itself.
    #
    #     Scoped 2026-08-31: flag ONLY a card assigned to a PM whose decision
    #     authority is claimed but which was NOT completed by the assigned PM
    #     profile. The phantom signature is a non-PM (or auto-decomposer)
    #     completing a PM-assigned decision card. A card the legitimate PM
    #     actually completed is real PM work and must not be flagged — checking
    #     who ran the card, not just the title, removes the crying-wolf on
    #     genuine approvals while still catching the phantom.
    pm_cards = con.execute(
           "SELECT t.id, t.title, t.assignee, t.status, "
           " COALESCE((SELECT r.summary FROM task_runs r "
           "            WHERE r.task_id=t.id ORDER BY r.id DESC LIMIT 1), '') AS summary "
           "FROM tasks t WHERE t.assignee IN ('jobsy','weroll-pm','gen-pm') "
           "AND t.status IN ('done') "
           # The assigned-PM profile must NOT have been the completer. A phantom is
           # a non-PM/auto run closing a PM-assigned decision card; a real PM
           # completion is genuine work and is excluded.
           "AND NOT EXISTS (SELECT 1 FROM task_runs c "
           "                WHERE c.task_id=t.id AND c.outcome IN ('completed','done') "
           "                AND c.profile IN ('jobsy','weroll-pm','gen-pm')) "
           "AND (LOWER(COALESCE(t.title,'')) LIKE '%decide the%' "
           " OR LOWER(COALESCE(t.title,'')) LIKE '%approve the%' "
           " OR LOWER(COALESCE(t.title,'')) LIKE '%design for%' "
           " OR LOWER(COALESCE((SELECT r.summary FROM task_runs r "
           "            WHERE r.task_id=t.id ORDER BY r.id DESC LIMIT 1),'')) LIKE '%pm decision%' "
           " OR LOWER(COALESCE((SELECT r.summary FROM task_runs r "
           "            WHERE r.task_id=t.id ORDER BY r.id DESC LIMIT 1),'')) LIKE '%approved%' "
           " OR LOWER(COALESCE((SELECT r.summary FROM task_runs r "
           "            WHERE r.task_id=t.id ORDER BY r.id DESC LIMIT 1),'')) LIKE '%decision locked%')"
       ).fetchall()
    for r in pm_cards:
        text = f"{r['title'] or ''} {r['summary'] or ''}"
        # an owner-confirmation anchor (confirmed_by / signed-off / "owner approved")
        # demotes this from a flag
        if re.search(r"(?:confirmed_by|signed.?off|owner approved|owner_signed|quoted owner)", text, re.I):
            continue
        problems.append(
            f"UNVERIFIED PM DECISION: {r['id']} ({r['assignee']}, {r['status']}) titled "
            f"'{r['title'][:60]}' completed claiming decision authority "
            f"('{r['summary'][:60]}') with no owner-confirmation anchor. "
            f"Owner did not sign this; treat as ADVISORY not authoritative."
        )

    # 2d. completed card whose own handoff reports a test-suite teardown stall
    #     yet claims the full-suite gate pass. A "full pytest / full suite pass"
    #     verification gate is NOT met if the handoff itself says the suite
    #     "stalls", "hangs", "times out" at teardown. Focused tests passing is
    #     not equivalent to the suite gate. If a card's AC requires the full
    #     suite and its completion summary reports a stall/hang, that gate is
    #     open — flag it for a scoped suite fix (not a blanket skip).
    stall_cards = con.execute(
        "SELECT DISTINCT r.task_id AS id, r.summary, t.title, t.assignee "
        "FROM task_runs r JOIN tasks t ON t.id=r.task_id "
        "WHERE r.status IN ('done','completed') AND t.tenant='backupbrain' "
        "AND (LOWER(r.summary) LIKE '%teardown%' "
        "     OR LOWER(r.summary) LIKE '%suite%stall%' "
        "     OR LOWER(r.summary) LIKE '%suite%hang%' "
        "     OR LOWER(r.summary) LIKE '%cannot complete%' "
        "     OR LOWER(r.summary) LIKE '%could not complete%' "
        "     OR LOWER(r.summary) LIKE '%pytest%stall%' "
        "     OR LOWER(r.summary) LIKE '%pytest%hang%')"
    ).fetchall()
    for r in stall_cards:
        s = r["summary"] or ""
        # only flag when a FULL/SUITE teardown stall is reported AND the card
        # claims the full-suite gate as a verdict. Avoid innocent 'hang'/'stall'.
        full_suite_gate = re.search(
            r"(full\s+(pytest|suite)|pytest.*\b-q\b|\.venv/bin/python.*pytest)", s, re.I)
        teardown_stall = re.search(
            r"(teardown|cannot complete|could not complete|stalls? (after|at)|hangs? (after|at|during))",
            s, re.I)
        if full_suite_gate and teardown_stall:
            problems.append(
                f"SUITE-STALL GATE OPEN: {r['id']} ({r['assignee']}) titled "
                f"'{r['title'][:60]}' handoff reports a full-suite teardown "
                f"stall/hang but gates verdicts on that suite. Gate NOT met — "
                f"flag for a scoped suite fix (fixture/lifespan teardown), "
                f"not a blanket skip."
            )

    # 2e. blocked/gave_up card that actually has an approval on record.
    #     A worker's clean exit gets scored as a protocol_violation -> failure_limit
    #     -> "gave up" -> blocked, EVEN when the implementation was reviewed and
    #     approved (see the "cards already in review lane" loop). This flags a
    #     card whose status is blocked but that has a completed run (or a review
    #     child) whose summary shows approval, so an operator can re-promote it
    #     instead of it sitting stranded while the work is actually done.
    approve_runs = con.execute(
        "SELECT t.id AS task_id, t.status AS tstatus, t.assignee, "
        "       r.outcome, substr(r.summary,1,90) AS s "
        "FROM task_runs r JOIN tasks t ON t.id=r.task_id "
        "WHERE t.status IN ('blocked') "
        "AND r.outcome IN ('completed') "
        "AND (LOWER(r.summary) LIKE '%approved%' "
        "     OR LOWER(r.summary) LIKE '%thumbs up%' "
        "     OR LOWER(r.summary) LIKE '%approve %')"
    ).fetchall()
    seen = set()
    for r in approve_runs:
        if r["task_id"] in seen:
            continue
        seen.add(r["task_id"])
        problems.append(
            f"APPROVED-BUT-STRANDED: {r['task_id']} ({r['tstatus']}, @{r['assignee']}) is "
            f"blocked/gave_up but has a completed approval on record "
            f"('{r['s']}'). Re-promote it to its next legal lane (review/ready) "
            f"instead of leaving it stranded — the work is done and approved."
        )

    # 2f. parked triage decision awaiting PM acceptance.
    #     The routing fix parks decision-shaped auto-decomposer children in
    #     'triage' so a phantom can never auto-promote into a build. But a
    #     parked decision advances only when the PM explicitly accepts it
    #     (kanban_unblock). Nothing dispatches a triage card, so a decision
    #     that nobody accepts sits forever. Flag any triage card that is
    #     decision-shaped (or auto-decomposer-created) and older than
    #     PARKED_DECISION_SECONDS, so the operator/PM sees it and accepts or
    #     rejects it instead of it silently stranding.
    parked = con.execute(
        "SELECT id, title, assignee, created_at, created_by "
        "FROM tasks WHERE status='triage' "
        "AND (LOWER(COALESCE(title,'')) LIKE '%decide%' "
        "     OR LOWER(COALESCE(title,'')) LIKE '%approve the%' "
        "     OR LOWER(COALESCE(title,'')) LIKE '%ratify%' "
        "     OR LOWER(COALESCE(title,'')) LIKE '%spec the%' "
        "     OR LOWER(COALESCE(title,'')) LIKE '%amend the%' "
        "     OR created_by='auto-decomposer')"
    ).fetchall()
    for r in parked:
        age = NOW - (r["created_at"] or NOW)
        if age <= PARKED_DECISION_SECONDS:
            continue
        problems.append(
            f"PARKED DECISION: {r['id']} (triage, @{r['assignee']}) titled "
            f"'{r['title'][:60]}' has sat unaccepted for {int(age // 60)}m. "
            f"Accept it (kanban_unblock) or reject it — a parked triage card "
            f"never auto-advances."
        )

    con.close()

    # --- check 3: auto-restart the dispatcher-owner gateway on confirmed stall ---
    # Chains on fix-20260902-dispatcher-takeover. Only fires when BOTH a
    # confirmed-ready stall AND a stale dispatcher heartbeat hold on two
    # consecutive passes (RESTART_REQUIRED_PASSES). Guarded by the 30-min
    # cooldown and the fresh-pidfile refusal. Fail-safe: absent heartbeat file
    # (takeover not yet deployed) => no restart, alert-only as before.
    existing_state = read_json_state(RESTART_STATE)
    decision, new_state = evaluate_restart(existing_state, stalled_ready, NOW)
    if decision.should_restart:
        _send_restart(decision, problems)
    write_json_state(RESTART_STATE, new_state)

    if audit and not problems:
        return "OK — no stale-diagnosis, deadlock, unverified PM decision, suite-stall, approved-but-stranded, parked-decision card, or dispatcher restart needed.\n"
    return "\n".join(f"[kanban-liveness-watch] {p}" for p in problems)


if __name__ == "__main__":
    out = run(audit="--audit" in sys.argv)
    if out:
        print(out)
