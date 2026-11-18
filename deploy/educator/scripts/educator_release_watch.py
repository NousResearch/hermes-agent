"""release-watch pre-run wake gate (educator profile, every 20 minutes).

No LLM, no writes outside the profile. Each run:

1. refreshes both read-only checkouts (``workspace/checkouts/…``);
2. reads the PRDs (Detailed Bucket Features) whose ``Release Comms Status``
   is ``Not started`` or ``In progress``;
3. reads the status of each PRD's ``JIRA`` tickets;
4. reads the Release Assets rows edited since the last run;
5. diffs all of it against its state file;
6. prints the wake-gate payload — ``{"wakeAgent": false}`` when nothing
   changed (the run costs $0), otherwise ``{"wakeAgent": true, "events": …}``.

Events (``type``):

- ``baseline`` — first run only: PRDs already eligible for a Kit. Nothing is
  drafted; the agent asks the team which (if any) to backfill.
- ``kit_start`` — a ``Not started`` PRD has a ticket at Ready for staging or
  later. ``retry: true`` when the PRD is still ``Not started`` a day after the
  first ``kit_start`` (the earlier run did not finish the Kit).
- ``ticket_status`` — a linked ticket moved into a watched status.
- ``all_done`` — every ticket of an ``In progress`` PRD is Done (once; re-armed
  if a ticket reopens).
- ``asset_approved`` — a Release Assets row moved to an approved status.
- ``asset_edited_after_approval`` — a human edited an approved row.
  Both carry the approval measurement (``educator_wake.measure_approval``):
  ``approved_chars``, ``snapshot_missing``, and with a snapshot
  ``draft_chars``, ``chars_changed``, ``chars_changed_exact``, ``baseline``
  (``draft`` | ``redraft``), ``assumption_chars`` when a re-draft exists; or
  ``measure_error``. The script also keeps the evidence itself (agent
  read-only, ``workspace/metrics/``): the Draft-time baseline, the approved
  copy and the row's line in ``edit-rate.csv``.
- ``veto_due`` — a veto-level action waited out its window; the agent runs the
  named tool call (the script has no Slack token to do it).
- ``notebook_not_updated`` — the last run that woke the agent did not rewrite
  ``workspace/notebook.md``.
- ``checkout_error`` / ``source_error`` / ``config_error`` — reported once per
  distinct error, again every 6 hours while it persists (``persisting_since``).
- ``gave_up`` — an event that failed ``MAX_ATTEMPTS`` runs in a row: the agent
  posts once that a human is needed (``event``, ``last_error``).

When the previous woken run **failed**, its events are carried over
(``carried_over: true``, ``attempt``) so a crash never loses a trigger, up
to ``MAX_ATTEMPTS`` attempts. The one-off ``baseline`` post is never carried:
a run that posted it and still ended failed must not post it again.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import educator_wake as w  # noqa: E402  (sibling module)

JOB = "release-watch"
SCRIPT_NAME = "educator_release_watch.py"
KIT_STATUSES = ("Not started", "In progress")
NOT_STARTED, IN_PROGRESS = KIT_STATUSES
# Jira statuses compared case-insensitively ("Ready for staging" in BA).
WATCHED = frozenset({"ready for staging", "in staging", "done"})
DONE = frozenset({"done"})
KIT_RETRY_AFTER = timedelta(hours=24)
VETO_REMIND_AFTER = timedelta(hours=1)
ASSET_OVERLAP = timedelta(minutes=5)
ERROR_RESURFACE_AFTER = timedelta(hours=6)
# Rows past Draft that never get a fallback baseline (not a reviewed draft).
NO_BASELINE_STATUSES = frozenset({"Published", "Won't do", "Not started"})
MAX_KIT_STARTS_PER_RUN = 3
MAX_ATTEMPTS = 3  # a carried event is tried at most this many times in all
NOT_CARRIED = frozenset({"checkout_error", "source_error", "config_error",
                         "notebook_not_updated", "gave_up",
                         # one-off post: re-sending it after a failed run that
                         # already posted it would duplicate the message
                         "baseline"})


def _settings(cfg: dict) -> dict:
    notion = cfg.get("notion") if isinstance(cfg.get("notion"), dict) else {}
    guard = notion.get("write_guard") if isinstance(notion.get("write_guard"), dict) else {}
    prds = guard.get("prds") if isinstance(guard.get("prds"), dict) else {}
    ru = cfg.get("release_updates") if isinstance(cfg.get("release_updates"), dict) else {}
    approved = ru.get("approved_statuses") or ["Approved"]
    if isinstance(approved, str):
        approved = [approved]
    kits = guard.get("release_kits") if isinstance(guard.get("release_kits"), dict) else {}
    return {
        "draft_status": str(kits.get("draft_status") or "Draft"),
        "prd_sources": [str(x) for x in prds.get("data_sources") or []],
        "prd_status_prop": str(prds.get("property") or "Release Comms Status"),
        "prd_jira_prop": str(ru.get("prd_jira_property") or "JIRA"),
        "assets_source": str(ru.get("release_assets_data_source_id") or ""),
        "status_prop": str(ru.get("status_property") or "Status"),
        "type_prop": str(ru.get("asset_type_property") or "Asset type"),
        "approved": {str(a) for a in approved} - {"Published"},
    }


def _norm(status: Optional[str]) -> str:
    return (status or "").strip().lower()


def _error_events(state: dict, source: str, key: str, message: Optional[str],
                  now: datetime) -> list[dict]:
    """Report a distinct error once, then again every ERROR_RESURFACE_AFTER
    while it persists (``persisting_since``); clear it when the source
    recovers. A lasting outage therefore keeps failing a run now and then,
    so the scheduler's incident alert and owner DM fire, without waking the
    agent every 20 minutes."""
    errors = state.setdefault("errors", {})
    prefix = f"{source}:{key}:"
    if message is None:
        for sig in [s for s in errors if s.startswith(prefix)]:
            errors.pop(sig)
        return []
    sig = prefix + message[:200]
    rec = errors.get(sig)
    if isinstance(rec, str):  # state written by an earlier version
        rec = {"since": rec, "reported": rec}
    for old in [s for s in errors if s.startswith(prefix) and s != sig]:
        errors.pop(old)
    if rec and now - (w.parse_ts(rec.get("reported")) or now) < ERROR_RESURFACE_AFTER:
        errors[sig] = rec
        return []
    since = (rec or {}).get("since") or w.iso(now)
    errors[sig] = {"since": since, "reported": w.iso(now)}
    if source == "checkout":
        ev = {"type": "checkout_error", "repo": key, "error": message}
    elif source == "config":
        ev = {"type": "config_error", "error": message}
    else:
        ev = {"type": "source_error", "source": source, "error": message}
    if rec:
        ev["persisting_since"] = since
    return [ev]


def _read_prds(s: dict) -> list[dict]:
    prds = []
    for ds in s["prd_sources"]:
        schema = w.notion_get(f"data_sources/{ds}")
        kind = ((schema.get("properties") or {}).get(s["prd_status_prop"]) or {}).get("type") or "status"
        flt = {"or": [{"property": s["prd_status_prop"], kind: {"equals": v}} for v in KIT_STATUSES]}
        for page in w.notion_query(ds, flt):
            props = page.get("properties") or {}
            people = (props.get("Comm Owner") or {}).get("people") or []
            prds.append({
                "id": str(page.get("id")),
                "title": w.page_title(page),
                "url": page.get("url"),
                "status": w.prop_text(props.get(s["prd_status_prop"])),
                "keys": w.jira_keys(w.prop_text(props.get(s["prd_jira_prop"]))),
                "assets": w.prop_names(props.get("Release Comms Assets")),
                "comm_owner": [{"id": p.get("id"), "name": p.get("name")} for p in people
                               if isinstance(p, dict)],
            })
    return prds


def _kit_event(prd: dict, statuses: dict, retry: bool) -> dict:
    return {"type": "kit_start", "prd_id": prd["id"], "prd_title": prd["title"], "prd_url": prd["url"],
            "assets": prd["assets"], "comm_owner": prd["comm_owner"],
            "jira": {k: statuses.get(k) for k in prd["keys"]}, "retry": retry}


def _prd_events(state: dict, prds: list[dict], statuses: dict, now: datetime,
                baseline: bool) -> tuple[list[dict], int]:
    events: list[dict] = []
    deferred = 0
    old_tickets = state.get("tickets") or {}
    old_prds = state.get("prds") or {}
    new_prds: dict = {}
    eligible = []
    kit_budget = MAX_KIT_STARTS_PER_RUN
    for prd in prds:
        prev = old_prds.get(prd["id"]) or {}
        rec = {"status": prd["status"], "keys": prd["keys"], "kit": prev.get("kit"),
               "all_done": prev.get("all_done", False)}
        known = [k for k in prd["keys"] if statuses.get(k) is not None]
        reached = any(_norm(statuses.get(k)) in WATCHED for k in known)

        if not baseline:
            for key in known:
                before, after = old_tickets.get(key), statuses[key]
                if before is not None and _norm(before) != _norm(after) and _norm(after) in WATCHED:
                    events.append({"type": "ticket_status", "prd_id": prd["id"], "prd_title": prd["title"],
                                   "key": key, "from": before, "to": after})

        if prd["status"] == NOT_STARTED and reached:
            kit = rec["kit"] or {}
            if baseline:
                eligible.append(prd)
                rec["kit"] = {"at": w.iso(now), "reason": "baseline"}
            else:
                first = not kit
                due_retry = (kit.get("reason") == "kit_start"
                             and now - (w.parse_ts(kit.get("at")) or now) >= KIT_RETRY_AFTER)
                if first or due_retry:
                    if kit_budget > 0:
                        kit_budget -= 1
                        events.append(_kit_event(prd, statuses, retry=not first))
                        rec["kit"] = {"at": w.iso(now), "reason": "kit_start"}
                    else:
                        deferred += 1

        all_done = bool(known) and len(known) == len(prd["keys"]) and all(
            _norm(statuses[k]) in DONE for k in known)
        if not all_done:
            rec["all_done"] = False
        elif prd["status"] == IN_PROGRESS and not rec["all_done"]:
            rec["all_done"] = True
            if not baseline:
                events.append({"type": "all_done", "prd_id": prd["id"], "prd_title": prd["title"],
                               "prd_url": prd["url"], "keys": prd["keys"]})
        new_prds[prd["id"]] = rec

    if baseline and eligible:
        events.append({"type": "baseline", "eligible_prds": [
            {"prd_id": p["id"], "prd_title": p["title"], "prd_url": p["url"],
             "jira": {k: statuses.get(k) for k in p["keys"]}, "assets": p["assets"]} for p in eligible]})
    state["prds"] = new_prds
    tracked = {k for p in prds for k in p["keys"]}
    state["tickets"] = {k: v for k, v in statuses.items() if k in tracked and v is not None}
    return events, deferred


def _asset_events(state: dict, s: dict, now: datetime) -> list[dict]:
    cursor = w.parse_ts(state.get("assets_cursor"))
    if cursor is None or not s["assets_source"]:
        state["assets_cursor"] = w.iso(now)
        return []
    bot = state.get("bot_user_id")
    if not bot:
        bot = str(w.notion_get("users/me").get("id") or "")
        state["bot_user_id"] = bot
    flt = {"timestamp": "last_edited_time",
           "last_edited_time": {"on_or_after": w.notion_ts(cursor - ASSET_OVERLAP)}}
    rows = w.notion_query(s["assets_source"], flt)
    assets = state.setdefault("assets", {})
    events = []
    for row in rows:
        rid = str(row.get("id"))
        props = row.get("properties") or {}
        status = w.prop_text(props.get(s["status_prop"]))
        edited = str(row.get("last_edited_time") or "")
        editor = str((row.get("last_edited_by") or {}).get("id") or "")
        prev = assets.get(rid) or {}
        info = {"row_id": rid, "title": w.page_title(row), "url": row.get("url"), "status": status,
                "asset_type": w.prop_text(props.get(s["type_prop"])),
                "destination": w.prop_text(props.get("Destination")),
                "product": ";".join(w.prop_names(props.get("Product")))}
        try:
            if status == s["draft_status"]:
                w.take_baseline(rid, info["asset_type"], status, now)
            elif (status not in s["approved"] and status not in NO_BASELINE_STATUSES
                  and not w.has_baseline(rid)):
                w.take_baseline(rid, info["asset_type"], status, now)
        except (w.SourceError, ValueError) as e:
            events += _error_events(state, "notion", "baseline", f"baseline snapshot failed: {e}", now)
        if status in s["approved"] and prev.get("status") not in s["approved"]:
            events.append({"type": "asset_approved", **info, **w.measure_approval(info, now)})
        elif (status in s["approved"] and prev.get("edited") and edited != prev.get("edited")
              and editor and editor != bot):
            events.append({"type": "asset_edited_after_approval", "edited_at": edited, **info,
                           **w.measure_approval(info, now)})
        assets[rid] = {"status": status, "edited": edited}
    state["assets_cursor"] = w.iso(now)
    return events


def _veto_events(state: dict, now: datetime) -> list[dict]:
    seen = state.setdefault("veto_seen", {})
    events = []
    for action, call in w.VETO_TOOL_CALLS.items():
        ids = sorted(w.due_pending_ids(action))
        if not ids:
            seen.pop(action, None)
            continue
        sig = ",".join(ids)
        last = seen.get(action) or {}
        if last.get("sig") == sig and now - (w.parse_ts(last.get("at")) or now) < VETO_REMIND_AFTER:
            continue
        seen[action] = {"sig": sig, "at": w.iso(now)}
        events.append({"type": "veto_due", "action": action, "pending_ids": ids, "tool_call": call})
    return events


def _event_id(ev: dict) -> tuple:
    return (ev.get("type"), ev.get("prd_id"), ev.get("row_id"), ev.get("key"), ev.get("action"))


def _carry_over(state: dict, previous: dict, have: set) -> list[dict]:
    """Events of a failed woken run, carried into this one — each at most
    MAX_ATTEMPTS times in all. Then ONE ``gave_up`` (with the scheduler's
    last error) replaces it, so a broken step cannot loop every 20 minutes
    forever on an essential, budget-exempt job. Errors, notebook checks,
    gave_up itself and the one-off baseline are never carried: they are
    re-detected or best-effort."""
    out = []
    for ev in (state.get("last_wake") or {}).get("events") or []:
        if ev.get("type") in NOT_CARRIED or _event_id(ev) in have:
            continue
        attempt = int(ev.get("attempt") or 1)
        original = {k: v for k, v in ev.items() if k not in ("carried_over", "attempt")}
        if attempt >= MAX_ATTEMPTS:
            out.append({"type": "gave_up", "attempts": attempt, "event": original,
                        "last_error": str(previous.get("error") or "")[:300]})
        else:
            out.append({**original, "carried_over": True, "attempt": attempt + 1})
    return out


def main(now: Optional[datetime] = None) -> dict:
    now = now or w.utcnow()
    state = w.load_state(JOB)
    baseline = state is None
    state = state or {"version": 1}
    s = _settings(w.load_profile_config())
    w.TRUNCATED_QUERIES.clear()
    events: list[dict] = []
    extra: dict = {}

    for repo, err in w.refresh_checkouts().items():
        events += _error_events(state, "checkout", repo, err, now)

    missing_cfg = [k for k, v in (("notion.write_guard.prds.data_sources", s["prd_sources"]),
                                  ("release_updates.release_assets_data_source_id", s["assets_source"]))
                   if not v]
    events += _error_events(state, "config", "sources",
                            ("config.yaml has no " + " / ".join(missing_cfg) + " — nothing is watched")
                            if missing_cfg else None, now)

    try:
        prds = _read_prds(s)
        events += _error_events(state, "notion", "read", None, now)
        try:
            keys = sorted({k for p in prds for k in p["keys"]})
            statuses = w.jira_statuses(keys)
            events += _error_events(state, "jira", "read", None, now)
            prd_events, deferred = _prd_events(state, prds, statuses, now, baseline)
            events += prd_events
            if deferred:
                extra["deferred_kit_starts"] = deferred
            missing = [k for k in keys if statuses.get(k) is None]
            if missing:
                extra["missing_jira_keys"] = missing
        except w.SourceError as e:
            events += _error_events(state, e.source, "read", str(e), now)
        events += _asset_events(state, s, now)
    except w.SourceError as e:
        events += _error_events(state, e.source, "read", str(e), now)

    events += _error_events(
        state, "notion", "truncated",
        (f"a Notion query stopped at its row cap ({', '.join(sorted(set(w.TRUNCATED_QUERIES)))}); "
         "results are incomplete") if w.TRUNCATED_QUERIES else None, now)
    events += _veto_events(state, now)

    job_id = w.find_job_id(SCRIPT_NAME)
    previous = w.previous_woken_run(job_id, state)
    if previous is not None:
        if w.notebook_missed(previous):
            events.append({"type": "notebook_not_updated", "job": JOB,
                           "last_run_started": previous.get("started_at") or previous.get("claimed_at")})
        if previous.get("status") == "failed":
            extra["previous_run_failed"] = True
            events += _carry_over(state, previous, {_event_id(e) for e in events})

    if not events:
        w.save_state(JOB, state)
        return w.emit({"wakeAgent": False})
    state["last_wake"] = {"at": w.iso(now), "events": events}
    w.save_state(JOB, state)
    return w.emit({"wakeAgent": True, "job": JOB, "now": w.iso(now), "events": events, **extra})


if __name__ == "__main__":
    main()
