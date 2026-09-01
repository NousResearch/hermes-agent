#!/usr/bin/env python3
"""Bob blocker-rate + review-falsifiability watchdog.

Why this exists
---------------
On 2026-08-22 the fleet moved off Western frontier models. Bob (the builder) took
a large capability drop. The mitigation is Rodge: a reviewer gating every merge.
That only works if someone watches whether Rodge is suddenly blocking far more
than it used to - otherwise the first signal is a bad merge.

2026-08-25 REPAIR
-----------------
This script had never counted a single review, and could not have.

  1. It read the ROOT state.db and filtered `sessions.profile_name = 'rodge'`.
     Root's profile_name is only ever NULL or '.hermes' - never a profile name.
     Kanban workers run with a profile-scoped HERMES_HOME, so Rodge's sessions
     live in ~/.hermes/profiles/rodge/state.db. The query matched zero rows.
  2. `reviews == 0` returned SILENTLY, which under the watchdog convention is
     indistinguishable from a healthy week. A permanently broken query looked
     exactly like a quiet one.

Both are fixed below: the correct DB is read, and a zero-review result is now
LOUD when Bob has actually been shipping.

2026-09-01 FALSIFIABILITY (this build)
--------------------------------------
The fleet reports 87.5% and 91% first-pass review approval. Both are raw
agreement on hand-picked batches. Computed across ALL cards since 29 Aug the
true figure is ~75%. Both numbers are true and they measure different
populations - but only one of them is the fleet's actual first-pass rate.

This script now computes the first-pass rate the way the reporter should have:
not from a hand-picked batch, but as a full-population count of review rounds
in the window, over ALL reviewed cards, with the denominator always printed
next to the percentage. A rate without its N is not a measurement.

Two new outputs, both derived rather than selected:

A. First-pass approval from the KANBAN lifecycle (all cards, no sampling).
   A review round opens on `review_requested` and closes on `completed`
   (approval) or `changes_requested`. The rate and per-author split are
   computed over every round in the window.

B. Spec-clause attribution on Rodge's Critical findings. Rodge's verdict
   template (SOUL change landing 2026-09-01) requires each Critical to carry
   a `spec:` clause - the acceptance criterion it violates. A Critical with
   no `spec:` is the reviewer checking the diff against itself rather than
   against the spec, so this build warns when any Critical lacks that clause.

Watchdog convention: silent unless something is wrong; empty stdout = silent
tick. --audit prints the full measurement for smoke-testing.

Tuning
------
THRESHOLD           - blocker rate that triggers an alert.
FIRST_PASS_MIN      - first-pass approval rate below this triggers an alert.
MIN_REVIEWS         - don't cry wolf on a thin week.
CLAUSE_MIN_SHARE    - share of Rodge's Criticals carrying a spec: clause below
                      which we warn (1.0 = any un-claused Critical is loud).
"""

import json
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.expanduser("~/.hermes/scripts"))
try:  # the live copy pairs with comms_style.py in ~/.hermes/scripts
    from comms_style import clean  # noqa: E402
except Exception:  # noqa: BLE001 - versioned copy runs standalone
    def clean(text):
        """Hyphen-only, whitespace-tidied fallback so the versioned watchdog is
        runnable outside the live scripts dir."""
        for ch in ("\u2014", "\u2013", "\u2010", "\u2011", "\u2012", "\u2015"):
            text = text.replace(ch, "-")
        return re.sub(r"[ \t]+\n", "\n", text)

HERMES_HOME = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
# Kanban workers get a profile-scoped HERMES_HOME. If we were launched inside a
# profile, walk back up to the real root (a root has config.yaml AND profiles/).
if not os.path.isdir(os.path.join(HERMES_HOME, "profiles")):
    _up = os.path.dirname(os.path.dirname(HERMES_HOME))
    if os.path.isdir(os.path.join(_up, "profiles")):
        HERMES_HOME = _up

REVIEWER_PROFILE = "rodge"
BUILDER_PROFILE = "bob"
# Rodge's own store - NOT the root one.
STATE_DB = os.path.join(HERMES_HOME, "profiles", REVIEWER_PROFILE, "state.db")
BUILDER_DB = os.path.join(HERMES_HOME, "profiles", BUILDER_PROFILE, "state.db")
KANBAN_DB = os.path.join(HERMES_HOME, "kanban.db")
BASELINE = os.path.join(HERMES_HOME, "state", "bob-blocker-baseline")

WINDOW_DAYS = 7
THRESHOLD = 0.45
MIN_REVIEWS = 4
FIRST_PASS_MIN = 0.60
CLAUSE_MIN_SHARE = 1.0
# If Bob shipped at least this many turns in the window and Rodge produced zero
# parseable reviews, that is a broken contract, not a quiet week.
MIN_BUILDER_ACTIVITY = 3

CRITICAL = re.compile(r"^##\s*Critical\s*\(Blockers\)\s*$", re.M | re.I)
MAJOR = re.compile(r"^##\s*Major\s*\(Should Fix\)\s*$", re.M | re.I)
ITEM = re.compile(r"^\s*-\s*\[[ xX]\]\s*\S", re.M)
SPEC_CLAUSE = re.compile(r"spec\s*:\s*\S", re.I)
REVIEW_ISH = re.compile(r"^##\s*(Summary|Verification|Security Notes)", re.M | re.I)


def section_body(text, header_re):
    m = header_re.search(text)
    if not m:
        return ""
    rest = text[m.end():]
    nxt = re.search(r"^##\s+", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


def assistant_messages(db, cutoff):
    """All assistant turns in the window. The DB is already profile-scoped, so
    there is nothing to filter on profile_name - which is exactly the bug."""
    if not os.path.exists(db):
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        return [r[0] for r in con.execute(
            """
            SELECT m.content
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE m.role = 'assistant'
              AND m.content IS NOT NULL
              AND COALESCE(s.last_activity_at, s.started_at, 0) >= ?
            """, (cutoff,)).fetchall()]
    except Exception as exc:  # noqa: BLE001
        print(f"warning: bob watchdog could not read {db} - {exc}")
        return None


# --------------------------------------------------------------------------
# Kanban lifecycle: full-population first-pass approval (no sampling).
# --------------------------------------------------------------------------
def load_review_events(cutoff):
    """All lifecycle events for cards reviewed in the window, keyed by task."""
    if not os.path.exists(KANBAN_DB):
        return {}
    try:
        con = sqlite3.connect(f"file:{KANBAN_DB}?mode=ro", uri=True)
        rows = con.execute(
            """
            SELECT task_id, kind, payload, id
            FROM task_events
            WHERE created_at >= ?
              AND kind IN ('review_requested', 'completed', 'changes_requested')
            ORDER BY task_id, id
            """, (cutoff,)).fetchall()
    except Exception as exc:  # noqa: BLE001
        print(f"warning: bob watchdog could not read {KANBAN_DB} - {exc}")
        return None
    by_task = defaultdict(list)
    for task_id, kind, payload, eid in rows:
        by_task[task_id].append({"kind": kind, "payload": payload, "id": eid})
    return dict(by_task)


def _implementer_of(payload):
    try:
        p = json.loads(payload) if payload else {}
        val = p.get("implementer") if isinstance(p, dict) else None
        return val if isinstance(val, str) and val.strip() else None
    except (json.JSONDecodeError, TypeError):
        return None


def first_pass_metrics(cutoff):
    """Return dict of full-population review-round metrics for the window.

    A review round opens on `review_requested`; its verdict is the next
    `completed` (approval) or `changes_requested` for that task. Rounds still
    awaiting a verdict are not counted (not yet a measurement). Author is the
    `implementer` recorded on the round's review_requested payload.
    """
    by_task = load_review_events(cutoff)
    if by_task is None:
        return None
    totals = {"approvals": 0, "changes": 0, "rounds": 0}
    per_author = defaultdict(lambda: {"approvals": 0, "changes": 0})
    for evs in by_task.values():
        awaiting = False
        author = None
        for e in evs:
            kind = e["kind"]
            if kind == "review_requested":
                awaiting = True
                author = _implementer_of(e["payload"])
                continue
            if kind in ("completed", "changes_requested") and awaiting:
                awaiting = False
                totals["rounds"] += 1
                bucket = per_author[author or "unknown"]
                if kind == "completed":
                    totals["approvals"] += 1
                    bucket["approvals"] += 1
                else:
                    totals["changes"] += 1
                    bucket["changes"] += 1
    return {
        "totals": totals,
        "per_author": dict(per_author),
    }


# --------------------------------------------------------------------------
# Spec-clause attribution on Rodge's Critical findings.
# --------------------------------------------------------------------------
def spec_clause_stats(review_messages):
    """Share of Rodge's Critical findings that carry a `spec:` clause.

    Returns (critical_findings, with_clause, share) over all parseable reviews.
    """
    crit_findings = 0
    with_clause = 0
    for content in review_messages:
        if not content or not CRITICAL.search(content):
            continue
        body = section_body(content, CRITICAL)
        for line in body.splitlines():
            if not ITEM.search(line):
                continue
            crit_findings += 1
            if SPEC_CLAUSE.search(line):
                with_clause += 1
    share = (with_clause / crit_findings) if crit_findings else 1.0
    return crit_findings, with_clause, share


def render_author_breakdown(metrics):
    if not metrics:
        return ()
    rows = []
    for author, b in sorted(metrics["per_author"].items()):
        ap, ch = b["approvals"], b["changes"]
        tot = ap + ch
        if tot == 0:
            continue
        rows.append(
            f"{author}: {ap}/{tot} approved ({ap/tot:.0%}) "
            f"- {ch} reworked"
        )
    if not rows:
        return ()
    return tuple([""] + rows)


def main():
    audit = "--audit" in sys.argv
    cutoff = time.time() - WINDOW_DAYS * 86400

    # Existing: parse Rodge's verdict content.
    rows = assistant_messages(STATE_DB, cutoff)
    if rows is None:
        print(clean(
            f"warning: bob watchdog cannot measure: {STATE_DB} is missing or "
            f"unreadable.\n\n"
            f"Rodge's reviews are the only gate on Bob's merges. Until this "
            f"reads, nothing is watching that gate."))
        return 0

    reviews = blocked = major_only = 0
    malformed = 0
    for content in rows:
        if not content:
            continue
        if not CRITICAL.search(content):
            # Looks like a review but has no contract heading? Count it.
            if REVIEW_ISH.search(content):
                malformed += 1
            continue
        reviews += 1
        crit = bool(ITEM.search(section_body(content, CRITICAL)))
        maj = bool(ITEM.search(section_body(content, MAJOR)))
        if crit:
            blocked += 1
        elif maj:
            major_only += 1

    # New: first-pass approval across ALL reviewed cards + per author.
    fp = first_pass_metrics(cutoff)
    criticals, with_clause, clause_share = spec_clause_stats(rows)

    if audit:
        print(f"window            : last {WINDOW_DAYS} days")
        print(f"reviewer db       : {STATE_DB}")
        print(f"assistant turns   : {len(rows)}")
        print(f"parseable reviews : {reviews}  (heading present)")
        print(f"  -> with blocker : {blocked}")
        print(f"  -> major only   : {major_only}")
        print(f"MALFORMED reviews : {malformed}  (review-shaped, no '## Critical (Blockers)')")
        if fp is not None:
            t = fp["totals"]
            rate = (t["approvals"] / t["rounds"]) if t["rounds"] else 0.0
            print("")
            print(f"FIRST-PASS (all cards, window-denominator):")
            print(f"  approved      : {t['approvals']} / {t['rounds']} review rounds")
            print(f"  changes-req   : {t['changes']}")
            print(f"  first-pass    : {rate:.1%}  (denominator = {t['rounds']})")
            for a, b in sorted(fp["per_author"].items()):
                tot = b["approvals"] + b["changes"]
                if tot:
                    print(f"    {a}: {b['approvals']}/{tot} ({b['approvals']/tot:.0%})")
        print(f"CRITICAL spec: : {with_clause}/{criticals} carry a spec: clause "
              f"(share {clause_share:.0%})")
        return 0

    builder_rows = assistant_messages(BUILDER_DB, cutoff) or []

    # --- the case that used to be silent -----------------------------------
    if reviews == 0:
        if malformed > 0:
            print(clean(
                f"warning: Rodge produced {malformed} review-shaped outputs in "
                f"the last {WINDOW_DAYS} days and NONE of them carried the "
                f"'## Critical (Blockers)' heading.\n\n"
                f"The heading and the '- [ ] file:line' items are a machine "
                f"contract - this watchdog parses them literally. Prose instead "
                f"of that structure means every review reads as zero blockers, "
                f"whatever Rodge actually found.\n\n"
                f"Fix the review template enforcement in Rodge's SOUL or the "
                f"review card body, not this script."))
            return 0
        if len(builder_rows) >= MIN_BUILDER_ACTIVITY:
            print(clean(
                f"warning: Bob was active in the last {WINDOW_DAYS} days "
                f"({len(builder_rows)} assistant turns) but Rodge produced "
                f"ZERO reviews.\n\n"
                f"Either review cards are not being created, or the reviewer "
                f"is not running. Bob's merges are currently ungated."))
            return 0
        return 0  # genuinely quiet: no builder activity either

    rate = blocked / reviews

    prev = None
    try:
        with open(BASELINE) as fh:
            prev = float(fh.read().strip())
    except Exception:  # noqa: BLE001
        pass
    try:
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        with open(BASELINE, "w") as fh:
            fh.write(f"{rate:.4f}")
    except Exception:  # noqa: BLE001
        pass

    warns = []

    # Malformed reviews are worth flagging even when the rate looks fine.
    if malformed > 0 and reviews > 0:
        warns.append(clean(
            f"warning: {malformed} of Rodge's {reviews + malformed} reviews in "
            f"the last {WINDOW_DAYS} days omitted the '## Critical (Blockers)' "
            f"heading and were not counted. Any blockers in those reviews are "
            f"invisible to this watchdog."))

    # First-pass approval - full population, denominator always shown.
    if fp is not None:
        t = fp["totals"]
        fp_rate = (t["approvals"] / t["rounds"]) if t["rounds"] else 0.0
        if t["rounds"] >= MIN_REVIEWS and fp_rate < FIRST_PASS_MIN:
            warn_lines = [
                f"FIRST-PASS approval across ALL reviewed cards is {fp_rate:.1%} "
                f"over the last {WINDOW_DAYS} days (N = {t['rounds']} review "
                f"rounds; {t['approvals']} approved, {t['changes']} reworked).",
                "",
                f"That is below the {FIRST_PASS_MIN:.0%} floor. This is the "
                f"full-population figure - every reviewed card in the window, "
                f"not a hand-picked batch.",
            ] + list(render_author_breakdown(fp))
            warns.append(clean("\n\n".join(warn_lines)))

    # Spec-clause attribution: a Critical with no spec: clause is the reviewer
    # checking the diff against itself instead of against the spec.
    if criticals > 0 and clause_share < CLAUSE_MIN_SHARE:
        warns.append(clean(
            f"{criticals - with_clause} of Rodge's {criticals} Critical findings "
            f"in the last {WINDOW_DAYS} days carry NO 'spec: <clause>' "
            f"reference.\n\n"
            f"The verdict contract requires every Critical to cite the "
            f"acceptance criterion it violates. A Critical with no clause is "
            f"the reviewer checking the diff against itself rather than against "
            f"the spec. Share of Criticals with a clause: {clause_share:.0%} "
            f"(N = {criticals})."))

    if reviews < MIN_REVIEWS or rate < THRESHOLD:
        # Block rate is healthy; still surface the falsifiability warnings.
        if warns:
            print("\n\n".join(warns))
        return 0

    delta = ""
    if prev is not None:
        delta = f" (last week {prev:.0%}, {'+' if rate >= prev else ''}{(rate - prev) * 100:.0f} pts)"

    blocker_msg = clean(
        f"Bob blocker rate is {rate:.0%} over the last {WINDOW_DAYS} days{delta}.\n\n"
        f"Rodge blocked {blocked} of {reviews} reviews "
        f"({major_only} more had Major findings but no blocker).\n\n"
        f"Threshold is {THRESHOLD:.0%}. Since 2026-08-25 Bob runs "
        f"`openai/gpt-5.6-luna` and Rodge runs `moonshotai/kimi-k2.7-code`. If "
        f"this is sustained rather than one bad week, look at Bob's context "
        f"discipline first (tool_output.max_bytes, turn budget) - the "
        f"2026-08-25 measurement showed Bob's old model passed the same edit "
        f"task cleanly in isolation and only failed under accumulated context.")
    warns.insert(0, blocker_msg)
    print("\n\n".join(warns))
    return 0


if __name__ == "__main__":
    sys.exit(main())