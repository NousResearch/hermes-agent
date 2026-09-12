#!/usr/bin/env python3
"""Assignee-mismatch auditor watchdog for the kanban fleet (fix-C2, t_9efe82a4).

Detection complement to the mint-time guard (C1). C1 stops NEW misassignments at
mint; this daily scan catches any that still slip through — an LLM emit under an
override, a lane verb the role-map does not cover, or a card minted before C1
shipped. It only FLAGS and, when a mismatch is actionable, BLOCKS for the PM to
re-route. It never mutates another card's assignee.

What it reads
-------------
For every card created in the window it takes the MINT-TIME assignee from the
`created` event in task_events — NOT the current tasks.assignee column, which is
reassigned to the reviewer on request_review and would manufacture false
positives. Auto-decomposer children (created payload `{"by":"auto-decomposer",
...}`) carry no mint assignee at all: the record is a gap, and routing for those
is decided post-create by the decompose pipeline, so they are reported separately
(--gap) and never flagged as a conflict — keeping false positives near zero, the
card's calibration requirement.

The lane role-map is the same one C1 uses:
    build->bob, review->rodge, verify->steve-o, design->karl,
    deploy->default(held), PM/triage->jobsy.

A card is FLAGGED only when the mint assignee is a real worker/receiver profile
(bob/rodge/steve-o/karl/jobsy/axel/builder) AND it lands on a DIFFERENT lane than
the title/body implies AND the pair is not a documented legitimate cross-lane
(run cards with an explicit `assignee_override`, deploy->bob, triage->jobsy).
Legit review->rodge, verify->steve-o, triage->jobsy and deploy->default cards are
never flagged even when the reviewer/verifier profile differs from mint.

Action taken
------------
For each flagged card still actionable (status not done/archived, no completed
run), the script posts a `## [routing-audit] mismatch` comment naming
expected-vs-actual and blocks it kind=needs_input for Jobsy to re-route.
Idempotent: it writes a marker into the comment (task id + window) and skips a
card that already carries this run's marker, so a daily re-scan never dups.

Usage:
    assignee-mismatch-watch.py                 # silent cron default (last 24h)
    assignee-mismatch-watch.py --days 7        # 7-day baseline, report to stdout
    assignee-mismatch-watch.py --audit         # always print full mismatch table
    assignee-mismatch-watch.py --selftest      # exit 0/1 on self-test fixtures
    assignee-mismatch-watch.py --apply         # comment+block actionable flags
    assignee-mismatch-watch.py --gap           # also list auto-decomposer gaps

Exit code is 0 unless --selftest fails. Watchdog pattern: empty stdout = silent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time

HERMES_HOME = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
if not os.path.isdir(os.path.join(HERMES_HOME, "profiles")):
    _up = os.path.dirname(os.path.dirname(HERMES_HOME))
    if os.path.isdir(os.path.join(_up, "profiles")):
        HERMES_HOME = _up

KANBAN_DB = os.environ.get("KANBAN_DB") or os.path.join(HERMES_HOME, "kanban.db")

# ---------------------------------------------------------------------------
# Lane role-map (mirrors C1: tools/kanban_tools.py lint + kanban_decompose.py)
# ---------------------------------------------------------------------------
ROLE_MAP = {
    "build": "bob",
    "review": "rodge",
    "verify": "steve-o",
    "design": "karl",
    "deploy": "default",
    "pm": "jobsy",
}

# Real worker/receiver profiles that can be wrongly routed (flag candidates).
WORKER_PROFILES = {"bob", "rodge", "steve-o", "karl", "jobsy", "axel", "builder"}

# Profiles that route but are never a routing *defect* complaint: operators,
# switchboard and the fleet kernel owner (default == Agent Smith; smith is the
# same operator alias). Platform-kernel FIX cards legitimately run on them, so a
# FIX title minted to default/smith/switch is NOT a build-lane misroute.
OPERATOR_PROFILES = {"default", "switch", "smith", "brain", "gen-pm", "weroll-pm"}

# Legitimate cross-lane mint/implied pairs. Mint==implied is always fine. These
# document the explicit override cases the card calls out (deploy->bob etc.).
ALLOWED_CROSS_LANE = {
    ("deploy", "bob"),   # a release card running on bob (explicit override)
    ("pm", "jobsy"),     # triage/PM decision cards legitimately jobsy
    ("deploy", "default"),
}

# Title markers -> lane (same order/verbs as C1). A marker wins over a verb.
# [triage] checked first: a "[triage] [Bob] X" card is a PM-parked card (jobsy),
# NOT a build card, so the triage prefix must beat an inner [Profile] marker.
TITLE_LANE_PREFIX = [
    (re.compile(r"\[\s*triage\s*\]", re.I), "pm"),
    (re.compile(r"\[\s*(?:bob|builder)\s*\]", re.I), "build"),
    (re.compile(r"\[\s*rodge\s*\]", re.I), "review"),
    (re.compile(r"\[\s*(?:steve-?o?|steveo)\s*\]", re.I), "verify"),
    (re.compile(r"\[\s*karl\s*\]", re.I), "design"),
    (re.compile(r"\[\s*jobsy\s*\]", re.I), "pm"),
]

# Lane verbs that are unambiguous -> lane. Build, review, verify, design, deploy.
TITLE_LANE_VERB = [
    (re.compile(r"^\s*(?:re-?review|review)\b", re.I), "review"),
    (re.compile(r"^\s*(?:verify|qa\b|smoke|real-?click|re-?verify)\b", re.I), "verify"),
    (re.compile(r"^\s*(?:design|specif|spec\b|decide|approve)\b", re.I), "design"),
    (re.compile(r"^\s*(?:deploy|release)\b", re.I), "deploy"),
    (re.compile(r"^\s*(?:implement|build\b|create\b|fix\b|repair|rework|integrate|land\b|viralize|align|port)\b", re.I), "build"),
]

# Cards whose TITLE marks a liveness probe / cost-cap adjudication / PM re-scope.
# These are legitimately owned by a specific non-build profile and must never be
# flagged: "Wire proof: karl turn" is karl's own liveness card; "Adjudicate
# cost-cap" is steve-o's adjudication job (cost policy); "RE-SCOPE" is jobsy's PM
# re-scoping action; "Approve extension under the cap" is the cost adjudicator.
LEGIT_OWNER_TITLE = [
    (re.compile(r"wire proof\s*:", re.I), None),          # matched via embedded profile name below
    (re.compile(r"^\s*probe\b", re.I), None),
    (re.compile(r"cost-?cap|adjudicat|approve extension|requeue under", re.I), "jobsy"),
    (re.compile(r"re-?scope\b", re.I), "jobsy"),
    (re.compile(r"wire\s+brain", re.I), "jobsy"),
    (re.compile(r"pre-review-gate configuration|gate-?config adjudicat|commission search", re.I), "verify"),  # steve-o owns gate config
    # Decision/approval/sign-off gates — the auto-decomposer parks decision-shaped
    # children (approve the design/plan, decide, ratify, sign-off, approval gate)
    # on jobsy (triage) so a ghost PM-run cannot self-complete an unsigned call.
    # These carry no "jobsy" title marker yet are legitimately jobsy's; without
    # this a recovered auto-decomposer effective-assignee of jobsy would false-
    # flag them against a design/build lane. Mirrors kanban_decompose's
    # _DECISION_TITLE_RE routing.
    (re.compile(r"approval gate|sign-?off|decision gate|approve (?:the|a|an) (?:design|plan|approach|spec|decision|model|schema|architecture|strategy|source)\b", re.I), "pm"),
]

# Embedded profile name inside a 'Wire proof: <profile> turn' / liveness card.
WIRE_PROOF_PROFILE = re.compile(r"wire proof\s*[:—-]\s*([a-z][a-z-]*)\s+turn", re.I)

# Body markers that name the owner authoritatively (survive title verb ambiguities).
BODY_OWNER_RE = re.compile(
    r"(?im)(?:assignee|owner|implementer|routed to|run by)\s*[:=]?\s*"
    r"[\"']?(bob|rodge|steve-?o|steveo|karl|jobsy|default|axel|builder|smith)",
)


def _row(con, sql, args=()):
    r = con.execute(sql, args).fetchone()
    return dict(r) if r is not None else None


def mint_assignee(con, task_id, payload):
    """Return (mint_assignee|None, source). source in {payload, gap, none}."""
    if payload:
        try:
            pl = json.loads(payload)
        except Exception:
            pl = {}
    else:
        pl = {}
    if isinstance(pl, dict):
        if "assignee" in pl:
            a = pl.get("assignee")
            return (a if isinstance(a, str) else None), "payload"
        if isinstance(pl.get("by"), str) and pl["by"] == "auto-decomposer":
            return None, "gap"
    return None, "none"


def effective_assignee(con, task_id):
    """Recover who an auto-decomposer child was actually routed to.

    Auto-decomposer children carry NO mint assignee in their `created` event
    (recording gap, fix-C2), so the daily scan cannot read it the way it reads a
    payload-minted card. The decomposer still routed each child to a real profile
    before dispatch, and that profile is durably recorded as the FIRST dispatched
    run's profile (lowest ``task_runs.id``) — mint-stable and immune to the review
    reassignment that corrupts ``tasks.assignee`` (which flips to the reviewer on
    ``request_review``). Returns the profile, or None when the card was never
    dispatched (no run: genuinely undetermined, never flagged).

    Only real runs count (a ``scheduled``/``released`` placeholder is not a
    dispatch). The first dispatched run is the implementer the decomposer chose.
    """
    row = _row(
        con,
        "SELECT profile FROM task_runs WHERE task_id=? AND profile IS NOT NULL "
        "AND status NOT IN ('released','scheduled') ORDER BY id LIMIT 1",
        (task_id,),
    )
    return (row.get("profile") if row else None)


def embedded_owner(title):
    """Return the profile a liveness-'Wire proof: <profile> turn' card belongs to,
    plus two-word owner handoffs (Bob —, Rodge —). None if not such a card."""
    m = WIRE_PROOF_PROFILE.search(title or "")
    if m:
        nm = (m.group(1) or "").lower()
        if nm == "steve":
            return "steve-o"
        if nm in ROLE_MAP.values() or nm in ("axel", "builder", "smith"):
            return nm
    # "Rodge review ..." / "Bob — ..." / "Steve-o verify ..." explicit owner prefix
    t = (title or "").strip()
    for prof, pat in (("bob", r"^(?:\[bob\]|bob\s*[—-])"), ("rodge", r"^(?:\[rodge\]|rodge\s*[—-]|rodge review)"),
                      ("steve-o", r"^(?:\[steve-?o\]|steve-?o\s+[a-z-]+\b)"),
                      ("karl", r"^(?:\[karl\]|karl\s*[—-]|karl\s*:)")):
        if re.match(pat, t, re.I):
            return prof
    return None


def implied_lane(title, body):
    """Determine the lane the card title/body points at, or None if ambiguous."""
    t = (title or "").strip()
    # Liveness probes / cost-cap / PM re-scope cards carry their own owner.
    ow = embedded_owner(t)
    if ow:
        return "build" if ow == "bob" else ("review" if ow == "rodge" else
              ("verify" if ow == "steve-o" else ("design" if ow == "karl" else
              ("deploy" if ow in ("default",) else "pm"))))
    for pat, lane in LEGIT_OWNER_TITLE:
        if pat.search(t):
            return lane
    # Title owner markers beat everything.
    for pat, lane in TITLE_LANE_PREFIX:
        if pat.search(t):
            return lane
    # Body owner marker (authoritative when present).
    m = BODY_OWNER_RE.search((body or "")[:400])
    if m:
        name = (m.group(1) or "").lower()
        if "rodge" in name:
            return "review"
        if "steve" in name:
            return "verify"
        if "jobsy" in name:
            return "pm"
        if "karl" in name:
            return "design"
        if name in ("default", "deploy"):
            return "deploy"
        if name in ("bob", "builder"):
            return "build"
    for pat, lane in TITLE_LANE_VERB:
        if pat.match(t):
            return lane
    return None


def classify(con, task_id, title, body, payload, status, created_at):
    """Return a dict describing one card's routing decision."""
    mint, src = mint_assignee(con, task_id, payload)
    lane = implied_lane(title, body)
    expected = ROLE_MAP.get(lane) if lane else None
    return {
        "id": task_id,
        "status": status,
        "title": title or "",
        "created_at": created_at,
        "mint": mint,
        "mint_src": src,
        "lane": lane,
        "expected": expected,
        "flag": False,
        "reason": None,
    }


def is_flag(c):
    """True iff c is a genuine mint-time routing mismatch (worker profile)."""
    if c["mint_src"] == "gap" or c["mint"] is None:
        return False
    if c["lane"] is None or c["expected"] is None:
        return False
    if c["mint"] == c["expected"]:
        return False
    if c["mint"] not in WORKER_PROFILES:
        return False
    if (c["lane"], c["mint"]) in ALLOWED_CROSS_LANE:
        return False
    return True


def scan(con, days):
    """Return (flags, gaps). flags is sorted list of classify() dicts."""
    cutoff = time.time() - days * 86400
    flags, gaps = [], []
    rows = con.execute(
        """SELECT e.task_id id, e.payload, t.title, t.body, t.status, t.created_at,
                  t.completed_at, t.assignee cur_assignee
           FROM task_events e JOIN tasks t ON t.id=e.task_id
           WHERE e.kind='created' AND e.created_at >= ?""",
        (cutoff,),
    ).fetchall()
    seen = set()
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        c = classify(con, r["id"], r["title"], r["body"], r["payload"],
                     r["status"], r["created_at"])
        if c["mint_src"] == "gap":
            # fix-C2 auto-decomposer gap: the created payload records no mint
            # assignee, but the decomposer still routed the child to a profile.
            # Recover the EFFECTIVE assignee (first dispatched run) and evaluate
            # it against the implied owner, so a misrouted auto-decomposer child
            # (build-lane child that actually dispatched to rodge/steve-o/axel)
            # is detected instead of being blind-parked as a gap. A never-dispatched
            # card stays an undetermined gap (no evidence of a misroute).
            eff = effective_assignee(con, c["id"])
            if eff is None:
                gaps.append(c)
                continue
            c["mint"] = eff
            c["mint_src"] = "auto-effective"
            # The joby-style PM gate / operator-profile cards the decomposer
            # legitimately parks on jobsy/karl/switch are covered by the same
            # legit-lane exclusions in is_flag() — no new false-positive class.
            if is_flag(c):
                c["reason"] = (
                    f"auto-decomposer child effective assignee={eff} "
                    f"({c['mint_src']}) on {c['lane']}-lane card; expected {c['expected']}"
                )
                flags.append(c)
            continue
        if is_flag(c):
            # reconstruct expected-vs-actual reason
            c["reason"] = (
                f"mint assignee={c['mint']} ({c['mint_src']}) on "
                f"{c['lane']}-lane card; expected {c['expected']}"
            )
            flags.append(c)
    flags.sort(key=lambda c: c["created_at"])
    return flags, gaps


def actionable(con, c):
    """Card still worth flagging/blocking (not done/archived, nothing ran)."""
    if c["status"] in ("done", "archived", "gave_up"):
        return False
    if c["status"] == "blocked":
        # a card already parked as operator_hold / needs_input at mint is not a
        # NEW routing defect to re-route — someone already holds it for review.
        row = _row(con, "SELECT block_kind FROM tasks WHERE id=?", (c["id"],))
        if row and row["block_kind"] in ("operator_hold", "card_defect", "needs_input", "capability"):
            return False
    rr = _row(con, "SELECT COUNT(*) n FROM task_runs WHERE task_id=? AND outcome IN ('completed','done')", (c["id"],))
    if rr and rr["n"]:
        return False
    return True


def already_flagged(con, c):
    """True iff a prior run posted the [routing-audit] mismatch comment."""
    row = _row(con, "SELECT COUNT(*) n FROM task_comments WHERE task_id=? AND body LIKE '%[routing-audit] mismatch%'", (c["id"],))
    return bool(row and row.get("n"))


def _post_comment(con, c):
    """Insert the [routing-audit] mismatch comment (author = this watchdog)."""
    body = (
        f"## [routing-audit] mismatch\n\n"
        f"**Expected owner:** {c['expected']} ({c['lane']}-lane)  \n"
        f"**Mint assignee:** {c['mint']}  \n"
        f"**Reason:** {c['reason']}  \n"
        f"\nFlagged by the daily assignee-mismatch auditor. Re-routing is the PM's "
        f"(Jobsy's) call — this card is **not** self-corrected."
    )
    con.execute(
        "INSERT INTO task_comments(task_id, author, body, created_at) VALUES(?,?,?,?)",
        (c["id"], "assignee-mismatch-watch", body, int(time.time())),
    )


def _block(con, c):
    """Block the card kind=needs_input so it surfaces for the PM to re-route.

    Mirrors the lifecycle's public block writes: set tasks.block_kind, move the
    row to status 'blocked', and append a 'blocked' task_event. Idempotent guard
    lives in the caller (already_flagged / actionable)."""
    now = int(time.time())
    con.execute("UPDATE tasks SET block_kind='needs_input', status='blocked' WHERE id=?",
                (c["id"],))
    con.execute(
        "INSERT INTO task_events(task_id, kind, payload, created_at) VALUES(?,?,?,?)",
        (c["id"], "blocked",
         json.dumps({"reason": c["reason"], "kind": "needs_input",
                     "recurrences": 0, "source_status": c["status"],
                     "by": "assignee-mismatch-watch"}),
         now),
    )


def apply_actions(con, flags, commit=True):
    """Comment + block each actionable mismatch. Returns the list acted on.

    Idempotent: a card already carrying the [routing-audit] mismatch marker is
    skipped, so a daily re-scan never duplicates a flag. Only actionable cards
    (not done/archived, nothing ran, not already held) are touched."""
    acted = []
    for c in flags:
        if not actionable(con, c):
            continue
        if already_flagged(con, c):
            continue
        _post_comment(con, c)
        _block(con, c)
        acted.append(c)
    if commit and acted:
        con.commit()
    return acted


def render_table(flags, gaps, show_headers=True):
    lines = []
    if flags:
        lines.append(f"[routing-audit] MINT-TIME ASSIGNEE MISMATCHES ({len(flags)})")
        lines.append("expected-vs-actual | card | status | title")
        for c in flags:
            lines.append(
                f"  {c['expected']} != {c['mint']} | {c['id']} | {c['status']:9s} | {(c['title'] or '')[:58]}"
            )
    if gaps and show_headers:
        lines.append("")
        lines.append(f"[info] auto-decomposer cards, no mint assignee recorded (gap, not flagged): {len(gaps)}")
        for c in gaps:
            lines.append(
                f"  - {c['id']} {c['status']:9s} routed={c.get('mint') or '?'} | {(c['title'] or '')[:58]}"
            )
    return "\n".join(lines)


def run(args):
    # --apply needs write access (post comment + block); default is read-only.
    con = sqlite3.connect(KANBAN_DB if args.apply else f"file:{KANBAN_DB}?mode=ro", uri=not args.apply)
    con.row_factory = sqlite3.Row
    flags, gaps = scan(con, args.days)
    show_gaps = args.audit or args.gap
    body = render_table(flags, gaps, show_headers=show_gaps)

    if args.apply:
        acted = apply_actions(con, flags)
        con.close()
        if acted:
            print(f"[routing-audit] flagged+blocked {len(acted)} actionable mismatch(es) "
                  f"for Jobsy to re-route. Full table:\n{body}")
        else:
            print(body or "no actionable mismatches (all flags already done/archived/held)")
        return body, acted

    con.close()
    return body, []


# ---------------------------------------------------------------------------
# Self-test (AC1): calibrate against the known 7-day baseline with zero FP on
# legit review/verify/triage/deploy cards.
# ---------------------------------------------------------------------------
def run_selftest():
    """Assert the audit logic flags genuine build-lane->wrong cases and never a
    legit review/verify/triage/deploy card. Returns (ok, message)."""
    import tempfile

    failures = []
    D = {
        "fake": [
            # (title, body, created_payload, expected_flag)
            ("[Bob] Implement orgagent health router", "",
             '{"assignee": "rodge", "status": "todo"}', True),          # build->rodge
            ("[Bob] Fix brain search grounding", "",
             '{"assignee": "axel", "status": "ready"}', True),          # build->axel
            ("[Bob] Implement Slack draft launcher", "",
             '{"assignee": "steve-o", "status": "todo"}', True),        # build->steve-o
            ("[Bob] Implement Gmail launcher action", "",
             '{"assignee": "bob", "status": "todo"}', False),           # correct
            ("Review orgagent PR against AC 1-7", "",
             '{"assignee": "rodge", "status": "todo"}', False),         # legit review->rodge
            ("Verify AC 3-6 against served backend", "",
             '{"assignee": "steve-o", "status": "todo"}', False),       # legit verify->steve-o
            ("[triage] Implement the Slack draft launcher", "",
             '{"assignee": "jobsy", "status": "blocked"}', False),      # legit triage->jobsy
            ("Deploy: squash-merge orgagent PR + restart", "",
             '{"assignee": "default", "status": "blocked"}', False),    # legit deploy->default
            ("Adjudicate cost-cap for test card", "",
             '{"assignee": "steve-o", "status": "blocked"}', False),    # steve-o legit adjudicator
            ("Implement worker-pool fix", "",
             '{"by": "auto-decomposer", "from_decompose_of": "x"}', False),  # gap, not flagged
            ("[Rodge] Review E1 promotion manifest", "",
             '{"assignee": "rodge", "status": "todo"}', False),         # legit
            ("[Steve-o] QA E1 promotion", "",
             '{"assignee": "steve-o", "status": "todo"}', False),       # legit
            ("[Karl] C2 auditor design", "",
             '{"assignee": "karl", "status": "todo"}', False),          # legitimate design->karl
            ("[Karl] design note", "",
             '{"assignee": "rodge", "status": "todo"}', True),          # design->rodge mismatch
            ("DECIDE pre-review-gate configuration + commission car", "",
             '{"assignee": "steve-o", "status": "todo"}', False),       # steve-o owns gate config
            ("FIX pre-review-gate focused-tests parser: honor multi-path", "",
             '{"assignee": "smith", "status": "todo"}', False),         # smith = operator kernel lane, not a build misroute
            ("Wire proof: karl turn on the restored OpenRouter config", "",
             '{"assignee": "karl", "status": "todo"}', False),          # karl's own liveness card
            ("Wire proof: steve-o turn on the restored config", "",
             '{"assignee": "steve-o", "status": "todo"}', False),       # steve-o's own liveness card
        ]
    }
    tmp = tempfile.NamedTemporaryFile(suffix=".db")
    con = sqlite3.connect(tmp.name)
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE tasks(id TEXT PRIMARY KEY, title TEXT, body TEXT, assignee TEXT,
            status TEXT, created_at INTEGER, completed_at INTEGER);
        CREATE TABLE task_events(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT,
            run_id INTEGER, kind TEXT, payload TEXT, created_at INTEGER);
        CREATE TABLE task_runs(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT,
            profile TEXT, outcome TEXT, summary TEXT, status TEXT);
        CREATE TABLE task_comments(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT,
            body TEXT, created_at INTEGER);
    """)
    now = int(time.time())
    for i, (title, body, payload, expect) in enumerate(D["fake"]):
        tid = f"t_st{i}"
        con.execute("INSERT INTO tasks(id,title,body,assignee,status,created_at) VALUES(?,?,?,?,?,?)",
                    (tid, title, body, json.loads(payload).get("assignee"), "todo", now - i))
        con.execute("INSERT INTO task_events(task_id,kind,payload,created_at) VALUES(?,?,?,?)",
                    (tid, "created", payload, now - i))
    con.commit()
    flags, gaps = scan(con, 30)
    flag_ids = {c["id"] for c in flags}
    for i, (title, body, payload, expect) in enumerate(D["fake"]):
        tid = f"t_st{i}"
        flagged = tid in flag_ids
        if flagged != expect:
            failures.append(
                f"{tid} {title!r}: expected flag={expect} got flag={flagged}"
                + (f" (mint={json.loads(payload).get('assignee')})" if payload else "")
            )
    con.close()
    if failures:
        return False, "SELF-TEST FAILED\n" + "\n".join("  ✗ " + f for f in failures)
    return True, f"self-test OK: {len(D['fake'])} cases, "
    f"{len(flag_ids)} flagged all correct, 0 false positives on legit review/verify/triage/deploy cards"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=1, help="window in days (default 1, cron daily)")
    ap.add_argument("--audit", action="store_true", help="print full mismatch table always")
    ap.add_argument("--gap", action="store_true", help="also report auto-decomposer gap cards")
    ap.add_argument("--apply", dest="apply", action="store_true", default=True,
                    help="comment+block actionable flags (default ON for the watchdog; use --no-apply to only report)")
    ap.add_argument("--no-apply", dest="apply", action="store_false",
                    help="report-only: never mutate cards")
    ap.add_argument("--selftest", action="store_true", help="run self-test fixtures and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        ok, msg = run_selftest()
        print(msg)
        return 0 if ok else 1

    body, acted = run(args)
    if args.audit:
        print(body or "[routing-audit] no mismatches in window")
    return 0


if __name__ == "__main__":
    sys.exit(main())