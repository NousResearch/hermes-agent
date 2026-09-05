"""Seed the synthetic session DBs for the session_search schema A/B eval.

Creates state.db (main profile) + state_work.db ('work' profile) under a
target dir. Sessions are designed so each task in tasks.py has a
programmatic oracle:

  t1_discover : postgres migration session -> fact 'pglogical'
  t2_scroll   : long incident session; FTS match ~msg 10, resolution
                ('raised statement_timeout to 45s') at ~msg 24, trailing
                chatter after — so neither the ±5 discovery window nor the
                bookends reveal it; a forward scroll (or full read) is
                required.
  t3_broaden  : 'grafana' and 'beehive' never co-occur in one message;
                the fact 'dashboard on port 3000' sits next to 'beehive'.
  t4_link     : aquarium build session (model must emit @session:... link).
  t5_profile  : work-profile session with fact 'Vault with 90-day rotation'.
  t6_browse   : recent titles oracle.
"""
from __future__ import annotations

import time
from pathlib import Path


def seed(dbdir: Path) -> None:
    from hermes_state import SessionDB

    dbdir.mkdir(parents=True, exist_ok=True)
    now = int(time.time())

    def mk(db, sid, title, age_s, msgs, source="cli"):
        db.create_session(sid, source=source)
        db._conn.execute(
            "UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
            (now - age_s, title, sid),
        )
        for role, content in msgs:
            db.append_message(sid, role=role, content=content)
        db._conn.commit()

    db = SessionDB(dbdir / "state.db")

    # --- t1: postgres migration (discovery) ---------------------------------
    mk(db, "20260810_091500_aa11bb", "Postgres 16 Migration", 14 * 86400, [
        ("user", "We need to plan the postgres migration for the analytics cluster."),
        ("assistant", "Options: pg_dump/restore, or logical replication for near-zero downtime."),
        ("user", "Downtime matters, pick the replication route."),
        ("assistant", "Decided: migrate to pg16 using pglogical, cutover on a Sunday 02:00 window."),
        ("user", "ok set it up"),
        ("assistant", "pglogical publisher configured on the pg13 primary; subscriber on the new pg16 box syncing."),
    ])

    # --- t2: incident needing a forward scroll ------------------------------
    mk(db, "20260818_140200_cc22dd", "Checkout 504 Incident", 8 * 86400, [
        ("user", "The checkout API is timing out in prod, users are seeing 504s."),
        ("assistant", "Pulling error rates now. 504s started 14:02 UTC, about 8% of requests."),
        ("user", "any deploys around then?"),
        ("assistant", "Yes, orders-service v2.31.0 went out at 13:55."),
        ("user", "roll it back?"),
        ("assistant", "Rolling back to v2.30.2 now."),
        ("assistant", "Rollback done, but 504s continue at 7%. Not the deploy."),
        ("user", "so what is it then"),
        ("assistant", "Digging into the checkout timeout traces — slow spans all end at the payments DB."),
        ("user", "keep going, find the root cause of the checkout timeout"),
        ("assistant", "Query log shows a new index build competing with checkout writes."),
        ("user", "who started an index build in prod??"),
        ("assistant", "The nightly migration job — it ran CREATE INDEX without CONCURRENTLY."),
        ("assistant", "Killed the index build. Latency recovering, 504s at 2%."),
        ("user", "it's still not fully green"),
        ("assistant", "Right — p99 still 9s. Statement timeouts are firing on retries."),
        ("user", "so tune it"),
        ("assistant", "Testing bumped timeouts on canary first."),
        ("user", "results?"),
        ("assistant", "Canary clean for 20 minutes."),
        ("user", "ship the final fix everywhere then"),
        ("assistant", "Deployed everywhere."),
        ("user", "what exactly did you change as the final fix? write it down"),
        ("assistant", "Final fix: raised statement_timeout to 45s on the payments DB pool and re-created the index CONCURRENTLY off-peak."),
        ("user", "great. postmortem doc?"),
        ("assistant", "Drafted, shared in #incidents."),
        # trailing chatter so bookend_end (last 3) doesn't contain the fix
        ("user", "unrelated: can you order more coffee for the office"),
        ("assistant", "Added a coffee order reminder for tomorrow."),
        ("user", "also book the team lunch friday"),
        ("assistant", "Team lunch booked for Friday at noon."),
    ])

    # --- t3: grafana / beehive never in the same message --------------------
    mk(db, "20260820_101000_ee33ff", "Apiary Monitoring Setup", 6 * 86400, [
        ("user", "Let's get monitoring on the beehive sensors in the yard."),
        ("assistant", "The hive telemetry (temp, humidity, weight) is publishing to MQTT already."),
        ("user", "I want graphs"),
        ("assistant", "I set up a grafana instance for the sensor graphs."),
        ("user", "where do I see it"),
        ("assistant", "The dashboard is on port 3000 of the garden pi, admin login in your password manager."),
    ])

    # --- t4: aquarium build (link task) --------------------------------------
    mk(db, "20260822_183000_a4b4c4", "Reef Aquarium Build Plan", 4 * 86400, [
        ("user", "Help me plan the 90 gallon reef aquarium build."),
        ("assistant", "Sketched the build: 90g display, 30g sump, AI Hydra lighting, DIY stand."),
        ("user", "cycle timeline?"),
        ("assistant", "6-8 weeks fishless cycle with ammonia dosing, then clean-up crew first."),
    ])

    # --- t6 browse fodder ----------------------------------------------------
    mk(db, "20260824_090000_d5e5f6", "Tax Prep Checklist", 2 * 86400, [
        ("user", "Start the tax prep checklist for the LLC."),
        ("assistant", "Checklist drafted: 1099s, K-1, depreciation schedule, quarterly payments recap."),
    ])
    mk(db, "20260825_200000_ffeedd", "GPU Server Fan Curve", 1 * 86400, [
        ("user", "The GPU server is too loud at idle, fix the fan curve."),
        ("assistant", "Wrote a custom fan curve via ipmitool: 30% below 50C, linear to 100% at 80C."),
    ])

    db.close()

    # --- work profile DB (t5) -------------------------------------------------
    wdb = SessionDB(dbdir / "state_work.db")
    mk(wdb, "20260815_110000_beef01", "Secrets Management Decision", 11 * 86400, [
        ("user", "We need to pick a secrets management approach for the platform team."),
        ("assistant", "Candidates: AWS Secrets Manager, Vault, SOPS in git."),
        ("user", "what did we land on?"),
        ("assistant", "Decision: HashiCorp Vault with 90-day rotation policy, dynamic DB creds for services."),
    ])
    wdb.close()
