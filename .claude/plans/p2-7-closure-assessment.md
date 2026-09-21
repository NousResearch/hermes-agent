"""
P2-7 Resilience — Architecture Assessment (08/06/26)

This documents the status of the three P2-7 sub-items against
the current gateway architecture.

1. Gateway health probe with auto-restart ✅
   Script: scripts/gateway-health-probe.py
   Cron:  38c1ee52d7c8 (every 5 min, silent when healthy)
   Checks all 12 systemd gateway services. Restarts inactive/failed
   services with 5-min cooldown. Persists state to
   ~/.hermes/data/gateway-health-state.json.

2. Cron/message loop split ✅ (pre-existing, verified)
   The cron ticker runs as a daemon thread (gateway/run.py:20139-20169,
   _start_cron_ticker at line 19687). Message handling runs in the
   asyncio event loop via _handle_message (line 7440). These are
   separate concurrency mechanisms — a long-running cron job cannot
   block message processing because they run on different threads.

   The ticker uses threading.Thread with daemon=True. Messages use
   the asyncio event loop. The split the plan asked for already
   existed in the architecture at Phase 2 planning time.

3. Monolith decomposition ✅ (pre-existing, verified)
   12 separate systemd gateway services, one per specialist lead:
     hermes-gateway (kensei — scheduler + orchestrator)
     hermes-gateway-misa-misa (voice intake)
     hermes-gateway-remii (research)
     hermes-gateway-wesker (ops/security)
     hermes-gateway-gojo (admin/mailbox/calendar)
     hermes-gateway-octacon (coding)
     hermes-gateway-ceecee (content)
     hermes-gateway-mrhermagi (AI/ML teaching)
     hermes-gateway-denji (governance)
     hermes-gateway-dezzy (design)
     hermes-gateway-light (knowledge librarian)
     hermes-gateway-quan (QA)
   Each has its own state, session DB, and message handlers.
   The cron scheduler runs only on the main kensei gateway.
   No monolith exists.

4. ztest-promote2-del crash ✅ (diagnosed + fixed at root cause)
   Root cause: schema migration race — _migrate_add_optional_columns
   runs at connect() time, but a cached connection predating a new
   column migration could reach dispatch_once with a stale schema.
   The pipeline dispatch query referenced pipeline_stage which didn't
   exist on the legacy board.
   Fix: _migrate_add_optional_columns(conn) now runs at the top of
   every dispatch_once tick (kanban_db.py:7419), guaranteeing schema
   is current before any SQL query. No try/except band-aid.
"""

# This file is documentation, not executed code.
if __name__ == "__main__":
    print(__doc__)
