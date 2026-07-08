"""HERMETIC B1 End-to-End Canary (readonly, in-memory).

Validates the full B1 happy-path:

    objective text -> normalized -> classified -> discovered ->
    [B1 evidence_pack] -> execution_contract -> subgoals -> dry-run -> render

All effects are in-memory. No real:
- state.db, audit log, kanban DB, GBrain/Obsidian/notebooklm
- subprocess / network / LLM provider
- filesystem outside tmp_path
- git / push / PR / workflow

Baseline contract:
- HEAD frozen: 3b4b7cd23151fad8d167fb6538714094679aaeb9 (pilot dryrun)
- B1 wiring: commit 3a62bf03d (default-off; pass-through when off)
- Known-dirty in the parent suite:
  tests/test_executive_v2/canary_pilot/test_pilot_scope_guard.py::test_scope_head_is_frozen_sha
  (expects 3a62bf03d; not retargeted per operator scope)
"""
