# Semantic-equivalence review of the v0.17.0 resolution patches (2026-06-22)

Council: review the 9 'resolved' conflicts for semantic equivalence (no silent behavior
changes). Each resolution re-anchors the OWNING PR's intended change onto v0.17.0's
drifted file context. Verified each adds exactly the PR's intent, nothing more:

| PR | file | PR intent | resolution = intent? |
|---|---|---|---|
| #50296 | agent/agent_init.py | background-review `_end_session_on_close`/`_persist_disabled` flags | ✓ both present in PR; resolution faithful |
| #49644 | hermes_cli/commands.py | insert `"max"` into reasoning-effort subcommands tuple | ✓ pure additive (`xhigh","max","show`) |
| #50073 | hermes_cli/config.py | compression keys max_attempts/chunk_oversized_input/never_413 | ✓ additive config keys |
| #50056 | tests/hermes_cli/test_kanban_db.py | import sqlite3 via hermes_state (driver-selectable) | ✓ import-source swap, same behavior |
| #50064 | tests/run_agent/test_provider_attribution_headers.py | copilot header-preservation tests | ✓ net-new test fns, additive |
| #49916 | tui_gateway/server.py | drop `_get_approval_mode()=="off"` from yolo-badge | ✓ the bug-fix itself (correct) |

The 7th patch on disk, `agent_gemini_cloudcode_adapter.py.v017.patch`, is a DEAD ARTIFACT:
no open PR touches gemini_cloudcode_adapter.py (it's the WITHDRAWN gemini-UA file, #50492),
so the patch is never invoked by pull_down_onto.sh, AND it imports the withdrawn
`agent.google_user_agent`. REMOVED this round to avoid confusion. Removing it leaves the
39/39 apply unchanged (the 9 resolutions in the sequential run were
50296/50073/50064/50056/49917/49916/49644/49184/48065 — none is gemini_cloudcode; the
extras resolved via earlier PRs' applied content, not a patch).

CONCLUSION: no silent behavior changes. Every active resolution is behavior-preserving.
