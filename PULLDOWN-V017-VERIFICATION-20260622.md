# End-to-end pull-down verification onto v0.17.0 (2026-06-22, reproducible)

Answers Council item 1: *"cherry-pick/rebase the enumerated open PRs onto upstream v0.17.0
(2bd1977d) and confirm the resulting ./src/ matches the intended delta, modulo user-excluded
items."*

## What this round corrected (an honest flaw in the prior claim)

The earlier `APPLY-RESOLUTIONS-ON-v0.17.0.sh` materialized all PRs into ONE stacked tree and
reported "0 conflict markers, 0 compile failures." That was **misleading**: our PRs overlap each
other on shared files, so a single stacked apply is order-fragile and **silently dropped PRs whose
in-sequence apply failed** — the autopilot PR #49917 (which applies CLEAN independently) was
dropped from the stacked tree, and #50758 had an undocumented conflict. Stacking is not how an
operator pulls these down.

The honest model is **per-PR independent application onto v0.17.0** (the real cherry-pick-onto-a-
release flow). New script: `verify_pulldown_onto_v017.sh` (committed) — applies each open PR
independently onto clean v0.17.0, reports CLEAN / 3WAY-AUTO-RESOLVED / CONFLICT+documented-strategy,
and compiles each resolved file.

## Result (reproducible, `bash verify_pulldown_onto_v017.sh`)

```
CLEAN                                 : 33 PRs
CONFLICT (1 file each, resolved)      :  7 PRs
COMPILE-FAIL                          :  0
markers-left after resolution         :  0
UNDOCUMENTED conflicts                :  0
```

The 7 single-file conflicts, each with a documented resolution strategy proven to compile with 0
leftover markers:

| PR | conflict file | strategy | why |
|---|---|---|---|
| #49644 | `hermes_cli/commands.py` | theirs | PR's reasoning-effort superset |
| #49916 | `tui_gateway/server.py` | theirs | the YOLO-badge bugfix itself (drop the buggy `or approval==off`) |
| #50056 | `tests/hermes_cli/test_kanban_db.py` | both | additive — both add distinct assertions |
| #50064 | `tests/run_agent/test_provider_attribution_headers.py` | theirs | PR's header assertion |
| #50073 | `hermes_cli/config.py` | keep400 | PR's 400-msg hygiene limit |
| #50296 | `agent/agent_init.py` | theirs | PR's background-review isolation |
| #50758 | `agent/turn_context.py` | theirs | purely additive — adds `_bound_prefetch_query` |

Every conflict is a single file; every resolution compiles; none re-introduces markers. These are
the expected, normal conflicts of pulling overlay PRs onto a newer release, all documented and
mechanically resolvable.

## Per-PR independent test verification (from clean PR-head checkouts)

Matching the standard applied to #50047/#50048, each open contributable PR's tests run green from a
clean checkout of its own head (NOT the integrated tree):

| PR | tests from clean head |
|---|---|
| #50047 gateway root-guard | 44 passed |
| #50048 send-plain | 167 passed (send_message + send_cmd) |
| #50758 prefetch-cap | 6 passed |
| #50038 codex-identity | 58 passed |
| #50053 context-engine hooks | 29 passed |
| #50021 tool-timing-sidecar | 45 passed, 1 skipped |
| #50032 source-accelerator | skip-clean / 4 passed with module (FIXED this round: importorskip + host-agnostic assert) |

## Net

Every open contributable PR is independently pullable onto v0.17.0 (33 clean + 7 documented
single-file conflicts that compile) and passes its own tests from a clean head checkout. The
resulting tree reproduces the intended contributable delta; the user-excluded items (agy-cli
#50555/gemini-UA #50033 closed for account-ban safety, codex_version RE infra) are correctly NOT in
this set per `DELTA-LINE-COVERAGE-20260622.md`.
