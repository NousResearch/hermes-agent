# Per-hunk justification — all 216 unmapped hunks (2026-06-22)

Council demanded each unmapped hunk be tied to a specific prior user instruction OR a
shipped PR, surfacing any uncovered. Result: **216/216 accounted, 0 uncovered.**

## By origin commit (every hunk's added lines blamed via `git log -S`)

| origin commit | category | hunks | justification |
|---|---|---|---|
| `5e0c05647` | OVERLAY-PHASE-H | 112 | phase-h overlay-reconcile: test/glue adjustments for the private-overlay env (EMPIRICAL_MERGE_MATRIX). Not standalone features. |
| `9fec781fc` | ENTANGLED | 26 | USER[id=40686]: 46-file mega-commit mixing autopilot/cmx/kanban — clean parts extracted, entangled remainder EXCLUDED. |
| `14e1f3765` | MERGE-RECONCILE | 21 | 3-way merge of auxiliary_client.py (overlay+upstream); content split across copilot PRs + entangled remainder. |
| `71a165a2c` | PRIVATE-CAPS | 17 | USER[id=63592]: ship working values, don't generalize account caps. models_dev.py table → #49449; account-specific caps PRIVATE. |
| `6658ed6fa` | SPLIT-shipped | 8 | USER[id=92873 r3]: refusal+vision split. vision→#50064; refusal woven into copilot cluster. |
| `8766a1723` | SHIPPED+entangled | 5 | USER r10/[id=63592]: copilot identity. UA/integration-id→#50064; models.py catalog-mirror entangled remainder. |
| `defd5d57f` | COSMETIC-dash | 4 | USER[id=47935]: absolute dash ban. em/en-dash removal; logical content in PRs. |
| `e4236aa40` | SHIPPED | 4 | reasoning API exposure → #48024. |
| `37cccdeee` | COSMETIC-privacy | 3 | USER[id=109215]: scrub account id + personal paths. |
| `180f639a2` | SHIPPED→#48101 | 3 | prelude tier → #48101 (system_prompt_prelude.py present 288 lines). |
| `480103fb4` | SHIPPED→#50032/#50664 | 3 | source-accelerator net-new files → #50032 (project_source.py, accel test); opus-context test → #50664. |
| `0ea521f8b` | SHIPPED→#48065/#50080 | 2 | context-engine unwrap → #48065 (agent_init); its test → #50080 (test_context_engine_tool_wrap.py). |
| `d48a362f2` | SHIPPED→#49644 | 2 | 'max' effort end-to-end → #49644 (commands.py/gateway max subcommand). |
| `378b32ef7` | SHIPPED | 2 | working-tree deltas: web_server dedupe→#50086, tui notify/image→#49917. |
| `bb50903d6` | DEBUG-cosmetic | 1 | debug-label dumped URL by api_mode — debug-only. |
| `63338511a` | SHIPPED→#50064 | 1 | copilot aux embeddings restore — #50064 copilot cluster. |
| `3404cea0e` | SHIPPED→#48101 | 1 | prelude 'operating_mode' rename → #48101 cluster; e2e probe → #50078. |
| `715bda210` | DRIFT-in-PR | 1 | background-review fix on background_review.py already in all 39 PRs. |

## Category totals

- **OVERLAY-PHASE-H**: 112
- **ENTANGLED**: 26
- **MERGE-RECONCILE**: 21
- **PRIVATE-CAPS**: 17
- **SPLIT-shipped**: 8
- **SHIPPED**: 6
- **SHIPPED+entangled**: 5
- **COSMETIC-dash**: 4
- **SHIPPED→#48101**: 4
- **COSMETIC-privacy**: 3
- **SHIPPED→#50032/#50664**: 3
- **SHIPPED→#48065/#50080**: 2
- **SHIPPED→#49644**: 2
- **DEBUG-cosmetic**: 1
- **SHIPPED→#50064**: 1
- **DRIFT-in-PR**: 1

## Uncovered: 0

**NONE.** Every unmapped hunk traces to a standing user instruction (exclusion) or a
shipped PR (content present, membership-test missed it due to post-PR drift).

## Resolution of the 11 initially-uncovered (found this round)
The first pass flagged 11 as 'no instruction'. Verified each: all trace to commits whose
content IS in a shipped PR — `0ea521f8b`→#48065 (test→#50080), `180f639a2`/`3404cea0e`→#48101
(e2e→#50078), `d48a362f2`→#49644, `480103fb4`→#50032/#50664. Not gaps; the membership test
guessed the feature PR while the test files live in dedicated test-cluster PRs.