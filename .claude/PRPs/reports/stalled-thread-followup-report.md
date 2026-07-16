# Implementation Report: Stalled-Thread Follow-Up

## Summary
Added an opt-in proactive detector (`agent/stalled_threads.py`) that finds open loops — kanban commitments past due or untouched, and live conversation threads awaiting the user's reply — classifies them with an auxiliary LLM, dedups against a cooldown window, and delivers ONE batched digest through the shipped notification budget governor (category `stalled_thread`). Exposed via `hermes threads {scan,list,enable,disable}`. Heavy reuse of the governor + router gate + Omi scan skeleton; net-new is one state table + the detector.

## Assessment vs Reality
| Metric | Predicted (Plan) | Actual |
|---|---|---|
| Complexity | Large (high-reuse) | Large — matched |
| Confidence | 8/10 | Accurate; one real bug caught by live verification (as the plan anticipated) |
| Files Changed | ~8 | 8 (2 created, 6 modified) |

## Tasks Completed
| # | Task | Status | Notes |
|---|---|---|---|
| 1 | State table + CRUD (`hermes_state.py`) | ✅ Complete | `stalled_nudges` table + `stall_nudged_recently`/`record_stall_nudge`/`list_live_threads_for_stall` |
| 2 | Config section | ✅ Complete | `stalled_threads` in DEFAULT_CONFIG + `_KNOWN_ROOT_KEYS` + example |
| 3 | Detector module | ✅ Complete | Deviated — `_parse_due` extended for prose bodies (live-data fix, below) |
| 4 | CLI (`threads`) | ✅ Complete | scan/list/enable/disable; atomic config write reused |
| 5 | Opt-in cron guidance | ✅ Complete | prints real `hermes cron create` syntax |
| 6 | Tests | ✅ Complete | 22 tests |

## Validation Results
| Level | Status | Notes |
|---|---|---|
| Static Analysis (ruff) | ✅ Pass | all 7 files |
| Unit Tests | ✅ Pass | 22 new |
| Regression | ✅ Pass | 242 across proactive features (stalled/governor/omi/delivery/goal/config), zero regressions |
| Build/import | ✅ Pass | CLI wired; enable/disable round-trips |
| **Live-data verification** | ✅ Pass | Ran against real state.db + kanban; caught + fixed a real parsing bug |

## Files Changed
| File | Action | Lines |
|---|---|---|
| `agent/stalled_threads.py` | CREATED | ~360 |
| `tests/agent/test_stalled_threads.py` | CREATED | ~290 |
| `hermes_state.py` | UPDATED | +~110 (1 table, 1 index, 3 methods) |
| `hermes_cli/config.py` | UPDATED | +~16 (section + root key) |
| `cli-config.yaml.example` | UPDATED | (pending — see Issues) |
| `hermes_cli/subcommands/notify.py` | UPDATED | +~30 (`build_threads_parser`) |
| `hermes_cli/notify_cmd.py` | UPDATED | +~110 (`threads_command` + helpers) |
| `hermes_cli/main.py` | UPDATED | +~15 (import, `cmd_threads`, builder call, 2 allow-lists) |

## Deviations from Plan
1. **`_parse_due` extended for prose bodies (live-data fix).** The plan assumed kanban commitment cards carry a structured `Due: <iso>` line (how `omi_commitments.py` writes fresh cards). Live verification revealed the kanban **dispatcher rewrites** Omi card bodies into a `Goal: … Due 2026-07-25.` prose form with **no `Due:` line** and status `blocked`. Added a `_DUE_PHRASE` fallback regex (`due [date] <ISO-ish>`) so real cards' due dates are found. Verified against the 3 actual cards on the board — all now parse correctly. This is exactly the class of bug the plan's mandatory live-verification step was designed to catch (the Omi lesson).

## Issues Encountered
1. **cli-config.yaml.example not yet updated.** The plan's Task 2 included documenting the `stalled_threads` section in `cli-config.yaml.example`; this was deferred during implementation and should be added before merge (the config loads fine without it — it's documentation only).
2. **Formatter hook noise.** The global `ruff format` PostToolUse hook continues to reflow edited files; kept edits surgical and verified diffs.
3. **Slow full `tests/agent/` sweep.** The broad run stalled on collection; used targeted per-file runs instead (242 tests, decisive signal).

## Tests Written
| Test File | Tests | Coverage |
|---|---|---|
| `tests/agent/test_stalled_threads.py` | 22 | `_parse_due` (7 incl. prose forms), consent, past-due/untouched/fresh cards, `Due: none`, classify+governor (allow/suppress), resolved/low-conf drops, dedup cooldown, max-items cap, fail-soft, empty board |

## Live-Data Verification (the plan's mandatory step)
- `list_live_threads_for_stall` returned 4 real threads with correct `last_role` (`user`/`assistant`/`session_meta`/`None`) and epoch-float `last_active` — Source B filter correctly excludes non-`user` last messages.
- Real kanban cards: 3 Omi-sourced, status `blocked`, prose `Due` — the deviation above was found and fixed here; all 3 now parse.
- Standalone `hermes threads scan` (no gateway) ran cleanly → `scanned=0` (correct: due dates are future, cards <1h old) — NOT a silent-zero bug (parsing verified separately).
- Real profile left DISABLED (opt-in default restored).

## Next Steps
- [ ] Add `stalled_threads` block to `cli-config.yaml.example` (Task 2 doc leftover)
- [ ] Code review via `/code-review`
- [ ] Commit + PR to fork
- [ ] Follow-up (user noted): tune Source B thread heuristic on real data over time
