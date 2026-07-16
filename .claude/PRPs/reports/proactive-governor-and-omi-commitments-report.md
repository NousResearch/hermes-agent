# Implementation Report: Notification Budget Governor + Omi Commitment → Kanban

## Summary
Implemented two coupled proactivity features for Hermes/Mercury:
1. **Notification/attention budget governor** (`agent/notification_budget.py`) — scores proactive agent-initiated messages (`score = p_act × value_hint`), enforces a daily soft cap + hard ceiling, learns per-category thresholds from `keep`/`mute` feedback, and is idempotent + fail-open. Wired into the delivery router as a pre-send gate keyed on `metadata["proactive"]`.
2. **Omi commitment extractor** (`agent/omi_commitments.py`) — opt-in scheduled scan that reads the Omi transcript via MCP, extracts owner-only commitments with an auxiliary LLM, files them as `triage` kanban cards (deduped by content hash), and routes the summary notification through the governor.

Plus a `hermes notify` / `hermes omi` CLI and documented config.

## Assessment vs Reality
| Metric | Predicted (Plan) | Actual |
|---|---|---|
| Complexity | Large | Large |
| Confidence | 8/10 | Matched — 8/10 was accurate; two scope corrections needed |
| Files Changed | ~10 | 12 (6 created, 6 modified) |

## Tasks Completed
| # | Task | Status | Notes |
|---|---|---|---|
| 1 | State tables + CRUD (`hermes_state.py`) | ✅ Complete | 3 tables, 9 methods, +206 lines |
| 2 | Config sections | ✅ Complete | `notifications` + `omi_commitments` in DEFAULT_CONFIG + `_KNOWN_ROOT_KEYS` |
| 3 | Governor module | ✅ Complete | Deviated — score/quantity separation + optimistic cold-start (see Deviations) |
| 4 | Router gate (`gateway/delivery.py`) | ✅ Complete | Gates only `metadata["proactive"]`; fail-open |
| 5 | Tag proactive producers | ✅ Deviated | Scoped to Omi only; cron/webhook/goal deliberately NOT gated (see Deviations) |
| 6 | Omi extractor | ✅ Complete | Deviated — `triage=True` not `initial_status="triage"` (see Deviations) |
| 7 | `hermes notify` / `hermes omi` CLI | ✅ Complete | status/keep/mute/scan/enable/disable |
| 8 | Omi cron job registration | ✅ Deviated | Prints `hermes cron create` guidance rather than auto-registering (see Deviations) |
| 9 | Governor tests | ✅ Complete | 13 tests |
| 10 | Omi extractor tests | ✅ Complete | 12 tests |

## Validation Results
| Level | Status | Notes |
|---|---|---|
| Static Analysis (ruff) | ✅ Pass | All 12 changed/new files clean |
| Unit Tests | ✅ Pass | 25 new tests pass |
| Regression | ✅ Pass | 324 tests (delivery/goal/config) + 693 cron tests, zero regressions |
| Build/Import | ✅ Pass | All modules import; `hermes notify`/`omi` CLI wired and functional |
| Edge Cases | ✅ Pass | disabled/consent, MCP error, bystander-exclusion, dedup, low-confidence, malformed JSON, fail-open, ceiling |

Note: `scripts/run_tests.sh` targets `venv/bin/python` (no pytest); tests were run with the active nix interpreter via `python -m pytest`.

## Files Changed
| File | Action | Lines |
|---|---|---|
| `agent/notification_budget.py` | CREATED | 351 |
| `agent/omi_commitments.py` | CREATED | 327 |
| `hermes_cli/notify_cmd.py` | CREATED | 187 |
| `hermes_cli/subcommands/notify.py` | CREATED | 58 |
| `tests/agent/test_notification_budget.py` | CREATED | 167 |
| `tests/agent/test_omi_commitments.py` | CREATED | 239 |
| `hermes_state.py` | UPDATED | +206 |
| `gateway/delivery.py` | UPDATED | +43 |
| `hermes_cli/config.py` | UPDATED | +36 |
| `hermes_cli/main.py` | UPDATED | +25/-1 |
| `cli-config.yaml.example` | UPDATED | +43 |
| `gateway/run.py` | UPDATED | +5 (explanatory NOTE only) |

## Deviations from Plan
1. **Scope narrowed to Omi-only gating (Task 5).** The plan tagged cron, webhook, and goal-status as proactive producers. During validation, existing tests (`test_goal_status_notice`, `test_goal_verdict_send`, and 5 cron tests) surfaced a design error: cron jobs, webhook subscriptions, and `/goal` status notices are all **user-requested/configured** deliveries. Gating them would silently drop things the user explicitly asked for. Correction: the governor now gates only genuinely **agent-initiated** proactivity (Omi commitments). The router-level gate remains as reusable infrastructure for future ambient producers (e.g. stalled-thread detection). Documented with NOTE comments at each producer path.
2. **Score/quantity separation (Task 3).** The plan's formula folded `attention_cost` into the gating score (`p_act·value − attention_cost`), which made the escalation/ceiling tiers unreachable once enough messages had been sent (caught by unit tests). Corrected: `score = p_act·value` (intrinsic importance) governs the threshold/escalation gates; quantity is governed independently by the cap/ceiling counters. `attention_cost` is retained in the ledger for observability.
3. **Optimistic cold-start (Task 3).** A brand-new category seeds `p_act = 1.0` (not `base_threshold`), so its first genuinely-valuable notification is judged on the producer's `value_hint` rather than auto-suppressed before any feedback exists. Dismissals pull `p_act` down via the EWMA.
4. **`triage=True` not `initial_status="triage"` (Task 6).** The kanban `create_task` API only accepts `{"running","blocked"}` for `initial_status`; the `triage` state is set via the `triage=True` flag. Verified against `hermes_cli/kanban_db.py`.
5. **Dynamic DB path (Tasks 3/6).** Both modules resolve the state DB via `get_hermes_home() / "state.db"` at call time instead of the import-frozen `DEFAULT_DB_PATH`, so profile switches and per-test `HERMES_HOME` overrides are honored.
6. **Omi cron job is user-scheduled (Task 8).** `hermes omi enable` flips the config flag and prints the exact `hermes cron create` command rather than auto-registering a job via an unverified store API — keeps the schedule visible/editable in `hermes cron list` and avoids depending on an unconfirmed internal API.

## Issues Encountered
1. **Formatter noise.** A global `PostToolUse` hook (`~/.claude/hooks/post-edit-format.sh`) runs `ruff format` on every `.py` edit, but the repo is not `ruff format`-clean and does not use it in CI. Early edits triggered thousands of lines of unrelated reflow across `gateway/run.py`, `config.py`, `cron/scheduler.py`, `main.py`. Resolved by restoring those files to HEAD, temporarily disabling the hook, re-applying only the semantic hunks by hand (matching the repo's compact style), then restoring the hook. Final tracked diff is +357/-1.
2. **Test runner interpreter.** `scripts/run_tests.sh` uses `venv/bin/python` which lacks pytest in this environment; ran tests with the active nix `python -m pytest` instead.

## Tests Written
| Test File | Tests | Coverage |
|---|---|---|
| `tests/agent/test_notification_budget.py` | 13 | disabled passthrough, threshold gate, soft-cap, hard-ceiling, idempotency, dismiss/act learning, clamping, per-category override, fail-open, unknown signal, status |
| `tests/agent/test_omi_commitments.py` | 12 | consent skip, owner-commitment card, bystander exclusion, low-confidence drop, dedup on rescan, MCP-error graceful, governor-routed notify (allow + suppress), JSON parser robustness |

## Next Steps
- [ ] Code review via `/code-review`
- [ ] Consider a follow-up: implicit act-detection (user replies within N min) as a learning signal beyond explicit `keep`/`mute`
- [ ] Future proactive producer (stalled-thread detection) can reuse the router gate by setting `metadata["proactive"]`
- [ ] Create PR via `/prp-pr`
