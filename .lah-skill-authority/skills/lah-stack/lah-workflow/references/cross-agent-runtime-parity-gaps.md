# Cross-Agent Runtime Parity Gaps

Condensed findings from `LAH_HERMES_CODEX_WORKFLOW_RUNTIME_PARITY_AUDIT_V1` (2026-07-30).

Full report: `/home/deploy/lah-stack-repos/lah-stack-tools/docs/operator/audits/LAH_HERMES_CODEX_WORKFLOW_RUNTIME_PARITY_AUDIT_V1.md`

## Executive Verdict

**RUNTIME_PARITY_FALSE** — Hermès and Codex do NOT have execution parity despite sharing a skill name and gate structure.

## Key Structural Differences

| Dimension | Hermès | Codex |
|-----------|--------|-------|
| SKILL.md size | 90,941 bytes | 17,985 bytes (5× smaller) |
| Reference files | 94 files (~844KB) | 0 files |
| Scripts directory | Present (`scripts/`) | Absent |
| Templates directory | Present (`templates/`) | Absent |
| Documented pitfalls | ~70 entries | ~20 entries |
| Router engine | Independent (`dry-run-route.sh`) | Delegates to Hermès |
| `execute_code` | PROVEN | NOT_PROVEN |
| `patch` (fuzzy find-replace) | PROVEN | NOT_PROVEN |
| `todo` (task tracking) | PROVEN | NOT_PROVEN |
| `session_search` (FTS5) | PROVEN | NOT_PROVEN |
| `skill_view` (structured load) | PROVEN | PARTIAL (`cat` only) |
| Automatic skill loading | From system prompt | Manual (`[skill-name]`) |

## What Codex Loses Without Reference Files

Without the 94 reference files, Codex cannot load patterns for:

- **HTTP route runtime proof** — mock LAHB server, env isolation, 90-scenario taxonomy
- **Cross-process runtime proof** — `spawn()` with file-based coordination
- **X402 reconciliation** — orphan recovery, boundary crash windows, strict-object validation
- **Docker compose safe deployment** — `--force-recreate` name conflicts, SHA build args
- **I18N intent routing safety** — `normalizeText()` accent stripping, word-boundary patterns
- **Container runtime drift** — diffing `docker exec` files against `git show HEAD`
- **Provider boundary crash window** — dispatch timestamp placement, adapter-side fetch
- **Reasoning model budget** — `reasoning_content` consuming entire `max_tokens` budget
- **Commit integrity audit** — 8-phase contamination detection, STRATEGY B repair
- **Verification-supersession truth correction** — distinguishing clean absence from behavioral equivalence
- **Freeze-while-building** pattern — deepFreeze vs pre-frozen sub-objects
- **Feature branch integration readiness** — multi-repo branch naming, PR dependency ordering
- **GHA-unavailable merge bypass** — dead CI check detection, admin bypass pattern

## lah-start: Preflight-Only

`lah-start` is a **pre-flight gate orchestrator** that exits after producing a receipt:

| Function | Classification |
|----------|---------------|
| Repository identity | EXECUTABLE_HARD_GATE |
| CarteLogic bootstrap | EXECUTABLE_HARD_GATE |
| Context pack creation | EXECUTABLE_HARD_GATE |
| Memory gate | EXECUTABLE_HARD_GATE |
| Mission type gate profile | EXECUTABLE_HARD_GATE |
| Startup receipt | EXECUTABLE_HARD_GATE |
| Receipt reuse | EXECUTABLE_HARD_GATE |
| CodeGraph precheck | EXECUTABLE_FAIL_SOFT |
| Git policy observation | EXECUTABLE_FAIL_SOFT |
| Runtime brief injection | DECLARATIVE_ONLY (JSON to stdout) |
| State machine | EXECUTABLE_FAIL_SOFT (startup only) |
| Failure budget | DECLARATIVE_ONLY (brief text) |
| Roadmap mutation prevention | NOT_PROVEN (no mechanism) |
| Post-start gate enforcement | NOT_PROVEN (no daemon/watcher) |
| Agent launch | NOT_PROVEN (prints JSON, exits) |

## Root-Cause Ranking

| Cause | Rank |
|-------|------|
| Skill not automatically loaded in Codex | PROVEN_PRIMARY |
| Lack of durable gate state | PROVEN_PRIMARY |
| Lack of roadmap locking | PROVEN_PRIMARY |
| Differing agent harness architecture | PROVEN_PRIMARY |
| Unavailable `execute_code` and `patch` | PROVEN_CONTRIBUTING |
| Incomplete reference parity (94 vs 0) | PROVEN_CONTRIBUTING |
| Codex context compaction | PROVEN_CONTRIBUTING |
| Differing sub-agent implementations | LIKELY |
| Duplicated routing/bootstrap operations | POSSIBLE |
| Dynamic skill-selection overhead | POSSIBLE |
| Differing model behavior | POSSIBLE |

## Hermès Mission Evidence

At least 25 Hermès missions with proven gate evidence, including:

| Mission | Gates | Tests | Verdict |
|---------|-------|-------|---------|
| LAH_ONE_COMMAND_AUTONOMOUS_ROADMAP_EXECUTION_V1 | A1-A8, N1-N17 (70 tasks) | 133/133 legacy | CERTIFIED |
| LAH_STRATEGIC_PLANNER_PROCESS_SEPARATED_HANDOFF_V1 | 36 phases | 551/551 | 36/36 phases PASS |
| LAH_MAINTENANCE_AUTHORITY_V1 | 7 gates + sub-agents | 53 checks | CERTIFIED |
| CLOE_HOST_GATEWAY_PERSISTENT_PLUGIN_REPAIR_V1 | 14 gates | 10/10 preflight | CERTIFIED |
| LAH_REPOSITORY_HYGIENE_AUTHORITY_V1 | 5 gates + sub-agents | 16/16 | CERTIFIED_READ_ONLY |

## Codex Mission Evidence

Codex invoked `lah-workflow` for at least 5 CLOE missions (CLOE_PROVIDER_MODEL_ALIAS_BOUNDARY_REPAIR_V1, CLOE_DEDICATED_OPERATOR_BRIDGE_AUTH_REPAIR_V1, etc.) but end-to-end certification could not be confirmed from available evidence (volume limitation on session transcript reading).

