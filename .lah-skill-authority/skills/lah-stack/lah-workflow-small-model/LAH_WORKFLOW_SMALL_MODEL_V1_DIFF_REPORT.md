# LAH_WORKFLOW_SMALL_MODEL_V1_DIFF_REPORT

## ORIGINAL

- **path:** `/home/deploy/.hermes/skills/lah-stack/lah-workflow/SKILL.md`
- **lines:** 1024
- **bytes:** 100639
- **sha256:** `ba1407bea3d0a3d30a4bdb368bea3f0afdad05822489f8084fba248764d7e5`

## SMALL MODEL

- **path:** `/home/deploy/.hermes/skills/lah-stack/lah-workflow-small-model/SKILL.md`
- **lines:** 186
- **bytes:** 7448
- **sha256:** `654632226e5277ff87c3558bee593d8e6986a9048c90d6be98c4e8963c0a513`

## PRESERVED SEMANTICS

- Authority order (operator > constraints > runbook > workflow > domain > tool > historical)
- Gate sequence (0 through 11, same order and naming)
- STOP semantics for AMBIGUOUS/UNRESOLVED routing
- Safety constants pattern (Object.freeze)
- FastSafe 15-check enumeration
- Sub-agent discipline (max 3, context rules)
- Ciblé staging (no `git add .`)
- Commit message prefix convention
- PR merge with `--match-head-commit`
- Behavioral simulation gate (Gate: BEHAVIORAL_SIMULATION)
- Mission type awareness (CODE_CHANGE, READ_ONLY_AUDIT, DESIGN_ONLY, MIXED, PROMOTION_ONLY)
- Evidence authority hierarchy (runtime > provider > authorization > safety > deployment > certified history)
- LCM context discipline principles

## SIMPLIFIED SEMANTICS

- Mission-type branching logic: the small-model variant defers to `lah-workflow` for detailed mission-type branching (READ_ONLY_AUDIT, DESIGN_ONLY, MIXED, PROMOTION_ONLY) rather than duplicating it. Small models load `lah-workflow` for mission-type specifics when needed.
- Pitfall encyclopedia: the 40+ documented pitfalls from the canonical skill are not duplicated. Small models are instructed to consult `lah-workflow` references when a specific pitfall is suspected.
- Cross-agent execution differences: not included (small models run as Hermes agents, not Codex).
- Communication adaptative (caveman levels): preserved as a reference but not expanded.
- Linked references catalog: not duplicated; small model defers to canonical skill for reference resolution.
- Detailed command examples with full paths: replaced with compact references.
- Operator packet format details: deferred to `references/operator-packet-format.md` in the canonical skill.

## MOVED-TO-ON-DEMAND

- Mission-type branching (READ_ONLY_AUDIT, DESIGN_ONLY, MIXED, PROMOTION_ONLY details) → load `lah-workflow` when mission type requires it
- Pitfall resolution patterns → load `lah-workflow` references when a specific pitfall is suspected
- Gate 9 dead CI check protocol → load `lah-workflow` when CI failure is detected
- Gate 9 PR merge worktree verification → load `lah-workflow` for post-merge verification
- Container runtime drift patterns → load `lah-workflow` references when Docker is involved
- i18n intent routing safety patterns → load `lah-workflow` references for French-language missions
- Provider budget patterns → load `lah-workflow` references for reasoning model interactions
- Upstream npm package patterns → load `lah-workflow` references for npm-related defects
- SHA provenance patterns → load `lah-workflow` references for document provenance
- Base64/Base64URL patterns → load `lah-workflow` references for encoding tests

## REMOVED DUPLICATION

- Gate-pass checklists repeated across gates → consolidated into a single compact gate sequence table
- "Do not skip" warnings → replaced with explicit STOP semantics in the Stop Contract
- Sub-agent context rules repeated in Batch 1 and Batch 2 → stated once in Sub-Agent Discipline
- "No git add ." repeated in FastSafe and Gate 8 → stated once in the Gate Sequence
- Evidence authority patterns repeated across reference files → consolidated into a single Evidence Contract
- Multiple "STOP" semantics for different error conditions → unified in the Stop Contract table
- The 3-lane sub-agent batch pattern described in both Gate 0.5 and Sub-agents section → stated once
- Mission type skip rules described in both the table and individual gate sections → consolidated

## NEW SMALL-MODEL GUARDRAILS

1. **Mission Mode Classification** — every mission must be classified into exactly one of EXECUTE/DIAGNOSTIC/REPAIR/CERTIFY before execution. No silent mode switching.
2. **Certified Runbook Fast Path** — when a certified runbook exists, it takes priority over all workflow steps. No reconstruction from history.
3. **Search Budget** — explicit bounded discovery policy. EXECUTE: max 1 narrow lookup. CERTIFY: max 2 narrow evidence-resolution actions. DIAGNOSTIC/REPAIR: bounded but allowed.
4. **Anti-Archaeology** — broad recursive grep, broad find, filesystem archaeology, worktree archaeology, port scanning, route guessing, and repeated searches for the same fact are FORBIDDEN by default in EXECUTE and CERTIFY modes.
5. **Evidence Contract** — compact evidence classes with strict hierarchy. Historical receipt cannot override current provider state. Source code is not runtime evidence.
6. **Stop Contract** — 8 explicit STOP conditions with prescribed actions. No improvisation after STOP.
7. **LCM Context Discipline** — compact working state format. No full historical mission reloads. No repeated reconstruction of certified facts.
8. **Authority Order** — deterministic 7-level hierarchy. Lower authority cannot override higher authority STOP.
9. **Sub-Agent Discipline** — max 3 per call, 2 batches max. No discovery-only sub-agents.
10. **Deferral to Canonical** — mission-type branching, pitfall resolution, and detailed reference patterns are delegated to `lah-workflow` rather than duplicated.

## EXPECTED REDUCTION IN

- **Broad searches** — bounded by explicit search budget (1-2 narrow lookups vs unlimited)
- **Repeated searches** — forbidden; certified facts are treated as references, not re-queried
- **Tool calls** — reduced by eliminating discovery archaeology and mode-switching overhead
- **Context reconstruction** — LCM context discipline prevents full historical mission reloads
- **Architecture rediscovery** — certified runbook fast path eliminates re-discovery; CodeGraph is used once at Gate 1
- **Accidental mode switching** — explicit mode classification with STOP on unauthorized switching

## NOTE ON MEASUREMENT

No numerical performance improvement is claimed without measurement. The reductions above are structural expectations based on the removal of unbounded discovery and the introduction of explicit budgets. Empirical validation is required to confirm actual iteration count reduction.
