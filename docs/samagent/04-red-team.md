# 04 — Red team: killing the v0 plan

Method: for each item in `03-draft-plan-v0.md`, find the strongest evidence *against* it, then give a verdict.
Verdicts: **KILL** (remove) · **CUT** (shrink) · **CHANGE** (redesign) · **KEEP**.
Evidence tags as in `01-research.md`.

## Attacks

| # | v0 item | Attack | Evidence | Verdict | What replaces it |
|---|---------|--------|----------|---------|------------------|
| 1 | D0 hard-fork + slim | Deleting upstream files guarantees merge conflicts and breaks the roughly 1.26M-line test suite. Hermes moves fast (a commit references PR #88282). Within months the fork cannot take upstream fixes. | [R] repo scale, test size | **CHANGE** | **Overlay, not fork-and-delete.** All new code in new directories. Slim by *config and profile*, never by deleting. A tracked patch list of core edits (target ≤ 4). |
| 2 | D0 rewrite core in TS like Pi ("Option C") | Discards 860K lines of solved provider, credential, approval, compaction and gateway problems. Pi's own protocol is still changing. The differentiation is not in the loop. | [R] audit, [W1] Pi churn | **KILL** | Adopt Pi's *ideas* (lean tools, sticky routing, cache-warming economics). Possibly run Pi later as an external worker over RPC. |
| 3 | D1 nine permanent specialists | "More agents = smarter" is false for write-shaped work. Nine writers means nine sets of implicit decisions. Also about 15× tokens for multi-agent setups. Roles like "Docs" or "DevOps" rarely have independent write sets. | Cognition; MAST 36.9% misalignment; Anthropic 15× [W1] | **CHANGE** | **One conductor + workers defined by *module ownership*, not job title.** Fan-out only after the contract freeze and only for ≥ 2 independent leaves. Default is single-thread for small projects. Verifiers are separate agents. |
| 4 | D1 no evidence the swarm helps | It is an untested assumption. Pi's thesis (no sub-agents in core) and Arena's harness-tax data both say simple can win. | [W1] | **CHANGE** | **Ablation rule:** the swarm ships enabled only if SamBench shows it beats the lean single-agent arm on pass rate, or on wall-clock at equal pass rate. Otherwise it stays opt-in. |
| 5 | D2 15–20 question interview | Interview fatigue. **You skipped a 6-question interview in this very session**, which is a live data point. A blocking interview loses the users we want. | [I] this session | **CHANGE** | **Adaptive, skip-safe interview:** at most 5 questions, each with a recommended default and a "use defaults" button. It never blocks. Skipped answers become a visible **assumption ledger** the agents treat as changeable. |
| 6 | D3 per-turn auto-switching | Switching model mid-conversation loses the prompt cache and thinking signatures, and changes behaviour mid-task. Hermes and Pi both document this. Also raises cost, opposite of the goal. | [R] Pi docs, Hermes AGENTS.md | **CHANGE** | **Task-boundary routing.** Each sub-agent is a fresh context with its own model. Escalation = a *new* agent given a handoff brief from the ledger. Sticky within a task. |
| 7 | D3 learned router | Routers only pay off after you have golden-task eval suites and logged outcomes. We have neither. Blind routing degrades quality silently. | [W2] Databricks/AT&T | **KILL for v1** | Static policy table + measured per-model scorecards. Revisit a learned router only after ≥ N logged outcomes and only if it beats the table on SamBench. |
| 8 | D3 "local models first" for everything | Open models trail cloud models in Arena's aggregate ranking, and small models hallucinate tools. Forcing local for spec, contract and judge steps risks the highest-leverage phases. | [W1] Arena; [W2] tool-calling | **CHANGE (honestly)** | Local-first is the default *for bounded work* (edits, test-fix loops, exploring, summaries, memory extraction, private data). Spec, contract and judge default to the strongest available model **if a cloud key exists**. If none exists, or the user chooses local-only, local runs everything with lower autonomy and visible warnings. This deviation from "local first" is explicit and configurable. |
| 9 | D4 knowledge graph + embeddings + reflection | Over-engineered. Vendor memory benchmarks disagree by 2×. Hermes users already complain about reflection passes that burn tokens and hard-coded counters. Memory that is wrong and permanent is worse than none. | [W2] LongMemEval spread; community complaints | **CUT** | **Ledger v1:** SQLite + FTS5 (already used in Hermes), explicit supersession, markdown mirror. Writes only on *events* (decision made, user correction, verified outcome), no periodic reflection. Embeddings only if a resume/recall eval shows FTS5 is not enough. |
| 10 | D4 "remember everything, forever" | Privacy and poisoning risk. Memory that flows into cloud prompts leaks private facts. | [R] memory entries enter the system prompt today | **CHANGE** | **Sensitivity tags.** Private facts are excluded from any cloud-routed prompt. Injection is a budgeted block in the *user* turn, never the system prompt. |
| 11 | D5 "match Arena's speed" | Not measurable and not copyable: Arena's warm-pool/routing internals are not public, and we have no hosted fleet. A local agent's bottleneck is prefill, not sandbox startup. | [W1] Arena internals unpublished | **CHANGE** | A **speed budget** with named metrics (time-to-first-signal, time-to-first-preview, time-to-green, cache-hit rate) and evidenced levers: deterministic scaffolds, cache-stable prefixes, parallel workers, local warm model, speculative decoding, speculative read-only prefetch. |
| 12 | D6 new Electron app with 3D graph, editor, terminal, marketplace | The existing desktop app is 608K lines. A second full UI is a multi-year trap for one builder. The 3D graph and editor are demo-ware; "not an IDE" was your requirement. | [R] 608K lines; user constraint | **CUT** | **Mission Control:** 5 screens, a size cap (~15K lines), built first as a **dashboard plugin** (the Kanban tab is the precedent) on existing gateway contracts. Desktop wrapping is a later packaging step. No editor, no marketplace. |
| 13 | D7 every target and deploy host at once | Unbounded scope; each stack needs templates, verification recipes and security checks. | [I] | **CUT** | **Web apps only in v1**, with 2–3 template packs. Deploy is a gated, later adapter. Other stacks via skills after v1. |
| 14 | D8 "smarter than X" and D9 "feels smart" | Unfalsifiable; and the harness-tax study says harness changes move cost more than success. Marketing claims without numbers will not survive scrutiny. | [W1] | **KILL** | **Seven falsifiable hypotheses** (H1–H7 in `05-final-plan.md`) measured on **SamBench-Web** against Hermes-default and Pi at equal models. Publish whatever the numbers say. |

## More attacks on the plan's foundations

- **A15 — My research could be wrong.** I read docs and code and measured only tool-schema size (I did not run the agent end to end), and many web sources are SEO. *Response:* Phase 0 is a measurement week with **decision gates**. Nothing big is built before baselines exist.
- **A16 — 24 weeks and 6 phases for one person.** *Response:* 12 weeks with parallel tracks, kill criteria per phase, and the UI and memory tracks capped.
- **A17 — Security theatre.** Scanners catch known patterns; "hallucinated auth" is a logic bug. *Response:* generate **authz probes from the spec's role matrix** and run them against the *running* app, plus secure scaffolds so auth is not improvised by the model.
- **A18 — Verification by the same model that wrote the code shares its blind spots.** *Response:* fresh-context verifiers, different model family where possible, and judge against the **original brief**, not the writer's summary.
- **A19 — Autonomy for beginners is dangerous** (Replit). *Response:* default to *plan-approve-then-run*, sandboxed worktrees, dev/prod separation, approval gates for deploy, delete, secrets and spend, and hard caps.
- **A20 — Cost blow-up** from swarm + verifiers. *Response:* an estimate before the run, hard caps per run and per task, and an ablation that turns the swarm off when it does not pay.

## Pre-mortem: "It is December 2026 and SamAgent failed. Why?"

| Likely cause | Prevention built into the final plan |
|--------------|--------------------------------------|
| The swarm produced integration hell and cost 5–10× a single agent | Contract freeze, ownership guard, fan-out gate, ablation rule |
| Local models were too weak on the user's machine; agents looped | Scorecards on real hardware, argument validators, repeat-loop detector, verified escalation, honest capability display |
| Fork drift: could not merge upstream after month 2 | Overlay + patch list + a monthly upstream-merge rehearsal |
| UI scope exploded | Size cap, dashboard-plugin-first, Simple/Pro only, no editor |
| Nobody trusted the output | Every "done" links to evidence: test run, screenshot, security report, assumptions list |
| The interview annoyed users | Skip-safe, ≤ 5 questions, defaults |
| Solo burnout | Phase kill criteria, parallel tracks capped, weekly demo of a real end-to-end run |

## Decision gates (what would make me change the plan)

| Gate | When | Test | If it fails |
|------|------|------|-------------|
| **G0** | End of Phase 0 | Baselines run; spikes S1–S9 pass or have workarounds | Revisit architecture (overlay vs patch vs fork) |
| **G1** | End of Phase 1 | Lean profile cuts input tokens ≥ 30% at equal pass rate; local tool-call validity ≥ 95% on the chosen model | Re-scope local-first to "helper" role; cloud-primary defaults |
| **G2** | End of Phase 2 | Brief → approved spec in ≤ 5 questions and < 10 min; acceptance tests generated and red | Simplify spec schema; drop generated tests for hard criteria |
| **G3** | End of Phase 3 | Swarm beats lean single-agent on SamBench (H2) | Ship swarm opt-in; keep the pipeline single-threaded |
| **G4** | Phase 5 MVP | 4/5 novices ship a working app unaided (H7) | Cut features, not screens; redo Simple mode |

## Verdict summary

Kept from v0: interview-first, sub-agent swarm, local + cloud, external memory, a simpler UI, web-first, and the ambition to be measurably better.
Killed: hard fork and slim, TS core rewrite, nine permanent specialists, per-turn switching, learned router, KG memory with reflection, 3D UI/editor/marketplace, all-targets-at-once, unfalsifiable claims.
Changed: everything else, into the gated, measured design in `05-final-plan.md`.
