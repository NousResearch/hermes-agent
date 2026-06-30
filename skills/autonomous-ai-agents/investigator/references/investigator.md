# Investigator — design notes

## Layering (primitive vs orchestrator)

- **`information-gain` (the ranker)** is a pure, **report-only** primitive: given `(prompt + evidence)`
  it returns the next-best questions ranked by value of information. It never acts.
- **`investigator` (this skill)** is the **orchestrator**: it calls the ranker, answers the top
  questions with a full Hermes agent, folds facts back into one growing context, re-ranks, and
  responds. The split keeps the safe primitive callable anywhere and isolates the privileged loop.

Cross-skill call is in-process: the investigator resolves the ranker's `scripts/` via `HERMES_HOME`
(or `INFOGAIN_SCRIPTS_DIR`) and `import infogain`, then calls `infogain.run(problem, cfg, evidence=)`
and reads `result["all_scored"]`. The grounded answerer/responder delegate to the `ask` skill via
`model_utils.dispatch_single` (a full agent, isolated context).

## Capability ladder

Default is **full agency** (`act`). The ladder only **down-scopes**, and it maps onto the answerer's
toolsets + a prompt directive — it does **not** reinvent a permission system.

| level | toolsets | directive | reversibility |
|---|---|---|---|
| **act** (default) | file, web, terminal | none | unattended — may take real actions |
| **experiment** | file, web, terminal | "reversible experiments only (scratch/worktree)" | reversible by intent |
| **read** | file, web | "read-only; action-needing → NOT_FOUND" | no side effects |

v1 caveat: down-scoping below `act` is **instruction-level** (the directive), since Hermes toolset
read-only granularity is not yet verified. `act` (the default) is unaffected. A future hard sandbox for
`experiment` would pin `answer_cwd` to an auto-created git worktree.

## Vantage handling (the cross-cutting thread with the ranker's vantage family)

The ranker's **vantage** family (when enabled) emits questions whose answer is *access-relative* —
`a(question, vantage)` differs by environment / server / identity / credential / token. For such a
question the Investigator should:

1. **Identify the vantage axis** the question names (which env/credential/POV would change the answer).
2. **Investigate from the relevant vantage(s)** it has access to (a vantage = an access/capability
   requirement — being on server B, using token T).
3. If **multiple vantages are reachable**, investigate from each and **report the diff** — the variation
   across vantages is itself the finding (e.g. "prod and staging configs diverge").
4. If **only one is reachable**, flag the answer as **vantage-conditional** rather than absolute.

v1 records the vantage axis and investigates from the current vantage; multi-vantage comparison is a
fast-follow (needs the Investigator to acquire/switch credentials, which is an `act`-level capability).
