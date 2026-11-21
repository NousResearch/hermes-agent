# Run

Native research, when chosen: resolve effort, hold the plan gate, then run the acquisition loop once per dimension of the approved plan, in plan order.

## Effort

Three knobs bundled in a **preset**; any knob pins individually, and **what the user says in the request beats both**.

| Preset (`{workflow.preset}`) | subagents | sources/round | depth |
|---|---|---|---|
| `quick` | low (2) | 5 | 1 |
| `standard` (default) | normal (3) | 8 | 2 |
| `deep` | high (6) | 12 | 3 |

- **subagents** — parallel assistants: `none` (0 — inline, sequential; also the no-subagent-harness fallback), `low` (2), `normal` (3), `high` (6, cap 10 — beyond the 3–5 sweet spot only for genuinely wide work).
- **max_sources_per_round** — distinct sources actually read per dimension per round (cap 25).
- **max_depth** — rounds per dimension: initial pass plus lead-following follow-ups (cap 5). A cap, not a quota — dimensions stop early on coverage or novelty exhaustion.
- **validation** (orthogonal to preset, default `normal`) — rigor rises `normal` < `high` < `max`; level semantics live in `references/verification.md`. Verification happens per dimension as material lands, never as an end-of-run rewrite pass.

`{workflow.subagent_models}` is an ordered model preference for assistants — first available wins; empty means harness default. Keep the lead on the strongest model; researchers at most one tier down; judgment work never on the smallest tier.

## The plan gate

The one hard stop, kept light: decision, type and pack-derived dimensions pruned to it, shape, the **decomposition topology** — *breadth-first* (independent sub-questions: assistants split the dimensions), *depth-first* (one question that needs several perspectives: assistants split by angle or methodology, not by dimension), or *straightforward* (a focused ask: one assistant, a handful of calls, no fan-out — never overinvest in a simple query) — knobs in force and where each came from, which search surfaces exist (harness web search; installed search-shaped MCP tools; `{workflow.external_sources}` — check, don't assume), whether to run the fan-out as a workflow when the harness offers orchestration and `{workflow.use_workflows}` allows, and an honest time estimate (a standard run is minutes; deep runs are tens of minutes and many times the tokens).

Present as a compact checklist, get approval, then: bind `{doc_workspace}` under `{workflow.research_output_path}` — expand the folder name with `uv run scripts/recon_kit.py slug "<topic>" --type <type> --pattern "{workflow.run_folder_pattern}"` so the same topic always resolves to the same folder — seed `research.md` from `{workflow.research_template}`, init the memlog (`uv run {project-root}/_bmad/scripts/memlog.py init --workspace {doc_workspace} --field topic="<topic>" --field type="<type>" --field decision="<decision>" --field preset="<preset>"`), log the approved plan as a `decision`, and tell the user the path.

Each dimension then runs in **rounds** — up to the resolved `max_depth` — and the report grows as material lands: the user watches the document build, not a spinner. Every digest is written to `{doc_workspace}/digests/` the moment it exists — one file per assistant per round (`<dimension>-r<round>-<n>.md`), the digest shape below, raw enough to re-derive from.

## Rounds and lead-following

Round 1 pursues the plan's questions **broad-first**: short, wide queries to map what exists, narrowing as the shape emerges — not long specific queries that return nothing. After each round, harvest the leads: new entities worth chasing, unexpected connections, contradictions between sources, and questions the round opened. Contradictions get priority. Promising leads become the next round's brief; note mid-course discoveries in the checkpoint so the user sees the turn happening.

A dimension stops before its round cap when either holds:

- **Coverage** — its plan questions are answered, with the critical claims confirmed per the resolved `validation` level.
- **Novelty exhaustion** — a full round surfaced no new load-bearing claim or lead.

Say which one ended it. Hitting the round cap with open questions is reported as an open question, never silently dropped.

**Stop-and-write valve.** If the run is dragging well past the plan gate's estimate — rounds queuing, budgets mostly spent — stop spawning, synthesize from the digests already on disk, and report the remainder as open questions with a route (a Deepen later, or a drafted prompt for the user's own tool). A shorter honest report beats a longer stale one.

## The fan-out

Fan out researcher assistants for the round — concurrency per the resolved `subagents` level, split by the plan's **topology**: breadth-first gives each assistant independent sub-questions; depth-first gives each a distinct perspective or methodology on the *same* question; straightforward is one assistant with a small budget — never fan out what one focused assistant answers. Each assistant runs behind the **research firewall**: it gets its brief and nothing else — no project files, no ambient context. The brief contains:

- the questions it owns, the decision they serve, and the topic
- its search surfaces (specialized tools first — installed search-shaped MCP tools, `{workflow.external_sources}` entries whose directive matches — then generic search), plus `{workflow.preferred_sources}` first / `{workflow.banned_sources}` never
- the pack's source craft and freshness bars, and the source-quality card below
- its budgets — sources (the round's share of `max_sources_per_round`) and tool calls, scaled to its task: under 5 for a simple lookup, ~5 medium, ~10 hard, 15 for genuinely multi-part, 20 never exceeded. Either budget spent → synthesize what it has
- the query craft: short queries (roughly five words or fewer) beat hyper-specific ones that return nothing; broaden when results are sparse, narrow when abundant; never repeat an identical query on the same tool; after every tool result, pause and evaluate — what did this add, what gap remains, what's the best next query — before firing again
- the epistemics rules verbatim, and the return contract: a digest, not raw results — findings as claims, each with `{claim, source, publisher, pub_date, accessed, confidence, class}`, plus leads worth chasing and what it looked for and could not find

**On each return, write the digest to `{doc_workspace}/digests/` before doing anything else with it.**

Spawn assistants on `{workflow.subagent_models}` when set (first available wins); otherwise the harness default — judgment work never drops to the smallest tier. When subagents are unavailable (or `subagents` is `none`), run the same rounds yourself, sequentially, under the same budgets and the same files-first discipline.

When workflow orchestration was approved at the plan gate, run the fan-out as a workflow: dimensions as parallel pipelines, assistants returning structured digests. The budgets, digest contract, firewall, and stopping rules apply unchanged — and however the acquisition parallelizes, digests land as files and the lead alone writes `research.md`, committing sections in plan order.

## Source quality

One card, applied by every assistant and the lead alike. Prefer **primary sources** — filings, regulator text, official documentation, original papers, a company's own reported numbers — over aggregators and secondary reporting. Red flags that downgrade confidence on sight: speculative language ("could", "may", projections in future tense presented as findings), marketing register, passive voice with unnamed sources, cherry-picked or unsourced numbers, and aggregators recycling a single upstream report (that's one publisher, however many domains echo it). Answer engines (Perplexity Sonar, Grok, and kin) are aggregators too, however good the synthesis: chase their citations and cite those, never the engine. Conflicts resolve by recency, consistency with adjacent established facts, and publisher quality — never by averaging.

## Synthesize the dimension

When a dimension's rounds are done:

1. Verify at landing per `references/verification.md` — at `normal` validation this is a spot-check of the dimension's load-bearing claims, not a sweep.
2. Write the dimension's section per the pack's skeleton from its digest files — findings woven into prose answering the dimension's questions, every load-bearing claim cited inline `[n]`, confidence flagged where below high, contradictions reported with both sides cited. Append to `research.md` and add its sources to the running source table.
3. Log one memlog line per source batch (`--type source`) and one per load-bearing claim worth tracking for refresh — `--type claim`, text in the machine-readable shape `ref=[n] status=<verified|unverified|disputed|overturned> class=<class> pub=<YYYY-MM> — <claim>` so `scripts/recon_kit.py tally` and `staleness` can read the ledger; a later status change is a fresh claim line with the same `ref=` (last status wins).
4. Checkpoint: one or two lines in chat — what the dimension found, anything surprising, anything unresolved. Keep moving unless the user speaks up; a mid-run scope change is logged as a `decision` and the plan adjusts. Headless: skip checkpoints entirely.

When all dimensions are done, proceed to `references/synthesis.md` for final assembly.
