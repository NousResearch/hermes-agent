---
name: quality-analysis
description: The Analyze orchestrator for BMad agents. Runs the deterministic pre-pass, dispatches the quality lenses in parallel, merges their findings in-context, authors the synthesis layer, and renders the report deterministically via scripts/render_report.py. No per-subagent files.
---

**Language:** Use `{communication_language}` for all output.

# Analyze: Quality Analysis for a BMad Agent

Personality is investment, not waste. You analyze an agent to find where its capability prompts, structure, and wiring can be leaner or sharper, and you never recommend that the agent's voice be flattened. A rich persona is the deliverable, so the lenses apply the leanness bar to capability prompts and to leaked structure, not to persona voice, communication-style examples, domain framing, design rationale, or theory-of-mind.

`{target-agent-path}` is the agent directory under analysis, a directory containing a `SKILL.md`. You orchestrate: the pre-pass classifies and counts, the lenses judge, you synthesize, and the render script produces the report. You do not read the agent's raw files yourself, because the pre-pass and the lenses already do and your context is better spent merging their returns.

## Run folder

Each analyze run owns `{target-agent-path}/.analysis/<YYYY-MM-DD-HHmm>/` (create it first). It receives `findings.json`, `agent-analysis-report.html`, and `agent-analysis-report.md`. This run folder is the report location everywhere — the headless return points into it.

## Headless mode

If `{headless_mode}=true`, skip user interaction, take safe defaults, note any warning rather than asking, and emit the structured JSON described under Present. This is the builder's own headless mode and has nothing to do with a built autonomous agent's runtime Pulse Mode (`--pulse`); the two are different flags entirely.

## Pre-scan check

Confirm the agent is resolvable at `{target-agent-path}` and that a `SKILL.md` is present. In interactive mode, note any uncommitted changes in the agent tree so the user knows the report reflects the working copy; in headless mode record that as a warning and proceed. You do not commit, stage, or push anything.

## Run the deterministic pre-pass first

Run the pre-pass once, before any lens sees the agent, so every lens reads a compact classification and token picture instead of re-deriving it from raw text:

```bash
uv run scripts/prepass.py {target-agent-path}
uv run scripts/scan-path-standards.py {target-agent-path}
uv run scripts/scan-scripts.py {target-agent-path}
```

The two lint scanners return deterministic findings as JSON; carry their entries straight into the merged findings list with ids `lint-<n>`, keeping their severities. They are facts, not judgment, so no lens re-derives them.

It prints one JSON object on stdout, the pinned pre-pass shape:

```json
{
  "agent_type": "stateless | memory | autonomous",
  "is_memory_agent": true,
  "skill_md_tokens": 0,
  "files": [{ "path": "SKILL.md", "tokens": 0 }]
}
```

Hold that object. `agent_type` and `is_memory_agent` decide whether the conditional sanctum lens runs, and the token counts are the lengths the lenses reason about. Lengths come from tokens here, never line counts. The pre-pass reads the built agent's sanctum to classify it; it never reads the builder's `.memlog.md`, and neither do you.

## Dispatch the lenses in parallel

Hand each lens the pre-pass JSON and `{target-agent-path}`, and run them as parallel subagents. Each lens loads the bar its own spec file names plus `references/lens-contract.md`, stays in its lane, and returns its findings to you in-context. No lens writes a file or a per-subagent analysis document.

Six base lenses run for every agent:

| Lens | File | Owns |
| --- | --- | --- |
| Leanness | `references/scan-leanness.md` | The three minimal-baseline tests applied to capability prompts and leaked structure, with the persona carve-out held explicit. The only lens that fills `proposed_smallest` and `predicted_delta`. |
| Architecture | `references/scan-architecture.md` | Frontmatter, topology, progressive disclosure, activation soundness (the four-step waking spine and Pulse Mode), ordering, parallelization, read-avoidance. |
| Determinism | `references/scan-determinism.md` | The determinism test, the signal-verb scan, the script-opportunity categories, intelligence placement, and the transcript repeated-work signal. |
| Customization | `references/scan-customization.md` | The customize.toml surface, its abuse lenses branched by archetype, and confirmation it is the only config mechanism present. |
| Enhancement | `references/scan-enhancement.md` | Edge cases, experience gaps, delight, headless potential, facilitative patterns. |
| Agent cohesion | `references/scan-agent-cohesion.md` | Persona-capability alignment, gaps, redundancy, granularity, user-journey coherence. |

One conditional lens runs only when the pre-pass classified the agent as memory or autonomous:

| Lens | File | Runs when |
| --- | --- | --- |
| Sanctum architecture | `references/scan-sanctum-architecture.md` | `is_memory_agent` is `true`. Bootloader weight, sanctum templates, First Breath, CREED standing orders, the init script. Skipped entirely for a stateless agent. |

Read `is_memory_agent` from the pre-pass. If it is `true`, include the sanctum lens in the parallel dispatch so seven lenses run. If it is `false`, dispatch the six base lenses only and the report will carry no sanctum block.

Every lens returns the JSON in `references/lens-contract.md`. Only the leanness lens fills `proposed_smallest` and `predicted_delta`; those two fields let you route a defend-against-absence finding to the eval-runner's variant mode for a real cut-or-keep verdict rather than a guess, and that routing happens in the build flow, not here.

## Synthesize and render

Merge the lens returns into one findings list, keeping each finding's `id` so it stays traceable to the lens that raised it. Do this in your own context; there is no extract-and-reassemble round-trip.

Two org gates fold in here: if `{agent.build_standards}` is non-empty, check the agent against each directive (`skill:`, `file:`, or plain text) and add any miss as a conformance finding; if `{agent.evals_required}` is set, confirm `{target-agent-path}/evals/cases.json` satisfies it (`"baseline"` or `"any"`) and add a high-severity finding when it does not.

Then author the report yourself. You hold every finding in context, so no subagent is involved; never hand-write report HTML, and never edit the rendered file. The findings are the evidence; the synthesis is what a user must grasp in 30 seconds. All synthesis fields are yours to write:

- `verdict` — one line naming the overall state and the one or two findings that matter most. When the agent carries a rich persona, say it was treated as investment, not waste.
- `grade` — `excellent` (no high or critical, few medium), `good` (some high or several medium), `fair` (multiple high), `poor` (any critical). Lowercase.
- `summary` — 2-3 sentences: the agent's primary strength and primary opportunity. This is the first thing the user reads.
- `themes` — findings clustered by shared root cause, not by file. Ask: "if I fixed X, how many findings across lenses would that resolve?" 3-5 themes; findings that fit no theme stay ungrouped in `findings` only. Each theme's `action` is one coherent fix instruction for the whole cluster, and `finding_ids` lists the constituent findings.
- `strengths` — what works and must be preserved (the load-bearing persona belongs here), so a fix pass does not flatten it.
- `recommendations` — ranked by leverage: rank 1 resolves the most findings for the least effort. `resolves` lists the finding ids it would clear.

The agent blocks are optional portrait-and-context blocks, built from the pre-pass and what the lenses observed:

- `agent_profile` — `name`, `title`, `icon`, `agent_type` (straight from the pre-pass), one-line `mission`. Drawn from the agent's `[agent]` metadata.
- `capabilities` — `{ name, kind, note }` per capability, where `kind` is the form (prompt, script, multi-file, external skill) and `note` is one line on what it does.
- `detailed_analysis` — keyed by lens name, each value that lens's one-line `verdict`.
- `sanctum` — only for memory and autonomous agents: `{ present: true, location, files, note }` where `location` is `{project-root}/_bmad/memory/{skillName}/` and `note` states that the sanctum is the built agent's runtime memory, distinct from the builder's `.memlog.md`. Omit the block (or set `present: false`) for a stateless agent.
- `experience` — `journeys` as `{ name, steps }` for the main paths a user takes through the agent, and `headless` as one line on the agent's headless story.

`findings.json` is one object (schema_version 2):

```json
{
  "schema_version": 2,
  "subject": "<agent name or path analyzed>",
  "generated": "<ISO date>",
  "verdict": "<one-line overall assessment>",
  "grade": "excellent | good | fair | poor",
  "summary": "<2-3 sentence narrative>",
  "standards": {
    "canon": "<absolute path to this builder's references/prompt-quality-canon.md>",
    "principles": "<absolute path to this builder's references/agent-quality-principles.md>",
    "scripts": "<absolute path to this builder's references/script-standards.md>"
  },
  "agent_profile": { "name": "", "title": "", "icon": "", "agent_type": "", "mission": "" },
  "capabilities": [{ "name": "", "kind": "", "note": "" }],
  "detailed_analysis": { "leanness": "<lens verdict>", "architecture": "<lens verdict>" },
  "sanctum": { "present": true, "location": "", "files": [], "note": "" },
  "experience": { "journeys": [{ "name": "", "steps": "" }], "headless": "" },
  "themes": [
    {
      "title": "<root-cause name>",
      "root_cause": "<what is happening and why it matters>",
      "finding_ids": ["leanness-1", "determinism-2"],
      "action": "<one coherent fix for the whole theme>"
    }
  ],
  "strengths": ["<what works and should be preserved>"],
  "recommendations": [
    { "rank": 1, "action": "<what to do>", "resolves": ["leanness-1"] }
  ],
  "findings": ["<every lens finding unchanged, per references/lens-contract.md>"]
}
```

Rules:

- `standards` is always filled: resolve the three absolute paths from this builder's own `{skill-root}` at authoring time. The shell prepends them to every copied fix prompt, so the session that applies a fix holds the same bar that produced the findings.
- `findings` carries every lens finding unchanged — keep each finding's `id`, `lens`, and `severity` so it stays traceable. Carry `proposed_smallest` and `predicted_delta` only when the leanness lens supplied them; omit the keys otherwise.
- Severity counts are derived from the `findings` array by the script and the shell — there is no counts field to keep consistent.
- Every key except `schema_version`, `subject`, `generated`, `verdict`, and `findings` is optional: omit a key entirely rather than writing an empty placeholder. A clean pass is a real report.
- Keep `evidence` and `recommendation` to a sentence or two; the shell shows them in a collapsible row, not a document.

Write the island object to `{run-folder}/findings.json` and render:

```bash
uv run scripts/render_report.py {run-folder}/findings.json --shell assets/report-shell.html -o {run-folder}/agent-analysis-report.html --md {run-folder}/agent-analysis-report.md
```

If the script refuses, fix `findings.json` and re-run; never hand-edit the HTML. Open the HTML report for the user — it is the deliverable of Analyze; do not replace it with a chat summary of the findings. The shell fails loud: a malformed island shows a visible banner, never a blank page, and an empty findings array renders an explicit no-findings panel, so a clean agent still produces a real report.

## Record the run

Append one memlog event carrying the grade (init the memlog first if `{target-agent-path}/.memlog.md` does not exist):

```bash
uv run {project-root}/_bmad/scripts/memlog.py append --path {target-agent-path}/.memlog.md --type event --text "analyze: grade <grade>, <c> critical / <h> high / <m> medium / <l> low, report .analysis/<timestamp>/agent-analysis-report.html"
```

## Present

**IF `{headless_mode}=true`:** emit

```json
{
  "headless_mode": true,
  "status": "complete",
  "agent": "{target-agent-path}",
  "agent_type": "stateless | memory | autonomous",
  "grade": "excellent | good | fair | poor",
  "html_report": "{target-agent-path}/.analysis/<timestamp>/agent-analysis-report.html",
  "md_report": "{target-agent-path}/.analysis/<timestamp>/agent-analysis-report.md",
  "memlog": "{target-agent-path}/.memlog.md",
  "counts": { "critical": 0, "high": 0, "medium": 0, "low": 0 }
}
```

**IF interactive:** present the agent portrait (icon, name, title, type), the grade, the one-line verdict, the severity tally, the capability dashboard summary, and the top themes. Note that the persona was treated as investment and was not flagged as waste. Point to the HTML report path, say it opened in the browser, and offer to walk through findings, apply a fix, or route a leanness finding's `proposed_smallest` to a variant eval.
