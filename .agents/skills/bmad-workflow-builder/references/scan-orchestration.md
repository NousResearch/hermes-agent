# Scan Orchestration

How Analyze runs: a deterministic pre-pass, five LLM lenses in parallel, you merge and synthesize in-context, and a script renders the report. `{target-skill-path}` is the skill under analysis.

## Run folder

Each analyze run owns `{target-skill-path}/.analysis/<YYYY-MM-DD-HHmm>/` (create it first). It receives `findings.json`, `skill-analysis-report.html`, and `skill-analysis-report.md`.

## Run the deterministic pre-pass first

Run these in parallel so the lenses read metrics instead of re-deriving them:

- `uv run scripts/prepass-prompt-metrics.py {target-skill-path}`: per-file token counts (via `scripts/count_tokens.py`), frontmatter facts, and structural signals as JSON.
- `uv run scripts/prepass-workflow-integrity.py {target-skill-path}`: workflow-integrity checks as JSON.
- `uv run scripts/scan-path-standards.py {target-skill-path}`: path-convention lint (bare-paths-from-root, no double-prefix, no `./`).
- `uv run scripts/scan-scripts.py {target-skill-path}`: script-standards lint (PEP 723 metadata, shebangs, non-stdlib confirmation).

## Run the five lenses as parallel subagents

Hand each lens the pre-pass JSON and the skill path. Each loads the bar its own spec file names (the canon, the principles file, or its lane's spec) and returns its findings to you in-context.

| Lens | File | Owns |
| --- | --- | --- |
| Leanness | `references/scan-leanness.md` | The three minimal-baseline tests: the core test, the defend-against-its-own-absence test, the outcome-vs-prescription test. |
| Architecture | `references/scan-architecture.md` | Structure, frontmatter, file topology, progressive disclosure, three-mode soundness. |
| Determinism | `references/scan-determinism.md` | The intelligence-placement boundary: intelligence leaks and determinism leaks, cross-referenced to script opportunities. |
| Customization | `references/scan-customization.md` | `customize.toml` surface economics, and confirmation that it is the only config mechanism present. |
| Enhancement | `references/scan-enhancement.md` | Missing named patterns to add and over-applied patterns to cut. |

Each lens returns the JSON in `references/lens-contract.md`. The leanness lens also returns `proposed_smallest` and `predicted_delta` on defend-against-absence findings, which you can route to the eval-runner's variant mode for a cut-or-keep verdict.

## Apply the org gates

Two customize-driven gates run alongside the lenses, only when configured:

- **`{workflow.build_standards}`** — if non-empty, check the skill against each directive (`skill:`, `file:`, or plain text) and fold any miss into the findings as a conformance finding.
- **`{workflow.evals_required}`** — if set, confirm the skill has the required evals (`"baseline"` or `"any"`); if not, add a high-severity finding.

## Author the synthesis layer

Merge the lens returns into one findings list, keeping each finding's `id`. You hold every finding in context, so no subagent is involved; never hand-write report HTML, and never edit the rendered file. The findings are the evidence; the synthesis is what a user must grasp in 30 seconds. All synthesis fields are yours to write:

- `verdict` — one line naming the overall state and the one or two findings that matter most.
- `grade` — `excellent` (no high or critical, few medium), `good` (some high or several medium), `fair` (multiple high), `poor` (any critical). Lowercase.
- `summary` — 2-3 sentences: the skill's primary strength and primary opportunity. This is the first thing the user reads.
- `themes` — findings clustered by shared root cause, not by file. Ask: "if I fixed X, how many findings across lenses would that resolve?" 3-5 themes; findings that fit no theme stay ungrouped in `findings` only. Each theme's `action` is one coherent fix instruction for the whole cluster, and `finding_ids` lists the constituent findings so the report can show them under the theme.
- `strengths` — what works and must be preserved, so a fix pass does not flatten it.
- `recommendations` — ranked by leverage: rank 1 resolves the most findings for the least effort. `resolves` lists the finding ids it would clear.

## Schema (schema_version 2)

`findings.json` is one object:

```json
{
  "schema_version": 2,
  "subject": "<skill path analyzed>",
  "generated": "<ISO date>",
  "verdict": "<one-line overall assessment>",
  "grade": "excellent | good | fair | poor",
  "summary": "<2-3 sentence narrative>",
  "standards": {
    "canon": "<absolute path to this builder's references/prompt-quality-canon.md>",
    "principles": "<absolute path to this builder's references/skill-quality-principles.md>",
    "scripts": "<absolute path to this builder's references/script-standards.md>"
  },
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
  "experience": {
    "journeys": [{ "name": "", "steps": "" }],
    "headless": "<one line on the skill's headless story>"
  },
  "findings": ["<every lens finding unchanged, per references/lens-contract.md>"]
}
```

Rules:

- `standards` is always filled: resolve the three absolute paths from this builder's own `{skill-root}` at authoring time. The shell prepends them to every copied fix prompt, so the session that applies a fix holds the same bar that produced the findings.
- `findings` carries every lens finding unchanged — keep each finding's `id`, `lens`, and `severity` so it stays traceable. Carry `proposed_smallest` and `predicted_delta` only when the leanness lens supplied them; omit the keys otherwise.
- Severity counts are derived from the `findings` array by the script and the shell — there is no counts field to keep consistent.
- `grade`, `summary`, `themes`, `strengths`, `recommendations`, and `experience` are optional: omit a key entirely rather than writing an empty placeholder. A clean pass is a real report — empty `findings`, a grade that reflects it, and a verdict saying the lenses passed.
- Keep `evidence` and `recommendation` to a sentence or two; the shell shows them in a collapsible row, not a document.

## Write and render

Write the object to `{run-folder}/findings.json` and render:

```bash
uv run scripts/render_report.py {run-folder}/findings.json --shell assets/report-shell.html -o {run-folder}/skill-analysis-report.html --md {run-folder}/skill-analysis-report.md
```

If the script refuses, fix `findings.json` and re-run; never hand-edit the HTML. Open the HTML report for the user — it is the deliverable of Analyze; do not replace it with a chat summary of the findings. The markdown twin is the archival artifact of the same data.

The shell fails loud: a malformed island shows the parse-error banner, an unfilled shell shows a placeholder banner, and an empty findings array with a real subject renders an explicit no-findings panel — never a blank page and never fabricated findings.

## Record the run

Append one memlog event carrying the grade (init the memlog first if `{target-skill-path}/.memlog.md` does not exist):

```bash
uv run {project-root}/_bmad/scripts/memlog.py append --path {target-skill-path}/.memlog.md --type event --text "analyze: grade <grade>, <c> critical / <h> high / <m> medium / <l> low, report .analysis/<timestamp>/skill-analysis-report.html"
```

## Present

**IF `{headless_mode}=true`:** emit

```json
{
  "headless_mode": true,
  "status": "complete",
  "skill": "{target-skill-path}",
  "grade": "excellent | good | fair | poor",
  "html_report": "{target-skill-path}/.analysis/<timestamp>/skill-analysis-report.html",
  "md_report": "{target-skill-path}/.analysis/<timestamp>/skill-analysis-report.md",
  "memlog": "{target-skill-path}/.memlog.md",
  "counts": { "critical": 0, "high": 0, "medium": 0, "low": 0 }
}
```

**IF interactive:** present the grade, the one-line verdict, the severity tally, and the top themes. Point to the HTML report path, say it opened, and offer to walk through findings, apply a fix, or route a leanness finding's `proposed_smallest` to a variant eval.
