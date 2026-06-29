# Skill Quality Principles

BMad-specific knowledge for skills the builder produces. Loaded at build time so the author works to the bar from the start, and at analysis time so the lenses verify against the same bar. The universal bar — the destination shape, the tests, the two-version comparison, the reader, the habit — lives in `references/prompt-quality-canon.md`; load it alongside this file, apply it, and never restate it. What follows is only what the bare model would not know: BMad conventions, wiring, and the patterns and failure shapes BMad has paid for.

## Naming
- Skill name = folder name (kebab-case)
- Module skill: `{module-code}-{name}` (e.g. `bmm-create-prd`, `cis-brainstorm`)
- Standalone: `{name}`
- The `bmad-` prefix is reserved for official BMad creations

## Description format
Two parts: `[5-8 word summary]. [Use when user says 'specific phrase' or 'specific phrase'.]`

Quote the trigger phrases. Default to conservative (explicit) triggering, since most BMad skills are explicitly invoked. Organic triggering is reserved for skills that should activate on context (e.g. "Trigger when code imports the anthropic SDK").

Bad: `Helps with PRDs and product requirements.` It is too vague and will hijack unrelated conversations.

## Path conventions
All file references in a skill use bare paths from the skill root. The canonical Resolution rules block, stamped into any SKILL.md that references multiple internal files:

```
## Resolution rules
- Bare paths and `{skill-root}` (e.g. `references/press-release.md`) resolve from this skill's installed directory.
- `{project-root}` → the project working directory.
- `{skill-name}` → the skill directory's basename.
```

Additional rules:
- Forward slashes only (cross-platform).
- Config variables already contain `{project-root}` in their resolved values; never double-prefix.
- `references/` is for prompt content carved out of SKILL.md. `assets/` is for templates and other static content the workflow loads. `scripts/` is for deterministic code. Never put workflow content directly at skill root.

## Customization (customize.toml)
customize.toml is the only customization mechanism — no installer questions, no module.yaml authoring, no boolean-toggle config, no settings concept inside a built skill. The full spec (the ask, universal defaults, offered-when-relevant points, three-layer merge rules, forbidden mechanisms) lives in `references/customize-toml-guide.md`. The wiring rule worth carrying everywhere: SKILL.md must read declared values as `{workflow.<name>}` — a hardcoded path beside a declared scalar silently no-ops the override.

## Intelligence placement
Scripts handle plumbing (fetch, parse, validate, count, transform); prompts handle judgment (interpret, classify, decide). Crossing the boundary in either direction is a defect: a script using regex to decide what content means leaks intelligence into the script, and a prompt counting items or validating structure leaks determinism into the LLM. The determinism test, the signal-verb scan, and the pre-pass JSON pattern live in `references/script-opportunities-reference.md`.

## Workflows: inline first, carve by relevance
Default: write the entire workflow as named sections in SKILL.md (`## Discovery`, `## Constraints`, `## Finalize`, and so on). A multi-stage coaching workflow can live in one SKILL.md. Carving follows the canon's test: carve what only some branches need or what pushes SKILL.md past its token tier, keep a routing map in SKILL.md, and leave inline what is too small to repay the indirection. When you carve:
- **Descriptive filenames.** `references/press-release.md`, `references/customer-faq.md`, never `01-press-release.md`; the carve-out is a section, not a "step," and SKILL.md routes by name.
- Each carved-out file works standalone, since context compaction can drop SKILL.md mid-flow. No "as described in the overview."
- Progression conditions, where they exist, must be testable ("when X is captured, route to Y"). "When ready" is vague.
- The file uses `{communication_language}` (and `{document_output_language}` if it produces a doc).
- There are NO exit hooks in the system. Don't add `## On Exit` sections, because they would never run.
- **Gotchas stay in SKILL.md.** A rule whose trigger the model cannot recognize — a soft-delete column that poisons queries, a health endpoint that lies, three names for one ID — never carves to a reference however branch-specific it is, because the model cannot load a file for a situation it does not know it is in. When a user corrects a running skill, the cheapest durable fix is appending that correction as a gotcha line.

## Headless mode
When a skill supports headless invocation, the memlog absorbs every assumption made without the user: intent inference, proposed names, customization defaults, conflict resolutions, lint-fix calls, anything the user would have weighed in on interactively. Append these as typed `assumption` and `decision` entries through `{project-root}/_bmad/scripts/memlog.py` as they happen. The JSON return is the smallest set of paths the caller needs (typically `skill` plus the memlog path, plus the report path for analysis flows); the memlog carries the reasoning. `status` is `complete` or `blocked`; on `blocked`, include a one-line `reason` and still return the memlog path so the caller can read the detail. Without this discipline, headless silently buries its calls and the audit trail breaks on the next session.

## Subagent constraints
- Subagents CANNOT spawn other subagents. Chain through the parent.
- Don't read files in the parent if you can delegate the read; the parent stays lean.
- Subagent prompts must specify the exact return format and an "ONLY return X" constraint, or you get verbose prose back.
- **The implicit-read trap:** language like "review", "acknowledge", or "summarize what you have" causes the parent to read files even when you didn't ask for it. If a later stage delegates document analysis, earlier stages must NOT use that language. Use "note paths for subagent scanning; don't read them now".

## Length guidance
Length is measured in tiktoken tokens through `scripts/count_tokens.py` (`cl100k_base`, with a chars/4 fallback when tiktoken is unavailable). There is no line-count gate anywhere. The canon's tests still apply to every line; budgets are a guardrail, not the goal.

SKILL.md is tiered against two org-configurable thresholds, `{workflow.skill_md_token_desired}` (default 2000) and `{workflow.skill_md_token_budget}` (default 3000). The hard tier sits deliberately under the Agent Skills spec's 5,000-token recommendation, and the budget is a drift guardrail, not the leanness bar — the canon's tests still cut a ceremonial line in a 900-token file:

- **Under desired** — on target; no action.
- **Between desired and budget** — warn the user that SKILL.md is getting heavy and name the section most worth lifting, but do not block.
- **Over budget** — a hard finding. Bring it back under budget through progressive disclosure: lift the largest self-contained section to `references/` or `assets/` and leave a one-line pointer, rather than compressing prose into something the model has to decode. Repeat until under `{workflow.skill_md_token_budget}`.

| File kind | Token budget |
| --- | --- |
| SKILL.md | `{workflow.skill_md_token_desired}` aim / `{workflow.skill_md_token_budget}` hard |
| Multi-branch reference | ~4500 |
| Single-purpose reference | ~9000 |

When any reference file runs past its budget, lift a section the same way.

## Patterns BMad has seen pay off
Institutional names for patterns the LLM won't generate by default:

- **Open-floor opening**: Conversational skills start with an explicit invitation for the user to share everything they have (goals, references, examples, paths to artifacts) before any structured Q&A. The dump replaces most of the question script that would otherwise follow, and the agent then asks only what's missing. The form adapts to the input: a vague request gets "tell me everything", a path or URL gets "what do you want focused on?". It costs almost nothing token-wise and drastically improves the conversational feel.
- **Soft-gate elicitation**: "Anything else, or shall we move on?" at natural transitions. Users always remember one more thing when given a graceful exit.
- **Intent-before-ingestion**: Understand why the user is here before scanning artifacts, because without intent the scanning is noise.
- **Capture-don't-interrupt**: Out-of-scope insights mid-flow get captured silently rather than redirected. Users in flow share their best material unprompted.
- **Dual-output**: Human artifact plus an LLM distillate, when the artifact will feed downstream agents.
- **Parallel review lenses**: Fan out two or three review subagents (skeptic, opportunity-spotter, a contextually-chosen lens) before finalizing a significant artifact.
- **Three-mode architecture**: Guided, Yolo, Headless. Not every skill needs all three, but considering it during design prevents lock-in.
- **Graceful degradation**: Subagent-dependent features fall back to sequential when subagents are unavailable.
- **Plan-validate-execute**: For batch or destructive operations, produce an intermediate plan artifact, validate it against the source of truth with a script whose errors name the fix ("field 'signature_date' not found — available: …"), and only then execute. The validation script is the load-bearing piece, because it lets the model self-correct before anything irreversible runs.
- **Working state across turns**: a multi-turn skill that builds something holds state as a memlog (the decision trail), a structured working artifact (the work-in-progress that transforms into the output), both, or neither. The choice and the full treatment live in `references/working-state-patterns.md`.

## Writing
- One term per concept; pick it and stick to it.
- A default, not a menu: when several tools or approaches would work, name one and demote the alternatives to an escape-hatch clause ("use X; for scanned input use Y"). A list of equal options makes the model spend its turn choosing instead of working.
- Third person in descriptions ("Processes files", not "I help process files").
- Descriptive file names (`form-validation-rules.md`, not `doc2.md`).

## Failure Modes With Body Count

- **Description over-broadens** → Skill hijacks unrelated conversations. Fix: quote trigger phrases.
- **Vague progression conditions** ("when ready") → Stage never advances or advances early. Fix: testable conditions.
- **Stage references SKILL.md** ("as above") → Breaks on compaction. Fix: make stages self-contained.
- **Subagent prompt without explicit return format** → Verbose prose responses. Fix: "Return ONLY {schema}. No other output."
- **Parent reads then delegates analysis** → Context bloat that makes the delegation pointless. Fix: delegate the read.
- **Implicit-read trap** in a stage that precedes subagent delegation → Parent reads everything anyway. Fix: explicit "don't read these now".
- **Boolean toggles in customize.toml** → Author didn't decide what the skill does; the surface becomes a permutation forest. Fix: pick a default and let users fork if they want the other shape.
- **Hardcoded path in SKILL.md while customize.toml declares the scalar** → Override silently does nothing. Fix: SKILL.md must read `{workflow.<name>}`.
- **Identity, communication style, or principles in `[workflow]`** → The workflow wants to be an agent. Fix: point the author at agent-builder and remove it from the workflow surface.
- **Multi-turn producing skill with no working-state strategy** → state lives only in the conversation and dies on compaction or revisit. Fix: choose a memlog or a structured working artifact (`references/working-state-patterns.md`).
- **Working-state strategy buried under ceremony** → a memlog-discipline enumeration or a meta `## Workspace` section pays the pattern's cost without its value. Fix: thread it through the intents at the points that matter; `bmad-product-brief` is the model.
