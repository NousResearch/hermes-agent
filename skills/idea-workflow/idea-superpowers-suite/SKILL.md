---
name: idea-superpowers-suite
description: "Use when running the full idea workflow: capture a rough idea, expand it into design/UI/implementation docs, research similar products, and generate build-ready Markdown artifacts."
version: 1.2.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [idea-workflow, note-taking, product-design, ui-design, implementation, research, superpowers]
    related_skills: [idea-to-design-doc, idea-to-ui-design-brief, idea-to-implementation-doc, goal-prompt-pack, writing-plans]
---
# Idea Superpowers Suite

This skill is a superpowers-style umbrella workflow for idea development.

It coordinates the focused stages:
- `idea-to-design-doc` for product/design thinking
- `idea-to-ui-design-brief` for optional Full-mode UI direction and image-generation concepts
- `idea-to-implementation-doc` for build-ready implementation planning

Use this skill when the user wants a bigger, multi-step workflow that feels like a reusable system rather than a one-off prompt.

## Purpose

Turn a rough thought into a structured chain of artifacts:
1. Idea capture — create or update a local idea note from the user's high-level concept.
2. Clarifying interview — capture the rough plan, philosophy, target experience, and constraints.
3. Design doc — flesh the idea into a full product/UX/technical design document.
4. Optional UI design brief — in Full mode only, define screens, layout, visual direction, states, and optional image-generation concept prompts.
5. Research pass — compare similar products and identify what to emulate or avoid.
6. Implementation spec — translate the design into practical build strategy.
7. Build handoff — produce one Markdown file with tasks, prompts, tests, acceptance criteria, and Superpowers handoff guidance.
8. Optional Goal Prompt Pack — use standalone `goal-prompt-pack` when the user wants the video-style spec → UI mockups → final autonomous goal prompt workflow. This option can create the UI spec, generate or prepare actual mockups, translate them into written visual requirements, and produce a copy-paste-ready Codex Goal Mode / Hermes goal prompt. It does not require Superpowers and does not execute the build unless the user explicitly asks.
9. Spec review — judge readiness before execution.

## Design Philosophy

This workflow should:
- capture fast;
- clarify deeply;
- research before building;
- separate product thinking from engineering thinking;
- separate UI direction from implementation details when interface quality matters;
- save every stage as its own Markdown artifact;
- remain skippable and resilient when the user wants speed.

The UI brief is optional. It improves interface-heavy projects, but it must not make Lite mode heavier or break existing design-doc → implementation-doc flow.

## Mode Selection

At the start, choose one of two modes unless the user explicitly picks one:

- Lite mode — for sketches, small utilities, early thoughts, and ideas the user wants captured quickly.
- Full mode — for serious app/product ideas that may later be built by an AI coding agent or Superpowers.

Default to Lite mode when the idea is vague or exploratory. Default to Full mode when the user asks for a product spec, implementation plan, build handoff, research pass, UI/mockup direction, or agent-ready artifact.

## Default File Layout

Use local Markdown files first. If the user later wants Obsidian export, treat that as a separate step.

### Lite mode layout

```text
ideas/<idea-slug>.md
ideas/index.md
```

Optional only if requested:

```text
ideas/<idea-slug>.implementation.md
ideas/<idea-slug>.build-prompt.md
ideas/<idea-slug>.ui-brief.md
```

### Full mode layout with UI stage

```text
ideas/<idea-slug>/
  README.md
  00-idea-capture.md
  01-design-doc.md
  02-ui-design-brief.md
  03-implementation-spec.md
  04-agent-build-handoff.md
  05-spec-review.md
```

Optional Goal Prompt Pack add-on:

```text
ideas/<idea-slug>/
  05-goal-prompt-pack.md
  06-spec-review.md
```

When adding the Goal Prompt Pack, move or create spec review after the prompt pack so it can judge the actual goal prompt readiness.

### Full mode fallback layout when UI stage is skipped

If the UI stage is skipped, preserve the previous simpler numbering:

```text
ideas/<idea-slug>/
  README.md
  00-idea-capture.md
  01-design-doc.md
  02-implementation-spec.md
  03-agent-build-handoff.md
  04-spec-review.md
```

Optional Goal Prompt Pack add-on when UI stage is skipped:

```text
ideas/<idea-slug>/
  04-goal-prompt-pack.md
  05-spec-review.md
```

Use whichever numbering is already established in a package. Keep the README artifact map consistent.

## Workflow Stages

### Stage 1: Capture

When the user says they have an idea, create a working note with:
- title;
- short summary;
- rough problem statement;
- initial bullet points;
- mode: Lite or Full.

If needed, ask for a name, but prefer to proceed with a temporary title and refine later.

### Stage 2: Interview

Ask one question at a time. Use `references/interview-question-bank.md` for stronger prompts.

In Lite mode, ask only enough to capture the idea clearly, then draft.

In Full mode, cover:
- target user and problem;
- what the app should do;
- desired behavior and product experience;
- scope and non-goals;
- main screens or sections;
- what should be visible at a glance;
- where users take action;
- details that can be hidden until needed;
- desired visual taste: minimal, playful, professional, dense, dashboard-like, calm, command-center, etc.;
- empty, loading, error, success, first-run, and permission states where relevant;
- data, integrations, and platform constraints;
- data location and hosting preference;
- platform targets: web, desktop, mobile web, iOS, Android, etc.;
- topology: one app or multiple pieces such as desktop app + hosted web app + API + worker;
- authentication, secrets, API keys, public/private sharing, and client/server boundaries;
- what would make the result feel excellent, not merely functional.

Do not push into stack decisions too early unless the user has made a technical constraint explicit. When technical planning starts, recommend practical defaults first, then let the user accept or change them.

### Stage 3: Design Doc

Write `01-design-doc.md` to capture product direction and enough technical shape to support planning.

Suggested sections:
- one-line summary;
- problem / purpose;
- product philosophy;
- target user;
- core concept;
- desired behavior;
- key features;
- layout / information architecture;
- UX notes;
- technical shape;
- data / integrations / platform needs;
- hosting / data location / deployment preference;
- platform targets;
- recommended technical defaults and accepted/changed decisions;
- non-goals;
- open questions;
- next steps.

### Stage 4: Optional UI Design Brief

This stage is Full-mode only by default. Do not add it to Lite mode unless the user explicitly asks for UI, mockups, generated images, or visual design direction.

Load `idea-to-ui-design-brief` when this stage runs.

Create `02-ui-design-brief.md` when interface quality matters. It should define:
- purpose of the UI;
- product feel and design principles;
- primary screens/surfaces;
- screen-by-screen notes;
- layout and information architecture;
- component inventory;
- key user flows;
- states to design: empty, loading, error, success, first-run, permissions;
- visual direction;
- accessibility and responsiveness;
- content/copy notes;
- optional image-generation concepts;
- selected direction;
- handoff notes for implementation.

#### Optional image-generation pass

Image generation is concept exploration, not the implementation source of truth.

Use it when:
- the user wants possible UI designs;
- taste is ambiguous;
- a dashboard/app/landing page needs visual exploration before code;
- comparing 2-3 directions would help.

Default concepts:
1. Conservative — familiar, low-risk, easy to implement.
2. Strong-fit — best interpretation of the product brief.
3. Divergent — more distinctive, useful for taste discovery.

If image generation is unavailable or skipped, write the prompts and continue text-only. If images are generated, use vision/manual review to translate them back into written requirements, component notes, and acceptance criteria. Do not ask Superpowers to copy an image blindly.

### Stage 5: Research Pass

For the specific idea, look for similar products and note:
- what already exists;
- what is common / commodity;
- what is different;
- what should be avoided;
- where the idea fits.

Keep this practical and bounded. Research should strengthen the spec, not become an endless market report.

### Stage 6: Implementation Thinking

Translate the idea into `03-implementation-spec.md` if the UI brief exists, otherwise `02-implementation-spec.md`.

Include:
- major system pieces;
- data needs;
- recommended database/storage choice and why;
- hosting/deployment target;
- platform target decisions;
- app topology decisions;
- frontend structure and how it respects the UI design brief if present;
- backend/service needs;
- recommended technical stack defaults;
- security/secrets model;
- integration points;
- workflow/milestones;
- risks and tradeoffs;
- executable build tasks;
- testing and verification plan;
- acceptance criteria / done means;
- a build sequence a developer or AI coding agent could follow.

Use recommend-then-confirm for technical choices:
1. infer sensible defaults from product constraints;
2. present a concise recommendation table;
3. ask whether the user accepts or changes items;
4. record accepted defaults and explicit overrides;
5. if unsure, proceed with defaults as assumptions.

### Stage 7: Final Agent Build Handoff

Create or update `README.md` using `templates/idea-package-readme-template.md`.

Create the final handoff as:
- `04-agent-build-handoff.md` if the UI brief exists;
- `03-agent-build-handoff.md` if the UI brief was skipped.

Load `idea-to-implementation-doc` when creating the handoff because the canonical handoff template lives there.

The handoff must include:
- mission;
- product vision;
- non-negotiable requirements;
- UI/design direction or explicit UI assumptions;
- out of scope;
- technical architecture;
- implementation phases;
- build tasks;
- testing requirements;
- verification commands/checks;
- acceptance criteria;
- done means checklist;
- prompt for the build agent;
- explicit Superpowers handoff instructions.

### Stage 8: Optional Goal Prompt Pack

Run this stage when the user asks for a goal prompt, Codex Goal Mode prompt, autonomous build prompt, or the video-style workflow where the agent first creates a spec, then UI mockups, then the final goal prompt.

Load `goal-prompt-pack`. This is a standalone skill and must remain usable without the idea workflow or Superpowers. Inside the idea workflow, treat it as an optional packaging stage that can consume existing design/UI/implementation artifacts, fill gaps, generate or prepare actual UI mockups, and produce the final copy-paste-ready goal prompt.

Use this stage to create a file such as:

- `05-goal-prompt-pack.md` if the UI brief exists and the previous handoff is `04-agent-build-handoff.md`.
- `04-goal-prompt-pack.md` if the UI brief was skipped and the previous handoff is `03-agent-build-handoff.md`.

The Goal Prompt Pack should include:
- product spec summary;
- UI spec or UI assumptions;
- mockup generation prompts;
- generated/selected mockup references when available;
- mockup-to-written-requirement translation;
- final Codex Goal Mode prompt, preferably starting with `/goal` when Codex is the target;
- Hermes/generic variant notes when useful;
- acceptance criteria;
- blocker and stop rules;
- final report format.

Do not execute the goal prompt in this stage unless the user explicitly asks to start the build.

### Stage 9: Spec Review / Readiness Gate

Create:
- `06-spec-review.md` if the UI brief and Goal Prompt Pack exist;
- `05-spec-review.md` if the UI brief exists and Goal Prompt Pack is skipped;
- `05-spec-review.md` if the UI brief is skipped but Goal Prompt Pack exists;
- `04-spec-review.md` if both UI brief and Goal Prompt Pack are skipped.

Review whether:
- the product goal is clear;
- UI/design expectations are clear enough for a fresh agent to avoid generic output;
- requirements are testable;
- unresolved product decisions remain;
- unresolved technical decisions remain;
- acceptance criteria are concrete;
- done means is specific;
- a fresh agent could build from the file without asking obvious questions;
- testing and verification requirements are included;
- non-goals prevent scope creep.

Verdicts:
- PASS — ready to feed into Superpowers.
- PASS WITH CHANGES — mostly ready; patch listed issues first.
- FAIL — too ambiguous or incomplete.

### Stage 10: Review / Plan Review

If the user wants a critique pass, review artifacts separately:
- design doc stays product-focused;
- UI brief gives concrete screen/style/state guidance without prescribing unnecessary code;
- implementation doc stays build-focused;
- final handoff is agent-ready;
- spec review is honest about gaps;
- package summary does not duplicate all focused docs.

## Progression Rules and Override Phrase

### Lite mode rules

- Capture quickly; do not force research, UI brief, or implementation planning.
- Ask at most 3-5 clarifying questions before drafting unless the user wants more.
- Do not create `02-ui-design-brief.md`, implementation spec, build handoff, or spec review unless the user upgrades to Full mode or specifically asks for UI/mockup direction.
- If the user says the idea is just a note, keep it as a note.

### Full mode rules

- Do not move from capture to design until target user, problem, and core behavior are clear enough to summarize.
- Do not move from design to implementation if open product or UI questions would change architecture, MVP scope, or core interface.
- Do not create final build handoff until implementation phases, testing requirements, acceptance criteria, and done means are present.
- Do not mark spec review PASS if verification checks are missing, major product/technical decisions remain, or UI expectations are too vague for an interface-heavy app.
- If the UI stage is skipped, carry UI assumptions forward instead of blocking.

### User override phrase

The user can force progression with this exact phrase:

> GREENLIGHT NEXT STAGE

When the user says it, move to the next stage. Briefly note the risk, carry unresolved items under Open Questions / Assumptions, and continue. This does not allow unsafe actions or credential exposure.

## Operating Rules

- Keep the workflow modular.
- Prefer separate files for separate stages.
- Save progress as Markdown.
- Patch the working idea note when the user confirms product, architecture, or UI decisions.
- Keep reusable skill content public-safe; do not embed private product plans or credentials.
- If the user says stop, that's enough, enough for now, done, draft it, or write it up, stop questioning and draft immediately.
- If the user wants a lighter experience, use Lite mode.
- If a stage is skipped, continue with the previous workflow and record assumptions.

## Cost-Aware Worker Delegation

Default to GPT/orchestrator time for judgment, synthesis, user interaction, and quality gates; use cheaper bounded workers for the long middle of the workflow when delegation is available.

Good tasks to delegate to cheaper DeepSeek/Gemini/macdaddy-style workers:
- bounded public research on comparable products, patterns, constraints, and risks;
- first-pass design doc drafts from captured notes;
- UI brief variants, visual direction alternatives, and screen/state inventories;
- first-pass implementation specs, milestone breakdowns, and test-plan drafts;
- critique/spec-review passes that identify ambiguity, missing acceptance criteria, and scope risks;
- mechanical Markdown cleanup, README artifact maps, and consistency checks.

Keep GPT/orchestrator responsible for:
- interpreting the user's actual intent;
- asking clarifying questions;
- deciding Lite vs Full mode and stage progression;
- resolving product/architecture tradeoffs;
- merging worker output into one coherent artifact set;
- final build handoff quality and readiness verdicts.

Delegation rules:
1. Give each worker a narrow prompt, the current artifact text or summarized source notes, the target filename/stage, and a compact handoff contract.
2. For cost-minimized build-readiness reviews, prefer a single bounded `architect-grunt` background run for the long middle: read source idea, draft/update implementation spec, final agent handoff, spec review, README/status map, and a compact worker report in a workspace folder. The orchestrator should only do setup, occasional stall checks if needed, and the final quality gate.
3. Ask workers to report: what they did, sources/files used, key findings, recommended changes, risks/assumptions, blockers, verification performed, and next action.
4. Do not expose private notes, credentials, customer data, or sensitive repo/log content to public web tools or external workers without explicit user approval.
5. Treat worker drafts as input, not truth. The orchestrator must review, correct, and synthesize before saving or presenting final artifacts.
6. Delegation is optional for Lite mode; use it only when it clearly saves cost/time or improves quality. In Full mode, prefer delegation for research, drafts, critiques, and mechanical checks.

### Grunt-offloaded readiness review pattern

When the user asks to use “almost 0 tokens,” “offload the whole process to grunt,” or similar, do not narrate a plan only. Start a background worker immediately if the source artifact can be found. Use a workspace such as `~/Hermes/FleetWorkspace/<idea-slug>-review/`, pass the source path, require no source-file mutation unless explicitly approved, and ask for these outputs: `README.md`, implementation spec, agent build handoff, spec review verdict, and `grunt-report.md`. Finalize by reading the worker report/artifacts and issuing the orchestrator's own PASS / PASS WITH CHANGES / FAIL decision before telling the user it is ready for Superpowers.

For existing-app-vs-idea reviews, use the same pattern but make the worker explicitly read-only against the app source and frame the output around improving the existing app rather than greenfield design. If the app is on a worker machine, first confirm candidate paths, then launch grunt with both the app source path and any related idea/docs path. Useful outputs are: `01-app-review.md` (current state), `02-implementation-spec.md` (improvement plan), `03-agent-build-handoff.md` (next build pass), `04-spec-review.md` (readiness gate), and `grunt-report.md`.

If the spec review returns PASS WITH CHANGES due only to small clarifications, run a narrow second grunt pass instead of resolving from memory. Ask it to create `05-clarifications.md` plus a compact `clarification-report.md` with recommended decisions, rationale, exact patch wording for the handoff, verdict impact, and residual risks. Then the orchestrator should patch the handoff/spec-review docs and update the verdict to PASS only after verifying the blockers are actually incorporated.

For existing-app reviews, use `references/grunt-offloaded-app-review.md`: require the worker to maintain `status.txt`, run the worker in the background with `grunt-run.log`, optionally create a temporary no-agent cron watcher for Telegram/origin progress updates, remove the watcher when complete, and spot-check key source claims before presenting the final verdict.

## Relationship to Focused Skills

- Use `idea-to-design-doc` when the user only wants the design stage.
- Use `idea-to-ui-design-brief` when the user wants UI direction, generated UI concepts, screen-by-screen mockup notes, or post-build UI redesign.
- Use `idea-to-implementation-doc` when the user only wants implementation planning.
- Use `goal-prompt-pack` when the user wants a standalone or plug-in package that creates the product/UI spec, UI mockup prompts or actual mockups, and the final autonomous build goal prompt for Codex Goal Mode, Hermes, or a generic coding agent. This skill does not require the idea workflow or Superpowers.
- Use `idea-superpowers-suite` when the user wants the full system.

## Relationship to Superpowers

The idea workflow is the front-end product/spec pipeline. Superpowers is the execution discipline pipeline.

Expected handoff:
1. `idea-superpowers-suite` captures the idea and asks clarifying questions.
2. `idea-to-design-doc` produces the design/spec document.
3. `idea-to-ui-design-brief` optionally produces UI direction and image-generation concept prompts.
4. `idea-to-implementation-doc` produces the build-ready Markdown handoff.
5. Superpowers validates, plans, implements, tests, reviews, and verifies.

The idea workflow should create high-quality input for Superpowers, not replace Superpowers execution.

## Reference Files

- `references/interview-question-bank.md` — reusable question bank for Lite and Full interviews.
- `references/example-cli-tool-build-handoff.md` — generic example final handoff for a small developer CLI tool.
- `references/example-saas-web-app-build-handoff.md` — generic example final handoff for a small SaaS/web app.
- `references/example-automation-script-build-handoff.md` — generic example final handoff for a practical automation workflow.

## Templates

- `idea-to-implementation-doc/templates/agent-build-handoff-template.md` — required structure for final agent/Superpowers build handoff documents.
- `templates/idea-package-readme-template.md` — README/status index for Full mode idea folders.

## Common Pitfalls

1. Making the UI stage mandatory. It is optional and Full-mode only by default.
2. Letting generated images become the source of truth. Translate images into written UI requirements.
3. Renumbering existing packages inconsistently. Preserve established numbering and update README maps.
4. Overloading Lite mode. Keep Lite fast unless the user asks for UI or Full mode.
5. Producing generic UI slop. Use product-specific screens, states, and actions.
6. Moving into code before product and UI decisions that affect scope are clear.

## Verification Checklist

- [ ] Mode was chosen or inferred correctly.
- [ ] Lite mode stayed lightweight unless explicitly upgraded.
- [ ] Full mode artifacts are separated by stage.
- [ ] UI brief exists only when useful/requested, or UI assumptions are carried forward.
- [ ] Optional image-generation prompts are treated as concepts, not implementation truth.
- [ ] README artifact map matches actual filenames and numbering.
- [ ] Final handoff includes UI direction or explicit UI assumptions.
- [ ] Spec review honestly reports unresolved product, UI, and technical gaps.
