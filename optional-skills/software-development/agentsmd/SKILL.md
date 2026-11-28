---
name: agentsmd
description: "Distill session lessons into a lean AGENTS.md — collect, distill, organically merge, verify commands, and write only after user approval."
version: 1.5.2
author: "Felix (kelegele/agentsmd-skill)"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [agents-md, project-rules, documentation, workflow, lessons]
---

# Lessons To AGENTS.md

## Goal

Turn one-off lessons into a small set of durable, reusable, executable rules in the project root's `AGENTS.md`. Four iron rules:

1. **Concise**: every rule is one line, imperative, directly executable. No narrative, no explanatory paragraphs.
2. **Compliant**: structure follows the open [agents.md](https://agents.md/) spec (see `references/agents-md-spec.md`) — pure Markdown, standard sections, written for AI coding agents.
3. **No bloat**: `AGENTS.md` is organic consolidation, not an append-only log. Hard cap **120 lines** (excluding blank lines); near the cap you must delete before adding. Content that still has long-term value but fails the root-file admission bar moves to `docs/` with a one-line reference (see Step 4). Rules that lost generality, were superseded, or duplicate others get deleted or merged.
4. **Confirm before writing**: every write — updating `AGENTS.md`, agent memory, progress/status files, moving collected files — must be shown to the user and approved first. Without approval, propose only; touch nothing.

## Workflow

### Step 1 Collect lessons

Extract lessons from this session/task:

- Errors, corrections, and rework the user explicitly pointed out (highest priority)
- Pitfalls actually hit: errors, environment issues, wrong approaches
- User preferences that repeated (2+ occurrences; a single occurrence doesn't count)
- Approaches that were validated to work

Exclude: one-off task details, temporary paths/parameters, obvious common sense, unverified guesses.

### Step 2 Distill rules

Process each lesson in three passes:

1. **Generalize**: raise the specific case into a general rule. "Wrong path alias in tsconfig.json broke the build" → "Run `npm run build` after changing path aliases".
2. **Compress**: one rule per line, format `- do X (reason ≤ half a line if needed)`. Example: `- Commits must pass lint and tests; CI blocks non-compliant commits`.
3. **Classify**: put it in the closest section (see the template's standard and optional sections). Only create a new section when nothing fits, and keep total sections ≤ 9.

**Admission bar**: a rule must pass all three — (a) likely to recur; (b) violating it causes real loss (rework/bugs/wasted time); (c) expressible in one line. Miss any one, discard.

### Step 3 Read current state

- First detect the scenario: **coding** or **work** (a non-code collaboration directory). Later steps follow the matching column in Scenario Adaptation.
- Check whether `AGENTS.md` exists in the project root.
- **Exists**: read it fully and build a list of existing rules (enumerate every one).
- **Missing**: build project understanding yourself, then generate the first `AGENTS.md`:
  1. Scan project structure, README, build/dependency configs (`package.json` / `pyproject.toml` / `Makefile` / `*.csproj` …), and CI configs to identify the stack, package manager, build and test commands.
  2. Combine with this session's lessons and generate the first version per the agents.md spec (skeleton: `assets/AGENTS-template.md`; granularity benchmark: `assets/AGENTS-example.md`). Omit standard/optional sections that don't apply — never leave empty placeholder sections.
  3. **Command verification**: every command written in the first version must be verified — run build/test/lint commands for real; commands impractical to run (deploy, migrations) must be cross-checked against `package.json` scripts / `Makefile` / CI configs for existence and exact spelling. Never write a command that fails or can't be verified.
  4. Apply Step 2's admission bar and concision rules to the first version too. Piling up generic common sense "to look complete" is forbidden.

### Step 4 Organic merge (core)

For each new rule, compare against every existing rule and do exactly one of the following. Unconditional appending is forbidden:

| Situation | Action |
|------|------|
| Equivalent rule exists | Discard the new rule; touch nothing |
| Partially overlaps | Merge into one more precise/more general statement |
| Is a special case of an existing rule | Replace the existing rule with the general version (one general > three special cases) |
| Contradicts an existing rule | Keep the new one (newer practice), delete the old |
| Existing rule clearly outdated (env/tooling changed) | Delete |
| New and passes the admission bar | Append to the matching section |

**Length control**:
- After merging, total length (excluding blank lines) ≤ 120 lines. Over the cap, prune in this order: outdated rules → duplicates/mergeable items → low-generality special cases → entries in the least-triggered sections.
- **Pruning ≠ always deleting**: pruned content that still has long-term reference value but fails the root-file admission bar (background, operational details, low-frequency special cases) moves to `docs/<topic>.md`, leaving a one-line reference in place (e.g. `- Deployment details: [docs/deploy.md](docs/deploy.md)`). Outdated and duplicate content is deleted outright, not migrated.
- Reference lines are scarce by design: keep only when truly useful on demand, at most 1–2 per section, so `docs/` doesn't bloat the root file indirectly.
- Aim for ≤ 8 entries per section; more means the section is accumulating details — converge downward.

### Step 5 Confirm and write

- Present the complete change plan to the user (must include):
  1. Rules to add (verbatim, with target section)
  2. Rules to merge/replace/delete, with reasons
  3. Content to migrate to `docs/`, target files, and the reference line left in the root file (writing `docs/` also requires approval)
  4. Updated total line count and remaining capacity
- Write `AGENTS.md` **only after user approval**, using your file-editing tools; leave all other sections untouched. Never write files before approval.

### Step 6 Sync memory and progress (conditional)

- **Agent memory**: if the current agent has a project memory mechanism (memory files/directories, brain, etc.), sync this session's lessons as project memory entries — follow the agent's own memory format, dedupe before writing.
- **Progress/status files**: if the project tracks progress (TODO.md, CHANGELOG.md, PROGRESS.md, STATUS.md), update the relevant entries for this change.
- **Same hard gate**: list the files to update and a content summary, and get user approval before writing. No memory mechanism or relevant files → skip silently, don't mention it.

### Step 7 Collect docs and data (optional closing)

After `AGENTS.md` is written, check for scattered output files (not yet grouped by topic, dumped at the top level) and group each by nature:

- **Documents** (for humans: MD/HTML/PPT/PDF) → `docs/<topic>/`
- **Data files** (machine-readable: CSV/XLSX/JSON/SQL dumps, exported results, datasets) → `data/<topic>/`

Work-scenario differences are in Scenario Adaptation.

1. **Check references before proposing**: for each candidate file, grep the whole project first (code, scripts, build/CI configs). Referenced files never move — moving breaks things. Files whose nature is unclear (document or data?) get flagged for the user; don't decide for them.
2. **Scan only, touch nothing**: present the list (path, nature, suggested destination) for approval — approve all or pick individually; "keep in place" is always allowed. Exceptions that stay in the root: project config files (`AGENTS.md` / `CLAUDE.md` / `skills-lock.json` …), `README`, `LICENSE`. **Never move, rename, or modify any file before the user approves.**
3. **Execute after approval**: move files → fix every relative-path reference (links in MD/HTML) → grep the whole project to verify no broken links (covering the reference points from step 1) → report the move list and reference fixes.

## Section conventions

Prefer the template's standard sections; semantic mapping:

- Build/environment lessons → `Build & Test Commands` or `Dev Environment Tips`
- Code style, naming, structure → `Code Style`
- Workflow constraints (commits, PRs, branches) → `PR / Workflow Instructions`
- Project-specific recurring pitfalls → `Project-Specific Gotchas`

Optional sections only when the project really has that fixed process; same admission bar applies, scarce by design:

- Build artifacts, env config, deploy commands, CI/CD → `Build & Deployment`
- Secret handling, permission model, security test requirements → `Security Notes`
- Frequent-error triage paths, log locations, debug entry points → `Debugging & Troubleshooting`

**Nesting**: a lesson that only applies to one subdirectory doesn't go in the root file — suggest a local `AGENTS.md` in that subdirectory (agents honor the nearest file when working there); the root keeps globally applicable content only.

## Scenario adaptation (coding / work)

In Step 3, detect the scenario first. Both share one workflow; only three spots differ:

- **Detection**: any code-project marker (package.json / pyproject.toml / Makefile / *.csproj / source dirs / CI config) → coding; only documents, spreadsheets, and reference material → work (report libraries, collaboration directories, etc.).

| Aspect | coding | work |
|------|--------|------|
| Initial scan & verification | Scan build/dependency/CI configs; run each command or cross-check it | Scan README, directory structure, naming conventions; no commands to verify — instead verify referenced paths exist |
| Section semantics | Build & Test / Code Style / PR rules | Data definitions & source freshness, file naming & archiving, delivery & review process; mostly custom sections, still ≤ 9 total |
| Step 7 collection & references | docs→`docs/`, data→`data/`; grep code references | Deliverables→`docs/`, data drafts→`data/`, reference material→`references/`; check MD/HTML relative links and paths inside AGENTS.md |

## Reference resources

- `references/agents-md-spec.md` — key points of the agents.md spec; required reading before creating an AGENTS.md for a project.
- `assets/AGENTS-template.md` — section skeleton reference; prune to the project's reality, never keep empty sections that don't apply.
- `assets/AGENTS-example.md` — the official agents.md example (pnpm/Turborepo monorepo); use it to calibrate the granularity and density of a good finished file.

## About

- Repo & updates: [kelegele/agentsmd-skill](https://github.com/kelegele/agentsmd-skill)
- Author site: [kelegele.com](https://kelegele.com)
