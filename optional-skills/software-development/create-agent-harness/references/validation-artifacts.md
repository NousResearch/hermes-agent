# Validation Artifacts

Reference for `create-agent-harness` Phase 4 — anti-patterns, quality checklist, final report and handoff.

### Anti-patterns

| Anti-pattern | Fix |
| --- | --- |
| Two parallel structures, `.agents/` plus `.claude/` | Migrate and **remove** the legacy one |
| `AGENTS.md` or `DEVIN.md` kept as source of truth | Merge into `CLAUDE.md` and remove, or make `AGENTS.md` a thin reference |
| `.claudeignore` kept as protection | Convert to `permissions.deny`; the CLI does not read it |
| `applyTo` in rules | Convert to `paths:` |
| Guardrails only in prompts | Add computational controls in `settings.json` and hooks |
| Speculative hooks or `deny` entries | Remove — generate only what the repo evidences |
| Branch protection implemented as a local hook | Use server-side branch protection plus `global-rules.md` |
| Unlimited context | Apply the token budget and the compaction ladder |
| A context source with no artifact | Map it in the inventory or declare it unnecessary |
| Knowledge entry with no cited source | Add the source path — ungrounded knowledge causes hallucination |
| Decisions taken with no auditable trail | Append to `## Decisions` with rationale and discarded alternatives |
| Everything loaded always-on | Reclassify as on-demand and reference it from `CLAUDE.md` |
| No verification loop | Mandatory lint, test, CI |
| Stateless sessions | `memory.md` plus dated long-term files |
| Memory as an unbounded log | Promote durable entries and reset `memory.md` |
| Two files holding the same state | One concept, one location |
| `.claude/MEMORY.md` stores state or history | Move records to `.claude/memory/{YYYYMMDD}-memory.md` and keep `MEMORY.md` as docs only |
| Duplicated info across files | Reference, do not copy |
| Content invented without evidence | Replace with `TODO:` and ask |

### Quality checklist

- [ ] `CLAUDE.md` within 1000 lines, repository-specific, no generic content
- [ ] `AGENTS.md` present as thin reference or symlink to `CLAUDE.md` for non-Claude platforms
- [ ] `.claude/settings.json` is valid JSON with `permissions` and `hooks`; `deny` empty unless the project requires entries
- [ ] `.claude/rules/global-rules.md` present, always-on, no `paths:`
- [ ] `CLAUDE.md` explicitly connects every always-on artifact: native (`CLAUDE.md`, `.claude/rules/global-rules.md`) and read-ritual (`.claude/memory/memory.md`, `.claude/CONTEXT.md`, `.claude/RULES.md`)
- [ ] Domain rules use `paths:`, never `applyTo`
- [ ] Skills carry the tripartite description: What / When / Do NOT
- [ ] Five sub-agents present (`engineer`, `plan`, `review`, `test`, `architecture`), stack-specialized, `name:` matching the file name
- [ ] Each sub-agent declares a verification loop
- [ ] `plan` sub-agent uses the SPEC SDD template and writes to `.specs/` before implementation
- [ ] `.claude/CONTEXT.md`, `.claude/RULES.md`, `.claude/MEMORY.md`, `.claude/TOOLS.md`, `.claude/WORKFLOWS.md` and `.claude/README.md` generated
- [ ] `.claude/MEMORY.md` is on-demand protocol documentation and stores no state or history
- [ ] `.claude/TOOLS.md`, `.claude/WORKFLOWS.md` and `.claude/README.md` are classified and referenced as on-demand
- [ ] Context Engineering section complete — see the checklist in 3.21
- [ ] Every context source mapped to an artifact or explicitly declared unnecessary
- [ ] Governance controls in place for data exposure, hallucination and traceability
- [ ] Memory protocol wired into `CLAUDE.md`, `global-rules.md` and the Agent Loop
- [ ] Hooks executable and registered in `settings.json`, if generated
- [ ] `.devin/config.json` with `read_config_from: { claude: true }` when Devin CLI is targeted
- [ ] Migration validated: 4.1, 4.2 and 4.3 all pass
- [ ] Every artifact backed by repository evidence

### Final report

```text
## Claude Code Harness Artifacts

### Root
- [ ] CLAUDE.md (single source of truth, within 1000 lines)
- [ ] AGENTS.md (thin reference or symlink to CLAUDE.md)

### .specs/ (one per feature)
- [ ] SPEC-{YYYYMMDD}-{feature}.md — SDD spec, approved before implementation

### .claude/
- [ ] settings.json — permissions (allow/ask/deny) + hooks
- [ ] rules/global-rules.md (mandatory, always-on)
- [ ] rules/{domain}.md (with paths:)
- [ ] skills/{slug}/SKILL.md (one per stack domain)
- [ ] agents/engineer.md (`orchestrator` runner), agents/plan.md (SPEC SDD writer), agents/review.md (`qa-analyst` + `code-review-and-quality`), agents/test.md (`quality-test-implementation`), agents/architecture.md (`architecture` skill) — all mandatory
- [ ] memory/memory.md (short-term)
- [ ] memory/{YYYYMMDD}-memory.md (long-term)
- [ ] commands/{slug}.md (if applicable)
- [ ] hooks/{slug}.sh (if applicable)
- [ ] knowledge/{slug}.md (if applicable)
- [ ] CONTEXT.md (context engineering — always-on via CLAUDE.md reference)
- [ ] RULES.md (guardrails — always-on via CLAUDE.md reference)
- [ ] MEMORY.md (protocol documentation — on-demand, no state/history)
- [ ] TOOLS.md (tools and MCP — on-demand reference)
- [ ] WORKFLOWS.md (automation — on-demand reference)
- [ ] README.md (harness infrastructure — on-demand reference)

### docs/ (recommended)
- [ ] README.md, technologies.md, packages.md, plugins.md, features.md, api.md

### .devin/ (only with Devin CLI integration)
- [ ] config.json — read_config_from: { claude: true }

### Platform-specific (generate only for targeted platforms)
- [ ] .opencode/ (skills, hooks, config — for OpenCode)
- [ ] .cursor/ (skills, hooks, config — for Cursor)
- [ ] .gemini/ (skills, hooks, config — for Gemini CLI / Antigravity IDE)
- [ ] .gemini/antigravity-cli/ (skills, hooks — for Antigravity CLI / agy)

## Legacy migrated and removed
- [ ] {origin} → {destination in .claude/} (removed)

## Validation
- [ ] 4.1 migration — every check returned empty
- [ ] 4.2 structure — every mandatory file present, settings.json valid, no MISSING in the six `.claude/*.md` reference files (CONTEXT.md, RULES.md, MEMORY.md, TOOLS.md, WORKFLOWS.md, README.md), `plan` sub-agent wired to `.specs/`
- [ ] 4.3 memory — both tiers present and wired

## Open TODOs (human decision required)
- [ ] {item} — reason
```

### Handoff

1. Seed `.claude/memory/memory.md` with the current state: branch, last commit, test baseline, active work item `none`, next action.
2. Seed `.claude/memory/{YYYYMMDD}-memory.md` with a `## Decisions` entry recording that the harness was created or migrated, which agent loop was chosen and why, plus any naming-collision decisions from Phase 2.
3. Commit on the dedicated branch with a descriptive message and propose the pull request to `main`. **Never push directly to a protected branch.**
4. Report the open TODOs one final time. Stop.

