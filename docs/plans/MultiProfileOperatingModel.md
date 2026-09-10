# KENSEI Multi-Profile Operating Model

Owner: Sahil Saghir
Operator: KENSEI
Created: 2026-05-03 22:39 BST
Status: Active v1 operating model, specialist profiles created and smoke-tested
Canonical context: `/home/kensei/repos/KenseiAgent/docs/NorthStar.md`

This document defines how KENSEI should operate as an orchestrator over specialised Hermes profiles using the Hermes Kanban coordination layer. It pauses the need for a custom Command Center until the native Hermes dashboard plus Kanban workflow has been proven or found lacking.

---

## 1. Core decision

KENSEI is the Sahil-facing orchestrator profile, not just another worker.

KENSEI receives Sahil's raw input, clarifies intent, decides whether the work is worth doing, creates the task graph, assigns work to specialist profiles, arbitrates disagreements, and brings decisions back to Sahil.

Specialist profiles are isolated team leads. They own their domain, execute assigned Kanban tasks, spawn short-lived subagents where useful, validate their output, and report back through structured handoffs.

Hermes Kanban is the coordination spine. It handles task routing, parent and child dependencies, comments, shared workspaces, profile lanes, runtime caps, durable state, and dashboard visibility.

Obsidian is the canonical human-readable knowledge layer. Hermes memory is for durable preferences and operational facts, not hidden project documentation.

---

## 2. Operating principles

1. KENSEI routes substantial work through Kanban instead of doing everything directly.
2. Specialist profiles collaborate through Kanban tasks, comments, parent outputs, shared workspaces, and completion metadata, not free-form cross-agent chatter.
3. Profiles are isolated by default, but context is shared deliberately through task bodies, comments, workspace files, and Obsidian links.
4. KENSEI arbitrates disagreements and presents tradeoffs to Sahil. No fake consensus.
5. Specialist memory is tightly scoped to workflow preferences, source preferences, tool habits, and lessons for that domain.
6. Project facts, decisions, and documentation live in Obsidian or repo docs, not hidden inside one profile's memory.
7. Custom Command Center work is paused until Hermes Kanban and dashboard have been exercised end-to-end.
8. Manual approval remains required for destructive actions, external sends, purchases, social posting, and public exposure.

---

## 2.1 Sahil decisions from profile interview

These decisions are now locked for the first implementation pass.

1. Profile IDs stay boring and functional. Aliases/nicknames can be added later, but docs, Kanban assignees, cron prompts, and runbooks use canonical role names.
2. `default` remains KENSEI. Do not create a separate sticky `kensei-orchestrator` profile unless a migration need appears.
3. General Assistant gets sensitive/admin tools only when task-scoped. Mailbox, calendar, booking, and browser/admin access are not default ambient powers.
4. Research Lead owns all research: AI/news/agents, social media research, Reddit/X scraping, web scraping, technical research, product research, validation, and general/semi-specialised investigations.
5. Coding Lead may edit files and run tests. Commits only when explicitly asked. No push without approval.
6. Content Lead drafts only for now. Scheduling/posting is future, explicit approval only.
7. Multi-platform posting still matters. The solution must be free or near-free first, with costs flagged before recommendation.
8. Knowledge Librarian may write to a defined KENSEI workspace inside Obsidian. No git push unless approved or handled by a defined sync job.
9. Ops Lead diagnoses freely and proposes fixes. Restarts, destructive actions, public exposure, and security-sensitive changes require approval.
10. Ops Lead is security plus DevOps: safe configuration, service health, performance, monitoring, backups, maintenance, and operational hardening.
11. Workspace defaults should follow Hermes operational best practice: `scratch` for research/content/admin, `worktree` for code edits, `dir:<path>` only for canonical repo/vault work.
12. First pilot is research -> content -> knowledge.
13. Second pilot is a small content drafting pipeline that repeats the same flow and touches the wider profile model.

---

## 3. Day 1 profile roster

| Profile | Role | Owns | Default model policy | Notes |
|---|---|---|---|---|
| `default` | Sahil-facing orchestrator and digital twin | Intake, clarification, task graph, arbitration, roadmap, final recommendation | Main provider, currently `openai-codex/gpt-5.5`, later likely Kimi or explicit premium model | Keep as KENSEI. No separate orchestrator profile for now. |
| `general-assist` | Secretary/general assistant | Personal admin, mailbox management, shopping, appointment/gym/reservation booking requests, reminders, future voice calls | `ollama-cloud/kimi-k2.6:cloud` | Distinct from KENSEI. KENSEI decides, General Assist executes admin. |
| `research-lead` | Research specialist | Web research, AI/news/agents, social media, Reddit/X scraping, web scraping, technical research, product research, validation, source finding, synthesis | `ollama-cloud/kimi-k2.6:cloud` or larger research model when needed | Own profile from day one to avoid research tool/context bleed. |
| `coding-lead` | Software build lead | Implementation plans, coding execution, tests, reviews, worktrees, subagent coordination | `ollama-cloud/qwen3-coder:480b` | Can spawn coding subagents and review subagents. |
| `content-lead` | Content strategy and drafting lead | LinkedIn/X drafts, brand voice drafts, launch copy, app copy | `ollama-cloud/kimi-k2.6:cloud` | Draft-only at first. Multi-platform scheduling/posting is a later approval-gated phase. |
| `knowledge-librarian` | Documentation and knowledge management | Obsidian notes, decision records, runbooks, session-to-note distillation | `ollama-cloud/kimi-k2.6:cloud` | Writes to vault only when explicitly assigned or approved. |
| `ops-lead` | Security, DevOps, and operations | Cron, gateway, MCP auth, backups, service health, monitoring, security checks, performance, upkeep | `ollama-cloud/kimi-k2.6:cloud` | Diagnoses freely, proposes fixes, blocks on restarts/destructive/security-sensitive actions. |

Not creating for now:

- `pm-lead`, overkill. KENSEI handles PM/product thinking with Sahil.
- `mailbox-lead`, not yet. Mailbox work sits under General Assist plus existing mailbox digest cron/skill unless it becomes heavy and interactive.
- Per-project profiles, not yet. Use Obsidian/project notes and repo context instead.

---

## 3.1 Telegram routing and digest ownership

### Unified Daily Digest (Topic 1)

All morning briefs land in Topic 1 ("Daily Digest") in sequence. Each section is self-contained dark-themed HTML with its emoji header:

| Time | Section | Emoji | Job |
|------|---------|-------|-----|
| 07:00 | System Health | 🖥️ | system-report-daily |
| 07:30 | Today's Calendar | 📅 | calendar-brief-daily |
| 08:00 | Inbox Management | 📬 | mailbox-digest-daily |
| 08:15 | AI/Tech Research | 🔬 | research-digest-daily |

### Profile output channels (Topics 19-23)

| Topic | Purpose | Primary profile | Delivery rule |
|-------|---------|-----------------|---------------|
| 19 | Content Drafts | content-lead | Drafts go to Kanban first. KENSEI previews drafts here for Sahil approval. |
| 20 | Build Log | coding-lead | PR summaries, scaffold progress, repo changes. KENSEI summarises from Kanban runs. |
| 21 | Ops & Alerts | ops-lead | Critical alerts: gateway down, disk >85%, Gmail tokens expired, digest failed 2x. |
| 22 | Research Deep Dives | research-lead | Longer investigations and market scans that warrant standalone reading. |
| 23 | Archive / Low-priority | any | Weekly summaries, routine health OKs, non-urgent logs, old topic catch-all. |

### DM (Direct Message)

Reserved for:
- System report critical flags
- Memory curator output
- Anything KENSEI decides needs immediate attention

### Routing principle

Default: specialist profiles complete Kanban tasks first. KENSEI owns Sahil-facing Telegram delivery.
Exception: cron jobs with explicit `deliver` targets (daily digests, memory curator) route directly.
Profile workers never push to Telegram directly. KENSEI is always the final filter.

---

## 4. Shared knowledge model

Use this split:

| Layer | Purpose | Examples |
|---|---|---|
| KENSEI memory | Durable Sahil preferences and stable environment facts | Communication preferences, approval gates, VPS facts |
| Specialist profile memory | Domain-specific workflow lessons | Research source rules, coding validation habits, content voice pitfalls |
| Hermes sessions | Conversation recall | “What did we decide last time?” |
| Kanban comments and metadata | Active work handoff and audit trail | Blockers, parent outputs, acceptance criteria, validation results |
| Shared workspace | Task files and temporary artifacts | Specs, draft files, build outputs, review notes |
| Obsidian vault | Canonical human-readable knowledge | Decisions, project notes, runbooks, strategy notes |
| Repo docs | Source-controlled technical docs | Architecture, build plans, app-specific AGENTS.md |

Obsidian stays the main knowledge graph. Do not create a second project dossier system outside it.

If a project needs a home note, create it inside Obsidian, not elsewhere. Keep it lightweight and linkable.

---

## 5. Standard workflow

### Example: Sahil wants to build X

1. Sahil gives KENSEI the idea.
2. KENSEI clarifies goal, user, success criteria, constraints, and risk.
3. KENSEI writes a short working brief.
4. KENSEI creates Kanban tasks with parent/child dependencies.
5. Research Lead validates or enriches the brief, if needed.
6. Coding Lead plans and builds. It may spawn subagents for implementation and review.
7. Coding Lead completes with structured metadata: files changed, tests run, risks, decisions, recommended next task.
8. Content Lead consumes the coding/research handoff and creates content outputs, if useful.
9. Knowledge Librarian updates Obsidian or repo docs with decisions and durable context.
10. Ops Lead deploys or monitors only when required and approved.
11. KENSEI reviews all outputs, arbitrates disagreements, and brings Sahil the decision summary and next recommended move.

---

## 6. Handoff contracts

Every profile should complete Kanban tasks with concise `summary` plus structured `metadata`.

### Coding Lead handoff

```json
{
  "changed_files": [],
  "tests_run": [],
  "tests_passed": null,
  "validation": [],
  "decisions": [],
  "risks": [],
  "next_recommended_profile": "content-lead|knowledge-librarian|ops-lead|null",
  "approval_needed": []
}
```

### Research Lead handoff

```json
{
  "sources_read": 0,
  "source_urls": [],
  "recommendation": "",
  "confidence": "low|medium|high",
  "tradeoffs": [],
  "open_questions": [],
  "next_recommended_profile": "coding-lead|content-lead|knowledge-librarian|null"
}
```

### Content Lead handoff

```json
{
  "target_audience": "",
  "channel": "linkedin|x|app-copy|email|other",
  "drafts_created": [],
  "voice_used": "",
  "context_inputs": [],
  "approval_needed": ["publish|schedule|send"],
  "next_recommended_profile": "knowledge-librarian|null"
}
```

### Knowledge Librarian handoff

```json
{
  "notes_created": [],
  "notes_updated": [],
  "decisions_captured": [],
  "links_added": [],
  "gaps_found": [],
  "sync_needed": true
}
```

### General Assistant handoff

```json
{
  "task_type": "mailbox|shopping|admin|reminder|booking|other",
  "actions_taken": [],
  "actions_requiring_approval": [],
  "external_messages_drafted": [],
  "spend_or_purchase_required": false,
  "next_recommended_profile": "kensei-orchestrator|ops-lead|knowledge-librarian|null"
}
```

### Ops Lead handoff

```json
{
  "systems_checked": [],
  "changes_made": [],
  "commands_run": [],
  "health_status": "green|amber|red",
  "risks": [],
  "rollback_path": "",
  "approval_needed": []
}
```

---

## 7. Implementation checklist

Detailed execution checklist now lives at:

`/home/kensei/repos/KenseiAgent/docs/MultiProfileImplementationChecklist.md`

This section keeps the high-level sequence only.

### A. Validate live Kanban baseline

- [ ] Confirm `hermes kanban --help` lists expected commands.
- [ ] Confirm `hermes kanban stats` runs cleanly.
- [ ] Confirm dashboard exposes Kanban board.
- [ ] Confirm gateway dispatcher is running or can be nudged.
- [ ] Confirm task logs and run history are accessible.
- [ ] Confirm profile lanes appear once profiles exist.

### B. Define and create profiles

- [ ] Decide exact profile names.
- [ ] Keep `default` as KENSEI Orchestrator or create `kensei-orchestrator` alias.
- [ ] Create `general-assist`.
- [ ] Create `research-lead`.
- [ ] Create `coding-lead`.
- [ ] Create `content-lead`.
- [ ] Create `knowledge-librarian`.
- [ ] Create `ops-lead`.
- [ ] Set explicit model/provider per profile.
- [ ] Verify `hermes profile list` and `hermes kanban assignees` show profiles.

### C. Write profile prompts

- [x] KENSEI Orchestrator prompt draft written with route-not-execute behaviour.
- [x] General Assistant prompt draft written, including booking requests.
- [x] Research Lead prompt draft written.
- [x] Coding Lead prompt draft written.
- [x] Content Lead prompt draft written.
- [x] Knowledge Librarian prompt draft written.
- [x] Ops Lead prompt draft written.
- [x] Each prompt draft includes scope, non-goals, approval gates, handoff contract, and escalation rules.

### D. Assign skills and tools

- [ ] KENSEI Orchestrator has Kanban orchestration skills.
- [ ] General Assistant has mailbox/general admin skills where available.
- [ ] Research Lead has web/research/Tavily/RSS skills.
- [ ] Coding Lead has coding, testing, GitHub, review, planning skills.
- [ ] Content Lead has brand voice and social content skills.
- [ ] Knowledge Librarian has Obsidian and documentation skills.
- [ ] Ops Lead has Hermes ops, cron, MCP, system health skills.
- [ ] Avoid loading giant MCP toolsets into profiles that do not need them.

### E. Configure shared workspace rules

- [ ] Decide default workspace type for each profile: `scratch`, `dir:<path>`, or `worktree`.
- [ ] Decide where shared Kanban artifacts live.
- [ ] Define when to use git worktrees.
- [ ] Define when Obsidian writes are allowed.
- [ ] Define cleanup/GC expectations.

### F. Define Obsidian operating pattern

- [ ] Confirm vault path on VPS.
- [ ] Confirm GitHub remote and sync script.
- [ ] Define where KENSEI architecture notes live.
- [ ] Define where project home notes live, if used.
- [ ] Define note naming conventions.
- [ ] Define when Knowledge Librarian commits/pushes vault changes.
- [ ] Prevent secrets from entering the vault.

### G. Pilot end-to-end workflow

- [x] Pick one small real task: free/near-free multi-platform posting options.
- [x] KENSEI creates task graph.
- [x] Research Lead completes a parent task.
- [x] Content Lead completes a downstream task.
- [x] Knowledge Librarian captures the result in Obsidian.
- [x] KENSEI reviews routing/model/Kanban pain points.
- [x] Critical fix applied: specialist profile `kanban.db` and `kanban/` paths now point at the shared default KENSEI board.
- [x] Decide whether to add dedicated Telegram topics for Content, Build, and Ops updates.
- [x] All four digests consolidated to Topic 1 (Daily Digest). Topics 19-23 repurposed: 19=Content, 20=Build, 21=Ops, 22=Research Deep Dives, 23=Archive.

### H. Decide Command Center fate

- [ ] Use Hermes dashboard and Kanban for at least one real pilot.
- [ ] Capture missing UI/UX gaps.
- [ ] Decide whether custom Mission Center is still needed.
- [ ] If needed, define it as a Sahil-facing overlay, not a replacement for Kanban.

---

## 8. Profile interview checklist

Use this before writing final prompts.

For each profile:

1. What is this profile responsible for?
2. What should this profile never do?
3. Which tools does it need by default?
4. Which tools should be available only by explicit task assignment?
5. Which skills should always be loaded?
6. What memory should it keep privately?
7. What knowledge should go to Obsidian instead of memory?
8. What approval gates apply?
9. What does a good handoff look like?
10. When should it block and ask Sahil/KENSEI?
11. When can it spawn subagents?
12. What model/provider should it use by default?
13. What should its workspace default be?
14. What failure modes are unacceptable?
15. What does “done” mean for this profile?

---

## 9. Answered questions and remaining decisions

Answered by Sahil on 2026-05-03:

1. Names stay functional and boring. Aliases can come later.
2. `default` remains KENSEI.
3. General Assistant gets mailbox/admin tools task-scoped.
4. Research Lead covers all research categories, including social scraping and product validation.
5. Coding Lead may edit and test. Commits only when asked. No push without approval.
6. Content Lead drafts only. Posting/scheduling comes later and must be approval-gated.
7. Knowledge Librarian may write to the defined KENSEI area in Obsidian. No git push unless approved or part of a sync job.
8. Ops Lead diagnoses freely, proposes fixes, and asks before restart/destructive/security-sensitive action.
9. First pilot is research -> content -> knowledge.
10. Second pilot is a small content drafting pipeline repeating that flow.

Remaining decisions before execution:

1. Pick any aliases/nicknames, if wanted. Not required for v1.
2. ~~Approve whether to execute the profile creation checklist.~~ Done 2026-05-04.
3. ~~Approve the exact first pilot topic.~~ Done: multi-platform posting options. Pilot A completed.
4. Decide after two pilots whether native Kanban/dashboard is enough or whether a custom Mission Center overlay is justified.
5. ~~Decide Telegram topic routing for profiles.~~ Done: unified Daily Digest Topic 1, profiles mapped to 19-23 via KENSEI relay.

---

## 10. Current recommendation

Do not build the custom Command Center yet.

First, configure the profile roster, use Hermes Kanban as the coordination layer, run one real end-to-end pilot, and then decide what UI is missing.

The target spine is:

KENSEI Orchestrator + Specialist Hermes Profiles + Hermes Kanban + Obsidian knowledge graph.
