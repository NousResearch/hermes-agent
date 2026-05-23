# Town-Hermes Feedback Protocol

**Status:** DRAFT / NOT ACTIVE
**Created:** 2026-05-18
**Priority:** 5 of 7 \(defer until Skills 1-4 stabilize\)

## Purpose

Define how insights, corrections, and operational findings from the Town environment \(memories, audit findings, operator feedback\) can be systematically routed back to Hermes agent configurations and knowledge files.

---

## Current State \(As-Is\)

```
Hermes -> email -> Town (Spec 090, operational)
  - Hermes jobs send structured emails to djschulz@gmail.com
  - Town routines trigger on [Hermes] subject prefix
  - Town creates tasks / DMs operator

Town -> Hermes (NO FORMAL PATH)
  - Town memories accumulate (20+ global as of May 18)
  - Town doc review findings accumulate (7 criticals, 6 warnings, 6 gaps)
  - Operator feedback via chat accumulates
  - None of this systematically reaches Hermes agent prompts or .learnings/ files
```

---

## Proposed Feedback Channels \(Design Only\)

### Channel 1: Memory Sync \(Town -> Hermes `.learnings/`\)

**Concept:** Periodically export actionable Town memories to a Hermes-readable format.

| Town Memory Type | Hermes Target | Sync Frequency |
| --- | --- | --- |
| Global corrections \(style, format\) | `.learnings/memory.md` | Weekly |
| Routine-specific findings | `.learnings/projects/{routine}.md` | Weekly |
| Watchlist corrections \(ticker changes\) | `production_data/` config or `tools/` scripts | On detection \(manual\) |
| 13F filing alerts \(do-not-re-alert\) | Hermes dedup state | On detection \(manual\) |

**Constraint:** Hermes `.learnings/memory.md` has a 100-line cap. Sync must be selective, not bulk.

### Channel 2: Audit Finding Routing \(Town -> Hermes Knowledge Layer\)

**Concept:** When the doc review log or drift baseline identifies findings relevant to Hermes operations, route them to the knowledge layer's contradiction ledger.

| Finding Type | Hermes Target | Example |
| --- | --- | --- |
| Skill text error | `artifacts/ops/contradiction_ledger/latest.md` | C2: inst\_delta\_z scope error |
| Infrastructure alert | `artifacts/ops/knowledge_layer/latest_state.json` | G6: 6 stale agents |
| CI red state | Ops supervisor anomaly input | C4: CI red 10+ days |

**Constraint:** Town does NOT write to the Hermes repo. The routing is via email or operator manual action.

### Channel 3: Operator Decision Feedback \(Town -> Hermes Governance\)

**Concept:** When the operator makes a decision in Town \(approves a spec, resolves an audit finding, calibrates a threshold\), that decision should be recorded in a format Hermes can consume.

| Decision Type | Current Path | Proposed Path |
| --- | --- | --- |
| Spec approval | Town chat -> email -> operator manually commits | Same + structured decision record |
| Threshold calibration | Town memory update | Same + governance log entry |
| Audit finding resolution | Town doc review log update | Same + resolution propagated to affected skill |

---

## Self-Improving Skill Integration

The self-improving skill defines dual storage:

- **Town:** `add_memory()` / `add_memory(routine_slug=...)`
- **Hermes:** `.learnings/memory.md`, `.learnings/projects/`, `.learnings/domains/`

**Current gap:** These are independent stores with no reconciliation.

**Proposed resolution \(design only\):**

- Town memories are the **operator-facing** store \(what the human sees and edits\)
- Hermes `.learnings/` files are the **agent-facing** store \(what agents load at session start\)
- Reconciliation is manual, operator-initiated, and infrequent \(monthly or at major milestones\)
- The operator decides which Town learnings are worth propagating to Hermes

---

## Town Platform Capabilities Reference \(May 2026\)

Understanding what Town can do is prerequisite to designing effective feedback channels. This section documents Town's actual capabilities so Hermes agents can reason about what is and isn't possible on the Town side.

### Company & Product

- **Product:** Town AI Assistant \(town.com\) — multi-platform AI work assistant
- **Founders:** Jean-Denis Greze \(ex-CTO Plaid\) and Tony Vincent \(ex-Google AI Director of Product\)
- **Funding:** $18M Series A from First Round Capital \(March 2025\)
- **Status:** Open beta, free tier, rapid development cycle
- **Pivot history:** Originally AI tax advisory for SMBs \(2024\); pivoted to general AI work assistant

### Core Capabilities Relevant to Hermes Feedback

| Capability | What It Does | Hermes Feedback Relevance |
| --- | --- | --- |
| Persistent memory | Global + routine-specific preferences stored across sessions | Primary store for operator corrections, watchlist updates, dedup state |
| Content Library | Structured collections \(biotech-screener, investment-frameworks, ai-projects\) with permanent storage and shareable URLs | Hosts 39 skills as of May 2026; stores drift baselines, ledgers, and audit artifacts |
| Routine automation | Trigger-based routines \(email, calendar, schedule, RSVP\) with 3 control modes \(read-only, HITL, autonomous\) | Runs 20+ routines including catalyst tracker, 13F monitor, SEC EDGAR scans, morning briefings, drift baselines |
| Email management | Search, read, draft, send, label, archive across connected Gmail accounts | Receives Hermes output emails; can parse, label, and act on \[Hermes\] subject prefix |
| Calendar management | Full CRUD + RSVP on Google Calendar events | Maintains PDUFA calendar, catalyst calendar, meeting scheduling |
| Document creation | Town Docs \(rich-text, side-panel editing\) and Google Docs | Creates meeting notes, research briefs, people profiles \(Memories/people/\) |
| Voice/audio | iOS recording, transcription, audio briefing generation | Morning briefings stored in morning-briefings collection |
| Web research | Web search + fetch with AI-analyzed summaries | Ad-hoc research, competitive intel, person research |
| Code execution | Sandboxed Python/shell with optional internet access | Data processing, calculations, file manipulation |
| Skill system | 39 named skills encoding DEM methodology, pipeline rules, and operational procedures | Hermes knowledge encoded as Town skills; authoritative for Town-side agents |
| Per-tool permissions | Granular control over which tools routines can use and in what mode | Constrains what automated feedback channels can actually do |
| MCP protocol | Extensible tool integrations via Model Context Protocol servers | Granola \(meeting notes\), potentially extensible to Hermes-native MCP |

### Platform Access Channels

| Channel | Capability | Feedback Path Viability |
| --- | --- | --- |
| Web app \(chat\) | Full tool access, side-panel docs, interactive UI | Primary operator interaction; highest-fidelity feedback |
| Email \(dedicated address\) | Forward tasks, trigger routines | Hermes -> email -> Town routine trigger \(Spec 090, operational\) |
| iOS app | Voice commands, meeting capture, approvals | Mobile approval of routine actions; voice memo capture |
| Slack | Task forwarding, notifications | Potential lightweight feedback channel |
| WhatsApp | Text/photo/voice task input | Operator quick-response channel |
| Desktop \(macOS\) | iMessage, Contacts, full web features | Extended data access \(contacts, local messages\) |

### What Town CAN Do \(Relevant to Feedback Design\)

1. **Store and retrieve structured memories** — global and routine-specific, with full CRUD
2. **Search and read emails** — including parsing Hermes-generated email output for structured data
3. **Create and edit documents** — Town Docs, Google Docs, Google Sheets
4. **Execute code** — sandboxed Python for data processing, file manipulation
5. **Run trigger-based routines** — incoming email triggers can fire Town routines on Hermes output
6. **Maintain a Content Library** — permanent, organized, shareable file storage with collections
7. **Manage skills** — create, read, and update skill documents that encode methodology
8. **Search across sources** — federated search across email, Drive, Content Library, integrations
9. **Delegate to sub-agents** — invoke named sub-routines \(research-person, general-purpose\) for complex tasks
10. **Generate audio** — text-to-speech for briefings, summaries, narration

### What Town CANNOT Do \(Constraints for Feedback Design\)

1. **Cannot write to Hermes repo** — no git push, no file mutation on Hermes infrastructure
2. **Cannot schedule or control Hermes cron jobs** — no access to WSL2/systemd/Docker
3. **Cannot approve or reject Hermes specs** — governance is operator-mediated, not automated
4. **Cannot reactivate suppressed Hermes agents** — per Spec 090 boundary
5. **Cannot access Hermes runtime state** — no direct API to agent fleet status, session logs, or knowledge layer
6. **Cannot initiate Hermes sessions** — communication is email-only \(Hermes -> Town\) or operator-mediated \(Town -> operator -> Hermes\)
7. **Cannot modify Hermes `.learnings/` files** — no file-level access to Hermes persistent storage
8. **Cannot access Hermes production data** — no direct read of `production_data/`, CIK lookups, or pipeline state

### Integration Count & Competitive Context

Town connects to \~12 named integrations \(Gmail, Calendar, Drive, Docs, Slack, Notion, Dropbox, GitHub, Linear, HubSpot, Asana, MCP\). This is significantly fewer than Lindy AI \(200+\) but Town differentiates on graduated autonomy, meeting capture, audio briefings, and persistent memory depth. See `ai-landscape-monitoring` skill for full competitive analysis.

### Operational Role in DEM Stack

```
Hermes (WSL2/Docker)          Town (Cloud SaaS)
  30-agent fleet         <-->   20+ routines
  .learnings/ memory     <-->   Persistent memories (20+ global)
  Skills (repo-governed) <-->   Skills (39, doc-governed)
  Knowledge layer        <-->   Content Library (3 main collections)
  Cron scheduling        <-->   Trigger-based routines
  Production pipeline    <-->   Operator interface + ad-hoc research
```

**Communication path:** Hermes -> structured email -> Town incoming\_email trigger -> routine processes output -> operator reviews in chat/email

**Feedback gap \(unchanged\):** Town -> Hermes has NO formal automated path. All feedback is operator-mediated via manual commits, email instructions, or chat-to-action workflows.

---

## What Town Is NOT \(Reinforced from Spec 090\)

- NOT a scheduler or cron controller for Hermes
- NOT a repo mutator or spec approver
- NOT allowed to reactivate suppressed agents
- NOT the authoritative source for any production state
- NOT an automatic feedback channel \(all feedback is operator-mediated\)