# 06 — Aro Workbench Prototype vs. Hermes/SamAgent: Comparison & Integration Plan

Status: **Decision document** — read with `03-redesign-plan.md` (the P0–P7 plan this supersedes in part).
Date: Oct 2026 · Prototype: `prototype/src/workbench/` (repo root) (the attached coding-agent-design-system, rebranded **Aro Workbench**, live AI wired).

---

## 1. What the prototype is now

The attached design system ("Conductor — Agent Workbench") was ported into this Next.js
project as `prototype/src/workbench/` (repo root) and merged with the best of the previous Aro prototype:

| Kept from old prototype | Replaced by design system |
| --- | --- |
| **Aro logo** (geometric A, emerald on dark tile) — `LogoMark` | Old titlebar/sidebar/statusbar/inspector/components |
| **Live AI** — `/api/chat` (z-ai server-side) streams the final reply word-by-word after scripted harness steps | Old chat canvas, tool cards, diff cards |
| Aro brand voice in the system prompt (terse, SOUL-derived) | Old demo engine/store (superseded by richer seeded sessions) |

Port quality: single `/` route, client-only mount, 5 themes, full keyboard layer, mobile
drawer adaptation, lint-clean, browser-verified (screenshots 10–22 in `screenshots/`).

**Prototype surface map** (all interactive, seeded demo data + one live AI endpoint):

- **Two products**: `Code` (coding agents on repos) ⇧⌘E `Agent` (one assistant, any task)
- **Chat canvas**: user bubbles, thinking, tool cards (read/edit/write/bash/grep/mcp/browser/test/plan),
  unified/split diffs, approval cards (allow-once / always / deny + risk), checkpoints, notices,
  interrupt (Esc), quick-context pills, slash commands, @-context menu, effort selector, env selector
  (local/worktree/cloud), dictation (Web Speech), model + agent pickers incl. local providers
- **Sessions**: projects w/ inherited defaults (agent/model/mode/rules/budget), pin/archive/rename/
  duplicate/delete, grouping by project/date/agent, drag-thread-to-project, archive filter, search
- **Context pane** (right rail): plan / changes / context (window budget, pinned, MCP, rules) / fleet /
  git / browser / editors / terminal
- **Views**: Tasks (work graph: acceptance criteria, deps, cost, dispatch), Skills (bundled/hub/agent-
  authored), Connectors (MCP), Projects, Review (diff-centric + checkpoints + staged accept),
  Git (repos/commits/PRs/branches), Browser (preview + console + pick-element→agent), Fleet
  (parallel runs, best-of-N variants, spend caps), Settings (general/keyboard/agents/providers/rules/
  notifications/voice/account + **design system reference**), Agents registry (9 agents w/ strengths,
  success rates, routing policy)
- **Voice**: draggable orb, hands-free command routing ("open tasks", "new thread"), speak-back, mic level viz
- **Chrome**: command palette ⌘K, shortcut registry ⌘/, todo popup ⌘T, layout control w/ wireframe
  presets (standard/focus/review/terminal), multi-tab terminal (agent/shell/dev/problems)
- **Live AI**: final agent reply per message is a real model answer (GLM via z-ai), streamed word-by-word

---

## 2. Capability matrix — prototype vs. Hermes/SamAgent reality

| Prototype concept | Hermes/SamAgent backend reality | Verdict |
| --- | --- | --- |
| Sessions/threads, titles, search | `SessionDB` SQLite + FTS5 search, checkpoints | ✅ wire up (search is richer in backend) |
| Permission modes: plan / readonly / agent / full | approvals system, `hermes security`, vault, egress rules | ✅ direct mapping; rename to Aro terms |
| Tool timeline cards (64 tools) | real tool events over tui_gateway JSON-RPC contracts | ✅ contracts already typed (Pydantic→TS) |
| Diff cards, review view, checkpoints | git review & worktrees in desktop; SessionDB checkpoints | ✅ upgrade: hunk-level actions are *better* in prototype |
| Terminal panel (multi-tab) | xterm.js 6 + node-pty + 7 backends (local/docker/ssh/modal/daytona/…) | ✅ swap seeded lines for pty bridge |
| Fleet: parallel runs, worktrees, best-of-N | `hermes worktree`, kanban multi-agent orchestration, serverless backends, `moa` | ✅ **backend exists, no UI today — big win** |
| Agent fleet (9 external CLIs + routing policy) | single Aro core + `delegate_task` subagents | ⚠️ **prototype invents a product direction** (see §5.2) |
| Skills (bundled/hub/agent-authored) | 58 + 152 skills, agentskills.io, skill curator | ✅ strong match; add skill-run telemetry |
| Connectors (MCP) | 65 optional MCP servers + built-in MCP client | ✅ match |
| Context window budget viz | prompt caching invariants + trajectory compressor | ✅ map % to real telemetry |
| Voice orb + dictation | Quick Entry voice, ASR/TTS plugins | ✅ route to Hermes ASR/TTS (privacy + quality) |
| Browser view + pick-element | computer-use tool, noVNC Bot Desktop | ✅ reuse desktop browser pane |
| Live AI reply | api_server :8642 (OpenAI-compatible) | ✅ point `/api/chat` at it for "live backend" mode |
| **Cron / automations** | cron scheduler — a core moat | ❌ **missing from prototype — add** |
| **Messaging channels** (~30 platforms) | gateway + 22 platform plugins | ❌ **missing from prototype — add** |
| **Memory graph / user modeling** | memory providers (honcho/mem0/…), Memory Graph in desktop | ❌ prototype only shows a context % — add P5 view |
| Artifacts | desktop artifacts surface | ❌ not in prototype |
| Onboarding / doctor | Python runtime friction (the #1 gap from `02-gap-analysis`) | ❌ not in prototype — add P7 flow |
| Usage/insights | usage/insights CLI groups | ❌ partial (Settings shows spend only) |

---

## 3. What to include FROM the prototype INTO the Hermes codebase

Recommended path stays **Option A** (re-skin the existing Electron renderer; zero backend
changes — the JSON-RPC narrow waist already carries everything the prototype shows):

| Phase | Port into `apps/desktop` renderer | Effort | Effect |
| --- | --- | --- | --- |
| R1 · Tokens | `index.css` theme layer (5 themes, obsidian default), `ui.tsx` primitives, fonts, DesignSystemView as internal reference page | S | whole-app visual identity in days |
| R2 · Transcript | `Transcript.tsx` step family (ToolCard/DiffView/ApprovalCard/Thinking/Checkpoint/Prose) mapped to streamed tool events; composer with modes/effort/env chips | M | the "diff-first, tool-timeline" core UX parity with Codex app |
| R3 · Sessions | `Sidebar.tsx` model (projects w/ inherited defaults, grouping, archive) over SessionDB via the shared gateway client | M | replaces current list; FTS5 search behind same input |
| R4 · Review + Terminal | `ReviewView` (split/unified, staged accept, checkpoints) on real diffs; `TerminalPanel` chrome over existing xterm bridge | M | parity++ (hunk actions beat current desktop) |
| R5 · Fleet + Context | `FleetView` over kanban/delegate_task runs; `ContextPane` sections over real telemetry (context %, MCP, rules) | M | the "parallel task threads" Codex-app pattern, on Hermes guts |
| R6 · Voice + polish | orb → Quick Entry/ASR/TTS; command palette, shortcut registry, layout presets, todo popup | S–M | the fit-and-finish layer |
| R7 · Brand | Aro wordmark rules, `LogoMark`, NOTICE/attribution strings (MIT) | S | already specified in `docs/brand/BRAND.md` |

Do **not** port: the two-product split (`Code`/`Agent`) — Hermes already *is* both; collapse
into one product with mode presets. Keep the mock agent-fleet registry out until §5.2 is decided.

---

## 4. What to change IN the prototype to align with Hermes (next iteration)

Priority order — each is scoped to days, not weeks:

1. **P-A · Cron & Channels views** (make the moats visible)
   - Cron: schedule list (human-readable "every weekday 9am"), next-run, last result, pause, spend cap
   - Channels: the ~30 messaging platforms as connectors with per-platform identity + approval rules
   → turns the prototype from "another Codex clone" into **the only agent workbench that shows
   cron + messaging + serverless**.
2. **P-B · Live backend toggle** — Settings → Backend: `demo` (today's seeds) | `live`
   (point `/api/chat` at a real Hermes `api_server :8642` / any OpenAI-compatible endpoint).
   Makes the prototype a real client for the actual system.
3. **P-C · Memory graph** (P5 of the original plan) — node graph of agent-curated memories +
   "what Aro learned this week"; wire the Context pane's memory row to it.
4. **P-D · Hunk-level review actions** — accept/reject per hunk + inline comment → agent follow-up
   (parity with Codex desktop's strongest feature).
5. **P-E · Onboarding/doctor strip** — first-run checklist (runtime, provider key, workspace pick)
   addressing the #1 gap from `02-gap-analysis.md`.
6. **P-F · Artifacts surface** — pin outputs (docs/decks/data) from agent sessions.

---

## 5. Decisions to make (blocking)

1. **Integration target** — confirm Option A (re-skin `apps/desktop` renderer) as the vessel for
   §3, vs. Option B (new Tauri shell) / C (ACP client). Recommendation: **A**, evaluated at R3.
2. **The agent fleet question** — the prototype's defining idea is *many agents, one workbench*
   (routing claude-code/codex/cursor/aro…). Hermes is a single self-improving core. Choose:
   a) **Harness direction**: Aro routes work to external agents (build the orchestrator for real) —
      differentiates hard vs. every competitor; needs adapter work.
   b) **Core direction**: one Aro agent, fleet view becomes subagents/delegate_task + best-of-N —
      simpler, matches today's backend.
   Recommendation: ship (b) in R5, prototype (a) behind the existing registry UI.
3. **Voice routing** — keep browser Web Speech (zero setup) vs. Hermes ASR/TTS plugins (quality,
   privacy, offline). Recommendation: browser engine as default, Hermes daemon engine when attached
   (the settings row already models this).
4. **Naming in data** — prototype uses `aro` agent id, `aro.*` localStorage keys, `aro:` events;
   deep-rebrand TODO (`04-rebrand-log.md`) still has `~/.hermes`→`~/.aro` etc. Sequence it before R1
   so strings align.

---

## 6. Verification log (this merge)

- lint: clean (0 errors, 0 warnings)
- dev server: `GET / 200`, no runtime errors, `POST /api/chat 200` (live AI confirmed)
- golden paths browser-verified: send message → tool steps → **live AI reply** ("Done. The toggle
  is in settings, and it updates the theme store…"), ⌘K palette → Review/Tasks, Settings → design
  system reference, Agent product switch, theme switch (obsidian/daylight), mobile drawer
- VLM scores: desktop 9/10 (no defects), mobile drawer 8/10, main chat 6/10 at 390px (dense by nature)
