# 03 — Aro Desktop Redesign Plan (P0–P7)

Sep 2026 · 8-phase plan to take Aro Desktop from feature-rich/visually dated to Codex-app-class. Design language: dark-first, zinc-950, emerald accent, Zed/Linear restraint (see `../brand/BRAND.md`).

Guiding order: **make it look right → make the core loop right → close the code gap → surface the moats → polish → prove it.**

---

## Phase overview

| Phase | Name | Goal | Rough scope |
|---|---|---|---|
| P0 | Foundation | Design tokens, dark-first | days |
| P1 | Chat canvas | The core loop, readable | weeks |
| P2 | Workspace shell | 3-pane app, keyboard-first | weeks |
| P3 | Close the code gap | Repo + diff-centric review | weeks |
| P4 | Parallel tasks board | Parallelism as a first-class surface | weeks |
| P5 | Make the moat visible | Memory/skills/cron UI | weeks |
| P6 | Polish & onboarding | Micro-interactions, first-run, usage | weeks |
| P7 | Verification | E2E, visual regression, a11y | continuous |

### P0 — Foundation
- Design tokens: zinc scale (950 base, 900 raised, 800 overlay), emerald accent (`#10B981/#34D399/#6EE7B7`), amber warning, rose error; spacing 4/8 grid; radius ≤ 8px; 1px borders over shadows.
- Tailwind 4 CSS-first token config; semantic CSS variables (`--bg-raised`, `--accent`) so light mode (crisp Linear-style white) falls out of the same tokens.
- Typography: Geist/Inter UI scale (13–15px), monospace for terminal/code/diffs.
- A11y baseline: WCAG AA contrast in both modes, visible focus rings (emerald), reduced-motion support, screen-reader landmarks.
- Motion: framer-motion, 150–250ms ease-out, opacity/translate only, nothing bouncing.

### P1 — Chat canvas
- Message stream: user/assistant bubbles, markdown + streaming, code blocks with copy button.
- **Tool timeline**: every tool call = a compact card in sequence — icon (search/edit/bash/think), one-line title, status color (emerald done / amber running-pulse / rose failed / zinc queued), collapsed by default when succeeded.
- Card chips: duration + token/cost per call, visible on the card, aggregated per turn.
- **Inline diff viewer** in edit-tool cards: +/- monospace lines, emerald/rose tinted.
- **Approval prompts**: explicit Approve / Approve-all / Deny card before risky tools, with the diff above the buttons.
- **Interrupt-and-redirect**: stop button always available; redirect = queue a message that lands after the current tool completes.
- Slash-command autocomplete in the composer (fuzzy, keyboard-navigable, emerald active row).

### P2 — Workspace shell
- 3-pane layout: **sessions sidebar** / **conversation canvas** / **context inspector** (collapsible right pane).
- Command palette (⌘K/Ctrl-K): fuzzy, actions + sessions + settings, keyboard-only operation of the whole app.
- Status bar footer: model + provider badge, connection state, backend type, session cost meter.
- Keyboard-first: shortcuts for new session, toggle panes, cycle tasks; all primary actions reachable in ≤2 keystrokes.

### P3 — Close the code gap
- **Repo tree** in context inspector: file browser with dirty-state indicators.
- **PR-style review mode**: session changes grouped as a reviewable changeset — file list, per-file diff, **hunk-level accept/reject**, per-hunk comments (become agent instructions).
- Backend already provides worktrees, git checkpoints, and session snapshots — this phase is presentation, not plumbing.
- Terminal stays (xterm.js) but docked in context inspector as a tab, not a rival surface.

### P4 — Parallel tasks board
- Unify the existing **kanban** + **live subagents** into one Tasks surface: columns Queued / Running / Review / Done.
- Task cards: title, assigned subagent, status, **live terminal preview** (streaming last lines), **cost meter**, elapsed time.
- One-click fan-out ("do these 10 items in parallel") spawning isolated subagents; drag a card to Review to open its diff (P3 review mode).
- Cancellation + merge-back of subagent results into the parent session.

### P5 — Make the moat visible
- **Memory Graph first-class**: nodes/edges visualization of what Aro knows; inspect, edit, delete memories; show *when* and *from which session* a memory formed.
- **Skill-creation moments**: when Aro creates a skill, show the card ("I learned this — want to keep it?"); skill library browser with usage counts and success rates.
- Session search UI (FTS5 already exists — expose it): search across history with LLM summaries.
- **"What Aro knows about you"**: a readable dossier view of the user model (Honcho), exportable.
- **Cron timeline**: scheduled automations on a calendar/timeline, next-run, last result, edit inline.

### P6 — Polish & onboarding
- Micro-interactions: streaming caret, tool-card status transitions, tasteful hover states; empty states that teach ("No sessions yet — press ⌘N").
- **First-run wizard**: pick provider (39, incl. Ollama local) → API key → pick a backend (local now, Modal later) → sample session proving the loop works. Target: value in <5 min.
- Usage dashboard: cost by day/model/session, per-task budgets, warnings before limits (never Zed's fail-first).
- Light mode pass, localization audit (en/zh-Hans exist), reduced-motion + screen-reader final pass.

### P7 — Verification
- Playwright E2E: new session → tool runs → approval → diff accept → task completes; multi-connection flows (local + remote backend registry).
- Visual regression (screenshot diffs) on the 12 canonical screens, both themes.
- A11y audit (axe) to zero criticals; keyboard-only walkthrough script in CI.
- Perf budget: cold start <2s, chat interaction <100ms input latency, 10k-message session scroll stays 60fps.

---

## Implementation options

| | A. Restyle existing Electron renderer | B. New Tauri/Next frontend | C. ACP client + `hermes-acp` backend |
|---|---|---|---|
| **What** | Keep apps/desktop (Electron 40 + React 19 + Tailwind 4); rewrite renderer screens against the existing IPC | Greenfield frontend on `tui_gateway` JSON-RPC contracts + dashboard FastAPI (:9119) + OpenAI-compatible API (:8642) | Build Aro Desktop as a pure ACP client to the existing `hermes-acp` server |
| **Pros** | Fastest; xterm/node-pty, voice, updater, notarization/MSIX pipelines all kept; no re-integration risk | Cleanest UI; smallest, fastest shell; web/desktop parity from one codebase | Most standards-based; Zed/VS Code interop for free; least bespoke protocol work |
| **Cons** | Inherits 756k-LOC desktop codebase; some legacy structure constrains layout | Big-bang risk; must re-do packaging, updater, pty, voice, deep links | ACP is young: approval flows, memory/skills surfaces, cron, kanban don't map to ACP yet — moats invisible |
| **Verdict** | ✅ **Do first** | ✅ **Do later** (P2–P3 era, once contracts are proven by A) | 🟡 Track; contribute missing surfaces upstream |

**Recommendation:** **A first, B later.** Phase A de-risks: it validates the design system and chat canvas on shipping infrastructure and delivers user value in weeks. The tui_gateway/dashboard/API contract trio (validated meanwhile) becomes the seam for B — a Tauri shell with the same renderer — without throwing away P0–P5 design work. Keep `hermes-acp` healthy as a distribution channel (editors), not as the desktop's spine.
