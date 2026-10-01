# 05 — Aro Desktop Prototype Spec

Sep 2026 · build target: **this repo** (`/home/z/my-project`, Next.js 16 + React 19 + Tailwind 4 + shadcn/ui + framer-motion). **Single route** (`/`) simulating the Aro Desktop app — a click-through-able, live-AI-backed vision of P1–P5 of the [redesign plan](03-redesign-plan.md) before committing to Option A/B on the real Electron app.

## Stack (already in `package.json`)

- Next.js 16 App Router, React 19, TypeScript, single route (`src/app/page.tsx`)
- Tailwind CSS 4 (CSS-first tokens) + shadcn/ui (Radix) + `lucide-react` icons
- `framer-motion` — subtle animations (150–250ms, opacity/translate only)
- `cmdk` — ⌘K command palette; `react-resizable-panels` — pane layout
- `z-ai-web-dev-sdk` — live AI via a **server-side only** API route; keys never reach the client
- `zustand` — client state (sessions, tasks, context)

## Design tokens (from `BRAND.md`, enforced here)

| Token | Value |
|---|---|
| Base | zinc-950 `#09090B`; raised zinc-900; overlay/hover zinc-800; 1px zinc-800 borders |
| Accent | emerald-500 `#10B981` (+ 400 `#34D399`, 300 `#6EE7B7` for gradient/links) |
| Status | running = amber pulse · done = emerald · failed = rose · queued = zinc |
| Type | Geist → `system-ui` stack, 13–15px UI; monospace for terminal/diffs/code |
| Motion | 150–250ms ease-out, opacity/translate only; respect `prefers-reduced-motion` |
| A11y | 44px hit targets, visible emerald focus rings, ARIA roles everywhere, WCAG AA |

No blue/indigo/purple anywhere. One emerald action per view. Borders over shadows.

## 1. Desktop shell

- **Custom titlebar**: 36–40px, drag region, **traffic lights** (red/amber/green circles, left), centered app title "Aro", right-side window controls affordance.
- **Sidebar** (240px, collapsible ⌘B): brand block (Aro mark + wordmark, emerald), **Sessions** list (grouped Today/Yesterday/Earlier; active = emerald left-rule + zinc-800 bg; cost chip per session), **Views** nav — Chat · Tasks · Context · Settings (icon + label rows, keyboard-numbered).
- **Status bar footer** (28px): model badge (e.g. `GLM-4.7 · openrouter`), backend chip (`local`), connection dot (emerald/amber/rose), session cost meter (e.g. `$0.042`), latency readout.

## 2. Chat canvas (default view)

- **Message stream**: user (right-aligned zinc-800 bubble) / assistant (full-width, markdown-rendered, streaming with caret); system/status notices as zinc-500 one-liners.
- **Tool timeline cards** (the core pattern): every simulated tool call renders as a compact card in the stream —
  - types: `search` (magnifier), `edit` (pencil), `bash` (terminal), `think` (brain), `skill` (sparkles)
  - header: icon · title (e.g. `rg "memory provider" --type py`) · status color chip · **duration chip** (`1.2s`) · **cost chip** (`0.8k tok`)
  - collapsed by default when done; expandable to show body/output; running state = amber pulsing dot; failed = rose + error text inline
- **Diff cards**: edit-tool expansion shows inline diff — monospace, `-` lines rose-tinted, `+` lines emerald-tinted, file header bar.
- **Approval card**: before a risky simulated tool — summary + diff + **Approve / Always allow / Deny** (44px buttons, Approve = emerald).
- **Interrupt-and-redirect**: Stop button in composer while streaming; typing during a run queues the message (shown as pending) and lands it as a redirect after the current step.
- **Composer**: multiline input, **slash-command autocomplete** (`/new /model /skills /usage /compress` — fuzzy filter, ↑↓ + Enter), model badge chip (click → model quick-switch), send = emerald, ⌘Enter hint.

## 3. Tasks board view

- Columns: **Queued / Running / Review / Done** (kanban via dnd-kit).
- **Subagent task cards**: title, status pill, **progress bar** (emerald), **cost meter** ($ + tokens), elapsed, **live terminal preview** (last 3 lines, monospace, streaming).
- **One-click fan-out**: input "split into N tasks" or a seeded multi-item job → spawns N cards into Queued → they progress to Running concurrently (simulated, staggered durations) with independent cost meters.
- Drag to **Review** → opens the linked diff card; Done cards show total cost + duration summary.

## 4. Context inspector (right pane, ⌘I toggle)

Tabs (Radix Tabs, icon + label):

| Tab | Content |
|---|---|
| **Files** | Repo tree (collapsible folders, dirty-file dot), click → file preview w/ line numbers; "recently edited by Aro" section |
| **Terminal** | Simulated xterm-style pane: last commands + outputs, running-line with cursor blink |
| **Memory** | "What Aro knows" — memory items (fact · source session · age), plus mini **Memory Graph** (nodes/edges, emerald on zinc, framer-motion layout) |
| **Skills** | Skill cards (name, description, times-used, success-rate), "+ Aro just learned: …" creation moment card |
| **Usage** | Session cost breakdown by model/tool, small bar chart (recharts), budget bar |

## 5. ⌘K command palette (cmdk)

- Fuzzy over: actions (New session, Toggle inspector, Switch view, Change model, Run task…), sessions (jump), settings. Emerald active row, kbd hints (↵ ⌘K esc). Opens centered-over-canvas with backdrop blur-zinc.

## 6. Live AI + simulation engine

- **`/api/chat` route** (server-side only): `z-ai-web-dev-sdk` chat completions; system prompt carries the Aro persona (SOUL.md voice: direct, terse). Streams back to the canvas.
- **Simulated tool-step engine**: after a user turn, run a scripted sequence — `think` → `search` → `bash` → `edit (diff)` → approval gate → assistant summary — each with randomized durations/costs, then hand the accumulated context to the live model for the final message. Tool cards are rendered from these events, proving the timeline UX with real AI text.
- **Seeded demo sessions**: 3–4 pre-baked sessions (e.g. "Refactor memory providers", "Cron: weekly audit", "Fix gateway flaky test") with full tool timelines, diffs, and costs — instant content on first paint, no typing required.

## 7. Non-goals (this prototype)

Real agent execution, real xterm/pty, persistence (Prisma stays unused), auth, multi-window, light mode (tokens defined, not exercised). This is the **design validation surface** for P1–P5 — after sign-off, the same tokens/patterns port to the real renderer (Option A) or the new Tauri frontend (Option B).

## Acceptance checklist

- [ ] Loads dark-first, single route, no blue/indigo anywhere
- [ ] Titlebar traffic lights + drag affordance render correctly
- [ ] Tool cards: all 5 types, status colors, duration+cost chips, collapse/expand
- [ ] Diff card with +/- tinted monospace lines; approval card gates the flow
- [ ] Interrupt stops streaming; queued redirect lands after current step
- [ ] Slash autocomplete in composer (fuzzy, keyboard-driven)
- [ ] Tasks board: 4 columns, live progress, fan-out, drag, terminal previews
- [ ] Context inspector: all 5 tabs populated; Memory Graph animates
- [ ] ⌘K opens fuzzy palette; all actions work
- [ ] Live AI reply via /api/chat (server-side SDK only)
- [ ] Keyboard-only walkthrough possible; focus rings visible; 44px targets
