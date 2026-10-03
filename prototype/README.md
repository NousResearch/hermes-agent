# Aro Workbench — Prototype

The **Aro Workbench** is the design-system-driven frontend prototype for
Aro Desktop: one session, every surface — multi-agent threads, diff-centric
hunk-level review, parallel runs, cron & channels automations, a memory
graph of what the agent learned, artifacts, skills, connectors, and a
switchable live-AI backend.

Built from the winning design direction (see
[`../docs/research/06-prototype-vs-hermes-integration.md`](../docs/research/06-prototype-vs-hermes-integration.md))
and merged with the previous Aro prototype's live-AI wiring.

## What's inside

```
src/
  app/            Next.js 16 App Router — single "/" route + /api/chat
  workbench/      The workbench itself (ported design system, rebranded Aro)
    App.tsx         shell: titlebar, sidebar, chat, context pane, terminal, statusbar
    components/     26 components (composer, transcript, views, voice orb, …)
    data/           seeded agents / sessions / tasks / git / providers / cron / channels / memory / artifacts
    lib/            app context, shortcuts, dictation, live-AI + backend-switch helpers
src/components/ui/  shadcn/ui set (template scaffolding, currently unused)
```

- **Stack**: Next.js 16 (App Router) · React 19 · TypeScript · Tailwind CSS 4
- **Live AI**: `src/app/api/chat/route.ts` calls the z-ai SDK server-side;
  scripted harness tool steps play out first, then the real model reply
  streams word-by-word (see `src/workbench/lib/live.ts`).
- **Backend toggle**: Settings → Backend switches the closing chat reply
  between the demo sandbox (`/api/chat`) and any live OpenAI-compatible
  endpoint (default `http://localhost:8642/v1` — the Aro agent's own
  api_server), with connection testing and graceful demo fallback.
- **Round-2 views**: Automations (cron schedules + ~30-platform channels —
  the messaging moat made visible), Memory graph ("what Aro learned this
  week", clustered nodes, pin/forget, provider switch), hunk-level review
  (per-hunk accept/reject/comment → agent follow-up), onboarding/doctor
  strip, and an Artifacts gallery with session provenance.
- **Themes**: obsidian (default) · daylight · nord · ember · paper.
- Runs client-only (localStorage persistence) on a single route.

## Run it

```bash
cd prototype
bun install        # or npm/pnpm install
bun run dev        # http://localhost:3000
```

## Status

Prototype — demo data is seeded; the backend is switchable between the
demo sandbox and a real OpenAI-compatible endpoint. The plan for wiring it
to the real agent (tui_gateway JSON-RPC, dashboard API, OpenAI-compatible
`:8642`) is
[`../docs/research/06-prototype-vs-hermes-integration.md`](../docs/research/06-prototype-vs-hermes-integration.md).
Round-2 status and the verification log live in §6–§7 of that doc.
