# Aro Workbench — Prototype

The **Aro Workbench** is the design-system-driven frontend prototype for
Aro Desktop: one session, every surface — multi-agent threads, diff-centric
review, parallel runs, skills, connectors, and a live AI backend.

Built from the winning design direction (see
[`../docs/research/06-prototype-vs-hermes-integration.md`](../docs/research/06-prototype-vs-hermes-integration.md))
and merged with the previous Aro prototype's live-AI wiring.

## What's inside

```
src/
  app/            Next.js 16 App Router — single "/" route + /api/chat
  workbench/      The workbench itself (ported design system, rebranded Aro)
    App.tsx         shell: titlebar, sidebar, chat, context pane, terminal, statusbar
    components/     22 components (composer, transcript, views, voice orb, …)
    data/           seeded agents / sessions / tasks / git / providers
    lib/            app context, shortcuts, dictation, live-AI helpers
src/components/ui/  shadcn/ui set (template scaffolding, currently unused)
```

- **Stack**: Next.js 16 (App Router) · React 19 · TypeScript · Tailwind CSS 4
- **Live AI**: `src/app/api/chat/route.ts` calls the z-ai SDK server-side;
  scripted harness tool steps play out first, then the real model reply
  streams word-by-word (see `src/workbench/lib/live.ts`).
- **Themes**: obsidian (default) · daylight · nord · ember · paper.
- Runs client-only (localStorage persistence) on a single route.

## Run it

```bash
cd prototype
bun install        # or npm/pnpm install
bun run dev        # http://localhost:3000
```

## Status

Prototype — demo data is seeded; the "backend" is the single live-AI endpoint.
The plan for wiring it to the real agent (tui_gateway JSON-RPC, dashboard API,
OpenAI-compatible `:8642`) is
[`../docs/research/06-prototype-vs-hermes-integration.md`](../docs/research/06-prototype-vs-hermes-integration.md).
