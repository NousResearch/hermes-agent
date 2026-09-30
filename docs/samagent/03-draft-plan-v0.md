# 03 — Draft plan v0 (written to be attacked)

> This is my **first** plan, before any self-criticism. It is deliberately ambitious and contains mistakes. `04-red-team.md` attacks every numbered item. The plan you should follow is `05-final-plan.md`.

## Vision (v0)

"SamAgent: the smartest coding agent. Say what you want, answer a few questions, and a team of AI specialists builds it end to end in one shot, using local models first, remembering everything, with a beautiful new app."

## D0 — Architecture

Hard-fork Hermes and slim it: delete the messaging gateway, most tools, most providers, and the desktop app. Rename to SamAgent. Rewrite the UI from scratch as a new Electron app.

## D1 — Agent society

A permanent team spawned for every project:

`Product Manager · Architect · Frontend · Backend · Database · QA · Security · DevOps · Docs` — nine specialists working in parallel from the PRD. A "Chief" agent coordinates them.

## D2 — Interview

A thorough discovery interview (15–20 questions) covering users, features, design, stack, scale, and edge cases before any work starts.

## D3 — Model router

A **learned router** that picks local or cloud on **every turn**, using a classifier trained on our logs. It auto-switches between agents and models mid-conversation to save money and keep everything local when possible.

## D4 — Memory

A **knowledge graph with embeddings** of the user, projects, code, and conversations. A reflection pass every N turns writes new memories. Everything is remembered forever.

## D5 — Speed

"Match Arena's speed" using parallel agents plus a cloud sandbox pool we host.

## D6 — UI

A new Electron desktop app with a 3D live graph of agents at work, a chat, a code editor pane, a terminal, a browser preview, a memory explorer and a plugin marketplace.

## D7 — Targets

Web apps, mobile apps, games, desktop apps and CLIs from day one. Built-in one-click deploy to every major host.

## D8 — Claims

Marketing line: "smarter than Cursor, Claude Code, Hermes and Pi."

## D9 — Roadmap (24 weeks)

| Phase | Weeks | Content |
|-------|-------|---------|
| 1 | 1–4 | Fork, slim, rebrand |
| 2 | 5–8 | Agent society + interview |
| 3 | 9–12 | Learned router + local model stack |
| 4 | 13–16 | Knowledge-graph memory |
| 5 | 17–20 | New Electron app |
| 6 | 21–24 | Deploy integrations, marketplace, launch |

Success metric: "it feels smart".
