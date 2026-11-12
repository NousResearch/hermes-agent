# agents.md Spec Summary

Source: https://github.com/agentsmd/agents.md — "a simple, open format for guiding coding agents"

## What it is

`AGENTS.md` is a "README for AI coding agents": a dedicated, predictable place that gives the agent project context and instructions. The format is deliberately minimal — **no required schema, no mandatory fields**, just a Markdown file.

## Core spec

1. **Filename**: must be `AGENTS.md`, at the repository root.
2. **Format**: plain Markdown (same syntax as README), directly readable by agents.
3. **Nesting**: subdirectories may carry their own `AGENTS.md`; when working on files there, the nearest `AGENTS.md` wins (more specific instructions override root-level). Root file therefore holds globally applicable content only.
4. **Content nature**: write "what the agent doesn't know but will get wrong" — build/test commands, code style conventions, project-specific pitfalls, workflow rules. No vague vision statements, no self-introductions.
5. **Sections**: no officially mandated sections, but the community's high-frequency set is:
   - `# Project Overview` (one or two sentences)
   - `## Dev Environment Tips`
   - `## Build & Test Commands`
   - `## Code Style`
   - `## Testing Instructions`
   - `## PR Instructions`

   High-frequency optional sections (only when the project really has that fixed process): `## Build & Deployment`, `## Security Notes`, `## Debugging & Troubleshooting`.
6. **Style** (from the official minimal example):
   - Imperative sentences, mostly `- ` list items
   - Commands given in full executable form (including package-manager prefix)
   - Each item is an action instruction, not background description
   - Short overall (official example is < 30 lines)

## Compatibility

`AGENTS.md` is natively recognized by major coding agents (Codex, Cursor, Copilot, …) as their project-level instruction file.

## How this skill maps to the spec

- Concise → one imperative line per rule, ≤ 120 lines total
- Compliant → keep the community's high-frequency sections (standard + optional), globally applicable content only
- No bloat → nesting guidance: a lesson that only applies to a subdirectory goes into a local `AGENTS.md` there instead of bloating the root; when slimming, content with long-term value migrates to `docs/<topic>.md` with a one-line reference rather than being hard-deleted
- Verifiable → first-version commands must be run for real or cross-checked against build/CI configs (see SKILL.md Step 3)
