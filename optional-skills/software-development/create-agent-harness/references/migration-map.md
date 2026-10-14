# Migration Map

Reference for `create-agent-harness` Phase 2 — canonical mappings from legacy harness artifacts into `.claude/`.

## Canonical migration map

| Origin (legacy) | Destination (Claude Code) | Action | Conversion |
| --- | --- | --- | --- |
| `AGENTS.md` / `DEVIN.md` (root) | `CLAUDE.md` (root) | Merge, then remove origin | Rewrite in the `CLAUDE.md` format |
| `CLAUDE.md` (root, already present) | `CLAUDE.md` (root) | Complete — merge missing mandatory sections, never overwrite | Add `## Memory Protocol` (always-save rule for prompts, history and knowledge), the Agent Loop read ritual and the always-on connection when absent |
| `.agents/subagents/*.md` | `.claude/agents/{slug}.md` | Convert, then remove `.agents/` | `allowed-tools` becomes `tools` |
| `.agents/skills/*` or root `skills/*` | `.claude/skills/{slug}/SKILL.md` | Move, then remove origin | Ensure `name` and `description` frontmatter |
| `.agents/rules/*` or root `rules/*` | `.claude/rules/{slug}.md` | Convert, then remove origin | `applyTo` becomes `paths:` |
| `.agents/knowledge/*` or root `knowledge/*` | `.claude/knowledge/{slug}.md` | Move, then remove origin | — |
| `.agents/MEMORY.md`, `SESSION_STATE.md`, root `memory/*` | `.claude/memory/` | Consolidate, then remove origin | Split into short-term and long-term |
| `.devin/config.json` | `.claude/settings.json` | Migrate permissions and hooks; reduce to the minimum | Devin JSON to the Claude schema |
| `.devin/agents/{name}/AGENT.md` | `.claude/agents/{slug}.md` | Convert, then remove `.devin/agents/` | Devin frontmatter to Claude frontmatter |
| `.windsurf/`, `.cursorrules`, `GEMINI.md`, `copilot-instructions.md` | `.claude/rules/` or `CLAUDE.md` | Extract useful rules, then remove originals | Prose to rule or section |
| `.claudeignore`, `.devinignore`, `.windsurfignore`, `.aiignore`, `.cursorignore` | `.claude/settings.json` (`permissions.deny`) | Convert patterns, then remove the files | `glob` becomes `Read(glob)` |
| `.cursor/` | `.claude/rules/` or `.claude/skills/` | Extract reusable content, then remove | Keep only if platform-specific quirks are required |
| `.gemini/` | `.claude/rules/` or `.claude/skills/` | Extract reusable content, then remove | Keep only if platform-specific quirks are required |

> `.claudeignore` is **not read** by the Claude Code CLI. Exclusions belong in `permissions.deny`. Keeping the file gives a false sense of protection.

## Naming and collision rules

- **Slug** — kebab-case ASCII derived from the original name: `Backend.NET` becomes `backend-dotnet`.
- **Name collision** — two legacy artifacts mapping to the same destination: suffix with the origin domain (`-from-agents`, `-from-devin`) and record the decision in long-term memory. Never overwrite silently.
- **Conflicting content** — keep both as distinct sections at the destination and mark `TODO: review duplication`. Never discard content without human confirmation.
- **Sub-agents** — the file name `{slug}.md` and the frontmatter `name:` must match exactly.

## Frontmatter conversions

```diff
---
- allowed-tools: Read, Grep
+ tools: Read, Grep
name: review
description: ...
---
```

```diff
---
- applyTo: "**/*.cs"
+ paths:
+   - "**/*.cs"
---
```

> `applyTo` is **not** interpreted by Claude Code. Use `paths:`. A rule without `paths:` is always-on.

## Removing legacy artifacts

Run only for the artifacts actually migrated, and only after confirming their useful content exists at the destination.

```bash
git rm -r .agents .windsurf .devin/agents 2>/dev/null
git rm AGENTS.md DEVIN.md GEMINI.md copilot-instructions.md \
       .cursorrules .cursorignore .aiignore \
       .claudeignore .devinignore .windsurfignore 2>/dev/null
git rm SESSION_STATE.md 2>/dev/null

# ONLY if these directories were loose at the repository root
git rm -r skills rules knowledge memory 2>/dev/null
```
