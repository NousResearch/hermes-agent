---
name: agent-tool-docs
description: Read a local LLM-wiki before using a complex tool or skill.
version: 0.1.0
author: Saša Bogdanov (sbstratex), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [wiki, knowledge-base, skills, onboarding, llm-tool-docs]
    related_skills: [llm-wiki, obsidian]
---

# Agent Tool Docs

Before using any complex external tool, skill, SDK, or vendor API, consult the local LLM wiki for that capability. This skill tells an agent where the wiki lives, how to navigate it, and how to keep it current.

## When to Use

- The user asks to use a tool/skill/API the agent has not used in this session.
- A diagram, integration, connector, or vendor workflow is requested.
- A task involves a documented external capability (Archify, apaleo, Mews, Piviq SDK, Claude Code, etc.).

## Don't use for
- Simple built-in Hermes tools that are self-explanatory (`read_file`, `terminal`, `web_search`).
- Tasks already covered by a more specific skill loaded in this session.

## Canonical wiki root

The local LLM-wiki root is configurable:

```
LLM_WIKIS_ROOT (env) or ~/Projects/llm-wikis/ (default)
```

Resolve the root first with `terminal` if the environment variable may be set; otherwise use the default. Every supported capability has its own subdirectory:

| Wiki | Path under root | Covers |
|---|---|---|
| Archify | `archify/` | Diagram rendering and validation |
| Piviq SDK | `piviq/` | Piviq product/core APIs and acceptance rules |
| Hospitality Vendor APIs | `hospitality-vendors/` | apaleo, Mews, Cloudbeds, and other PMS/connector docs |

## Reading order for any wiki

1. `<root>/INDEX.md` — master catalog and per-wiki first-read checklist.
2. `<wiki>/SCHEMA.md` — conventions, taxonomy, and how a new LLM should use this wiki.
3. `<wiki>/index.md` — page catalog.
4. `<wiki>/log.md` — recent changes.
5. The relevant concept/entity/comparison page for the requested task.

## How to keep wikis current

- When a vendor releases a new API version or schema, mirror the new docs into the wiki's `raw/articles/` and update `log.md`.
- When a project decision changes how a tool is used, update the concept page, bump `updated:` in frontmatter, and append to `log.md`.
- When adding support for a new external capability, create a new wiki subdirectory using the same structure and register it in `<root>/INDEX.md`.
- Commit and push wiki changes the same day; the wiki root is a git repository.

## Verification

A usable wiki must satisfy:

- `SCHEMA.md` exists and explains conventions.
- `index.md` lists every page.
- `log.md` records the latest change.
- Every wiki page has at least two outbound `[[wikilinks]]`.
- Every raw source file carries `source_url`, `ingested`, and `sha256` frontmatter.

If any check fails, warn the user and fall back to the upstream source URL noted in the raw file.