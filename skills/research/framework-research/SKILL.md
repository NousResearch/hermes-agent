---
name: framework-research
description: "Systematic methodology for researching external frameworks, tools, or systems — extracting architecture, concepts, use cases, and integration patterns from documentation and code."
version: 1.0.0
author: Fuad Al Fajri
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Research, Frameworks, Tools, Documentation, Integration]
    category: research
    related_skills: [grounded-citations, research-paper-writing]
---

# Framework & Tool Research

Systematic methodology for researching an external framework, tool, or system from its documentation, codebase, and community resources. Produces a structured summary covering architecture, core concepts, use cases, and integration analysis.

## When to Use

- User asks to research, investigate, explore, or evaluate an external tool, framework, system, or library
- User asks how to integrate or combine an external tool with their stack
- User asks to compare tools, or explain how a library works

Don't use for: one-off quick lookups ("what does this package do") — plain `web_search`/`web_extract` suffices. This skill is for producing a structured, reusable research summary.

## Workflow

### Phase 1: Surface Reconnaissance (5-10 min)

1. **GitHub README** — navigate to the project's GitHub repo. Read the README top-to-bottom for:
   - Elevator pitch ("what is X")
   - Architecture diagram / overview
   - Core concepts listed
   - Quickstart code snippets
   - Key features list

2. **Documentation map** — find the docs index efficiently:
   - **Mintlify-hosted docs**: hit `/llms.txt` (e.g. `https://docs.crewai.com/llms.txt`) — returns every page as a raw markdown URL
   - **Docusaurus/GitBook**: look for a `sidebar.json`, `sidebars.js`, or `_category_.json` pattern
   - **ReadTheDocs**: check the sitemap at `/sitemap.xml`
   - **Raw markdown repos**: check `/docs/` directory on GitHub

3. **Identify core concept pages** — from the index, pick:
   - Architecture / Overview
   - Core building blocks (agents, tasks, workflows, etc.)
   - Configuration / Installation
   - API reference (high-level)
   - Integration guides

### Phase 2: Deep Dive via Raw Markdown (fastest path)

Once you have the URLs of concept pages:

1. **Curl the raw markdown** — these are faster and more complete than browser navigation:
   ```bash
   curl -s https://docs.example.com/v1.0/en/concepts/agents.md | head -300
   ```
   - Mintlify pattern: `/v{version}/en/concepts/{topic}.md`
   - GitHub raw: `https://raw.githubusercontent.com/org/repo/main/docs/{topic}.md`

2. **Read sequentially by concept layer** — follow the dependency order:
   - Foundation: What is it? Architecture overview
   - Building blocks: Agents → Tasks → Workflows/Flows → Orchestration
   - Execution: Configuration, processes, state management
   - Extensions: Tools, plugins, integrations, MCP, APIs
   - Production: Observability, checkpointing, deployment, security

3. **Capture code snippets** — extract representative examples for each concept (they demonstrate API shape and usage patterns)

### Phase 3: Structured Synthesis

Organize findings into a report covering:

| Section | Content |
|---------|---------|
| **What it is** | One-paragraph elevator pitch, license, stars, version |
| **Architecture** | Tier/component diagram, data flow explanation |
| **Core Concepts** | Each building block with parameters and relationships |
| **Key Features** | Feature table with descriptions |
| **Use Cases** | 3-5 concrete scenarios with agent/task breakdown |
| **Integration Analysis** | How it maps to/complements Hermes Agent |

### Phase 4: Integration Analysis with Hermes

For integration analysis, evaluate:

| Hermes Capability | Framework Counterpart | Relationship |
|-------------------|----------------------|--------------|
| `terminal` tool | Any CLI | Hermes can spawn as subprocess |
| `web_search` | Search tools | Redundant — pick one |
| `browser` | Web scraping tools | Redundant — pick one |
| Skills (filesystem) | Skills/Knowledge | Complementary |
| Memory | Memory | Complementary |
| MCP tools | MCP support | Both support MCP |
| `computer_use` | — | Unique to Hermes |
| Autonomous orchestration | Flows/Workflows | Complementary layers |

Define 3-4 concrete integration approaches with a recommendation.

### Phase 5: Save Deliverable

Write the structured report to a file and present a concise summary to the user with:
- Architecture overview
- Core concept map
- Recommended integration approach
- Key decision factors

## Common Pitfalls

- **Don't rely on browser for plain markdown docs** — curl the raw markdown URLs directly (faster, no truncation, no JS dependencies)
- **Don't read pages in random order** — concepts build on each other; follow the dependency chain
- **Don't stop at the landing page** — Mintlify SPA apps often redirect all URLs to `/`; use `llms.txt` or direct versioned `/v{...}/en/...` paths
- **Don't over-invest in enterprise-only features** — flag them as "enterprise" but focus on open-source core
- **Don't fabricate integration points** — if there's no natural integration point, say so honestly

## Verification Checklist

- [ ] Output file answers: what problem does this framework solve?
- [ ] Output file covers: core building blocks and their relationships
- [ ] Output file lists: concrete use cases
- [ ] Output file states: how it can integrate with Hermes (or honestly says it can't)
- [ ] Every claim traces to an extracted page (see `grounded-citations` for citation ledger workflow)
