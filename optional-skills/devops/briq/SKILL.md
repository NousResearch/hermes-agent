---
name: briq
description: Sandboxes and real services on microVMs, upstream-maintained.
version: 1.0.0
author: briq (briq-cloud)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [sandbox, postgres, database, mcp, preview, microvm, cloud]
    category: devops
    related_skills: [docker-management]
    upstream:
      repo: briq-cloud/briq-mcp
      path: skills/briq
---

# briq (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [briq-cloud/briq-mcp](https://github.com/briq-cloud/briq-mcp): the project
> ships the skill as a single `SKILL.md` under `skills/briq/`. `hermes skills
> install official/devops/briq` pulls the current file live from that repo
> (quarantined and scanned like any hub install) — this directory holds only
> the catalog metadata, so the vendored copy can never go stale.

briq runs services and sandboxes for agents, each on its own microVM, through
one MCP server (`https://api.briq.cloud/mcp`). A sandbox is a microVM with
Python, Node, git and a shell, for scripts, tests and builds that should not
run on the user's machine. A service is Postgres, MySQL, Redis, MongoDB, a
headless Chrome, an app from a GitHub repo or any Docker image, with connection
strings and HTTPS preview URLs. The skill tells the agent when to reach for
briq, how to connect it (including the device-code sign-in for an agent on a
server or in a chat app), and how to keep spending in check.

## Prerequisites

- A briq account; the user signs in once and sets the agent's budget. In
  Hermes: `hermes mcp add briq --url https://api.briq.cloud/mcp --auth oauth`,
  then `hermes mcp login briq --flow device` (it prints a link and a code to
  approve from any device).
- For scripts and CI without MCP, the CLI: `npm install -g @briq-cloud/cli`
  (Node 22.6 or later).

Full documentation: https://briq.cloud/docs/setup/agents
