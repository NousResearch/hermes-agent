---
name: printing-press
description: Use when turning an API, website, OpenAPI/GraphQL spec, HAR capture, or repeated service workflow into an agent-native CLI/MCP server using mvanhorn/cli-printing-press. Preflight the binary, run the lean research -> generate -> shipcheck loop, and fall back to Hermes-native skill/CLI design when the toolchain is unavailable.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [printing-press, cli, mcp, api, openapi, skills, automation]
    homepage: https://github.com/mvanhorn/cli-printing-press
    related_skills: [agent-native-cli-factory, service-to-skill-prompting, hermes-agent-skill-authoring, claude-code, codex]
---

# CLI Printing Press

## Overview

Use this skill as the Hermes bridge for [mvanhorn/cli-printing-press](https://github.com/mvanhorn/cli-printing-press): a Claude-Code-oriented toolchain that turns APIs, websites, API docs, OpenAPI/GraphQL specs, or HAR captures into agent-native Go CLIs and MCP servers.

Its core idea is that every API has a **secret identity**: the thing the service really is from an agent's point of view. A good printed CLI is not just endpoint wrappers. It should expose compact, low-token, compound commands that agents can use repeatedly without hunting through docs, copying context, or making many brittle browser/API calls.

The upstream project can generate:

- A Go CLI, usually named `<service>-pp-cli`
- An MCP server, usually named `<service>-pp-mcp`
- Claude Code skills for the generated capability
- Research briefs, proofs, scorecards, and provenance manifests
- Local SQLite-backed sync/search/workflow commands when useful

## When to Use

Use this skill when the user wants to:

- "print" a CLI for an API, SaaS service, website, or workflow
- convert an OpenAPI spec, GraphQL schema, API docs URL, or HAR capture into a reusable tool
- compress a repeated browser/API task into a deterministic command
- generate an MCP server for an API/service
- evaluate whether an existing API wrapper is agent-native enough
- use the specific `mvanhorn/cli-printing-press` repository or `/printing-press` skill

Do not use this skill when:

- the task is a one-off lookup that a direct API/browser call can finish cheaply
- the service has no stable interface, no useful repeated workflow, or no access
- the user only needs a prose summary or a normal Hermes skill, not a CLI/MCP artifact
- installing/running Go or Claude Code is out of scope for the current environment

If the workflow decision is unclear, load `adaptive-workflow-routing` first. If the correct output is a hand-authored Hermes skill or tiny bespoke wrapper rather than the upstream Printing Press, use `agent-native-cli-factory` and `service-to-skill-prompting` instead.

## Upstream Repository Facts

Canonical upstream:

```text
https://github.com/mvanhorn/cli-printing-press
```

Observed repository shape:

```text
cmd/printing-press/        Go binary entry point
internal/openapi/          OpenAPI parser
internal/graphql/          GraphQL parser
internal/docspec/          docs-to-spec generation
internal/generator/        Go template renderer
internal/profiler/         API archetype / feature recommender
internal/pipeline/         resumable pipeline and ship checks
internal/vision/           feature scoring model
catalog/                   known API catalog entries
skills/printing-press/     main Claude Code skill
```

The upstream skill set includes:

- `printing-press` — main generation skill
- `printing-press-polish` — improve and fix an existing printed CLI
- `printing-press-publish` — package/publish a CLI
- `printing-press-reprint` — regenerate an existing CLI with newer machinery
- `printing-press-score` — score generated output
- `printing-press-output-review` — review output quality
- `printing-press-import` — import/adapt existing outputs
- `printing-press-retro` — retrospective improvement loop
- `printing-press-catalog` — deprecated; main skill handles catalog checks

## Preflight

Before promising to run Printing Press, check local capability. Do not assume it is installed.

```bash
command -v printing-press || true
printing-press --version 2>/dev/null || true
go version || true
command -v claude || true
```

Interpretation:

- If `printing-press` exists and version is compatible with the upstream skill, use it.
- If Go is missing, do not try `go install` until the user has accepted installing toolchain dependencies.
- If Claude Code is missing or unreliable, prefer the raw `printing-press` binary when possible, or use the Hermes-native fallback below.
- If the current environment is James's bob-prime and Go is absent, state that the skill is installed but the binary/toolchain needs setup before real printing runs.

Install command when Go is available and installation is in scope:

```bash
go install github.com/mvanhorn/cli-printing-press/v4/cmd/printing-press@latest
```

Then verify:

```bash
printing-press --version
printing-press --help
```

## Input Routing

Choose the strongest available source:

1. **OpenAPI / Swagger URL or file** — best deterministic substrate.
2. **GraphQL SDL/schema** — good if complete and auth patterns are clear.
3. **Official API docs URL** — use docs-to-spec flow; verify generated assumptions.
4. **HAR capture** — useful for private/sniffed browser APIs; redact secrets first.
5. **Website with no docs** — browser-sniff only if permitted and safe.
6. **Service name only** — let catalog/discovery resolve a likely spec, then verify.

Never persist raw secret values, cookies, bearer tokens, session IDs, or unredacted HARs in generated artifacts, manuscripts, READMEs, or commits. Store only env var names and placeholders.

## Standard Hermes Workflow

### 1) Define the secret identity

Before generation, write a one-paragraph answer:

```text
<Service> is not just <surface description>; for agents it is <secret identity>.
The CLI should optimize for <top repeated workflows>.
```

Examples:

- A chat platform may be a searchable knowledge base.
- A project tracker may be a dependency graph and stale-work detector.
- A payments API may be a reconciliation and anomaly investigation tool.
- A travel site may be a constrained option search engine.

### 2) Choose the smallest useful contract

Identify the first printed contract:

- `list/get/create/update/delete` wrappers only if the API truly needs them
- `sync` into local SQLite when repeated reads/searches matter
- `search` for local/offline querying
- `stale`, `blocked`, `health`, `similar`, `timeline`, `diff`, or other domain commands when the secret identity demands them
- `--json` on commands agents will compose
- compact text output for Telegram/CLI humans

### 3) Run Printing Press when available

With Claude Code skills installed, upstream expects commands like:

```text
/printing-press Notion
/printing-press Discord codex
/printing-press --spec ./openapi.yaml
/printing-press --har ./capture.har --name MyAPI
/printing-press https://postman.com/explore
```

When driving from Hermes, prefer concrete shell commands where the binary supports them, and use Claude Code only for bounded work where reliability is proven. Keep the run scoped and require artifacts as output.

### 4) Shipcheck before declaring success

A printed CLI is not done because it builds. Run or request the upstream shipcheck equivalents:

- build/test
- dogfood/behavioral checks against representative targets
- verify every generated command, especially `--json`
- scorecard
- README/SKILL honesty review
- secret scan of generated artifacts

Fix small dogfood failures before ship. Do not bury obvious bugs as future work if they are 1-3 file edits.

### 5) Publish or store deliberately

Default output locations in upstream docs are under:

```text
~/printing-press/.runstate/
~/printing-press/library/
~/printing-press/manuscripts/
```

If integrating into James's workspace, place stable wrappers under the appropriate project repo, not scattered temp directories. Do not commit bulky runstate, raw captures, build products, or secrets.

## Hermes-Native Fallback

If the upstream toolchain is unavailable, still use the Printing Press design pattern:

1. Load `agent-native-cli-factory` and `service-to-skill-prompting`.
2. Identify the repeated workflow and secret identity.
3. Write a small bespoke CLI or script with a narrow contract.
4. Add a Hermes skill that documents when and how to use it.
5. Verify on the original stuck case.
6. Record pitfalls in the skill, not in memory.

Fallback output can be a Hermes skill alone when code generation would be overkill.

## Quality Bar for Generated CLIs

A good printed CLI should:

- Have a small number of strong commands rather than hundreds of thin wrappers.
- Prefer stable flags and explicit env var names for auth.
- Emit compact `--json` for agent composition.
- Avoid interactive prompts in default agent flows.
- Include `sync`/SQLite only when repeated local queries are actually useful.
- Include domain-specific commands that match the secret identity.
- Provide deterministic errors with actionable messages.
- Include a README with real examples and honest limitations.
- Include a generated skill/MCP description only if those surfaces are accurate.

## Common Pitfalls

1. **Endpoint wrapper trap.** A CLI that exposes every endpoint one-for-one is usually not agent-native. Add compound commands around the actual repeated questions.

2. **Skipping behavioral tests.** `go build` proves compilation, not usefulness. Test commands against representative examples and error paths.

3. **Leaking secrets through HARs.** HAR/browser captures can contain cookies, bearer tokens, and PII. Redact before archiving or committing.

4. **Over-documenting before code exists.** Use the lean loop: brief -> generate -> fix high-value gaps -> shipcheck. Avoid phase theater.

5. **Treating Claude Code as mandatory.** The upstream workflow is Claude-Code-oriented, but Hermes can use the binary, generated artifacts, or fallback design pattern directly.

6. **Installing toolchains without scope.** Go/Claude Code installation changes the machine. Check preflight and only install when the user's request clearly includes setup or they approve it.

7. **Publishing generated artifacts blindly.** Generated READMEs/skills can overclaim. Review semantic honesty before sharing or committing.

## Verification Checklist

- [ ] Preflight checked `printing-press`, `go`, and optional `claude`
- [ ] Secret identity and narrow CLI contract are stated
- [ ] Best available source selected: spec, schema, docs, HAR, website, or catalog
- [ ] Secrets/PII handling is explicit before captures or archives
- [ ] Generated or fallback CLI builds and exposes help
- [ ] Representative commands, `--json`, and error paths were dogfooded
- [ ] Scorecard/verification results are recorded when using upstream Printing Press
- [ ] README/SKILL/MCP descriptions are semantically honest
- [ ] Bulky generated state and raw captures are not committed

## Quick Commands

Inspect upstream:

```bash
gh repo view mvanhorn/cli-printing-press --web
```

Install binary when Go is available:

```bash
go install github.com/mvanhorn/cli-printing-press/v4/cmd/printing-press@latest
```

Start upstream Claude Code workflow from a clone:

```bash
git clone https://github.com/mvanhorn/cli-printing-press.git
cd cli-printing-press
claude --plugin-dir .
```

Then in Claude Code:

```text
/printing-press <service-or-spec>
```

Hermes fallback prompt shape:

```text
Use printing-press design: identify the service's secret identity, choose the narrowest useful CLI/MCP/skill contract, build the smallest deterministic wrapper, and verify it on the original repeated workflow.
```
