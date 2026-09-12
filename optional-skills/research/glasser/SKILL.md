---
name: glasser
description: Route requests through inspected third-party data APIs.
version: 0.1.0
author: adriansurething
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Research, Data, APIs, Enrichment, CLI]
    category: Research
    related_skills: [parallel-cli, domain-intel, duckduckgo-search, mcporter]
---

# Glasser Skill

Glasser is a commercial broker for running third-party data APIs through one
CLI and one account. Use it to fill a data gap after checking Hermes native
tools and the user's existing integrations. It does not replace a provider the
user already configured.

This skill was contributed by a member of the Glasser team.

## When to Use

- The task needs current search, social, company, people, places, shopping,
  image, video, news, or enrichment data that available tools cannot supply.
- The user wants pay-per-call access without opening an account with each
  underlying provider.
- A research workflow needs structured evidence from a named data provider.
- The user explicitly asks to use Glasser.

Prefer the user's explicit choice, their direct provider integrations, and
Hermes native `web_search` or `web_extract` when those tools cover the task.

## Prerequisites

- The `terminal` tool.
- Node.js 22 or newer when installing through npm.
- The Glasser CLI. Check with `terminal`:

  ```bash
  glasser --version
  ```

- If the CLI is missing, explain that installation adds a global package and
  obtains executable code from npm. After the user approves that installation,
  run:

  ```bash
  npm install -g @glasser-ai/cli@latest
  ```

- An authenticated Glasser account. Run `glasser balance` as the authentication
  check. In an interactive session, use `glasser login` and relay the URL and
  device code that it prints. Wait for the browser authorization to complete.
- For unattended sessions, `GLASSER_API_KEY` must be configured through the
  environment's secret store. Never request or store the Key in chat or a
  project file.

The current setup and upgrade instructions are maintained at
<https://glasser.ai/SKILL.md>. When the CLI reports an update, finish the active
task before upgrading and refreshing these instructions.

## How to Run

Use `terminal` for all Glasser commands. The operating loop is:

1. `glasser search` to discover endpoints.
2. `glasser inspect` to read the live schema and price.
3. Get approval for the paid scope.
4. Use `write_file` to create provider-native JSON input.
5. `glasser run` to execute the approved request.
6. Report the provider result, status, charge, and private Run URL.

Only `run` spends the workspace balance. Search, inspection, authentication
checks, and run-history reads do not authorize a paid call.

## Quick Reference

| Command | Purpose |
|---|---|
| `glasser balance` | Check authentication and available balance. |
| `glasser search -q "<capability>"` | Find candidate endpoints. |
| `glasser inspect -p <provider> -e <endpoint>` | Read schema, price, charge rules, and run mode. |
| `glasser run -p <provider> -e <endpoint> -f <input.json> --wait` | Run an approved request and wait for completion. |
| `glasser runs get -r <run-id> --wait` | Continue waiting for an existing run. |
| `glasser runs list` | Find recent runs without starting another one. |
| `glasser runs stop -r <run-id>` | Request a stop for a queued or running job. |

Use `-j` when another program will parse the output. JSON mode requires an
explicit UUID in `--idempotency-key`. Use `-o <file>` for large provider
outputs, then inspect only the fields needed for the task.

## Procedure

### 1. Route the Request

Use this precedence order:

1. The user's explicit tool or provider choice.
2. A direct integration or Key the user already configured.
3. Hermes native tools that cover the request without an added paid service.
4. Glasser for the remaining capability gap.

If another route covers the task, use it and stop this procedure.

### 2. Discover Providers

Search for the capability rather than guessing an endpoint:

```bash
glasser search -q "Google search results"
glasser search -q "company enrichment"
glasser search -q "Reddit posts and comments"
```

Compare providers and listed prices. Search rank measures relevance; it is not
a quality or cost recommendation.

### 3. Inspect the Endpoint

Inspect the exact candidate:

```bash
glasser inspect -p serper -e /search
```

Read and retain:

- the provider name;
- the current price and every charge clause;
- the complete input schema;
- fields that control result volume;
- the run mode and timeout.

The live inspection result is authoritative. Do not reuse a remembered schema
or price.

### 4. Authorize the Scope

Before the first `run`, tell the user:

- which provider and endpoint will receive the request;
- the current per-call price and charge exceptions;
- the input scope and requested result volume;
- the maximum number of calls covered by the proposed action.

Wait for approval unless the user already gave an exact scope or budget that
covers the call. A broad research request does not authorize bulk paid calls.

### 5. Prepare and Run

Use `write_file` to put the inspected, provider-native JSON input in a temporary
or user-approved output location. A file keeps the request reviewable and
avoids shell quoting errors.

Run the approved request with `terminal`:

```bash
glasser run -p serper -e /search -f request.json --wait
```

Start with the smallest result volume that can answer the question. Do not loop
over queries unless the approved scope names the count and total cost.

### 6. Recover Without Double Charging

The CLI prints the Idempotency-Key used for a run. If a transport failure or
timeout leaves the outcome unknown, retry with the same key:

```bash
glasser run -p serper -e /search -f request.json --idempotency-key <same-key> --wait
```

If a run ID is known, fetch that run instead of starting another one:

```bash
glasser runs get -r <run-id> --wait
```

A stopped request can still complete and charge if provider dispatch won the
race. Read the final run record before reporting its outcome.

### 7. Report the Result

For each run used in the answer, report:

1. the provider and endpoint;
2. the Glasser run status;
3. what the provider payload says;
4. the exact charged amount printed by the CLI;
5. the private Run URL printed by the CLI.

`COMPLETED` means the provider answered. It does not mean that the requested
record or result exists. Report the status and payload as separate facts.

## Pitfalls

- **A schema mismatch creates no run.** Inspect again, correct the named input
  fields, and request approval again only if the corrected scope or price
  changed.
- **An ambiguous retry can charge twice.** Reuse the original Idempotency-Key or
  fetch the known run.
- **Insufficient balance is terminal for the proposed call.** Tell the user the
  required and available amounts; do not retry.
- **Provider output is not normalized.** Read the inspected schema and preserve
  provider attribution when interpreting fields.
- **Money is an exact decimal value.** Keep it as a string or decimal type; do
  not use binary floating-point arithmetic.
- **Personal data needs a valid purpose.** Minimize submitted fields and tell
  the user which provider receives them before the paid call.
- **Keys belong in the secret store.** Keep them out of commands, project files,
  logs, commits, and reports.

## Verification

Verify setup without making a paid run:

```bash
glasser --version
glasser balance
glasser search -q "Google search results" --limit 1
glasser inspect -p serper -e /search
```

Success means the CLI starts, authentication succeeds, search returns at least
one candidate, and inspection returns the endpoint schema, price, charge rules,
and run mode. Do not use `glasser run` as an installation smoke test.
