---
name: parallel-cli
description: Use Parallel for web research, enrichment, and monitoring.
version: 1.2.0
author: George Pickett (@grp06), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Research, Web, Search, Deep-Research, Enrichment, CLI]
    related_skills: [duckduckgo-search, mcporter]
---

# Parallel CLI Skill

Use `parallel-cli` through Hermes's `terminal` tool when the user explicitly requests Parallel or needs Parallel-specific research, enrichment, entity discovery, or monitoring. For ordinary one-off lookups, prefer Hermes native `web_search` and `web_extract`.

Parallel is a hosted service with a free tier and paid usage. This skill covers agent-safe, non-interactive CLI workflows; it does not replace Hermes's native web tools.

## When to Use

Use this skill when:

- The user mentions Parallel or `parallel-cli`.
- The task needs structured enrichment, FindAll entity discovery, or recurring monitors.
- A long-running research job should be launched asynchronously and resumed by ID.
- Search or extraction results need to be saved as authoritative JSON for later use.

Do not use it by default for simple web lookups when Hermes native tools are sufficient.

## Prerequisites

Run the preflight through `terminal`:

```bash
parallel-cli --version
parallel-cli auth --json
parallel-cli <command-group> --help
```

The commands below are verified against `parallel-cli` 0.7.1. If a documented command is missing, identify the install method and upgrade through that same method before retrying.

Install with one method appropriate to the active terminal backend:

| Method | Command |
|---|---|
| Homebrew | `brew install parallel-web/tap/parallel-cli` |
| npm | `npm install -g parallel-web-cli` |
| uv | `uv tool install "parallel-web-tools[cli]"` |
| pipx | `pipx install "parallel-web-tools[cli]"` |
| pip | `pip install "parallel-web-tools[cli]"` |

Linux and macOS also support the standalone installer:

```bash
curl -fsSL https://parallel.ai/install.sh | bash
```

### Authentication

If `parallel-cli auth --json` reports `authenticated: false`, choose the flow that matches the runtime.

For an attended local session:

```bash
parallel-cli login
```

For headless or SSH sessions with a person available to authorize, start the device flow with:

```text
terminal(command="parallel-cli login --no-browser --json", background=true, notify_on_complete=true)
```

Then:

1. Call `process(action="poll", session_id="...")` until the `device_code` event appears.
2. Show the user `verification_uri_complete`, or the verification URL plus `user_code`.
3. Wait for the user to authorize.
4. Poll or wait on the same process until it emits `auth_success`.
5. Re-run `parallel-cli auth --json`; continue only when `authenticated` is `true`.

The device flow does not require a PTY. For unattended CI or automation, set `PARALLEL_API_KEY` securely instead of starting device login. Never print the key.

## How to Run

1. Use `terminal` for every `parallel-cli` command.
2. Prefer `--json` when Hermes must parse stdout.
3. Use `-o` for large or reusable results, then read the saved file instead of relying on possibly truncated terminal output.
4. Choose a writable output path valid for the active terminal backend; do not assume a Unix temp directory exists.
5. Use `--no-wait` for long jobs, capture the returned identifier, and resume with `status` or `poll`.
6. Cite only URLs present in the returned Parallel payload.

## Quick Reference

| User need | Command | State to retain |
|---|---|---|
| Current fact or webpage results | `search` | Saved JSON path when used |
| Content from known URLs | `extract` | Saved JSON path when used |
| Synthesized investigation | `research run` | `run_id`, `interaction_id` |
| Add fields to existing rows | `enrich run` | `taskgroup_id` |
| Fast company/person list | `findall entity-search` | `entity_set_id` |
| Comprehensive entity dataset | `findall run` | `findall_id` |
| Recurring change detection | `monitor` | `monitor_id`, `event_group_id`, `next_cursor` |

Common flags:

- `--json`: print machine-readable output.
- `-o` / `--output`: save authoritative output to a file.
- `--no-wait`: return immediately after creating a long-running job.
- `--previous-interaction-id <interaction_id>`: reuse prior research context in a research or enrichment request.

## Procedure

### Search

Use a natural-language objective for intent and repeated `-q` flags for important literal queries:

```bash
parallel-cli search \
  "Identify the latest React 19 changes and primary sources" \
  -q "React 19 release notes" \
  -q "React 19 migration" \
  --mode basic \
  --max-results 10 \
  --excerpt-max-chars-total 27000 \
  --json \
  -o react-19-search.json
```

Use `turbo` for simple, latency-sensitive English or Japanese lookups, `basic` for the default balance, and `advanced` for harder multi-step searches. Add `--location us` for geography-sensitive results, `--after-date YYYY-MM-DD` for recency, or include/exclude domain filters when source scope matters.

Read the saved JSON and extract each result's title, URL, date, and useful excerpts. Skip navigation noise and never invent a citation.

### Extract

Use extraction when the URL is already known:

```bash
parallel-cli extract https://example.com \
  --objective "Find pricing and plan limits" \
  -q "pricing" \
  --json \
  -o extracted-page.json
```

Use `--full-content` for long articles or PDFs when excerpts are insufficient. If the response contains errors or no results, report the failure rather than fabricating content.

### Research

Launch long research asynchronously:

```bash
parallel-cli research run \
  "Compare leading AI coding agents by pricing, model support, and enterprise controls" \
  --processor pro-fast \
  --text \
  --no-wait \
  --json
```

Capture both returned fields:

- `run_id` controls `status` and `poll`.
- `interaction_id` carries context into a later research or enrichment request.

```bash
parallel-cli research status <run_id> --json
parallel-cli research poll <run_id> --timeout 540 -o research-report
parallel-cli research run "Drill into the top result" \
  --previous-interaction-id <interaction_id> \
  --no-wait \
  --json
```

`research poll -o research-report` writes `research-report.json` and also writes `research-report.md` when the task used `--text`. If polling times out, the server-side task continues; reuse the same `run_id`.

### Enrich

Use enrichment when the input entities already exist. Ask for suggested columns only when the user's desired output is unclear:

```bash
parallel-cli enrich suggest "Find CEO and recent funding" --json
```

Start a non-interactive inline run:

```bash
parallel-cli enrich run \
  --data '[{"company":"Anthropic"},{"company":"Mistral"}]' \
  --intent "Find headquarters and employee count" \
  --target enriched.csv \
  --no-wait \
  --json
```

Capture `taskgroup_id`, then poll to a JSON file:

```bash
parallel-cli enrich status <taskgroup_id> --json
parallel-cli enrich poll <taskgroup_id> --timeout 540 -o enriched-results.json
```

For files, replace `--data` with `--source-type`, `--source`, and explicit `--source-columns`. A prior research `interaction_id` may be passed with `--previous-interaction-id`; enrichment does not return a new interaction ID.

### FindAll

Use synchronous entity search only for a fast, best-effort company or person list:

```bash
parallel-cli findall entity-search \
  "AI startups in healthcare" \
  --entity-type companies \
  --match-limit 25 \
  -o healthcare-ai-entities.json
```

Capture `entity_set_id`. Entity-search results are not individually verified and the ID cannot be used with full FindAll enrichment or extension.

Use a full run for comprehensive discovery, match evaluation, exclusions, enrichment, or entity types beyond companies and people:

```bash
parallel-cli findall run \
  "Find AI coding agent startups with enterprise offerings" \
  --generator core \
  --match-limit 25 \
  --no-wait \
  --json

parallel-cli findall status <findall_id> --json
parallel-cli findall poll <findall_id> --timeout 540 -o findall-results.json
```

Capture `findall_id`. Filter obvious placeholder entities and validate falsifiable criteria before presenting results.

### Monitor

Confirm the query, frequency, and webhook before creating a persistent monitor:

```bash
parallel-cli monitor create "Track Tesla SEC filings" --frequency 1d --json
```

Capture `monitor_id` and verify creation with `get` rather than scanning the list:

```bash
parallel-cli monitor get <monitor_id> --json
parallel-cli monitor events <monitor_id> --json
parallel-cli monitor events <monitor_id> --event-group-id <event_group_id> --json
parallel-cli monitor update <monitor_id> --frequency 1w --json
parallel-cli monitor trigger <monitor_id> --json
parallel-cli monitor cancel <monitor_id> --json
```

Events are newest-first. When a response includes `next_cursor`, pass it back with `--cursor`; pagination is ignored when `--event-group-id` is set. Monitor queries and snapshot task-run IDs are immutable, and `trigger` emits an event only when it detects a material change. Always obtain explicit user confirmation before `cancel`, which is irreversible.

## Pitfalls

- Do not silently choose Parallel when Hermes native web tools are sufficient.
- Do not omit `--json` when Hermes must parse stdout.
- Do not rely on large terminal stdout when the command supports `-o`.
- Do not confuse `run_id` with `interaction_id`, or use a generic research ID for FindAll.
- Do not use `--previous-interaction-id` outside research and enrichment.
- Do not treat `entity-search` as comprehensive or assume its `entity_set_id` supports full FindAll operations.
- Do not start device login when no person is available to authorize it; use `PARALLEL_API_KEY` for unattended work.
- Do not cancel a monitor without explicit confirmation.
- Exit codes are `0` success, `2` bad input, `3` authentication, `4` API error, and `5` timeout.

## Verification

- [ ] `parallel-cli auth --json` reports `authenticated: true`.
- [ ] The selected command exits successfully.
- [ ] Any requested output file exists and contains valid JSON.
- [ ] Every asynchronous identifier is captured under its exact field name.
- [ ] `status` or `poll` confirms completion before final results are reported.
- [ ] Every citation URL appears in the returned Parallel payload.
- [ ] A created or updated monitor is confirmed with `monitor get <monitor_id> --json`.
- [ ] Any irreversible monitor cancellation had explicit user confirmation.
