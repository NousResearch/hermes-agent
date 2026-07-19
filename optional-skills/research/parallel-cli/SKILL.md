---
name: parallel-cli
description: Optional Parallel web research, enrichment, and monitoring.
version: 1.2.0
author: Kshitij (kshitijk4poor), George Pickett (grp06), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: [parallel-cli]
metadata:
  hermes:
    tags: [Research, Web, Search, Deep-Research, Enrichment, CLI]
    related_skills: [duckduckgo-search, mcporter]
    requires_toolsets: [terminal]
---

# Parallel CLI Skill

Use `parallel-cli` through Hermes's `terminal` tool when the user explicitly requests Parallel or needs Parallel-specific research, enrichment, entity discovery, or monitoring. Prefer Hermes native `web_search` and `web_extract` for ordinary one-off lookups.

Parallel is an optional third-party hosted service with a free tier and paid usage. This skill provides agent-safe operating guidance; it is not a complete CLI reference.

## When to Use

Use this skill when:

- The user mentions Parallel or `parallel-cli`.
- The task needs structured enrichment, FindAll entity discovery, or recurring monitoring.
- A long research task should be launched asynchronously and resumed by ID.
- Search or extraction output should be saved for later use.

Do not select Parallel silently when Hermes native web tools are sufficient.

## Prerequisites

Run a preflight through `terminal`:

```bash
parallel-cli --version
parallel-cli auth --json
parallel-cli search --help
```

Replace `search` with the selected leaf command, such as `research run --help`. Installed `--help` is authoritative for execution. If a required command is unavailable, report the installed version and ask before installing or upgrading unless the user explicitly requested setup.

Install with one method appropriate to the active terminal backend:

| Method | Command |
|---|---|
| Homebrew | `brew install parallel-web/tap/parallel-cli` |
| uv | `uv tool install "parallel-web-tools[cli]"` |
| pipx | `pipx install "parallel-web-tools[cli]"` |

### Authentication

If `parallel-cli auth --json` reports `authenticated: false`, choose the flow that matches the runtime.

For an attended local session:

```bash
parallel-cli login
```

For headless or SSH sessions with a person available to authorize, start device login in the background:

```text
terminal(command="parallel-cli login --no-browser --json", background=true, notify_on_complete=true)
```

Poll the returned `session_id` with `process(action="poll", session_id="...")` until the `device_code` event appears, show the user its verification URL and code, and wait for authorization. Continue only after the process emits `auth_success` and `parallel-cli auth --json` reports `authenticated: true`. Kill the process if the user declines or the code expires.

For unattended automation, use `PARALLEL_API_KEY` instead of device login. Never print the key.

## How to Run

1. Run every `parallel-cli` command through `terminal` and inspect the exact leaf command's `--help` before using version-sensitive options.
2. Do not interpolate quote-heavy, multiline, or otherwise untrusted user text directly into shell syntax. Use `write_file`, then pass `-` with stdin redirection or use `--input-file` where supported.
3. Treat Parallel as an external data processor. Do not send credentials, private files, personal records, or sensitive datasets unless the user clearly intends that data to be processed by Parallel.
4. Use default processor tiers and result limits unless the user requests or approves a more expensive scope.
5. Use `--json` when Hermes must parse stdout and no output file is requested. Use `-o` for reusable results, then treat the saved file as authoritative and read it with `read_file`.
6. Use a writable path valid for the active terminal backend; create its parent when needed and do not assume `/tmp` exists.
7. Before using `-o`, `--output`, or `--target`, choose a new path unless the user explicitly approves overwriting an existing file.
8. For long jobs, use `--no-wait`, retain the returned identifiers by field name, and resume with the corresponding `status` or `poll` command.
9. Cite only URLs present in the returned Parallel payload.

## Quick Reference

| User need | Command | State to retain |
|---|---|---|
| Current web results | `search` | Saved JSON path when used |
| Content from known URLs | `extract` | Saved JSON path when used |
| Synthesized investigation | `research run` | `run_id`, `interaction_id` |
| Add fields to existing rows | `enrich run` | `taskgroup_id` |
| Fast company/person list | `findall entity-search` | `entity_set_id` |
| Comprehensive entity dataset | `findall run` | `findall_id` |
| Recurring change detection | `monitor` | `monitor_id`, `event_group_id`, `next_cursor` |

## Procedure

### 1. Search and Extract

Save reusable search results to JSON:

```text
parallel-cli search "Identify the latest React changes and primary sources" --max-results 10 -o react-search.json
```

After writing complex user-controlled text to `objective.txt`, use the active shell's stdin form:

```bash
parallel-cli search - -o user-search.json < objective.txt
```

```powershell
Get-Content -Raw objective.txt | parallel-cli search - -o user-search.json
```

Read the saved JSON and cite only returned URLs.

Extract known URLs when discovery is unnecessary:

```bash
parallel-cli extract https://example.com --objective "Find pricing and plan limits" -o extracted-page.json
```

Use `--full-content` only when excerpts are insufficient. Report extraction errors or empty results instead of fabricating content.

### 2. Research

Launch long research asynchronously:

```bash
parallel-cli research run "Compare leading AI coding agents by pricing and enterprise controls" --text --no-wait --json
```

Capture both returned fields:

- `run_id` controls `research status` and `research poll`.
- `interaction_id` carries context into a later research or enrichment request.

```bash
parallel-cli research status RUN_ID --json
parallel-cli research poll RUN_ID --timeout 540 -o research-report
parallel-cli research run "Drill into the top result" --previous-interaction-id INTERACTION_ID --no-wait --json
```

Replace `RUN_ID` and `INTERACTION_ID` with the returned values. A text-schema poll writes JSON metadata and a Markdown report; verify both files before reporting completion. If processor choice matters, inspect `parallel-cli research processors --json` rather than relying on a memorized tier list.

### 3. Enrich

Inspect the input row count and confirm the requested output fields and processor before launch. Prefer file input and explicit columns over an open-ended `--intent`:

```bash
parallel-cli enrich run --source-type json --source companies.json --target enriched.json --source-columns '[{"name":"company","description":"Company name"}]' --enriched-columns '[{"name":"headquarters","description":"Company headquarters"},{"name":"employee_count","description":"Latest employee count"}]' --processor core-fast --no-wait --json
```

Capture `taskgroup_id`, then check or poll that task group:

```bash
parallel-cli enrich status TASKGROUP_ID --json
parallel-cli enrich poll TASKGROUP_ID --timeout 540 -o enriched-results.json
```

In the async form above, `--target` is not written at launch; `enrich poll -o` saves the final results.

Use a prior research `interaction_id` only with `--previous-interaction-id`. Retain a new context ID only when the current command actually returns one.

### 4. FindAll

Use entity search for a fast, best-effort company or person list:

```bash
parallel-cli findall entity-search "AI startups in healthcare" --entity-type companies -o healthcare-ai-entities.json
```

Entity-search results are not individually verified, and `entity_set_id` is not a full FindAll run ID.

Use a full run for comprehensive discovery, match evaluation, or exclusions:

```bash
parallel-cli findall run "Find AI coding agent startups with enterprise offerings" --no-wait --json
parallel-cli findall status FINDALL_ID --json
parallel-cli findall poll FINDALL_ID --timeout 540 -o findall-results.json
```

Use the returned `findall_id` for `status` and `poll`. Validate falsifiable criteria and obvious placeholder entities before presenting results.

### 5. Monitor

Create a monitor only when the user wants recurring tracking and the query, frequency, and delivery behavior are clear:

```bash
parallel-cli monitor create "Track Tesla SEC filings" --frequency 1d --json
parallel-cli monitor get MONITOR_ID --json
```

Use read operations without mutating state:

```bash
parallel-cli monitor list --json
parallel-cli monitor events MONITOR_ID --json
parallel-cli monitor events MONITOR_ID --event-group-id EVENT_GROUP_ID --json
```

Events are newest-first. Reuse `next_cursor` with `--cursor`; `--event-group-id` selects one execution instead of a page.

Run `update` or `trigger` only when the user has requested that exact side-effecting action. Confirm the exact monitor before irreversible cancellation:

- Update: `parallel-cli monitor update MONITOR_ID --frequency 1w --json`
- Trigger: `parallel-cli monitor trigger MONITOR_ID --json`
- Cancel after confirmation: `parallel-cli monitor cancel MONITOR_ID --json`

Verify creation or updates with `monitor get`. Monitor queries and snapshot task-run IDs are immutable; create a new monitor to change them.

## Pitfalls

- Do not combine `--json` and `-o` when a downstream parser expects stdout to contain only JSON; read the saved file instead.
- Do not confuse execution IDs with context IDs, or reuse an identifier with the wrong command family.
- Do not report asynchronous results before `status` or `poll` confirms completion.
- A poll timeout does not cancel server work; retain the ID and resume the same job instead of launching a duplicate.
- Do not treat `entity-search` as comprehensive or reuse its `entity_set_id` with full FindAll commands.
- Monitor cancellation is irreversible; create a new monitor to resume tracking.

## Verification

Verify authentication before a network operation:

```bash
parallel-cli auth --json
```

- [ ] Authentication reports `authenticated: true` before a network operation.
- [ ] The selected command exits successfully, or its failure is reported accurately.
- [ ] Every expected JSON file parses; text-schema research also produced non-empty Markdown.
- [ ] Every asynchronous identifier is retained under its returned field name and used with the matching command family.
- [ ] Every citation URL appears in the returned Parallel payload.
- [ ] Persistent Monitor actions match explicit user intent, and cancellation had confirmation.
