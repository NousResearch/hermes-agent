---
name: parallel-cli
description: Optional vendor skill for Parallel CLI — agent-native web search, extraction, deep research, enrichment, FindAll, and monitoring. Prefer JSON output and non-interactive flows.
version: 1.1.1
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Research, Web, Search, Deep-Research, Enrichment, CLI]
    related_skills: [duckduckgo-search, mcporter]
---

# Parallel CLI

Use `parallel-cli` when the user explicitly wants Parallel, or when a terminal-native workflow would benefit from Parallel's vendor-specific stack for web search, extraction, deep research, enrichment, entity discovery, or monitoring.

This is an optional third-party workflow, not a Hermes core capability.

Important expectations:
- Parallel is a paid service with a free tier, not a fully free local tool.
- It overlaps with Hermes native `web_search` / `web_extract`, so do not prefer it by default for ordinary lookups.
- Prefer this skill when the user mentions Parallel specifically or needs capabilities like Parallel's enrichment, FindAll, or monitor workflows.

`parallel-cli` is designed for agents:
- JSON output via `--json`
- Non-interactive command execution
- Async long-running jobs with `--no-wait`, `status`, and `poll`
- Research and enrichment context chaining with `--previous-interaction-id`
- Search, extract, research, enrichment, entity discovery, and monitoring in one CLI

## When to use it

Prefer this skill when:
- The user explicitly mentions Parallel or `parallel-cli`
- The task needs richer workflows than a simple one-shot search/extract pass
- You need async deep research jobs that can be launched and polled later
- You need structured enrichment, FindAll entity discovery, or monitoring

Prefer Hermes native `web_search` / `web_extract` for quick one-off lookups when Parallel is not specifically requested.

## Installation

Try the least invasive install path available for the environment.

### Homebrew

```bash
brew install parallel-web/tap/parallel-cli
```

### npm

```bash
npm install -g parallel-web-cli
```

### Python package

```bash
pip install "parallel-web-tools[cli]"
```

### Standalone installer

```bash
curl -fsSL https://parallel.ai/install.sh | bash
```

If you want an isolated Python install, `pipx` can also work:

```bash
pipx install "parallel-web-tools[cli]"
pipx ensurepath
```

The commands below are verified against `parallel-cli` 0.7.1. If a command or option is missing, upgrade through the same installation method used above before retrying.

Standalone, Homebrew, and npm installs support direct enrichment arguments, but not YAML configs, `enrich plan`, or deployment commands. Those features require the appropriate Python extras.

## Authentication

Interactive login:

```bash
parallel-cli login
```

Headless / SSH / CI:

```bash
parallel-cli login --no-browser --json
```

API key environment variable:

```bash
export PARALLEL_API_KEY="***"
```

Verify current auth status:

```bash
parallel-cli auth
```

## Core rule set

1. Always prefer `--json` when you need machine-readable output.
2. Prefer explicit arguments and non-interactive flows.
3. For long-running jobs, use `--no-wait` and then `status` / `poll`.
4. Cite only URLs returned by the CLI output.
5. Save large JSON outputs to a temp file when follow-up questions are likely.
6. Use background processes only for genuinely long-running workflows; otherwise run in foreground.
7. Prefer Hermes native tools unless the user wants Parallel specifically or needs Parallel-only workflows.

## Quick reference

```text
parallel-cli
├── auth
├── login
├── logout
├── search
├── extract / fetch
├── research run|status|poll|processors
├── enrich run|status|poll|suggest (+ plan/deploy with Python extras)
├── findall run|entity-search|ingest|status|poll|result|enrich|extend|schema|cancel
└── monitor create|list|get|update|cancel|events|trigger
```

## Common flags and patterns

Commonly useful flags:
- `--json` for structured output
- `--no-wait` for async jobs
- `--previous-interaction-id <id>` for research or enrichment follow-ups that reuse earlier context
- `--max-results <n>` for search result count
- `--mode turbo|basic|advanced` for search behavior
- `--include-domains domain1.com,domain2.com`
- `--exclude-domains domain1.com,domain2.com`
- `--after-date YYYY-MM-DD`

Read from stdin when convenient:

```bash
echo "What is the latest funding for Anthropic?" | parallel-cli search - --json
echo "Research question" | parallel-cli research run - --json
```

## Search

Use for current web lookups with structured results.

```bash
parallel-cli search "What is Anthropic's latest AI model?" --json
parallel-cli search "SEC filings for Apple" --include-domains sec.gov --json
parallel-cli search "bitcoin price" --after-date 2026-01-01 --max-results 10 --json
parallel-cli search "latest browser benchmarks" --mode turbo --json
parallel-cli search "AI coding agent enterprise reviews" --mode advanced --json
```

Use `turbo` for the fastest pass, `basic` for the default balance, and `advanced` for deeper searches.

Useful constraints:
- `--include-domains` to narrow trusted sources
- `--exclude-domains` to strip noisy domains
- `--after-date` for recency filtering
- `--max-results` when you need broader coverage

If you expect follow-up questions, save output:

```bash
parallel-cli search "latest React 19 changes" --json -o /tmp/react-19-search.json
```

When summarizing results:
- lead with the answer
- include dates, names, and concrete facts
- cite only returned sources
- avoid inventing URLs or source titles

## Extraction

Use to pull clean content or markdown from a URL.

```bash
parallel-cli extract https://example.com --json
parallel-cli extract https://company.com --objective "Find pricing info" --json
parallel-cli extract https://example.com --full-content --json
parallel-cli fetch https://example.com --json
```

Use `--objective` when the page is broad and you only need one slice of information.

## Deep research

Use for deeper multi-step research tasks that may take time.

Common processor tiers:
- `lite` / `base` for faster, cheaper passes
- `core` / `pro` for more thorough synthesis
- `ultra` for the heaviest research jobs

### Synchronous

```bash
parallel-cli research run \
  "Compare the leading AI coding agents by pricing, model support, and enterprise controls" \
  --processor core \
  --json
```

### Async launch + poll

```bash
parallel-cli research run \
  "Compare the leading AI coding agents by pricing, model support, and enterprise controls" \
  --processor ultra \
  --no-wait \
  --json

parallel-cli research status trun_xxx --json
parallel-cli research poll trun_xxx --json
parallel-cli research processors --json
```

### Context chaining / follow-up

```bash
parallel-cli research run "What are the top AI coding agents?" --json
parallel-cli research run \
  "What enterprise controls does the top-ranked one offer?" \
  --previous-interaction-id trun_xxx \
  --json
```

Recommended Hermes workflow:
1. launch with `--no-wait --json`
2. capture the returned run/task ID
3. if the user wants to continue other work, keep moving
4. later call `status` or `poll`
5. summarize the final report with citations from the returned sources

## Enrichment

Use when the user has CSV/JSON/tabular inputs and wants additional columns inferred from web research.

### Suggest columns

```bash
parallel-cli enrich suggest "Find the CEO and annual revenue" --json
```

### Plan a config

`enrich plan` and YAML config files require a Python install with the `cli` extra. Check `parallel-cli enrich --help`; if `plan` is absent, use direct `enrich run` arguments instead.

```bash
parallel-cli enrich plan -o config.yaml
```

### Inline data

```bash
parallel-cli enrich run \
  --data '[{"company": "Anthropic"}, {"company": "Mistral"}]' \
  --target enriched.csv \
  --intent "Find headquarters and employee count" \
  --json
```

### Non-interactive file run

```bash
parallel-cli enrich run \
  --source-type csv \
  --source companies.csv \
  --target enriched.csv \
  --source-columns '[{"name": "company", "description": "Company name"}]' \
  --intent "Find the CEO and annual revenue"
```

### YAML config run

```bash
parallel-cli enrich run config.yaml
```

### Status / polling

```bash
parallel-cli enrich status <task_group_id> --json
parallel-cli enrich poll <task_group_id> --json
```

Use explicit JSON arrays for column definitions when operating non-interactively.
Validate the output file before reporting success.

## FindAll

Use for web-scale entity discovery when the user wants a discovered dataset rather than a short answer.

```bash
parallel-cli findall entity-search "AI startups in healthcare" --entity-type companies -n 25 --json
parallel-cli findall run "Find AI coding agent startups with enterprise offerings" --json
parallel-cli findall run "AI startups in healthcare" -n 25 --json
parallel-cli findall status <run_id> --json
parallel-cli findall poll <run_id> --json
parallel-cli findall result <run_id> --json
parallel-cli findall schema <run_id> --json
```

Use `entity-search` for a synchronous, best-effort ranked list. Use `run` for comprehensive discovery with match evaluation, polling, and enrichment.

## Monitor

Use for ongoing change detection over time.

```bash
parallel-cli monitor create "Track Tesla SEC filings" --frequency 1d --json
parallel-cli monitor list -n 10 --json
parallel-cli monitor get <monitor_id> --json
parallel-cli monitor events <monitor_id> --json
parallel-cli monitor events <monitor_id> --event-group-id <event_group_id> --json
parallel-cli monitor update <monitor_id> --frequency 1w --json
parallel-cli monitor trigger <monitor_id> --json
parallel-cli monitor cancel <monitor_id> --json
```

Frequency uses `<n><unit>` with `h`, `d`, or `w` (for example `1h`, `1d`, or `1w`). Confirm the query, frequency, and webhook before creating a monitor.

`list` returns active monitors by default; add `--status active --status cancelled` when cancelled monitors are also needed. Event responses are newest-first and may include `next_cursor`; pass it back with `--cursor` for the next page.

Monitor queries and snapshot task-run IDs are immutable. Create a new monitor to change them. `trigger` starts a real off-schedule run without changing the regular schedule and only emits an event when it detects a material change.

Always confirm before `cancel`; cancellation is irreversible.

## Recommended Hermes usage patterns

### Fast answer with citations
1. Run `parallel-cli search ... --json`
2. Parse titles, URLs, dates, excerpts
3. Summarize with inline citations from the returned URLs only

### URL investigation
1. Run `parallel-cli extract URL --json`
2. If needed, rerun with `--objective` or `--full-content`
3. Quote or summarize the extracted markdown

### Long research workflow
1. Run `parallel-cli research run ... --no-wait --json`
2. Store the returned ID
3. Continue other work or periodically poll
4. Summarize the final report with citations

### Structured enrichment workflow
1. Inspect the input file and columns
2. Use `enrich suggest` or provide explicit enriched columns
3. Run `enrich run`
4. Poll for completion if needed
5. Validate the output file before reporting success

## Error handling and exit codes

The CLI documents these exit codes:
- `0` success
- `2` bad input
- `3` auth error
- `4` API error
- `5` timeout

If you hit auth errors:
1. check `parallel-cli auth`
2. confirm `PARALLEL_API_KEY` or run `parallel-cli login` / `parallel-cli login --no-browser --json`
3. verify `parallel-cli` is on `PATH`

## Maintenance

Check current auth / install state:

```bash
parallel-cli auth
parallel-cli --help
```

Upgrade through the original installation method:

```bash
# Shell installer
parallel-cli update

# Homebrew
brew upgrade parallel-cli

# npm
npm update -g parallel-web-cli

# pip / pipx
pip install --upgrade "parallel-web-tools[cli]"
pipx upgrade parallel-web-tools
```

## Pitfalls

- Do not omit `--json` unless the user explicitly wants human-formatted output.
- Do not cite sources not present in the CLI output.
- For headless authentication, use `login --no-browser --json` and follow the returned device-authorization URL and code.
- Prefer foreground execution for short tasks; do not overuse background processes.
- For large result sets, save JSON to `/tmp/*.json` instead of stuffing everything into context.
- Do not silently choose Parallel when Hermes native tools are already sufficient.
- Remember this is a vendor workflow that usually requires account auth and paid usage beyond the free tier.
