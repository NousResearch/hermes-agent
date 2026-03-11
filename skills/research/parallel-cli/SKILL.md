---
name: parallel-cli
description: Use Parallel CLI for agent-native web search, content extraction, deep research, enrichment, FindAll entity discovery, and monitoring. Prefer JSON output and non-interactive flows.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Research, Web, Search, Deep-Research, Enrichment, CLI]
    related_skills: [duckduckgo-search, mcporter]
---

# Parallel CLI

Use `parallel-cli` when the user wants Parallel specifically, or when a terminal-native workflow would benefit from Parallel's web search, extraction, deep research, or enrichment stack.

`parallel-cli` is designed for agents:
- JSON output via `--json`
- Non-interactive command execution
- Async long-running jobs with `--no-wait`, `status`, and `poll`
- Search, extract, research, enrichment, entity discovery, and monitoring in one CLI

## When to use it

Prefer this skill when:
- The user explicitly mentions Parallel or `parallel-cli`
- The task needs richer workflows than a simple web search/extract pass
- You need async deep research jobs that can be launched and polled later
- You need structured data enrichment, FindAll, or monitoring capabilities

Prefer Hermes native `web_search` / `web_extract` for quick one-off lookups when Parallel is not specifically requested.

## Installation

Try the fastest install path available for the environment.

### Standalone binary

```bash
# macOS / Linux
curl -fsSL https://parallel.ai/install.sh | bash
```

Homebrew is also supported:

```bash
brew install parallel-web/tap/parallel-cli
```

npm downloads the prebuilt binary:

```bash
npm install -g parallel-web-cli
```

### Python fallback

```bash
pipx install "parallel-web-tools[cli]"
pipx ensurepath
```

## Authentication

Interactive login:

```bash
parallel-cli login
```

Headless / CI / agent usage:

```bash
export PARALLEL_API_KEY="***"
```

Verify auth:

```bash
parallel-cli auth --json
```

If auth requires browser interaction, run with `pty=true`.

## Core rule set

1. Always prefer `--json` so Hermes can parse structured output.
2. Prefer non-interactive flags and explicit arguments.
3. For long-running jobs, use `--no-wait` and then `status` / `poll`.
4. Cite only URLs returned by the CLI output.
5. Save large JSON outputs to a temp file when follow-up questions are likely.
6. Use background processes only for genuinely long-running workflows; otherwise run in foreground.

## Quick reference

```text
parallel-cli
├── auth
├── login
├── logout
├── search
├── extract / fetch
├── research run|status|poll|processors
├── enrich run|status|poll|plan|suggest|deploy
├── findall run|ingest|status|poll|result|enrich|extend|schema|cancel
└── monitor create|list|get|update|delete|events|event-group|simulate
```

## Search

Use for current web lookups with structured results.

```bash
parallel-cli search "What is Anthropic's latest AI model?" --json
parallel-cli search -q "bitcoin price" --after-date 2026-01-01 --json
parallel-cli search "SEC filings for Apple" --include-domains sec.gov --json
```

Useful flags:
- `-q` for supplemental keyword queries
- `--after-date YYYY-MM-DD` for recency
- `--include-domains domain1.com,domain2.com` to constrain sources

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
```

Use `--objective` when the page is broad and you only need one slice of information.

## Deep research

Use for deeper multi-step research tasks that may take time.

### Synchronous

```bash
parallel-cli research run "Compare the leading AI coding agents by pricing, model support, and enterprise controls" --json
```

### Async launch + poll

```bash
parallel-cli research run "Compare the leading AI coding agents by pricing, model support, and enterprise controls" --no-wait --json
parallel-cli research status trun_xxx --json
parallel-cli research poll trun_xxx --json
parallel-cli research processors --json
```

Recommended Hermes workflow:
1. launch with `--no-wait --json`
2. capture the returned run/task ID
3. if the user wants to continue other work, keep moving
4. later call `status` or `poll`

## Enrichment

Use when the user has CSV/JSON/tabular inputs and wants additional columns inferred from web research.

### Suggest columns

```bash
parallel-cli enrich suggest "Find the CEO and annual revenue" --json
```

### Plan config

```bash
parallel-cli enrich plan -o config.yaml
```

### Non-interactive run

```bash
parallel-cli enrich run \
  --source-type csv \
  --source companies.csv \
  --target enriched.csv \
  --source-columns '[{"name": "company", "description": "Company name"}]' \
  --intent "Find the CEO and annual revenue"
```

### Status / polling

```bash
parallel-cli enrich status <task_group_id> --json
parallel-cli enrich poll <task_group_id> --json
```

Use explicit JSON arrays for column definitions when operating non-interactively.

## FindAll

Use for web-scale entity discovery when the user wants a set of entities matching natural-language criteria.

```bash
parallel-cli findall run "Find AI coding agent startups with enterprise offerings" --json
parallel-cli findall status <run_id> --json
parallel-cli findall poll <run_id> --json
parallel-cli findall result <run_id> --json
parallel-cli findall schema <run_id> --json
```

This is a better fit than ordinary search when the user wants a discovered dataset rather than a short answer.

## Monitor

Use for ongoing change detection over time.

```bash
parallel-cli monitor create --help
parallel-cli monitor list --json
parallel-cli monitor get <monitor_id> --json
parallel-cli monitor events <monitor_id> --json
parallel-cli monitor delete <monitor_id> --json
```

Use this when the user wants recurring tracking of a page or source rather than a one-time fetch.

## Input patterns for agents

Read from stdin when convenient:

```bash
echo "What is the latest funding for Anthropic?" | parallel-cli search - --json
echo "Research question" | parallel-cli research run - --json
```

## Error handling and exit codes

The CLI documents these exit codes:
- `0` success
- `2` bad input
- `3` auth error
- `4` API error
- `5` timeout

If you hit auth errors:
1. check `parallel-cli auth --json`
2. confirm `PARALLEL_API_KEY` or run `parallel-cli login`
3. verify `parallel-cli` is on `PATH`

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

## Pitfalls

- Do not omit `--json` unless the user explicitly wants human-formatted output.
- Do not cite sources not present in the CLI output.
- `login` may require PTY/browser interaction.
- Prefer foreground execution for short tasks; do not overuse background processes.
- For large result sets, save JSON to `/tmp/*.json` instead of stuffing everything into context.
