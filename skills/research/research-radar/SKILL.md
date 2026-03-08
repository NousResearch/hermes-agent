---
name: research-radar
description: Multi-source research radar for signal monitoring, trend extraction, and decision-ready intelligence briefings.
version: 3.0.0
metadata:
  hermes:
    tags: [research, radar, intelligence, trends, markdown, automation, change-detection]
    related_skills: [github-pr-workflow]
---

# Research Radar

Research Radar is a reusable Hermes skill for monitoring a topic across multiple public sources, extracting high-signal developments, and generating structured intelligence briefings in Markdown.

Use this skill when the user wants a report that is more than a simple summary:
- recent developments
- recurring themes
- risks
- opportunities
- recommended actions
- saved output for later review or automation

This skill can also compare the current report with the most recent previous run and highlight what changed.

## When to use

Use this skill for:
- AI / software / startup landscape monitoring
- market or ecosystem watchlists
- founder / operator / investor briefings
- “what changed in the last 7 days?” style analysis
- recurring intelligence reports scheduled with Hermes cron

Do not use this skill when:
- the user only wants a short paragraph
- a proprietary/private data source is required
- the task is real-time trading, legal, or medical advice

## Inputs

Collect or infer:
- topic
- time horizon
- target audience
- output filename

Defaults:
- time horizon: last 7 days
- target audience: technical generalist
- output filename: `research_briefing.md`

## Available tools

Prefer these tools only:
1. `terminal`
2. `execute_code`
3. `write_file`
4. `read_file`

Do not rely on unavailable tools such as `web_search`.

## Source strategy

Use multiple lightweight public sources when possible.

Preferred sources:
- Hacker News Algolia API for discussion/activity signals
- GitHub Search API for relevant repositories/projects
- arXiv API for research/paper signals

If one source fails, continue with the others and state that coverage was partial.

## Workflow

### Step 1 — Frame the task
Restate:
- topic
- time horizon
- target audience
- filename

### Step 2 — Gather signals
Use terminal-accessible public sources.

Examples:
- Hacker News Algolia API
- GitHub Search API
- arXiv API

Aim to collect:
- discussion signals
- code / repo signals
- research / paper signals

### Step 3 — Filter and synthesize
Identify:
- 3 to 5 key developments
- 2 to 4 emerging themes
- 2 to 4 risks
- 2 to 4 opportunities
- 3 recommended actions

Prefer concrete, recent, decision-relevant information.

### Step 4 — Compare with the last run when available
If a previous report exists for the same topic, read it and compare it against the current findings.

Look for:
- newly appearing developments
- developments that are no longer prominent
- changing risks
- changing opportunities

If a meaningful comparison is possible, add a section called:

## What Changed Since Last Run

Include short bullet points describing the differences.

If no previous report exists, skip this section and continue normally.

### Step 5 — Write the report
Write the final report using the standard template structure below.

## Standard Report Structure

# Research Radar: <topic>

## Executive Summary
Short paragraph.

## Key Developments
- bullet points

## Emerging Themes
- bullet points

## Risks
- bullet points

## Opportunities
- bullet points

## Recommended Actions
- bullet points

## What Changed Since Last Run
- bullet points if a previous report exists

## Sources
- list of links or source names

## Metadata
- topic
- time horizon
- generated date

## Style requirements

- Be specific and concise
- Prefer high-signal findings over generic commentary
- Avoid hype
- Make the report useful for decision-making
- Save the final report to the requested markdown filename

## Verification

After writing the report:
1. confirm the file exists
2. confirm the file is readable
3. summarize in one sentence what was produced

## Pitfalls to avoid

- Do not overclaim certainty from weak signals
- Do not invent sources
- Do not use only one source if multiple are available
- Do not produce a generic summary without risks/opportunities/actions
- Do not force a change-detection section when there is no meaningful prior report

## Example requests

- “Use research-radar to produce a 7-day briefing on AI coding agents and save it to `ai_agents_briefing.md`.”
- “Use research-radar to monitor robotics startups for the last 14 days and save the report to `robotics_watch.md`.”
- “Use research-radar to generate a founder-focused briefing on open-source agent tooling.”
- “Use research-radar to compare the latest AI coding agents landscape against the previous run and highlight what changed.”
