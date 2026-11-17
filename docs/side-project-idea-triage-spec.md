# Side Project Idea Triage Helper Spec

## Problem

Joe collects side-project ideas faster than he can execute them. The nightly Hermes loop needs a small offline helper that can turn a Markdown idea dump into a ranked shortlist without contacting external services or mutating live systems.

## Goals

- Parse human-maintained Markdown idea notes into structured idea records.
- Score ideas with a simple, transparent heuristic aligned with Joe's operating manual:
  - high leverage / compounding value
  - low effort and reversible first step
  - fits personal systems, investment framework, health, information aggregation, or side projects
  - penalizes vague ideas and blocked ideas
- Emit Markdown by default for morning review, with JSON available for automation.
- Stay offline, deterministic, and easy to test.

## Non-goals

- No network calls, LLM calls, deployment, messaging, or integration with real people.
- No writes to source notes unless a future explicit command is added.
- No attempt to be a full product-management system.

## Input contract

A Markdown file may contain ideas as `## Idea title` / `### Idea title` sections. Inside each section, optional fields can be written as bullets:

```markdown
## Investor signal digest
- Area: information aggregation
- Impact: 5
- Effort: 2
- Confidence: 4
- First step: Parse watchlist links and produce a daily brief
- Blocked: false
- Notes: Compounds BD top-of-funnel quality
```

Accepted numeric fields are `impact`, `effort`, `confidence`, and `urgency`, each clamped to 1-5. `Area`, `First step`, `Blocked`, and `Notes` are optional. Non-idea prose before the first idea heading is ignored.

## Output contract

Default Markdown output includes:

1. A title: `# Side Project Idea Triage`
2. A ranked table with rank, title, area, score, impact, effort, confidence, and first step.
3. A `## Top recommendation` section explaining why the first idea won.
4. A `## Watch / defer` section for blocked or low-score ideas.

JSON output returns `ideas` sorted by score with the parsed fields and score explanation.

## Scoring

Base score:

```text
impact * 3 + confidence * 2 + urgency + area_bonus - effort * 2 - penalties
```

Area bonus:

- +3: information aggregation, investment, personal systems, health, side projects, dev tools
- +1: learning, writing, finance
- +0: everything else

Penalties:

- +8 penalty if blocked
- +4 penalty if no first step
- +2 penalty if notes/title contain vague markers such as `someday`, `maybe`, or `research more`

Higher score ranks first.

## Verification plan

- Unit tests for parsing well-formed ideas and ignoring template/instructional prose.
- Unit tests for ranking, blocked penalties, and missing first-step penalties.
- CLI smoke test on a temporary Markdown sample in both Markdown and JSON modes.
