# MOE legal-writing quick commands

Personal local notes for the Hermes setup on this machine.

This file documents the legal-writing shortcuts and the intended workflow.
It is written for personal use and assumes the local Hermes config in:
- `~/.hermes/config.yaml`

## Purpose

These shortcuts are meant to make the legal thought-leadership workflow fast and consistent.

They sit on top of the `legal-writing` skill suite and use Hermes quick-command aliases.

## Installed quick commands

### `/moe-review`
Alias to:
- `/legal-thought-leadership-review`

Use when you want the full review flow:
- structure
- argument
- style
- domain framing
- humanization
- synthesis into a revision plan

Example:
- `/moe-review Review this LinkedIn draft for partner-level thought leadership.`

### `/moe-revise`
Alias to:
- `/legal-thought-leadership-revise`

Use when you want a full rewritten draft rather than critique only.

Example:
- `/moe-revise Turn this draft into a polished client alert.`

### `/moe-style`
Alias to:
- `/law-style-editor`

Use when the substance is mostly right and the main need is:
- clarity
- concision
- rhythm
- readability
- voice preservation

Example:
- `/moe-style Tighten this piece without making it sound generic.`

### `/moe-argument`
Alias to:
- `/law-argument-editor`

Use when the main need is:
- stronger thesis
- tighter doctrinal reasoning
- better use of authorities
- sharper policy logic

Example:
- `/moe-argument Strengthen the thesis and legal reasoning.`

## Recommended workflow

### 1. First-pass diagnosis
Start with:
- `/moe-review`

Use this when the draft is still taking shape or when you want the full set of expert lenses.

### 2. Full rewrite
If the review shows the piece needs a full rewrite, run:
- `/moe-revise`

This should produce a cleaner, publishable draft while preserving the core position.

### 3. Focused follow-up passes
After the big review or rewrite:
- use `/moe-style` for polish
- use `/moe-argument` for legal depth and thesis strength

## Skill mapping

Underlying skills currently installed in `legal-writing`:
- `legal-thought-leadership-review`
- `legal-thought-leadership-revise`
- `law-draft-review-orchestrator`
- `law-revision-synthesizer`
- `law-structure-editor`
- `law-argument-editor`
- `law-style-editor`
- `law-domain-editor`
- `humanizer`

## Implementation notes

### Quick-command behavior
Hermes now supports two quick-command types:
- `exec` — run a shell command directly
- `alias` — rewrite to another slash command

The `/moe-*` commands are all `alias` commands.

### Config location
Aliases live in:
- `~/.hermes/config.yaml`

Current entries:
- `moe-review -> /legal-thought-leadership-review`
- `moe-revise -> /legal-thought-leadership-revise`
- `moe-style -> /law-style-editor`
- `moe-argument -> /law-argument-editor`

## Guardrails

Use these tools as drafting and editorial support, not as a substitute for authority verification.

Working assumptions for this workflow:
- do not fabricate cases, statutes, guidance, or speeches
- preserve the intended commercial/legal audience
- prefer crisp, publishable prose over bloated academic prose
- keep Canadian/US legal framing explicit when relevant

## Verification

Related Hermes test file:
- `tests/test_quick_commands.py`

Last verified locally with:
- `/Users/rhx/.hermes/hermes-agent/venv/bin/pytest tests/test_quick_commands.py -q`

## Local-only note

This setup is intended for personal local use.
Do not open a public PR with these personal shortcut choices unless they are deliberately generalized first.
