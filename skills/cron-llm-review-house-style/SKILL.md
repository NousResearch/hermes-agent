---
name: cron-llm-review-house-style
description: Convert script-only Hermes cron jobs into script→LLM-review→pretty-delivery
  pipelines so every scheduled message is evaluated for false alarms/actionable items
  and formatted in a consistent emoji+hyperlink house style. Use when a user wants
  prettier cron output, LLM evaluation of cron responses, or to stop false-alarm/all-clear
  noise.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags:
    - cron
    - formatting
    - llm-review
    - telegram
    - house-style
    - false-alarm
    created_by: agent
    related_skills:
    - script-first-cron-design
    - messaging-platform-formatting
    config:
    - key: cron-llm-review-house-style.enabled
      description: Enable cron-llm-review-house-style skill behavior
      default: true
      prompt: Enable cron-llm-review-house-style skill?
    category: devops
platforms:
- linux
- macos
- windows
---


# Cron LLM-Review + House Style

## Overview

Pattern for upgrading Hermes cron jobs from raw script-only delivery to a
**script (precheck) → LLM review → pretty delivery** pipeline. The LLM both
**evaluates** the script output (is anything actionable / wrong / a false alarm?)
and **formats** it in a consistent house style with emojis and clickable links.

This is the inverse tradeoff of pure `script-first-cron-design`: you spend tokens
to gain judgment + presentation. Use it when the user explicitly wants prettier
output AND/OR LLM evaluation of every response. Combine the two skills: keep the
deterministic script as the data collector, add the LLM as reviewer/formatter.

## When to Use

- User asks to "fancy up" / prettify cron job responses.
- User wants the LLM to evaluate every cron response for errors/actionable items.
- A script-only watchdog produced a false alarm that got delivered verbatim.
- You want consistent emoji + hyperlink formatting across all scheduled messages.

Do NOT use for jobs where the user wants zero token cost and trusts the raw
script output — keep those `no_agent: true`.

## Key Mechanics (verified against Hermes source)

1. **`no_agent: false` + a `script`** = the script runs first as a precheck and
   its stdout is injected into the LLM prompt under `## Script Output`. The LLM's
   final response is what gets delivered. (`hermes cron edit <id> --agent` flips
   off no-agent mode; `--no-agent` flips it back on.)
2. **Silent suppression**: if the LLM final response is EXACTLY `[SILENT]`,
   delivery is suppressed but output is still saved to `cron/output/<id>/` for
   audit. Sentinel is `SILENT_MARKER = "[SILENT]"` in `cron/scheduler.py`.
3. **Zero-cost quiet ticks**: if the precheck script prints NOTHING (empty
   stdout), the scheduler skips the LLM entirely — so frequent watchers
   (every 15m/2h) only cost tokens when the script actually emits data. Design
   scripts to stay silent on non-events.
4. **Toolsets**: restrict via `enabled_toolsets` to the minimum
   (`terminal`, `file`, optionally `skills`) to keep cost/permissions low.

## Conversion Steps

1. List jobs, never guess IDs: `hermes cron list` or the `cronjob` tool.
2. For each script-only job to convert:
   - `hermes cron edit <id> --agent --prompt "<job intro>\n\n<shared review prompt>"`
   - Set minimal toolsets via the `cronjob` update action `enabled_toolsets`.
3. **Strip contradictory old instructions** like "deliver the script stdout
   verbatim and do not use an LLM" / "Empty stdout means no update" — they fight
   the new review step. (Regex them out of the prompt.)
4. For jobs that were ALREADY LLM-driven, just append the shared review prompt
   so they share the same evaluation+formatting standard.
5. Keep the one-off/transient jobs (e.g. a single-fire reminder) as-is unless
   asked.

## Shared Review Prompt

Store a reusable prompt file (e.g. `/opt/data/scripts/cron_review_prompt.md`) and
append it to every converted job's prompt. It must instruct the LLM to:

- **Evaluate**: flag anything actionable; detect false alarms / tooling glitches
  (e.g. `ModuleNotFoundError` misreported as an auth failure, pre-tournament
  zeros that look like a data glitch, wrong-timezone timestamps, broken links,
  duplicates). Never relay a likely glitch as a confirmed problem — suppress it
  or label it clearly with the fix.
- **Format** (house style): emoji-led bold title + `━━━━━━━━━━━━━━━` divider;
  short emoji sections; bold labels; compact `•` bullets; **hyperlink every
  referenced item** with descriptive link text; status icons 🟢🟡🔴⚪; Pacific
  time friendly labels (never raw UTC/ISO); no tables; preserve privacy footers.
- **SILENT rule**: reply EXACTLY `[SILENT]` when nothing is actionable/worth
  showing (suppresses delivery, no all-clear noise).

## Shared Style Helpers

Optionally back the scripts with a `cron_style.py` module exposing `header`,
`section`, `bullet`, `kv`, `link`, `local_time`, `now_label`, `render` so the raw
script output is already close to house style and the LLM mostly reviews/links.

## Verification (always do this)

1. Run a job WITH data (e.g. a sports/schedule job): `cronjob run <id>` then
   `hermes cron tick`, and read `cron/output/<id>/<newest>.md` — confirm the
   `## Response` section is pretty + links work + false-alarm caveats appear.
2. Run a HEALTHY watchdog (e.g. auth guard) the same way — confirm the output
   file is empty / `[SILENT]` and nothing was delivered.
3. Re-list jobs to confirm `no_agent`, `enabled_toolsets`, and prompt changed.

## Pitfalls

1. Leaving the old "do not use an LLM / deliver verbatim" line in the prompt — it
   contradicts the review step and confuses the model. Strip it.
2. Forgetting that empty script stdout skips the LLM — good for cost, but means
   the LLM can't "speak up" when the script SHOULD have produced output but
   didn't. For health checks, make the script emit a terse line on real failure,
   nothing on success.
3. Over-broad toolsets — cron LLM jobs only need `terminal`/`file`/`skills`.
4. A script that crashes on a missing dependency can masquerade as a domain
   error (the google-auth-guard `invalid_grant` false alarm). Give Google
   scripts a `uv run --with google-api-python-client --with google-auth-oauthlib
   --with google-auth-httplib2` fallback and only classify a real auth error when
   the dependency actually loaded.
5. Cost on frequent watchers — acceptable only because empty-stdout ticks skip
   the LLM. If a watcher emits output every tick, reconsider before converting.
