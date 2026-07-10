---
name: x-pulse
description: "Summarize the most-discussed X posts on chosen topics."
version: 1.0.0
author: HumphreySun98
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Social-Media, X, Twitter, Trends, Digest, Grok]
    related_skills: [xurl]
    requires_tools: [x_search]
---

# X Pulse Skill

Build a sourced digest of what's being discussed on X (Twitter) about a set of
topics, using the native `x_search` tool (xAI Grok's X index). It is for
*reading* the current conversation — trending discussion, reactions, claims —
not for posting; use the `xurl` skill to post or DM. xAI has **no** trending or
engagement-sort parameter, so "hottest" is expressed as a well-formed query and
the value here is the multi-topic sweep, dedup, and scheduled delivery.

## When to Use

- User asks "what's happening / trending on X about <topic>?"
- User wants a recurring digest of X discussion (a "daily X pulse")
- User wants sourced reactions to a launch, event, or claim from X

Do NOT use for general web search (use `web_search`) or to publish to X (use
the `xurl` skill).

## Prerequisites

- xAI credentials so the `x_search` tool is registered — either a SuperGrok
  OAuth login (`hermes auth add xai-oauth`) or `XAI_API_KEY` in `~/.hermes/.env`.
  This skill only appears when `x_search` is available (`requires_tools`).
- No extra packages; the digest script is pure Python stdlib.

## How to Run

1. Call the native `x_search` tool once per topic (it answers a single query at
   a time) and collect each JSON result.
2. Write the collected results to a JSON file with `write_file`.
3. Run the digest builder through the `terminal` tool:
   ```
   python scripts/build_digest.py --input results.json --title "X Pulse" --date 2026-07-10
   ```
4. Deliver the markdown it prints with `send_message`, or schedule the whole
   flow with `cronjob`.

## Quick Reference

| Need | How |
|---|---|
| Hottest discussion on a topic | `x_search(query="most-discussed posts about <topic> in the last 24h, cite the posts")` |
| Restrict to accounts | `x_search(query=..., allowed_x_handles=[...])` — max 20, mutually exclusive with `excluded_x_handles` |
| Recent window | `x_search(query=..., from_date="YYYY-MM-DD", to_date="YYYY-MM-DD")` |
| Merge topics into one digest | `python scripts/build_digest.py --input results.json` |
| Daily delivery | `cronjob(action="create", schedule="0 9 * * *", prompt="Run the x-pulse skill for <topics> and send the digest")` |

`build_digest.py` input is a JSON list of `{"topic": "<label>", "result": <x_search JSON>}`.

## Procedure

1. Resolve the topic list from the user (2–6 topics works well).
2. For each topic, call `x_search` with a query that asks for the **most
   discussed / highest-engagement** posts in the desired window and requests
   citations, e.g. `"the most-discussed posts about AI safety in the last 24
   hours — summarize the main threads and cite the posts"`.
3. Assemble a JSON list of `{"topic", "result"}` (parse each `x_search` return)
   and save it with `write_file` to e.g. `results.json`.
4. Run `python scripts/build_digest.py --input results.json --title "X Pulse"
   --date <today>` through the `terminal` tool.
5. Send the resulting markdown with `send_message` to the requested chat, or
   register a `cronjob` that repeats steps 1–5 on a schedule.

## Pitfalls

- **No trending API.** xAI's X Search has no popularity/sort parameter — the six
  supported knobs are `allowed_x_handles`, `excluded_x_handles`, `from_date`,
  `to_date`, `enable_image_understanding`, `enable_video_understanding`. "Hot"
  lives entirely in the query wording.
- **Unsourced answers.** When a result has `degraded: true`, the answer came
  from the model's own knowledge, not the live X index. The digest flags these
  with a ⚠️ line — treat them as unverified.
- **Handle limits.** Up to 20 handles per filter; `allowed_x_handles` and
  `excluded_x_handles` cannot be combined in one call.
- **Date ranges.** Dates are `YYYY-MM-DD`; a `from_date` later than today
  returns nothing. Keep windows recent for "what's hot right now".

## Verification

```
echo '[{"topic":"AI","result":{"success":true,"answer":"Big week for open models.","citations":[{"url":"https://x.com/a/status/1","title":"post"}],"degraded":false}}]' | python scripts/build_digest.py --title "X Pulse"
```

Prints a markdown digest with the topic section and a deduped Sources list.
