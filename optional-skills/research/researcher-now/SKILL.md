---
name: researcher-now
description: "Give Hermes Researcher superpowers — analyze any article, YouTube video, tweet, or podcast; run durable, extensible research spanning many hours and thousands of cited sources (all returned); and maintain realtime topics that keep a living brief current. Designed with machine payments so no human in the loop required."
version: 0.2.0
author: Researcher (researcher.now)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Research, Deep-Research, Web-Extraction, YouTube-Transcript, Personas, MPP, HTTP-402, Keyless]
    related_skills: [mpp-agent, stripe-link-cli]
  homepage: https://researcher.now
  source: https://github.com/researcher-now/skill
---

# Researcher

**Give Hermes Researcher superpowers.**

Use Researcher when the work should become a durable, inspectable research run instead of a one-off answer. A good run gives the user a watch URL immediately, then stores sources, artifacts, costs, report output, and run-scoped chat context.

## Product Model

Think in five verbs:

1. Plan: preview a vague or high-stakes request before spending.
2. Run: start durable server-side work and share the watch URL immediately.
3. Inspect: read the truth bundle, markdown, sources, source extractions, usage, warnings, and events.
4. Iterate: fund (raise the budget in place), fork, continue, deepen, focus, steer, re-extract, or regenerate the report.
5. Export: save markdown plus structured sources/source-extractions downstream.

## Stay Current

- Contract version: `2026-06-26`. Keyless access now works over MCP too (`https://researcher.now/mcp`, no API key — paid tools answer 402 with an MPP session challenge; authorizing it provisions an account and returns a durable key). Every API response carries an
  `X-Researcher-Contract` header, and run-creating responses a
  `contractVersion` field. When it differs from the version above, re-read
  `https://researcher.now/agent.txt` (and `GET /v1/skill`) before continuing.
- Response fields are authoritative over your own saved notes: prices,
  shapes, and statuses you recorded from earlier runs go stale — what the API
  returns now is the truth. Never report a cost from memory; use the
  response's `estimatedCostUsd` or the run's settled cost.
- Time-boxed change notices arrive as `X-Researcher-Notice` headers and
  `contractNotices` body fields for ~2 weeks after a change — read and obey
  them.
- If a response includes `pluginUpdateNote`, tell the operator to run
  `pip install -U researcher-now`.

## Setup

Dead-simple keyless (no key, no signup): point any x402/MPP client at researcher.now with a funded wallet — it pays per request and the first paid call returns a durable key in `X-Researcher-Agent-Key` to reuse. One line:

```bash
tempo wallet login   # one time (curl -L https://tempo.xyz/install | bash)
tempo request https://researcher.now/v1/runs --json '{"source":{"type":"topic","topic":"…"},"limits":{"maxCostUsd":2}}'
```

Full recipes (HTTP/mppx/MCP): `references/keyless-quickstart.md`. Or use a key:

- Required input: `RESEARCHER_API_KEY` or `RESEARCHER_TOKEN`, a customer Researcher API token from `https://researcher.now/account/?setup=agent`.
- Optional input: `RESEARCHER_BASE_URL`; default it to `https://researcher.now`.
- Use `RESEARCHER_API_KEY` as the bearer token. Accept `RESEARCHER_TOKEN` as an alias if the environment already uses it.
- If no customer token is available and a human is present, send them to `https://researcher.now/account/?setup=agent`.
- Keyless provisioning (no human): `POST /v1/runs` (or `/v1/analyze`) with no `Authorization`. With no payment attached you receive `402` plus a tempo MPP session challenge (`WWW-Authenticate`, `intent:"session"`); authorize it from your wallet and retry with the session credential. The verified channel payer becomes a persistent account, and a paid run returns a durable `rk_` key in `agentApiKey` (and the `X-Researcher-Agent-Key` response header). Store it and use it as `RESEARCHER_API_KEY` for all later reads, management, and runs. Writes can stay keyless by re-authorizing a session each call.
- Never search local files, shell history, or logs for customer tokens.
- Never use admin-token, payment-bypass, or internal-only routes in customer workflows.

## Choose The Operation

- Use `POST /v1/analyze` with `{ "url": ... }` for a SINGLE article, YouTube video, X/Twitter video post, or Spotify / Apple Podcasts episode when the user wants it read/summarized/analyzed now: responds in seconds with formatted markdown or a speaker-aware transcript PLUS the structured analysis (claims, facts, quantitative data, quotes), then the run continues to a full synthesized report on the same `viewerUrl`. Check the response `status`: `"running"` means the report is being written; `"awaiting_funding"` means the analysis was a free preview — relay the `viewerUrl` so the user can fund and finish the report. Podcast episodes always answer with `deferred: true` (`reason: "podcast"`) and no inline transcript — audio transcription takes minutes; relay the `viewerUrl` immediately and poll `GET /v1/runs/:id/job` instead of waiting on the HTTP call. Prefer this over a deep run for one-URL asks — only start a deep run for multi-source synthesis.
- Use `POST /v1/runs` with `preflightPlan:true` before spending when the customer prompt is vague, high-stakes, or needs an explicit success checklist.
- Use `POST /v1/runs` for all new durable work. Pick `source.type` as `topic`, `url`, `feed`, or `video`; include `requestedBy` and an `Idempotency-Key`.
- Use `source.type:"video"` for a YouTube URL, podcast/video link, or transcript-first task. Do not scrape video as a generic webpage.
- Use `source.type:"url"` for a known webpage/domain extraction and `source.type:"feed"` for one-shot RSS, Atom, or JSON Feed ingestion.
- Use `GET /v1/me` to confirm whether the bearer token is a customer or admin token before spending.
- Use `GET /v1/runs`, `/v1/library`, or `/v1/library/recall?q=*` to enumerate existing runs and recall.
- Use `GET /v1/runs/:id/job` for live run state. Read top-level `status`, `currentStage`, `currentActivity`, and `state`; legacy `job.status` remains nested under `job`.
- Use `POST /v1/webhooks` with `event:"run.complete"` before starting unattended or multi-run work. If no webhook receiver is available, stream or poll until `succeeded`, `failed`, or `cancelled`.
- Use `GET /v1/exports/pending` at session start and before commissioning new research: it lists terminal runs no agent on this account has seen yet — including runs started on the researcher.now web app, the API, or Discord. Surface each run's watch URL to the user, then acknowledge with `POST /v1/exports/ack {"ids":[...]}` so they stop appearing. Full results stay available via the normal run endpoints.
- Use `POST /v1/topics` when the user wants standing coverage of a subject ("keep an eye on X", "track Y going forward") instead of a one-off report. Body: `{ "prompt": "...", "title"?, "seeds"?: { "xAccounts"?: [], "githubProfiles"?: [], "feeds"?: [], "domains"?: [] }, "cadence"?: "realtime" | "hourly" | "daily" | "weekly" (default daily; realtime scans roughly every 10 minutes), "monthlyBudgetUsd"? }`. Make `prompt` the subject plus the user's stake in it. Creating a topic fires a bootstrap deep-research run automatically; seeds become standing watch targets (X accounts, GitHub profiles, RSS feeds, websites) plus an automatic wide-search lane. `seeds.xAccounts` accepts up to 1,000 handles (large lists are scanned in batches); `seeds.githubProfiles` takes up to 100 GitHub usernames watched via the public events API (releases, new repos, pushes, opened pull requests), surfaced on `GET /v1/topics/:id` as watch targets with kind `github`. Same customer key as `/v1/runs`.
- Use `GET /v1/topics` to list topics and `GET /v1/topics/:idOrSlug` for one topic plus recent events (including `scan_digest` items), watch targets, and docs. The path segment accepts either the topic's `slug` or its uuid, and every topic object carries a `slug` (a URL-friendly form of its title, unique within your account). Rename a topic with `PATCH /v1/topics/:id` body `{ "title": "..." }` (1-200 chars after trimming): it regenerates the slug, returns `{ topic }` with the new title and slug, and appends a `title_updated` event. Lifecycle: `bootstrapping` -> `active` once the bootstrap run completes | `bootstrap_failed` if the bootstrap run could not start. Scanning is independent of that lifecycle: the topic scans on its cadence (appending `scan_digest` events) from creation, including while bootstrapping, and stops only when paused or archived.
- Read topic docs from `GET /v1/topics/:id`: `docs.thesis` and `docs.working` are two versioned markdown documents, each `{ version, contentMarkdown, summary, createdAt }` or `null` before seeding. `thesis` holds the stable top-level beliefs seeded from the bootstrap research run (numbered claims with confidence and falsification conditions) and is never auto-edited — synthesis only flags tensions against it. `working` is the living idea layer: new scan items are continuously synthesized into it under pressure triggers (enough new items, staleness, first items after seeding), not on a fixed clock; each reorganization appends a new version. Watch for `docs_seeded` and `doc_synthesized` events — the `doc_synthesized` `message` is the what-changed summary and its `metadata.tensions` flags thesis claims under pressure, referencing claim ids — plus `docs_seed_failed` / `doc_synthesis_failed` on errors and `docs_seed_skipped` when seeding was skipped.
- Read Daily briefs from `GET /v1/topics/:id`: `dailyBriefs` is an array, newest first (up to 30), each `{ id, briefDate, windowStart, windowEnd, contentMarkdown, summary, itemCount, thesisDeltas, sourceItems, createdAt }` — an immutable, dated, once-per-day narrative of what happened in `[windowStart, windowEnd]` that matters to the topic, thesis-relative when a thesis doc exists. `thesisDeltas` is `[{ claimId, direction, note }]` where `direction` is `reinforced | contradicted | raised` and `claimId` references a thesis claim id (like `C1`) or is null; quiet days produce no brief. `sourceItems` is the day's notable scanned items (deduped by url, ranked by engagement/recency, capped at 12), each carrying the same optional visual fields as `scan_digest` items below. The response also carries `briefsSeenAt` (when the briefs were last marked seen, or null); `POST /v1/topics/:id/briefs/seen` marks them seen and returns `{ briefsSeenAt }`. Each written brief appends a `daily_brief_generated` event whose `message` is the brief's one-line summary.
- `scan_digest` items and brief `sourceItems` carry optional visual fields so clients can render tweets-as-tweets and og thumbnails, each present only when the provider supplied it: `text` (full post text), `authorName`, `authorAvatarUrl`, `media` (`[{ type: photo|video|animated_gif, url, previewUrl? }]`), `image` (thumbnail/og image for non-tweet items), and `domain`. All existing item fields are unchanged.
- Consult an expert persona when the user asks what a specific notable person thinks, would say, or has said about something ("what would Paul Graham say about X", "ask Buffett about Y", "how would Elon approach Z"). `GET /public/entities` lists the durable personas (Paul Graham, Warren Buffett, Elon Musk, Patrick Collison, Jeff Bezos, Stanley Druckenmiller, and more) with their slug, name, and what they are known for. `POST /v1/entities/:slug/chat` with `{ "message": "..." }` returns an answer in that persona's voice grounded in their actual writing, talks, and posts, with citations to exact sources (`answer` plus `citations`), and a `sessionId`. Each persona is a research-grounded corpus, not improvised mimicry; a small per-question fee applies. Multi-turn memory: pass the returned `sessionId` back on the next `POST /v1/entities/:slug/chat` (in the body alongside `message`) and the persona remembers the conversation; resume later by listing sessions with `GET /v1/entities/:slug/chat/sessions` (`list_persona_sessions` on MCP) and reusing a `sessionId`. Depth dial: pass `"depth": "quick"` (default) or `"depth": "deep"` for a wider reranked retrieval on hard questions. Async: pass `"async": true` to get a `turnId` immediately and poll `GET /v1/entities/:slug/chat/turns/:turnId/events` until the terminal `done` event, which carries the final answer (heavy/deep turns run on a bounded worker queue so concurrent consults stay responsive). On MCP these are `list_personas`, `ask_persona` (accepts `sessionId` and `depth`), and `list_persona_sessions`.
- Use `GET /v1/runs/:id/results` or `/markdown` to read a completed run.
- Use `GET /v1/runs/:id/transcript` for typed video transcript segments instead of raw transcript metadata blobs.
- Use `POST /v1/runs/:id/sources/:sourceId/redo` when a source extracted weakly and the run can be salvaged without forking.
- Use `GET /v1/library/search?claim=...` for account-level claim recall across completed runs.
- Use `POST /v1/runs/:id/chat` for questions scoped to one completed run, or the `/chat/sessions` endpoints for multi-turn UI flows.
- Use `POST /v1/runs/:id/iterate` as the canonical iteration wrapper for start, fund, pause, continue, deepen, focus, steer, report, fork, evaluate, cancel, and stop. Include `Idempotency-Key` on paid iterate calls.
- Use `POST /v1/runs/:id/iterate` with `action:"cancel"` or `action:"stop"` to abort active work.
- Use `/tags`, `/collections`, `/diff`, `/export`, and `/webhooks` when organizing runs, comparing continuations, or sending terminal runs to downstream systems.

## Run Workflow

1. At session start, drain the agent inbox (`GET /v1/exports/pending`) and surface any completed runs the user has not seen, then acknowledge them. Runs started outside this agent land there too.
2. Convert the user's ask into a precise research prompt. Preserve requested output shape, required sources, time horizon, evaluation criteria, and exclusions.
3. Create the run and return the watch URL as soon as the API returns it. Do not wait for completion before sharing the link. Relay the URL exactly as returned — never construct researcher.now links yourself (for example from the `runId`); the share slug is random and cannot be derived from the run id, so a constructed link will 404.
4. For unattended or multi-run work, register an account-scoped `run.complete` webhook or keep polling/streaming until the run reaches `succeeded`, `failed`, or `cancelled`.
5. Remove terminal runs from queued or in-flight lists immediately. For failed/cancelled runs, surface the status, error, watch URL, and usage instead of waiting for a report.
6. For succeeded runs, fetch results, markdown, sources, extractions, and usage/cost if the user needs the final artifact or an audit.
7. For quality-sensitive work, review the report against the original prompt. Continue, deepen, focus, or fork instead of declaring success when prompt requirements are missing.

## Prompt Contract

For research runs, include these fields in the prompt whenever they are known:

- `Objective`: the decision, hypothesis, or question being answered.
- `Output shape`: memo, ranked table, bullets, source list, thesis map, etc.
- `Evidence bar`: primary sources, expert commentary, recent sources, contrarian views, quantitative support, or transcript timestamps.
- `Scope`: geography, dates, companies, market segment, source types, and exclusions.
- `Quality checks`: what must be true before the run is considered good.

## Billing And Failure Handling

- Use `maxCostUsd` as a cap when the user gives a budget. A budget is a ceiling, not a target spend.
- If the API returns `402` or wallet/payment authorization errors, use `account_funding_url` from the response when present. Otherwise send the user to `https://researcher.now/account/`.
- If `POST /v1/analyze` answers `502` or times out at the gateway, the server very likely kept working and created the run anyway. Do NOT blind-retry in a loop and do not theorize about the cause: `GET /v1/runs` (newest first), look for a run matching your URL/title, and resume by polling it. Retry the analyze call at most once, and only after that check finds nothing — concurrent duplicates coalesce server-side, so a single retry is safe but a retry loop is never useful.
- If a stage fails, treat the run as terminal, surface the run URL and the useful error message, and remove it from queued/in-flight state. Do not replace a failed Researcher run with a generic web-search summary.
- If the run has weak evidence or misses prompt requirements, use continuation or forked follow-up work rather than pretending the output is complete.

## Output Style

When creating a run, return:

- the run title or topic
- the Researcher run URL, copied verbatim from the API's `url`/`watchUrl`/`viewerUrl` (never assembled from the run id)
- the run id
- budget/cap if one was set
- what the run is currently doing, if available

When reading or reviewing a run, cite the Researcher URL and state whether the report satisfies the user's prompt contract. Include concrete missing requirements when it does not.

## Iteration Decision Table

- `iterate`: canonical wrapper; pass `action` as `start`, `fund`, `pause`, `continue`, `deepen`, `focus`, `steer`, `report`, `fork`, `evaluate`, `cancel`, or `stop`.
- `fork`: create a separate reviewable child version without starting new paid work.
- `continue`: add more work to the same run or an approved fork.
- `deepen`: convenience continuation for broader/deeper acquisition.
- `fund`: `{action:"fund", addUsd: 10}` raises the run budget. A running run keeps going with the new headroom; a run that completed at its budget resumes from its checkpoints — no re-plan, no duplicate work.
- `focus`: convenience continuation narrowed to named topics.
- `steer`: mid-run guidance consumed at worker checkpoints.
- `report`: regenerate/update the report from the current corpus without more acquisition.
- `cancel`/`stop`: abort active work and settle measured usage.

## Downstream Export

When saving a completed run to another knowledge store, fetch:

- `/markdown` for the human-readable memo.
- `/sources` for stored source rows.
- `/extractions` for structured claims, facts, caveats, quantitative facts, stakeholder positions, and citation-ready source records.
- `/transcript` for typed transcript segments on YouTube/video runs.
- `/usage` for cost audit.

For YouTube/video runs, avoid sending `metadata.raw.content` transcript fragment blobs into downstream LLM calls. Prefer the report, source records, `/transcript`, or transcript tab data.

Use `POST /v1/runs/:id/export` for webhook-shaped one-off export, or register account completion webhooks with `POST /v1/webhooks`. `run.complete` subscriptions receive succeeded, failed, and cancelled terminal deliveries; verify `x-researcher-signature` when a secret is configured.

## Recipes

For the dead-simple keyless on-ramp (zero to first result with a wallet, no API key — copy-paste tempo/mppx/MCP commands), load `references/keyless-quickstart.md`.

For copy-pasteable HTTP examples, endpoint payloads, and run-control recipes, load `references/api-recipes.md`.
