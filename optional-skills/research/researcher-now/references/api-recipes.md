# Researcher API Recipes

Use `RESEARCHER_API_KEY` or `RESEARCHER_TOKEN` for bearer auth.

```bash
export RESEARCHER_BASE_URL="${RESEARCHER_BASE_URL:-https://researcher.now}"
export RESEARCHER_AUTH_HEADER="Authorization: Bearer ${RESEARCHER_API_KEY:-$RESEARCHER_TOKEN}"
```

## Health And Auth

```bash
curl -fsS "$RESEARCHER_BASE_URL/health"

curl -fsS "$RESEARCHER_BASE_URL/v1/me" \
  -H "$RESEARCHER_AUTH_HEADER"
```

## Plan Before Spending

Use `preflightPlan:true` for vague or high-stakes prompts.

```bash
curl -fsS -X POST "$RESEARCHER_BASE_URL/v1/runs" \
  -H "$RESEARCHER_AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: plan-$(date +%s)" \
  -d '{
    "preflightPlan": true,
    "requestedBy": "customer-agent",
    "source": {"type": "topic", "topic": "Personal AI adoption and food-market impact"},
    "depth": "standard",
    "instructions": "Return success criteria, evidence lanes, likely sources, and uncertainty."
  }'
```

## Create Runs

All new work starts through `POST /v1/runs`. Change `source.type` for topics, URLs, feeds, or videos.

```bash
curl -fsS -X POST "$RESEARCHER_BASE_URL/v1/runs" \
  -H "$RESEARCHER_AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: run-$(date +%s)" \
  -d '{
    "requestedBy": "customer-agent",
    "source": {"type": "topic", "topic": "Personal AI adoption and food-market impact"},
    "mode": "collection",
    "depth": "standard",
    "instructions": "Research the thesis, cite sources, map demand shifts, list implications, and flag uncertainty.",
    "limits": {"maxResearchLoops": 3, "maxCostUsd": 25}
  }'
```

```bash
curl -fsS -X POST "$RESEARCHER_BASE_URL/v1/runs" \
  -H "$RESEARCHER_AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: source-$(date +%s)" \
  -d '{
    "requestedBy": "customer-agent",
    "source": {"type": "url", "url": "https://example.com/report", "scope": "domain"},
    "depth": "standard",
    "limits": {"maxSources": 300, "maxCostUsd": 25}
  }'
```

```bash
curl -fsS -X POST "$RESEARCHER_BASE_URL/v1/runs" \
  -H "$RESEARCHER_AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: feed-$(date +%s)" \
  -d '{
    "requestedBy": "customer-agent",
    "source": {"type": "feed", "url": "https://example.com/feed.xml", "limit": 50},
    "limits": {"maxCostUsd": 10}
  }'
```

```bash
curl -fsS -X POST "$RESEARCHER_BASE_URL/v1/runs" \
  -H "$RESEARCHER_AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: video-$(date +%s)" \
  -d '{
    "requestedBy": "customer-agent",
    "source": {"type": "video", "url": "https://www.youtube.com/watch?v=...", "transcriptMode": "native"},
    "instructions": "Extract transcript-grounded concepts, quotes, references, and gaps."
  }'
```

## Read A Run

```bash
RUN_ID="..."

curl -fsS "$RESEARCHER_BASE_URL/v1/runs/$RUN_ID/job" -H "$RESEARCHER_AUTH_HEADER"
curl -N "$RESEARCHER_BASE_URL/v1/runs/$RUN_ID/stream" -H "$RESEARCHER_AUTH_HEADER"
curl -fsS "$RESEARCHER_BASE_URL/v1/runs/$RUN_ID/results" -H "$RESEARCHER_AUTH_HEADER"
curl -fsS "$RESEARCHER_BASE_URL/v1/runs/$RUN_ID/markdown" -H "$RESEARCHER_AUTH_HEADER"
curl -fsS "$RESEARCHER_BASE_URL/v1/runs/$RUN_ID/sources" -H "$RESEARCHER_AUTH_HEADER"
curl -fsS "$RESEARCHER_BASE_URL/v1/runs/$RUN_ID/extractions" -H "$RESEARCHER_AUTH_HEADER"
curl -fsS "$RESEARCHER_BASE_URL/v1/runs/$RUN_ID/transcript" -H "$RESEARCHER_AUTH_HEADER"
curl -fsS "$RESEARCHER_BASE_URL/v1/runs/$RUN_ID/usage" -H "$RESEARCHER_AUTH_HEADER"
```

Treat `succeeded`, `failed`, and `cancelled` as terminal statuses. Remove terminal runs from queued or in-flight lists immediately. For failed or cancelled runs, surface status, error, watch URL, and usage instead of waiting for a report.

## Completion Webhooks

For unattended or multi-run agents, register an account-scoped terminal webhook before starting work:

```bash
curl -fsS -X POST "$RESEARCHER_BASE_URL/v1/webhooks" \
  -H "$RESEARCHER_AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d '{
    "event": "run.complete",
    "url": "https://example.com/researcher-webhook",
    "secret": "at-least-8-chars"
  }'
```

```bash
curl -fsS "$RESEARCHER_BASE_URL/v1/webhooks" -H "$RESEARCHER_AUTH_HEADER"
curl -fsS -X DELETE "$RESEARCHER_BASE_URL/v1/webhooks/$WEBHOOK_ID" -H "$RESEARCHER_AUTH_HEADER"
```

`run.complete` subscriptions receive terminal deliveries for succeeded, failed, and cancelled runs. The `x-researcher-event` header and payload event identify the actual terminal event, such as `run.succeeded` or `run.failed`. When a secret is configured, verify `x-researcher-signature`: `sha256=` plus HMAC-SHA256 of the raw JSON body.

## Agent Inbox

Every terminal run on the account lands in the agent inbox as a pending export — including runs started on the researcher.now web app, the API, or Discord. No registration required. Drain it at session start:

```bash
curl -fsS "$RESEARCHER_BASE_URL/v1/exports/pending?limit=20" -H "$RESEARCHER_AUTH_HEADER"
```

Surface each run's `urls.watch` link to the user, then acknowledge what you consumed so it stops appearing as pending:

```bash
curl -fsS -X POST "$RESEARCHER_BASE_URL/v1/exports/ack" \
  -H "$RESEARCHER_AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d '{"ids": ["EXPORT_ID"]}'
```

Acknowledging is per-account: any agent on the account drains the same inbox. Full results stay available through the normal run endpoints after acknowledging.

## Standing Topics

Create a standing research topic when the user wants ongoing coverage of a subject instead of a one-off report. `prompt` is required — state the subject plus the user's stake in it. Seeds become standing watch targets (X accounts, GitHub profiles, RSS feeds, websites) plus an automatic wide-search lane; creating a topic fires a bootstrap deep-research run automatically:

```bash
curl -fsS -X POST "$RESEARCHER_BASE_URL/v1/topics" \
  -H "$RESEARCHER_AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Solid-state battery commercialization timelines. I advise an industrial client deciding when to commit a pack redesign; I care about manufacturing milestones, not lab records.",
    "title": "Solid-state batteries",
    "seeds": {
      "xAccounts": ["batteryresearcher"],
      "githubProfiles": ["batterylab"],
      "feeds": ["https://example.com/battery-news.xml"],
      "domains": ["batteryreports.example.com"]
    },
    "cadence": "daily",
    "monthlyBudgetUsd": 50
  }'
```

`title`, `seeds`, `cadence` (`realtime` | `hourly` | `daily` | `weekly`, default `daily`; `realtime` scans roughly every 10 minutes), and `monthlyBudgetUsd` are optional. `seeds.xAccounts` accepts up to 1,000 handles — paste a whole list; large lists are scanned in batches. `seeds.githubProfiles` takes up to 100 GitHub usernames (strip any `@` or `github.com/` prefix), watched via the public events API for releases, new repos, pushes, and opened pull requests; they appear on `GET /v1/topics/:id` as watch targets with kind `github`. The topic starts as `bootstrapping`, becomes `active` once the bootstrap run completes, or `bootstrap_failed` if the bootstrap run could not start — check `topic.status` on the 201 response. Scanning is independent of that lifecycle: the topic scans on its cadence (appending `scan_digest` events) from creation, including while bootstrapping, and stops only when paused or archived.

List topics and read one topic with its recent events (including `scan_digest` items), watch targets, and docs:

```bash
curl -fsS "$RESEARCHER_BASE_URL/v1/topics" -H "$RESEARCHER_AUTH_HEADER"
curl -fsS "$RESEARCHER_BASE_URL/v1/topics/$TOPIC_ID" -H "$RESEARCHER_AUTH_HEADER"
```

Every topic object carries a `slug` (a URL-friendly form of its title, unique within your account). `GET /v1/topics/:idOrSlug` accepts either the slug or the uuid, so `/v1/topics/ai-leaders` and `/v1/topics/$TOPIC_ID` resolve the same topic.

Rename a topic with `PATCH /v1/topics/:id` — this regenerates the slug and appends a `title_updated` event:

```bash
curl -fsS -X PATCH "$RESEARCHER_BASE_URL/v1/topics/$TOPIC_ID" \
  -H "$RESEARCHER_AUTH_HEADER" -H "Content-Type: application/json" \
  -d '{"title": "AI Leaders"}'
```

The response is `{ "topic": { ... } }` with the new `title` and `slug`.

The detail response includes `docs` — two versioned markdown documents, each `{ "version": 1, "contentMarkdown": "...", "summary": "...", "createdAt": "..." }` or `null` before seeding:

- `docs.thesis` — stable top-level beliefs seeded from the bootstrap research run: numbered claims with confidence and falsification conditions. Never auto-edited; synthesis only flags tensions against it.
- `docs.working` — the living idea layer: new scan items are continuously synthesized into it under pressure triggers (enough new items, staleness, first items after seeding), not on a fixed clock. Each reorganization appends a new version.

Doc updates show up in the topic's events as `docs_seeded` (both docs seeded from the bootstrap report) and `doc_synthesized` (each working-doc reorganization; its `message` is the what-changed summary and its `metadata.tensions` flags thesis claims under pressure, referencing claim ids). Errors surface as `docs_seed_failed` and `doc_synthesis_failed`; `docs_seed_skipped` means seeding was skipped (for example, no bootstrap report markdown was found).

## Chat With A Run

```bash
curl -fsS -X POST "$RESEARCHER_BASE_URL/v1/runs/$RUN_ID/chat" \
  -H "$RESEARCHER_AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d '{"message":"What are the strongest sources for the main conclusion?"}'
```

## Iterate

```bash
curl -fsS -X POST "$RESEARCHER_BASE_URL/v1/runs/$RUN_ID/iterate" \
  -H "$RESEARCHER_AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: iterate-$(date +%s)" \
  -d '{
    "action": "focus",
    "focus": ["contrarian evidence", "quantitative support"],
    "instructions": "Tighten the report against the original prompt contract.",
    "limits": {"maxCostUsd": 5}
  }'
```

Supported actions are `start`, `pause`, `continue`, `deepen`, `focus`, `steer`, `report`, `fork`, `evaluate`, `cancel`, and `stop`.

```bash
curl -fsS -X POST "$RESEARCHER_BASE_URL/v1/runs/$RUN_ID/sources/$SOURCE_ID/redo" \
  -H "$RESEARCHER_AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d '{"instructions":"Pay special attention to concrete caveats."}'
```

## Organize, Search, Export

```bash
curl -fsS "$RESEARCHER_BASE_URL/v1/runs/$RUN_ID/diff?against=$PARENT_RUN_ID" -H "$RESEARCHER_AUTH_HEADER"
curl -fsS -X POST "$RESEARCHER_BASE_URL/v1/runs/$RUN_ID/tags" -H "$RESEARCHER_AUTH_HEADER" -H "Content-Type: application/json" -d '{"tags":["customer-ready"]}'
curl -fsS "$RESEARCHER_BASE_URL/v1/library/search?claim=agent%20payment%20rails" -H "$RESEARCHER_AUTH_HEADER"
curl -fsS "$RESEARCHER_BASE_URL/v1/library/evidence?q=agent%20payment%20rails" -H "$RESEARCHER_AUTH_HEADER"
curl -fsS "$RESEARCHER_BASE_URL/v1/library/contradictions" -H "$RESEARCHER_AUTH_HEADER"
curl -fsS -X POST "$RESEARCHER_BASE_URL/v1/runs/$RUN_ID/export" -H "$RESEARCHER_AUTH_HEADER" -H "Content-Type: application/json" -d '{"target":{"type":"webhook","url":"https://example.com/researcher-webhook"}}'
```

## Enumerate Account Work

```bash
curl -fsS "$RESEARCHER_BASE_URL/v1/runs" -H "$RESEARCHER_AUTH_HEADER"
curl -fsS "$RESEARCHER_BASE_URL/v1/account/chats" -H "$RESEARCHER_AUTH_HEADER"
curl -fsS "$RESEARCHER_BASE_URL/v1/account/balance" -H "$RESEARCHER_AUTH_HEADER"
curl -fsS "$RESEARCHER_BASE_URL/v1/account/keys" -H "$RESEARCHER_AUTH_HEADER"
```
