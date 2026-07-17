# Hermes Operational Readiness Build Plan V71-V80

This plan picks up after V61-V70. V61-V70 moved the system from theoretical adapter contracts into guarded live-capable paths for production sweeps, promotion plans, GitHub secret-name scans, incident creation, artifact pointers, billing imports, and Khashi/Media project outcome adapters.

The next goal is not more dashboard surface area. The next goal is to make the live system function end to end in production: credentials configured, production checks running on a schedule, screenshots and logs stored somewhere durable, remote promotion executed through one shared Hetzner rail, incidents routed to the right human channels, provider evals scored, and all production dashboards emitting standard Hermes outcomes.

## Current Gap Assessment

| Area | Current State | Gap / Limitation | Enhancement Needed |
| --- | --- | --- | --- |
| Production sweeps | DNS/TLS/HTTP checks are implemented and approval-gated. | No production browser screenshot runner yet. | Add Playwright/browser screenshot execution with artifact capture. |
| Hetzner promotion | Promotion command plans are built and breaker-gated. | Real SSH transport and rollback artifact capture are not attached. | Wire shared Hetzner SSH promotion runner with command output, rollback plan, and post-deploy verification. |
| Secrets | GitHub secret/variable name scanner exists. | Hetzner env-file and provider-native scans are not implemented. | Add server-side presence checks that never expose values. |
| Billing | Manual billing import API exists. | Direct provider billing APIs are not integrated. | Keep manual import as source of truth first; add providers one at a time. |
| Project outcomes | Khashi VC and Media Engine emit `/api/hermes/outcomes`. | Other projects do not emit standardized outcomes yet. | Roll out the same endpoint to Hermes OS, Business Mapper, Media Business Operations, Investing System, Meal Assistant, and any future dashboards. |
| Breakers | Sweep and promotion paths are protected. | Scheduler, provider, project deploy, and autopilot paths still need breaker middleware. | Wrap each live entry point before increasing autonomy. |
| Incidents | Failed/blocked sweeps, deploys, missing secrets, and breaker blocks create records. | Incidents do not fan out to Discord/Telegram/email yet. | Add incident subscription fanout with dedupe and severity routing. |
| Artifacts | Artifact pointers and fingerprints are stored. | Durable artifact backend and retention jobs are not chosen. | Choose local mounted volume, S3/R2, or both; add lifecycle cleanup. |
| Provider evals | Eval records and golden batch scoring schema exist. | Real model calls and artifact scoring are still gated. | Add small paid-provider eval runner with budget breaker and manual approval. |
| Release trains | Release train plans can be recorded. | Multi-project live execution remains manual-first. | Execute after sweep, secret, artifact, incident, and promotion rails are proven. |

## Running Operator Checklist

Use this checklist to make the system functional in production. Do not store raw secret values in docs, commits, dashboard records, or incident payloads.

### GitHub / Repository

- [ ] Confirm `gh auth status` works locally for the repo owner.
- [ ] Add shared GitHub Actions secrets:
  - [ ] `HETZNER_HOST`
  - [ ] `HETZNER_USER`
  - [ ] `HETZNER_SSH_KEY`
  - [ ] `HETZNER_SSH_PORT` if not `22`
- [ ] Add shared GitHub Actions variable or secret:
  - [ ] `PRODUCTION_BASE_DOMAIN=tlccapitalgroup.com`
- [ ] Confirm project-specific production target mapping exists for every dashboard.
- [ ] Run Hermes GitHub secret-name scan for each repo after secrets are added.

### Hetzner / Production Server

- [ ] Confirm shared deploy root exists, expected default: `/root/apps/deploy`.
- [ ] Confirm shared promotion script exists, expected default: `/root/apps/deploy/scripts/promote-service.sh`.
- [ ] Confirm each production app checkout exists under the expected path.
- [ ] Confirm Docker and Docker Compose are installed.
- [ ] Confirm production Caddy routes exist for each dashboard domain.
- [ ] Confirm production `.env` files exist on the server for each app.
- [ ] Confirm production services restart through the shared promotion script, not bespoke per-project deploy commands.
- [ ] Confirm rollback path for each app is documented and executable.

### DNS / Routing

- [ ] Confirm every dashboard subdomain resolves to the Hetzner server.
- [ ] Confirm HTTPS certificates issue successfully for every dashboard route.
- [ ] Confirm `/health`, `/readyz`, `/dashboard-snapshot`, or equivalent health endpoints are reachable per project.
- [ ] Add each production URL to Hermes production sweep targets.

### Project Secrets / API Keys

- [ ] Khashi VC:
  - [ ] `KALSHI_BASE_URL`
  - [ ] `KALSHI_API_KEY` if the API requires bearer access.
  - [ ] `AMARI_AUTH_USERNAME`
  - [ ] `AMARI_AUTH_PASSWORD`
  - [ ] `AMARI_SESSION_SECRET`
  - [ ] `AMARI_ADMIN_TOKEN`, `AMARI_VIEWER_TOKEN`, `AMARI_RESEARCHER_TOKEN` if token access remains enabled.
- [ ] Media Engine:
  - [ ] `OPENAI_API_KEY` if OpenAI generation is enabled.
  - [ ] `DEEPSEEK_API_KEY` if DeepSeek routing is enabled.
  - [ ] `FIRECRAWL_API_KEY` if production story research requires Firecrawl.
  - [ ] `DISCORD_BOT_TOKEN`
  - [ ] `DISCORD_CLIENT_ID`
  - [ ] `DISCORD_GUILD_ID`
  - [ ] `DISCORD_DELIVERY_WEBHOOK_URL`
  - [ ] Provider keys for image/video providers actually used in production.
- [ ] Hermes / Agent layer:
  - [ ] GitHub access for secret scanning and workflow checks.
  - [ ] Provider keys only where the agent is allowed to make paid calls.
  - [ ] Budget breaker limits before enabling paid provider evals.

### Artifact Storage

- [ ] Choose artifact backend:
  - [ ] Local mounted Hetzner volume for first pass.
  - [ ] Cloudflare R2 / S3 for durable cross-server storage.
  - [ ] Hybrid: local hot cache plus R2/S3 long-term retention.
- [ ] Define retention windows for screenshots, logs, traces, receipts, invoices, evals, and generated media.
- [ ] Configure artifact base URL or signed-download strategy.
- [ ] Add cleanup jobs so artifact storage does not grow without bound.

### Incident Fanout

- [ ] Decide which channels receive incident notifications:
  - [ ] Discord operations channel.
  - [ ] Telegram operator channel.
  - [ ] Email or future support inbox.
- [ ] Define severity routing:
  - [ ] `critical`: immediate notification.
  - [ ] `high`: batched or immediate depending on production impact.
  - [ ] `warning`: dashboard-only unless repeated.
- [ ] Add dedupe windows so one broken route does not spam the channel.
- [ ] Add incident acknowledgement and resolution workflow.

### Governance / Breakers

- [ ] Keep global kill switch tested.
- [ ] Add breaker middleware to:
  - [ ] Khashi scheduler admission.
  - [ ] Khashi market sync / poller capacity changes.
  - [ ] Media Engine autopilot.
  - [ ] Media Engine provider generation calls.
  - [ ] Shared deploy/promotion commands.
  - [ ] Provider eval runner.
- [ ] Define budget caps for:
  - [ ] OpenAI.
  - [ ] DeepSeek.
  - [ ] Firecrawl.
  - [ ] Image generation.
  - [ ] Video generation.
  - [ ] Hosting/storage.

## V71-V80 Version Plan

| Version | Capability | Status | Outcome |
| --- | --- | --- | --- |
| V71 | Production Screenshot Runner | `[ ]` | Approved sweeps capture browser screenshots and store artifact pointers. |
| V72 | Real Hetzner Promotion Transport | `[ ]` | Hermes can run the shared promotion script over SSH with approval, breaker checks, logs, and rollback evidence. |
| V73 | Server Secret Posture Scanner | `[ ]` | Hermes can verify required production env names on Hetzner without exposing values. |
| V74 | Incident Notification Fanout | `[ ]` | Incidents route to Discord/Telegram/email by severity with dedupe and acknowledgement. |
| V75 | Durable Artifact Backend | `[ ]` | Screenshots, logs, traces, receipts, invoices, eval artifacts, and generated evidence are retained safely. |
| V76 | Remaining Project Outcome Adapters | `[ ]` | All production dashboards emit standard `/api/hermes/outcomes`. |
| V77 | Breaker Middleware Rollout | `[ ]` | Scheduler, provider, deploy, and autopilot live paths are all breaker-aware. |
| V78 | Provider Eval Execution | `[ ]` | Golden eval tasks can run through approved providers with budget caps and scored artifacts. |
| V79 | Billing Provider Integrations | `[ ]` | Provider billing exports/API data reconcile against manual imports. |
| V80 | Release Train Execution | `[ ]` | Multi-project production promotion can run only after sweeps, secrets, artifacts, incidents, and breakers pass. |

## V71 Production Screenshot Runner

- [ ] Add browser runner abstraction for production URL screenshots.
- [ ] Run only when production sweep is live, admin-approved, and breaker-allowed.
- [ ] Store screenshot artifact pointer, hash, viewport, URL, and timestamp.
- [ ] Attach screenshot artifacts to production sweep records.
- [ ] Fail the sweep when the screenshot is blank, unreachable, or route-mismatched.

## V72 Real Hetzner Promotion Transport

- [ ] Add SSH transport wrapper around the shared promotion script.
- [ ] Pass service key, version, migration flag, and production base domain.
- [ ] Capture stdout/stderr as redacted artifact pointers.
- [ ] Run post-deploy health and snapshot checks.
- [ ] Attach rollback instructions and previous-version evidence.

## V73 Server Secret Posture Scanner

- [ ] Add server-side env-name scanner command plan.
- [ ] Verify required names exist without printing values.
- [ ] Support per-project required secret manifests.
- [ ] Create incidents for missing required production env names.
- [ ] Track rotation age when metadata is available.

## V74 Incident Notification Fanout

- [ ] Add subscription target records.
- [ ] Add Discord webhook fanout.
- [ ] Add Telegram fanout if configured.
- [ ] Add dedupe key and cooldown handling.
- [ ] Add acknowledgement and resolved status updates.

## V75 Durable Artifact Backend

- [ ] Choose local volume and/or R2/S3.
- [ ] Add artifact write adapter.
- [ ] Add artifact read URL/signed URL strategy.
- [ ] Add lifecycle cleanup by artifact type and retention class.
- [ ] Add storage usage dashboard evidence.

## V76 Remaining Project Outcome Adapters

- [ ] Add `/api/hermes/outcomes` to Hermes OS.
- [ ] Add `/api/hermes/outcomes` to Business Mapper.
- [ ] Add `/api/hermes/outcomes` to Media Business Operations.
- [ ] Add `/api/hermes/outcomes` to Investing System.
- [ ] Add `/api/hermes/outcomes` to Meal Assistant.
- [ ] Add registry validation that flags missing outcome adapters.

## V77 Breaker Middleware Rollout

- [ ] Wrap Khashi scheduler admission.
- [ ] Wrap Khashi market sync capacity changes.
- [ ] Wrap Media Engine autopilot start/stop/capacity.
- [ ] Wrap Media Engine paid provider generation.
- [ ] Wrap provider eval execution.
- [ ] Add tests proving breakers block live actions.

## V78 Provider Eval Execution

- [ ] Define the first 10 golden tasks for coding, dashboard design, research, and operations.
- [ ] Add model/provider run adapters behind approval.
- [ ] Attach output artifacts and scored rubrics.
- [ ] Compare cost, latency, correctness, and regression risk.
- [ ] Promote provider-routing recommendations only after enough scored runs.

## V79 Billing Provider Integrations

- [ ] Keep manual billing import as baseline.
- [ ] Add OpenAI usage import when credentials and API access are available.
- [ ] Add Firecrawl usage import if available.
- [ ] Add hosting/storage monthly cost import.
- [ ] Add variance report between manual invoice totals and provider usage.

## V80 Release Train Execution

- [ ] Require green production sweep before promotion.
- [ ] Require secret scans to pass.
- [ ] Require breaker checks to pass.
- [ ] Require rollback artifact to exist.
- [ ] Execute promotion per project in controlled order.
- [ ] Stop train on first critical failure.
- [ ] Generate final release evidence report.

## Recommendation

Build V71, V72, V73, and V74 before increasing autonomy. Those four close the most practical production gaps: proving what is live, safely promoting it, knowing required secrets exist, and notifying you when something breaks.
