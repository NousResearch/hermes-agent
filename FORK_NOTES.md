# Rouse Gordon Hermes Agent Fork Notes

_Last updated: 2026-05-12_

This repository is a deployment-focused fork of the upstream Nous Research
[`hermes-agent`](https://github.com/NousResearch/hermes-agent). The upstream
project remains the base product; this fork adds Gordon Rouse's Railway-hosted
Telegram agent operations, personal automations, static wiki publishing workflow,
and safety/observability guardrails.

Use this file as the high-level map of what is intentionally different from
upstream. It is not a full changelog; inspect `git diff upstream/main...main` for
exact code changes.

## Primary goals of this fork

- Run Hermes as a long-lived **Railway Telegram gateway** instead of only a local
  CLI/gateway install.
- Preserve runtime-authored skills and operational scripts by syncing them back
  to this GitHub fork.
- Keep Gordon's personal workflows available after redeploys: wiki publishing,
  daily reports, job/weather/flight/todo automations, and self-restart skills.
- Reduce avoidable operational noise: false crash alerts, duplicate restart
  notices, repeated security prompts for known-good hosts, and unexpected model
  downgrades.
- Add guardrails against accidentally committing secrets from the Railway
  environment.

## Container and Railway deployment changes

### `Dockerfile`

Key differences from upstream:

- Adds `curl` because the self-restart/redeploy workflow calls Railway/Cloudflare
  endpoints and because Python `urllib` is more prone to Cloudflare/Tirith false
  positives in this deployment.
- Sets `TIRITH_IGNORED_RULES=lookalike_tld` by default so Railway's
  `backboard.railway.app` does not cause repeated approval prompts.
- Makes `/opt/hermes` owned by the `hermes` user so the entrypoint can replace
  selected image-baked directories with symlinks into the volume-backed Git
  checkout.
- Uses `CMD ["gateway", "run"]` so Railway starts the Telegram gateway by
  default.

### `docker/entrypoint.sh`

This is the biggest operational customization. It adds:

- **Secret-hygiene memory rule**: appends a persistent hard rule telling the
  agent not to reveal or write secret env var values.
- **Source-of-truth watcher bootstrap**: when `GITHUB_TOKEN` is present, clones
  `GITHUB_REPO` into the Railway volume and links `skills/`, `scripts/`,
  `tools/`, and `gateway/` from that checkout. This lets runtime skill/script
  edits survive restarts and be picked up without rebuilding the image.
- **Background source watcher**: starts `scripts/source_watcher.py`, which
  auto-commits safe runtime edits back to this fork.
- **Healthcheck server**: starts `scripts/healthcheck_server.py` so Railway has a
  simple HTTP endpoint even though Hermes itself is a polling bot.
- **Daily cost report daemon**: starts `scripts/cost_report_daemon.py` when the
  OpenRouter and Telegram env vars are configured.
- **Config enforcement on boot**: reapplies critical model/approval/compression
  settings so a fresh or late-mounted Railway volume does not silently revert to
  expensive/default upstream settings.

## Source watcher and secret safety

### `scripts/source_watcher.py`

Watches the volume-backed checkout and periodically commits/pushes runtime edits
back to GitHub. Important behavior:

- Stages and commits changed files from the runtime checkout.
- Runs a narrow Ruff gate for Python syntax/undefined-name failures before
  pushing.
- Refuses to commit files containing the literal value of env vars whose names
  look sensitive (`KEY`, `TOKEN`, `SECRET`, `PASSWORD`).
- Logs blocked commits to `/opt/data/logs/watcher-blocked.log` and sends Telegram
  alerts when configured.

### `.env.example`

Adds documentation for the GitHub token needed by the watcher and the optional
`GITHUB_REPO` override.

## Gateway behavior changes

### `gateway/run.py`

Deployment-specific changes include:

- Appends per-request metadata to a local JSONL request log for daily cost and
  usage reporting.
- Suppresses the gateway's old post-`/restart` success message because the
  Railway entrypoint already sends the single desired "I'm back!" notification.
- Treats `.planned_redeploy` as an expected self-restart marker so self-redeploys
  do not look like crashes.
- Exits cleanly on signal-driven shutdown in Railway. Upstream's systemd-oriented
  behavior exits non-zero to trigger `Restart=on-failure`; on Railway that creates
  noisy false "Deploy Crashed" notifications during ordinary redeploys.

## Model-routing changes

### `run_agent.py`

Adds a Codex model-substitution guard:

- Detects when the Codex/OpenAI backend serves a different model variant than
  requested, such as a smaller `mini`/`nano` tier.
- Treats that response as invalid so the configured fallback chain can route to
  MiniMax or another fallback provider.
- Sets a short per-model cooldown to avoid paying for repeated doomed Codex calls
  during an outage/quota window.
- Sends best-effort Telegram notifications when substitution starts and when the
  requested Codex model is served correctly again.

## Security prompt changes

### `tools/tirith_security.py` and `tools/approval.py`

- Adds `tirith_ignored_rules` config support plus `TIRITH_IGNORED_RULES` env var.
- Drops allowlisted Tirith findings and downgrades the verdict to `allow` if all
  findings were ignored.
- Includes Tirith `rule_id` values in approval descriptions so future false
  positives are easier to identify.

This is mainly to suppress the known-good Railway `.app` lookalike-TLD warning
without disabling Tirith entirely.

## Added operational scripts

This fork adds scripts that are specific to Gordon's deployment and workflows:

- `scripts/healthcheck_server.py` — tiny Railway healthcheck server.
- `scripts/source_watcher.py` — runtime change watcher/auto-committer.
- `scripts/cost_report.py`, `scripts/cost_report_daemon.py`,
  `scripts/cost_daemon_wrapper.py` — daily usage/cost reporting.
- `scripts/set_active_provider.py`, `scripts/audit_auth.py`,
  `scripts/add_minimax_auth.py`, `scripts/test_minimax.py` — provider/auth
  operations.
- `scripts/todo_manager.py`, `scripts/todo_wrapper.py` — Gordon's Telegram-driven
  daily todo list storage and formatting.

## Added tools

### `tools/publish_html.py`

Adds a static HTML publishing helper for Gordon's Cloudflare Pages workflows. It
supports publishing generated/static HTML artifacts from the agent environment
without requiring the user to manually run the deploy steps.

## Added and modified skills

This fork carries many deployment- and Gordon-specific skills in addition to
upstream skills. Major additions include:

- Hermes operations:
  - `skills/autonomous-ai-agents/hermes-agent/self-restart/`
  - `skills/autonomous-ai-agents/hermes-agent/self-rebuild/`
  - `skills/autonomous-ai-agents/hermes-agent/model-selection/`
  - deployment references for Railway, Cloudflare, Telegram history, request
    tracking, and provider migration.
- Static web/wiki publishing:
  - `skills/creative/html-to-cloudflare/`
  - `skills/research/llm-wiki/` extensions for Gordon's static HTML wiki and
    public standalone KBs.
- Personal productivity workflows:
  - daily LLM usage report,
  - daily todo list,
  - daily weather brief,
  - flights,
  - job search,
  - maps/commute references,
  - weather.
- Research/utility additions:
  - `skills/research/quadtree-refs/`
  - LinkedIn browser automation notes,
  - debugging references for the source watcher.

These skills intentionally contain Gordon-specific operational knowledge. Do not
upstream them without first removing private deployment assumptions.

## GitHub Actions changes

The upstream GitHub Actions workflows were removed in this fork. This repo is
primarily maintained by the Railway runtime watcher and manual/agent commits, not
by upstream's CI/release automation. If this fork starts accepting broader code
changes, reintroduce CI before relying on it as a general development branch.

## Static site and landing content

- `landing.html` is fork-specific static content.
- Cloudflare Pages and wiki publishing references live under the added skills,
  especially `skills/creative/html-to-cloudflare/` and
  `skills/research/llm-wiki/`.

## Upstream sync guidance

When pulling from upstream:

1. Treat upstream as product source; treat this file as the deployment overlay
   inventory.
2. Review conflicts especially in:
   - `Dockerfile`
   - `docker/entrypoint.sh`
   - `gateway/run.py`
   - `run_agent.py`
   - `tools/tirith_security.py`
   - `tools/approval.py`
   - `skills/`
3. Avoid blindly overwriting `skills/`, `scripts/`, and `tools/` symlink-aware
   behavior unless the Railway deployment path is being redesigned.
4. Re-check secret hygiene after merge conflicts; never paste real token values
   into docs, skills, logs, or commits.

## What should not be upstreamed as-is

- Gordon-specific skills, schedules, wiki references, and personal productivity
  workflows.
- Railway-specific restart/healthcheck assumptions unless generalized behind
  config flags.
- Request logging/cost reporting that assumes Telegram/OpenRouter env vars.
- Any static HTML or Cloudflare Pages workflow that names Gordon's deployment.

## What could be generalized upstream later

- Configurable Tirith ignored-rule support.
- Safer restart notification lifecycle for gateway deployments.
- Source watcher ideas, if rewritten as an opt-in plugin with strong secret
  scanning and explicit user setup.
- Codex model-substitution detection, if provider behavior remains an issue and
  can be implemented provider-neutrally.
