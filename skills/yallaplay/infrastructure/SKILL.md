---
name: infrastructure
description: Use for setup, dependencies, Hermes provider config, Bedrock/OpenAI routing, S3/storage, logs/cache, web-search fallback, generated storage, and environment admin.
version: 1.1.0
author: YallaPlay
license: Proprietary
metadata:
  hermes:
    tags: [infrastructure, setup, providers, storage, logs, bedrock, hermes, yallaplay]
    related_skills: [coder, codebase-readonly, knowledge]
---

# Infrastructure Skill

## Overview

Use this skill for environment setup, dependencies, provider/model routing, credentials hygiene, storage, generated logs/outputs, web-search fallback, and operational constraints. Keep infrastructure changes boring, reversible, and explicit.

This is the Hermes equivalent of the legacy `personas/infrastructure.md` workflow, adapted to the Hermes pilot repo.

## When to Use

- Installing dependencies, bootstrapping profiles, configuring providers, gateway/dashboard setup.
- S3/object storage, generated outputs/logs/cache, cleanup, environment health.
- Bedrock/OpenAI/OpenRouter routing, model IDs, credential checks, web-search fallback.
- Snowflake admin metadata such as role/warehouse/database setup; use Analytics for metric queries.

Load Engineering too for code changes. Load Collaboration/LiveOps when infrastructure actions affect teammates or production services.

## Required Context

Inspect relevant wiki/project docs:

- `yallaplay-wiki/engineering/infrastructure.md`
- `yallaplay-wiki/engineering/backend_services.md`
- `yallaplay-wiki/engineering/data_warehouse.md`
- `yallaplay-wiki/tools/product/app_insights.md`
- `docs/provider-notes.md`, `docs/bedrock-setup-status.md`, `docs/dashboard-cloudflare.md` when present.
- `config/hermes.example.yaml` and scripts under `scripts/` for reproducible setup.

For Hermes Agent itself, also load the `hermes-agent` skill and use official docs as source of truth.

## Setup and Dependencies

- Prefer reproducible scripts over manual shell state.
- Python is PEP 668-managed on this host; use a venv or existing project environment rather than global `pip` installs.
- Do not assume a command exists; check before using it.
- Keep example config in git, real config/secrets outside git.
- For generated runtime directories (`logs/`, `outputs/`, `.local/`, caches), confirm `.gitignore` coverage before producing large artifacts.

## Storage and Generated Files

- Generated CSV/charts belong under `outputs/` with timestamped/descriptive names.
- Query logs belong under `logs/` when produced by query tools.
- Avoid creating new cloud buckets/resources per task. Use existing documented storage and scoped prefixes.
- Do not wipe caches/logs/state en masse without an explicit narrow scope and reason.

## Provider and Credential Hygiene

- Never print or commit `.env`, `vars.toml`, OAuth caches, cloud credentials, Slack tokens, or private keys.
- Check provider/model configuration with read-only status/doctor commands first.
- Record stable provider/model/region gotchas in `yallaplay-wiki/engineering/infrastructure.md` or `docs/`.
- Active Hermes profile state is operational state. Changing it affects future sessions; say so and verify.

## Hermes Profiles and Skill Visibility

When Claudio skills exist in the repo but do not appear in `/skills`, distinguish repo-local source from profile-installed skills. YallaPlay-owned repo skills under `skills/yallaplay/` should be symlinked into the active profile's `skills/claudio-authored/` category for a clear visual split from bundled skills, then the current TUI needs `/reload-skills` or a new session.

See `references/hermes-profiles-yallaplay.md` for the YallaPlay profile split, skill visibility procedure, and promotion pattern from `claudio-lab` to a future `claudio-prod`.

## Web Search Fallback

- Prefer native web/search tools when available.
- If a local script fallback is needed, run it synchronously with a generous timeout.
- Do not fan out or background web-search subprocesses unless the tool explicitly supports it; legacy notes recorded VM pressure from background fan-out.

## Snowflake Admin Metadata

- Production warehouse reads belong to Analytics.
- Admin metadata (roles, grants, warehouse monitor setup, model/provider environment) belongs here.
- Default Snowflake role conventions from the legacy repo: `READONLY_USER` / `ANALYTICS_MONITOR`; target new grants at `ANALYTICS_MONITOR`, not `PUBLIC`, unless current docs say otherwise.

## Safety

- Do not edit, print, or commit secret files.
- Do not create broad cloud resources or change networking/gateway exposure without confirmation.
- Do not stop/delete the agent, scheduler, gateway, profile state, or wiki unless the user gives a narrow safe instruction.
- Prefer dry-runs and status commands before mutating environment state.

## Hermes TUI/Profile Reference

For recurring Claudio/Hermes TUI issues, use `references/hermes-tui-profiles-compression.md`. It covers:

- Why repo-local skills do not appear in `/skills` until installed into the active profile.
- When to run `/reload-skills`.
- TUI legibility formatting: prefer compact bullets over fenced blocks for short command/profile layouts when the user reports odd blank spacing.
- Why `/compress` may not reduce the visible context much when recent large tool outputs, loaded skills, tool schemas, or system/project instructions dominate.

## Common Pitfalls

1. **Global installs on PEP 668 hosts.** Use venv/uv or project scripts.
2. **Secret-adjacent debugging.** Show key presence, not values.
3. **Profile confusion.** `default` and `claudio-lab` have separate skills/config/state; verify active profile before changing it. `/skills` lists installed skills for the active profile, not repo-local `skills/` folders.
4. **Repo/profile skill drift.** YallaPlay-owned skills should be profile symlinks under the `claudio-authored` category pointing to repo `skills/yallaplay/`; if a real copied profile directory appears, reconcile it back to the repo and restore the symlink.
5. **TUI formatting friction.** If the user reports funky spacing, stop using the offending markdown shape; switch to compact bullets or plain text.
6. **Compression over-expectation.** `/compress` summarizes middle history but keeps system prompt, tools, loaded skills, memory, and protected recent tail; inspect recent large tool outputs before assuming compression failed.
7. **Generated artifact sprawl.** Keep outputs/logs clean and ignored.
8. **Network visibility surprises.** Dashboard/gateway/cloudflared changes can expose services; confirm before enabling.

## Verification Checklist

- [ ] Active profile/environment was verified when relevant.
- [ ] No secrets were printed or committed.
- [ ] Dry-run/status was used before mutation where possible.
- [ ] Generated files are under expected ignored locations.
- [ ] Setup/provider/storage changes are documented when reusable.
