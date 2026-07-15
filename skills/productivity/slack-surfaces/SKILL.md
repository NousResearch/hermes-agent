---
name: slack-surfaces
description: "Operate repo-defined Slack channels and canvases through the standalone slack-surfaces CLI."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Slack, Canvases, Channels, Operations, Documentation, Observability]
---

# Slack Surfaces

Use this skill when Hermes needs to manage Slack channels or canvases from
repository-defined specs without embedding the implementation into Hermes core.

This skill assumes the standalone `slack-surfaces` repository is installed
locally and available on `PATH`.

## Setup

```bash
./setup-hermes.sh
```

If you store the Slack bot token in 1Password, set this in `.env`:

```bash
SLACK_BOT_TOKEN_OP_REF=op://Second Brain/Slack_hermes_admin_app/Slack_bot_token
```

## Workflow

```bash
scripts/slack-surfaces-run.sh validate --spec slack-surfaces/*.yaml
scripts/slack-surfaces-run.sh sync --spec slack-surfaces/*.yaml
scripts/slack-surfaces-run.sh sync --spec slack-surfaces/*.yaml --apply
```

1. Validate the spec files first.
2. Run a dry-run sync to inspect planned changes.
3. Run `--apply` only when `SLACK_BOT_TOKEN` is available and live updates are intended.
4. Keep environment-specific Slack specs in the repo that owns the operational workflow, such as an infra repo.

For `hermes-agent-infra`, prefer:

```bash
cd ~/code/hermes-agent-infra
./scripts/run-slack-surfaces.sh validate --json
./scripts/run-slack-surfaces.sh sync --apply --json
```

## Notes

- Specs are versioned in Git, so Slack structure changes are reviewable.
- Hermes should call the wrapper through the terminal tool rather than reimplementing the sync logic in Hermes itself.
- The standalone repository also contains example specs and a GitHub Actions template for push-driven Slack updates.
