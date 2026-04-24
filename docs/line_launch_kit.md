# Hermes Agent x LINE Launch Kit

This document is the public-release playbook for the LINE integration.

The goal is not only to say "LINE works", but to claim a stronger story:

- Hermes Agent can run as a real LINE companion
- it can be operated as a stable always-on service
- it can share one identity across LINE and terminal when the terminal also targets the Primary host

## Positioning

Recommended headline:

`Hermes Agent on LINE: fixed webhook, Primary Mac runtime, shared memory with terminal`

Short version:

- LINE support for Hermes Agent
- fixed webhook via Cloudflare named tunnel
- Primary Mac steady-state operations
- shared Hermes identity across LINE + terminal when both hit the same host

What makes this more interesting than a basic adapter:

1. official Hermes docs do not currently present LINE as a built-in messaging surface
2. LINE requires a public webhook, unlike some chat integrations
3. the real value is operational completeness, not just API plumbing

## What To Show

The strongest demo has three beats:

1. User tells Hermes something on LINE
2. User opens terminal on another machine, but connects to the Primary-host Hermes
3. Hermes recalls the same fact or recent context

Suggested demo script:

1. On LINE: `今日の合言葉は紫のシャボン玉`
2. In terminal: `hermes`
3. Ask: `今日の合言葉は？`
4. Hermes answers correctly

Optional second demo:

1. Ask Hermes on LINE to use `/sethome`
2. Trigger a cron or cross-platform message
3. Show delivery back into LINE

## Public Repo Story

If publishing from a fork, the public-facing description should emphasize:

- LINE adapter
- stable webhook setup
- Primary-host operations
- shared-state usage pattern

Recommended repo blurb:

`Hermes Agent with LINE Messaging API support, fixed webhook operations via Cloudflare Tunnel, and a Primary-host setup for shared LINE + terminal memory.`

## Files Worth Calling Out

Core implementation:

- `gateway/platforms/line.py`
- `gateway/config.py`
- `gateway/run.py`
- `tools/send_message_tool.py`
- `toolsets.py`
- `hermes_cli/gateway.py`
- `hermes_cli/setup.py`
- `hermes_cli/platforms.py`
- `gateway/display_config.py`
- `agent/prompt_builder.py`
- `gateway/session.py`
- `run_agent.py`

Operational docs/scripts:

- `docs/line_named_tunnel.md`
- `docs/hermes_primary_host_model.md`
- `scripts/setup_line_named_tunnel.sh`
- `scripts/hermes-line-primary.sh`
- `scripts/hermes-primary-cli.sh`

## Launch Sequence

### 1. Clean the branch

- verify no secrets are committed
- verify docs use example hostnames/tokens
- make sure code compiles and gateway restarts cleanly

### 2. Capture demo assets

- LINE screenshot
- terminal screenshot
- short GIF or screen recording of the shared-memory recall

### 3. Publish in your own voice first

Recommended order:

1. GitHub fork / branch
2. X thread or single strong post
3. note or Zenn article
4. upstream issue / PR

This preserves authorship before the work is absorbed upstream.

### 4. Open the upstream proposal

Suggested path:

1. issue: propose official LINE support
2. summarize what is already implemented
3. link demo
4. open PR after maintainer feedback

## Messaging Angles

Use one of these depending on audience:

### Builder / OSS angle

`Hermes Agent officially covers Telegram, Discord, Slack, WhatsApp, Signal, and Email, but LINE was missing. I implemented a LINE adapter plus fixed-webhook operations so Hermes can actually live in LINE as a first-class surface.`

### Product angle

`I wanted Hermes to feel like a real companion in the app I actually use every day. So I wired Hermes Agent into LINE, gave it a fixed webhook and always-on Primary host, and made terminal + LINE talk to the same identity.`

### Ops angle

`The hard part was not the webhook itself. It was making LINE stable enough to run daily: fixed hostname, launchd services, Primary-host ownership, and predictable cross-surface memory.`

## Success Criteria

Call the launch successful when all of these are true:

1. a stranger can reproduce the setup from docs
2. LINE webhook survives restart without URL churn
3. `hermes` on your machine reaches the same Primary-host Hermes as LINE
4. demo assets clearly show cross-surface recall
5. your authorship is visible before upstream merge

