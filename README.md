# Hermes Agent - zapabob AI Engineering Portfolio Fork

<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://github.com/NousResearch/hermes-agent"><img src="https://img.shields.io/badge/Upstream-NousResearch-blueviolet?style=for-the-badge" alt="Upstream"></a>
  <a href="https://github.com/zapabob/hermes-agent"><img src="https://img.shields.io/badge/Fork-zapabob-black?style=for-the-badge" alt="This fork"></a>
  <a href="https://github.com/zapabob/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
</p>

This repository is my working AI engineering portfolio built on top of
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent). It
tracks upstream closely while adding the operational systems I actually use:
Windows-first runtime recovery, local LLM fallback, social publishing,
NotebookLM research packaging, VRChat tooling, and provider routing for
cost-aware AI work.

The point of this fork is not to replace upstream Hermes. It is to demonstrate
how I integrate an upstream agent platform into a durable personal AI
workstation: keeping security and bug fixes current, preserving fork-only
capability, and turning rough local workflows into reproducible commands,
plugins, tests, and recovery scripts.

## What This Shows

### Upstream Tracking With Fork Preservation

This fork keeps the official Hermes API and architecture as the baseline. The
sync workflow uses policy-driven merge tooling under `scripts/sync_all.py` and
`scripts/merge_tools/` so upstream features, vulnerability updates, and bug
fixes are imported without flattening local work.

The merge policy favors official behavior where upstream now covers the same
problem, then reapplies fork-specific advantages as overlays. That keeps this
repository useful as both a real runtime and a demonstration of maintainable
long-lived fork management.

### Windows-First Agent Operations

The fork hardens Hermes for a Windows workstation where source checkout,
desktop app, dashboard, gateway, local LLM servers, and scheduled tasks must
all survive restarts.

Implemented surfaces include:

- PowerShell launchers for Desktop, Dashboard, gateway, local secretary, and
  fallback LLM services.
- Task Scheduler registration and verification scripts for boot/logon
  autostart.
- Windows path, encoding, shell, log-rotation, and subprocess fixes.
- Runtime checks that verify ports, model endpoints, gateway state, and
  desktop process health rather than relying on optimistic startup messages.

### Local Secretary And Llama Fallback

The local secretary path uses llama.cpp-compatible OpenAI endpoints as a
private fallback when cloud providers are unavailable, expensive, or unsuitable.
It is treated as a real service: start scripts, health checks, context-size
validation, chat completion tests, and tool-calling checks all live in the
repo.

Useful entry points:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\start-llama-secretary.ps1
powershell -ExecutionPolicy Bypass -File scripts\windows\check-local-llm.ps1
```

### Cost-Aware Provider Routing

This fork includes provider work for OpenCode Zen `auto-free` and related
fallback behavior. The idea is practical: use free or low-cost cloud capacity
when it is available, then fall back to local inference when remote service
state changes.

The model routing is designed to derive truth from live catalog and runtime
signals instead of hard-coding assumptions that silently expire.

### LM-twitterer: Public AI Publishing From Hermes

`plugins/lm-twitterer` turns Hermes into a controlled X publishing assistant.
Text generation remains inside Hermes while X cookies are used only for posting
and replies. The plugin supports dry-runs, live posting, whitelist-gated
replies, cron installation, and topic validation to reduce accidental secret
leaks.

Fork additions include explicit text posting and local media attachment support
for workflows where the content has already been reviewed or produced by
another part of the system.

Example:

```powershell
hermes plugins enable lm-twitterer
hermes lm-twitterer install-deps --yes
hermes lm-twitterer auth-browser --screen-name YOUR_NAME --wait-seconds 600
hermes lm-twitterer post "public Hermes operations memo"
hermes lm-twitterer post --text "Reviewed release note" --media output\daily_posts\clip.mp4 --live
```

### NotebookLM Source Packaging

The NotebookLM plugin packages redacted operational logs, notes, and activity
into reusable research bundles. It is meant for turning day-to-day agent work
into source material for review, brainstorming, and longer-form analysis
without copying secrets into a cloud notebook.

```powershell
hermes plugins enable notebooklm
hermes notebooklm status
hermes notebooklm collect
hermes notebooklm brainstorm
```

### Gateway, Desktop, Dashboard, And TUI Integration

The fork keeps the main Hermes surfaces connected:

- CLI and Ink TUI for terminal work.
- Messaging gateway for Telegram, Discord, Slack, and other adapters.
- Electron Desktop as a separate desktop chat surface.
- Dashboard with embedded TUI and surrounding operational panels.

The design rule is simple: extend the existing Hermes surfaces and plugin
interfaces before adding core model tools. Core prompt and tool footprint stay
small; user-specific capability belongs at the edge.

### VRChat And Local Automation

The repository includes VRChat and Quest 2 operational tooling, OSC helpers,
runtime doctors, OpenXR repair scripts, and related skills. These are examples
of using an agent platform as a local systems operator rather than only as a
chat UI.

Relevant areas:

- `skills/gaming/vrchat/`
- `scripts/vrchat_runtime_doctor.py`
- `scripts/windows/vrchat_quest2_controller_doctor.ps1`
- `scripts/windows/run-vrchat-openxr-fix-admin.ps1`

## Engineering Principles

This fork follows the same core engineering constraints as upstream Hermes:

- Preserve per-conversation prompt caching.
- Keep the model tool surface narrow.
- Prefer plugins, skills, CLI commands, and service-gated tools over new core
  tools.
- Keep secrets in `~/.hermes/.env`; keep behavior in `config.yaml`.
- Validate with real runtime checks when a feature touches processes, files,
  ports, providers, or persistent state.
- Keep Windows behavior explicit rather than hoping POSIX assumptions transfer.

## Quick Start

On Windows, the supported bootstrap path is the PowerShell installer in
`scripts/install.ps1`. Clone-based development is still available when you want
to work directly from source.

```powershell
git clone https://github.com/zapabob/hermes-agent.git
cd hermes-agent
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[all,dev]"
python -m hermes_cli.main setup
```

Run the core surfaces:

```powershell
hermes --tui
hermes dashboard
hermes gateway run
```

Enable selected fork plugins:

```powershell
hermes plugins enable lm-twitterer
hermes plugins enable notebooklm
```

## Repository Map

| Area | Purpose |
| --- | --- |
| `scripts/sync_all.py` | Policy-based upstream sync for long-lived fork maintenance. |
| `scripts/windows/` | Windows service, desktop, gateway, local LLM, and recovery scripts. |
| `plugins/lm-twitterer/` | X publishing, replies, cron, explicit text, and media workflows. |
| `plugins/notebooklm/` | Research bundle collection and NotebookLM-oriented source packaging. |
| `tools/environments/` | Local shell, Windows path, subprocess, and terminal compatibility. |
| `gateway/` | Messaging gateway and platform adapters. |
| `apps/desktop/` | Electron desktop chat application. |
| `ui-tui/` | Ink-based TUI frontend. |
| `tui_gateway/` | JSON-RPC backend used by the TUI and desktop app. |
| `agent/` | Core conversation loop, provider adapters, prompt construction, and runtime helpers. |

## Portfolio Summary

As an AI engineer, this repository represents the kind of systems work I value:
agent runtimes that stay current with upstream, survive local machine drift,
respect security boundaries, and turn personal workflows into maintainable
software.

The strongest parts of this fork are not single demos. They are the connective
tissue: merge automation, plugin boundaries, runtime verification, Windows
recovery scripts, provider fallback, and tests that prove the operating surface
still works after upstream changes.

## Upstream Credit

Hermes Agent is developed by NousResearch. This fork is an applied operations
and portfolio layer on top of that work. Upstream documentation remains the best
starting point for the base platform:

- <https://hermes-agent.nousresearch.com/docs/>
- <https://github.com/NousResearch/hermes-agent>
