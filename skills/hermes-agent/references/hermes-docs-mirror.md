# Hermes Docs Mirror

**Location:** `~/references/hermes-docs/`
**Source:** https://hermes-agent.nousresearch.com/docs
**Pages:** 44 curated pages
**Last synced:** 15 May 2026
**Cron:** `hermes-docs-sync` (Mon 09:00, Topic 20)

## Purpose

Local mirror of the Hermes Agent documentation site. The canonical reference for understanding how Hermes works without searching through the codebase. Contains curated summaries of every major feature, guide, and reference page.

## How to Use

When the user asks "how does X work in Hermes" or you need to reference Hermes internals:

1. Check INDEX.md first for the page structure: `read_file ~/references/hermes-docs/INDEX.md`
2. Navigate to the relevant page under getting-started/, user-guide/, guides/, developer-guide/, integrations/, or reference/
3. Search across all docs: `grep -ri "search term" ~/references/hermes-docs/`

## Page Inventory

### Getting Started
- installation.md - Quick/manual install, prerequisites, troubleshooting
- quickstart.md - Fastest path to working setup
- learning-path.md - Docs by experience level

### User Guide
- configuration.md - Config file, directory structure, terminal backends
- cli.md - TUI, keybindings, session commands
- messaging.md - Telegram, Discord, Slack, 20+ platforms
- windows-native.md - Windows support and limitations
- security.md - Approval modes, authorisation, sandboxing

### Features (user-guide/features/)
- overview.md - Core capabilities at a glance
- tools.md - 70+ built-in tools, 28 toolsets
- skills.md - SKILL.md format, conditional activation
- memory.md - Cross-session memory, Mnemosyne
- mcp.md - MCP servers, filtering, prefix naming
- cron.md - Scheduling, lifecycle, workdir support
- delegation.md - Subagent spawning, context isolation
- code-execution.md - Programmatic tool calling via Python
- browser.md - Cloud/local/hybrid browser automation
- voice-mode.md - CLI mic, messaging replies, Discord VC
- hooks.md - Gateway, plugin, and shell hooks
- plugins.md - Custom tools, hooks, commands
- context-files.md - AGENTS.md, CLAUDE.md, SOUL.md
- personality.md - SOUL.md agent identity
- batch-processing.md - Training data generation
- rl-training.md - GRPO with LoRA via Tinker-Atropos

### Guides
- use-mcp-with-hermes.md - Practical MCP patterns
- use-voice-mode-with-hermes.md - Voice workflow setup
- tips.md - Best practices and tips
- daily-briefing-bot.md - Automated morning briefing tutorial
- team-telegram-assistant.md - Multi-user bot tutorial

### Developer Guide
- architecture.md - System overview, directory structure
- adding-tools.md - Tool vs skill decision, registry pattern
- creating-skills.md - SKILL.md authoring
- contributing.md - Dev setup, PR process

### Integrations
- providers.md - All 25+ inference providers
- integrations.md - MCP, search, browser, voice, plugins

### Reference
- cli-commands.md - All `hermes` CLI subcommands
- tools-reference.md - Full tool registry per toolset
- toolsets-reference.md - Tool groupings and presets
- faq.md - Common questions

### Skills Hub
- skills.md - Community skills, categories, registries

## Change Monitoring

A weekly cron (`hermes-docs-sync`, Mon 0900 Telegram Topic 20) crawls the live docs site and reports new or dead pages. New meaningful pages should be added to the curated mirror manually.
