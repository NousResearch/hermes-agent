# Hermes Agent — Development Guide

This file has been reduced to save context tokens (~18K tokens/session).
The full development guide has been moved to the **`hermes-agent-dev`** skill.

## Quick Reference

- **Load the full guide:** `skill_view(name='hermes-agent-dev')` or load the `hermes-agent-dev` skill
- **Full reference file:** The complete AGENTS.md content is at the skill's `references/AGENTS.md`
- **Codebase:** `/home/michel/.hermes/hermes-agent/`
- **Tests:** `scripts/run_tests.sh` (call this, not raw `pytest`)

### Key Architecture

Hermes = AI agent core (CLI, TUI, messaging gateway, desktop app).
- **Per-conversation prompt caching is sacred** — don't mutate past context mid-conversation
- **Core is a narrow waist; capability lives at the edges** — prefer plugins/skills over core tools
- **Footprint Ladder:** extend existing code → CLI + skill → service-gated tool → plugin → MCP → new core tool

### Quick Dev Commands

```bash
source .venv/bin/activate
scripts/run_tests.sh tests/agent/test_foo.py::test_bar  # one test
```

For full development guidance, contribution rubric, project structure, and known pitfalls, load the `hermes-agent-dev` skill.