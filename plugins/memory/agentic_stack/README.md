# agentic_stack memory provider

Wires Hermes's `MemoryProvider` hooks to an [agentic-stack](https://github.com/codejunkie99/agentic-stack) portable brain. Provides automatic episodic logging, system-prompt injection of the user's preferences and graduated lessons, a ripgrep-backed semantic search, and a health probe for an optional stealth-browser sidecar.

## What it does

- **Auto-logs every meaningful turn** via `sync_turn`. Importance is inferred by a small regex-based heuristic; entries below the configurable threshold are skipped as noise. Profile-tagged automatically via the brain's own provenance helpers.
- **Logs parent-side delegation outcomes** via `on_delegation` when the agent spawns subagents.
- **Writes a session rollup** via `on_session_end` at higher importance when a session had more than a trivial number of turns. First-pass implementation is heuristic (first question + closing reply summary); an LLM-backed rollup is a clean follow-up.
- **Injects brain content into every system prompt** via `system_prompt_block`:
  - `memory/personal/PREFERENCES.md` - user identity and collaboration style.
  - `memory/semantic/LESSONS.md` - graduated patterns from the brain's curation loop.
  - `memory/working/REVIEW_QUEUE.md` - only when the nightly dream cycle staged candidates needing human judgment.
  - An optional browser-backend-degraded warning when `CAMOFOX_URL` is set but unhealthy.
- **Prefetches relevant semantic-tier entries** via `prefetch(query)` using ripgrep with tokenized multi-word queries.
- **Exposes tools** for in-session brain interaction: `brain_search`, `brain_review_queue`, `brain_graduate`, `brain_reject`, `brain_log`.

## Activation

In a profile's `config.yaml`:

```yaml
memory:
  provider: agentic_stack
agentic_stack:
  brain_path: /path/to/.agent   # absolute path recommended; see note below
  auto_log: true
  log_threshold_importance: 4
  log_delegations: true
  session_rollup: true
  prefetch_enabled: true
  review_surface: true
```

Restart the gateway or start a new CLI session to pick up the provider.

### Note on `brain_path`

Absolute paths are preferred. When Hermes's terminal backend spawns a subprocess (e.g. one specialist profile shelling out to another), it exports a per-profile `HOME` for filesystem isolation. A `~/.agent` config value would tilde-expand using that per-profile HOME and misdirect to a nonexistent nested path. The plugin defends against this by using `pwd.getpwuid(os.getuid()).pw_dir` for tilde expansion rather than `os.path.expanduser`, but a plain absolute path sidesteps the issue entirely.

## Context / subagent safety

The provider respects the `agent_context` kwarg Hermes passes to `initialize`:
- `"primary"` (default): all hooks active, session is logged.
- `"cron"`, `"flush"`: writes disabled to avoid corrupting user representations with non-interactive traffic. System prompt injection still fires (cron jobs benefit from PREFERENCES/LESSONS context) but nothing gets written back to episodic memory.
- Subagents spawned via `delegate_task` (`skip_memory=True`) don't get a provider instance at all; only the parent sees their delegation result.

## Dependencies

- No new pip dependencies beyond Hermes's existing base. The brain's own Python harness is imported lazily at runtime from the configured `brain_path`, so the plugin can be installed and inspected even when no brain is mounted.
- Requires `rg` (ripgrep) for fast search; falls back to a pure-Python substring scan when `rg` is not on `$PATH`.

## File layout

```
plugins/memory/agentic_stack/
├── __init__.py      # AgenticStackProvider(MemoryProvider), register(ctx)
├── plugin.yaml      # manifest
├── client.py        # lazy imports from <brain_path>/harness/hooks/
├── reflector.py     # importance / success / skill heuristics
├── context.py       # ripgrep search, review-queue reader, CLI wrappers
└── README.md        # this file
```

## Upstream brain

This plugin is the Hermes-side companion to the agentic-stack repo's brain format. The brain itself lives outside Hermes so a user can take the same memory, skills, and protocols across harnesses (Claude Code, Cursor, Windsurf, OpenCode, OpenClaw, Hermes, Pi, standalone Python). See the [agentic-stack README](https://github.com/codejunkie99/agentic-stack) for the brain's tiered memory model, dream cycle, and review queue workflow.
