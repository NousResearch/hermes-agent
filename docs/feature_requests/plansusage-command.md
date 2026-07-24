# Feature Request: `/plansusage` Command

## Summary

Add a new slash command `/plansusage` that displays usage statistics (5-hour and weekly limits) for all configured inference providers, inspired by [CodexBar](https://github.com/steipete/CodexBar).

## Motivation

Hermes Agent supports multiple providers (OpenRouter, Codex, MiniMax, Nous, etc.) configured via `~/.hermes/config.yaml`. Users currently have no built-in way to quickly check their usage quotas and limits across all these providers from within the CLI or messaging gateway.

Tools like CodexBar solve this for individual services by providing a menu-bar widget and CLI that display:
- **Session usage**: tokens used in the current session
- **5-hour resets**: usage that resets every 5 hours (e.g., Codex tier limits)
- **Weekly resets**: usage that resets weekly

Bringing this capability directly into Hermes would provide a unified, multi-provider usage dashboard accessible via `/plansusage`.

## Proposed Implementation

### Command Design

```
/plansusage
```

**Output format** (example):
```
Provider Plans Usage
====================

codex (OpenAI Codex)
  Session:  ████████░░ 12,450 / 50,000 tokens
  5h reset: ██████░░░░ 60%  (resets in 2h 15m)
  Weekly:   ███░░░░░░░ 28%  (resets in 4d 2h)

minimax (MiniMax)
  Session:  ██░░░░░░░░ 3,200 / 100,000 tokens  
  5h reset: ██████████ 98%  (resets in 0h 45m) ⚠️
  Weekly:   █████████░ 87%  (resets in 6d 18h)

openrouter (OpenRouter)
  Session:  █░░░░░░░░░ 500 / unlimited
  5h reset: ████░░░░░░ 35%
  Weekly:   ░░░░░░░░░░ 5%
```

### Inspiration: CodexBar Provider Descriptor Pattern

CodexBar uses a `ProviderDescriptor` pattern in `Sources/CodexBarCore/Providers/`. Each provider has:
- Metadata (name, display name, limits)
- Fetch strategies for usage data (web scraping, API calls, browser cookies)
- Reset schedules (5h rolling, weekly calendar, etc.)

A similar approach in Hermes could define per-provider usage fetchers.

### Provider Configuration

The command should read from `~/.hermes/config.yaml` to discover all configured `plans` blocks and their provider associations. For example:

```yaml
plans:
  codex:
    provider: openai-codex
    # ... plan details
  minimax:
    provider: minimax
    # ... plan details
```

### Technical Approach

1. **New file**: `hermes_cli/plans_usage.py` — core usage fetching logic
2. **CommandDef entry** in `hermes_cli/commands.py`:
   ```python
   CommandDef("plansusage", "Show usage stats for all configured provider plans",
              "Info"),
   ```
3. **Handler** `_show_plans_usage()` in `cli.py`
4. **Gateway support**: command should work in both CLI and messaging gateway

### Fetching Strategy

Different providers require different data sources:
- **OpenAI Codex**: Browser cookie-based dashboard scraping (like CodexBar)
- **MiniMax**: API endpoint or web dashboard
- **OpenRouter**: API call to `/v1/usage` endpoint
- **Nous**: Portal API

### Additional Considerations

- **Caching**: usage data should be cached for a few minutes to avoid excessive API calls
- **Error handling**: gracefully degrade when a provider's usage endpoint is unavailable
- **Accessibility**: plain-text output suitable for Telegram/Discord (not just terminal)

## Alternatives Considered

- **Separate skill**: implement as a Hermes skill (`hermes skills install plans-usage`). This could work but the feature is generic enough to warrant core built-in status.
- **Only CLI**: implementing only for the CLI would leave messaging gateway users without the capability.

## References

- [CodexBar GitHub](https://github.com/steipete/CodexBar) — inspiration source
- [Hermes Slash Commands Reference](https://hermes-agent.nousresearch.com/docs/reference/slash-commands/)
- Hermes `hermes_cli/commands.py` — CommandDef registry
- Hermes `cli.py` — CLI command handlers
