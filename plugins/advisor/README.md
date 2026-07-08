# hermes-advisor-plugin

A persistent second-model review plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent):
a second model that reviews the main agent's work each turn and delivers
structured advice inline.

Inspired by and ported from [pi-omplike-advisor](https://github.com/pasky/pi-omplike-advisor)
by Petr Baudis (pasky), which brought the oh-my-pi advisor onto upstream pi's
extension surface. This is the same idea, adapted to Hermes' plugin system and
its different hook model.

I have been running this in my own Hermes setup (HermeL on gigul2) and it has
fired correctly every time — zero false positives so far.

## What it does

The advisor is a stateless reviewer that runs after every completed agent turn.
It receives the turn transcript (user message, tool calls, tool results,
assistant response), reviews it through a separate model, and delivers
structured advice back into the conversation.

Advice uses three severity levels:

| Tag | Meaning | Delivery |
|---|---|---|
| `[NIT]` | Non-urgent cleanup, missed opportunity | Injected immediately |
| `[CONCERN]` | Wrong direction, fragile approach, missing constraint | Held for reconfirmation |
| `[BLOCKER]` | Fundamentally unsound path | Held for reconfirmation |

Concerns and blockers are held across turns. On the next review, the advisor
sees them again in a reconfirmation preamble. If the advisor stays silent about
a previously held item, it is considered resolved and dropped. This prevents
stale advice from cluttering the conversation.

The advisor is **not** an executor. It cannot edit files, run commands, or
change session state. It only reads the turn transcript and delivers text advice.

## How it differs from the pi original

| Aspect | pi-omplike-advisor | Hermes plugin |
|---|---|---|
| Context model | Long-lived advisor agent with self-compaction | Stateless per-turn via `ctx.llm.complete()` |
| Turn detection | Native `turn_end` event | `post_llm_call` hook (fires once per turn, carries full history) |
| Advice delivery | `pi.sendMessage()` with steer + triggerTurn | `ctx.inject_message()` (CLI only) |
| Catch-up block | Stalls primary with exponential backoff while advisor settles | **Not available** — Hermes hooks are fire-and-forget |
| Slash command | `/advisor on\|off\|status` | Same plus `/advisor model`, `/advisor provider`, `/advisor providers`, `/advisor models`, `/advisor test` |

The biggest difference: Hermes plugin hooks are asynchronous callbacks. The
plugin cannot stall the agent loop while the advisor reviews. Advice always
arrives after the agent has moved to the next turn. The hold-and-reconfirm
pattern mitigates this — concerns are never delivered on first emission, only
on reconfirmation — but the catch-up block from the pi original is not
reproducible in Hermes' current hook model.

## Installation

The plugin ships with Hermes Agent. Enable it in `config.yaml`:

```yaml
plugins:
  enabled:
    - advisor
```

Or at runtime: `/advisor on`

### Prerequisites

- Hermes Agent v0.16+ (the plugin uses `ctx.register_hook`, `ctx.llm.complete`,
  and `ctx.register_command` — all stable since v0.16)
- A Hermes profile with at least one LLM provider configured

## Configuration

### Enable the plugin

The plugin is **disabled by default**. Enable it in config.yaml:

```yaml
plugins:
  enabled:
    - advisor
```

Or at runtime: `/advisor on`

### Trust gate for provider/model overrides

If you set a different model or provider for the advisor via `/advisor model`
or `/advisor provider`, Hermes' PluginLlm trust gate will block the override
unless you add:

```yaml
plugins:
  entries:
    advisor:
      llm:
        allow_provider_override: true
        allow_model_override: true
```

Without this, `ctx.llm.complete()` raises `PluginLlmTrustError` and the advisor
silently produces no output. The error is logged but not visible to the user.

If the advisor inherits the primary model (no `/advisor model` or
`/advisor provider` set), no trust gate config is needed.

The trust gate is resolved per-call, so config changes take effect on the next
turn without restarting.

### Advisor model

By default, the advisor uses the same model as the main agent. Override at
runtime:

```
/advisor model                      # interactive selector: pick provider + model
/advisor model mimo-v2.5            # different model, same provider (direct name)
/advisor provider custom:opencode-go  # different provider
/advisor config model deepseek-v4-pro  # alias for /advisor model
/advisor config provider custom:deepseek
```

**Interactive selector** (`/advisor model` with no arguments): opens the same
curses-based provider+model picker that `hermes model` uses. Select a provider,
then a model — the result is applied to the advisor's override while your primary
model config stays untouched.

The interactive selector runs `hermes model` in its own pseudo-terminal (PTY)
so its curses UI doesn't conflict with Hermes' prompt_toolkit terminal state.
This means it works reliably on iTerm2 and other terminal emulators.

Model and provider are independently settable. Set only `model` to use a
different model on the same provider. Set both to route to a completely
different provider and auth.

Check what's available:

```
/advisor providers           # list configured providers
/advisor models opencode-go  # list models for a provider
/advisor status              # show current config
```

Model/provider settings persist in `state.json` in the plugin directory.

**Real-world setup (author's daily driver):** the primary agent runs
**DeepSeek-V4-Flash** with `thinking=high` and the advisor runs
**MiMo-V2.5** — two mid-sized (~300B params) fast models, both routed
through the same OpenCode Go subscription. Set with `/advisor model mimo-v2.5`
after configuring `custom:opencode-go` as a provider in Hermes. This pairing
gives responsive primary work with a capable second opinion on every turn.

### Project guidance (WATCHDOG.md)

If a `WATCHDOG.md` file exists in the working directory, its contents are
appended to the advisor's system prompt as advisor-only review guidance. This
lets you tune what the advisor watches for — project-specific traps, style
rules, recurring pitfalls — without touching the main agent's prompt.

## Usage

```
/advisor          — show status
/advisor on       — enable automatic per-turn review
/advisor off      — disable (persisted)
/advisor status   — current state, model, provider, held notes
/advisor model <name>     — set advisor model
/advisor provider <name>  — set advisor provider
/advisor config           — same as /advisor status
/advisor config model <name>     — same as /advisor model
/advisor config provider <name>  — same as /advisor provider
/advisor providers        — list configured providers
/advisor models <provider> — list models for a provider
/advisor test <severity> <note>  — inject a test advisory (for testing delivery)
```

### Environment variables

| Variable | Effect |
|---|---|
| `ADVISOR_NO_REVIEW=1` | Skip live model reviews. Keeps the `/advisor test` delivery path for manual testing. |

## Caveats

- **`inject_message` is CLI only.** On Telegram, Discord, or other gateway
  platforms, advice is logged and stored in the state file but not delivered
  into the conversation. The user can check `/advisor status` to see held notes.
- **No catch-up block.** The advisor cannot stall the agent loop. Advice arrives
  after the agent has moved on. The hold-and-reconfirm pattern means concerns
  are never delivered on first emission, reducing the impact of asynchronicity.
- **`post_llm_call` fires once per turn at completion.** The advisor never sees
  intermediate thinking or mid-turn tool call results until the turn is fully
  done. For live mid-turn hints, `post_tool_call` would be needed (not implemented).
- **Plugin code changes require a full Hermes restart.** `/new` is not enough.
- **`plugins.enabled` must be a YAML list.** A scalar value like
  `plugins.enabled: advisor` silently fails. Use:
  ```yaml
  plugins:
    enabled:
      - advisor
  ```

## Troubleshooting

### Advisor stuck on an old issue

If the advisor keeps flagging something you've already addressed (like a stale
`[BLOCKER]` or `[CONCERN]` that won't clear), the held notes need resetting.
Do the the "IT Crowd" fixit:

```
/advisor off
/advisor on
```

This clears all held notes and starts fresh. Concerns and blockers are stored in 
`state.json` and survive agent restarts, so stale items can accumulate across 
sessions. The toggle is the surest reset.

## License

MIT — see [LICENSE](./LICENSE).

Same as pi-omplike-advisor and the original oh-my-pi advisor extension.

## Credits

- **Petr Baudis (pasky)** — author of [pi-omplike-advisor](https://github.com/pasky/pi-omplike-advisor),
  which inspired this port. The advisor system prompt, severity model, and
  hold-and-reconfirm pattern are directly adapted from that work.
- **oh-my-pi** — the original advisor concept, built for the pi agent ecosystem
  by [Can Bölük (can1357)](https://github.com/can1357).
- Author of this Hermes port: **Ljubomir Josifovski**.
- Built with assistance from agents **Hermes**, **pi**, **Codex** using models
  **DeepSeek-V4-Flash**, **GPT-5.5**.
