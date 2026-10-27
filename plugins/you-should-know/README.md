# you-should-know

A sidecar observer for Hermes: it scans the session's assistant output for
important things you might have missed — swallowed tool errors, unanswered
questions, risky actions taken, pending follow-ups, contradictions — and appends
a short digest to the turn's visible output.

Inspired by the Claude Code **"You should Know"** mod (announced by
[@ClaudeDevs](https://x.com/claudedevs), Oct 2026), which spins off a sideagent
to observe Claude's output. This is a from-scratch port to the Hermes plugin
surface: no Claude Code code is used.

## How it works

1. **Accumulate** — `post_llm_call` appends each turn's assistant text and
   `post_tool_call` appends one compact line per tool result to a bounded
   per-session buffer (keyed by Hermes home *and* session id, so multiplexed
   profiles never share state). The buffer is capped at `max_chars`
   (default 24k); oldest chunks drop first.
2. **Observe** — `transform_llm_output` (the sanctioned pre-persistence seam)
   runs one bounded observer pass per digest epoch through the host-owned
   plugin LLM (`ctx.llm.complete_structured`, JSON schema
   `{severity, title, detail}[]`) when the unobserved output reaches
   `min_observe_chars` (default 1500). Trivial sessions never trigger a call.
3. **Deliver** — findings are appended to the current turn's output *before*
   it is persisted, so the text you see is the text stored in SQLite and
   replayed next turn. Only the current turn's not-yet-written text is touched:
   prompt caching, strict role alternation, and the no-synthetic-user-message
   invariant all hold. Each epoch is reviewed once; a digest is never
   re-observed (the marker is stripped on accumulation — no feedback loop).

A digest at true session end is deliberately not attempted: by
`on_session_finalize` the final assistant row is already persisted and hook
results reach no user-visible surface.

## Enable

```bash
hermes plugins enable you-should-know
```

Once loaded the plugin is active unless opted out (mirrors the mod's single
enable command).

## Settings (`plugins.entries.you-should-know.settings` in `config.yaml`)

| key                | default | meaning |
|--------------------|---------|---------|
| `enabled`          | `true`  | kill-switch; hooks go inert when `false` |
| `max_chars`        | `24000` | observer-input + buffer bound (head+tail kept) |
| `min_observe_chars`| `1500`  | unobserved output needed to trigger a pass |
| `tool_line_chars`  | `400`   | per tool-result line kept in the buffer |
| `model`            | `""`    | model override for the observer pass |

Cost: at most one auxiliary LLM call per digest epoch (short sessions: one per
session), skipped entirely for trivial output.

## Cheap model

Two knobs, in priority order:

1. `settings.model` — passed as the `model=` override to `ctx.llm`. Requires
   the trust flag (fail-closed):
   ```yaml
   plugins:
     entries:
       you-should-know:
         llm:
           allow_model_override: true
         settings:
           model: "your-cheap-model"
   ```
   Without the flag the observer pass is skipped with a warning — never
   silently downgraded to another model.
2. The plugin's own auxiliary-task slot `you_should_know_observer`, pinnable
   without trust flags under `auxiliary:` in `config.yaml`:
   ```yaml
   auxiliary:
     you_should_know_observer:
       model: "your-cheap-model"
   ```

## Privacy

The observer pass sends recent session output to the configured model. It is a
local auxiliary call through the host's normal routing — no external service,
no telemetry. Disable per session with `enabled: false` if the transcript is
sensitive.
