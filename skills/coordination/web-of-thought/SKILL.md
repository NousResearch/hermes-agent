---
name: web-of-thought
description: "When to call wot_chat; design agents; modes & opt-in."
version: 1.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [debate, perspectives, critique, red-team, multi-agent, synthesize, wot, web-of-thought]
    related_skills: []
---

# Web-of-Thought (`wot_chat`)

**Opt-in only.** Unavailable unless `HERMES_ENABLE_WOT=1` (or `true`/`yes`/`on`).
Even then, include the `wot` toolset deliberately.

## Provider

Inner agents use **Hermes active session credentials** by default
(`resolve_hermes_endpoint` → runtime / config / `resolve_provider_client`).

Override with env (lab multi-backend):

- `LLM_BASE_URL` + optional `LLM_API_KEY` + `LLM_DEFAULT_MODEL`
- else `OLLAMA_URL` / localhost

Transcript `backend.endpoint_source` shows which source won.

## When to use

Reach for `wot_chat` when **at least two** apply:

- Multi-perspective question (tradeoffs, orthogonal review axes)
- Single answer likely brittle/biased
- Real downside cost if wrong
- You would otherwise simulate “another angle” in one reply

## When NOT to use

Simple facts, casual chat, time-sensitive tiny asks, tight token budgets
(`agents × rounds` multiplies spend).

## Designing agents

Minimal system prompts (1–2 sentences). No long role-play bios.

```json
{"name": "alpha", "system_prompt": "Argue the case for. Brief."}
{"name": "beta",  "system_prompt": "Argue the case against. Brief."}
{"name": "gamma", "system_prompt": "Synthesize alpha and beta."}
```

Names: alphanumeric / `_` / `-` (spaces auto-sanitized).

## Modes

| Mode | Behavior |
|---|---|
| `parallel` (default) | Concurrent rounds; peers see completed messages next round |
| `streaming` | Concurrent + chunked peer partials absorbed **next turn only** — **not** mid-token injection into an in-flight peer request |
| `sequential` | Round-robin full transcript |
| `queue` | Wake on `@Name` DM, or `#tag` matching `interests`, or any inbox if interests empty. Round 1 wakes all |

Routing in content:

- `@Peer: …` → direct message (Channel delivers only to Peer)
- `#topic` → tags for queue interests

## Cost

Prefer `max_rounds: 2–3`. Set `token_budget` for hard caps. 3×3 is usually enough.

## Result fields

1. `errors` — surface failures
2. `agents_done` / `rounds_run` / `stop_reason`
3. `transcript` — synthesize the user answer from this
4. `backend.endpoint_source` / `default_model` — verify provider wiring

## Example

```json
{
  "agents": [
    {"name": "alpha", "system_prompt": "Argue the case for. Brief."},
    {"name": "beta",  "system_prompt": "Argue the case against. Brief."},
    {"name": "gamma", "system_prompt": "Synthesize. Brief."}
  ],
  "task": "Should we migrate internal services to gRPC this quarter?",
  "mode": "parallel",
  "max_rounds": 3,
  "token_budget": 12000
}
```

## Anti-patterns

1. Calling WoT for everything
2. Per-agent `model` in tool args (stripped at tool boundary)
3. Long ceremonial prompts
4. Ignoring `errors`
5. Treating multi-agent output as manufactured truth
