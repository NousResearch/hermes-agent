# macs_dump — Hermes observability plugin

SLIP-style conditional **Decision-Chain Dump** for `NousResearch/hermes-agent`.
When a trigger fires (tool fails N times, API error, latency over budget,
abnormal finish, approval denied) it freezes the full decision chain of that
turn — system prompt, input, full LLM response, tool sequence, subagent tree,
timings/resources — into a `macs.dump.v0` JSON artifact.

- **Read-only** observer hooks (`hermes.observer.v1`); no middleware; **zero core
  changes**; **fail-open** (never breaks the agent).
- Dumps → `~/.hermes/macs-dump/<date>/` plus `index.jsonl`.

## Install (dogfood)

```bash
pip install -e <path-to-macs-repo>     # makes `macs` importable
cp -r integrations/hermes-plugin  <hermes>/plugins/observability/macs_dump
```

## Config (`plugin.yaml` consumers / ctx.config)

| key | default | meaning |
|-----|---------|---------|
| `dir` | `~/.hermes/macs-dump` | where dumps are written |
| `ttl_s` | `900` | per-turn buffer TTL |
| `maxlen` | `200` | per-turn event ring size |
| `triggers` | sensible defaults | list of `{predicate, ...params}` |

## Hooks subscribed

`on_session_start`, `on_session_end`, `pre_api_request`, `post_api_request`,
`api_request_error`, `pre_tool_call`, `post_tool_call`, `subagent_start`,
`subagent_stop`.

## Upstream PR note

For merging into Hermes, vendor the small core into this plugin folder so it is
self-contained (zero external deps). Relates to #6741 / #6642.
