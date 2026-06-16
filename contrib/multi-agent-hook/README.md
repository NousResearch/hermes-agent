# multi-agent hook

Gateway hook implementing the #9514 single-daemon multi-agent architecture
with the IBM CICS TCB model.

## Install

```bash
cp -r contrib/multi-agent-hook ~/.hermes/hooks/
hermes gateway restart
```

## What it does

| Event | Action |
|---|---|
| `agent:start` | Resolves agent via routing table → loads AgentContext → writes session hint |
| `agent:end` | Snapshots pseudo-conversational state to disk |

## Requires

- [#47027](https://github.com/NousResearch/hermes-agent/pull/47027) (AgentContext, RoutingTable, Pool)
- `~/.hermes/agent_routing.yaml` (see example below)

## Example routing config

```yaml
# ~/.hermes/agent_routing.yaml
routing:
  - topic: telegram:-100X:101
    agent: coding-agent
  - topic: telegram:-100X:201
    agent: cs-agent
  - dm: telegram:user123
    agent: personal-agent
default_agent: coding-agent
```

## Upgrade path

The hook API is **observer-only** (read context, fire side effects). It CANNOT
inject AgentContext into the LLM prompt — that requires a future
`agent:pre_dispatch` hook. See the handler docstring for details.
