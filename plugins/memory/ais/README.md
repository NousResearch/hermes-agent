# AIS Memory Provider

Cross-session knowledge-graph memory for Hermes — Ed25519 DID key
auth, RRF hybrid retrieval, temporal reasoning.

## Requirements

- `pip install hermes-memory-ais`
- An AIS account at [agentsandswarms.ai](https://agentsandswarms.ai)
- An Ed25519 keypair enrolled once per device via the npm client:
  ```bash
  npx aismemory enable-key-auth
  ```

The same keypair authenticates the npm `aismemory` MCP client, the AIS
REST API, and this Hermes plugin — one key, all surfaces.

## Setup

```bash
hermes memory setup    # select "ais"
```

The wizard prompts for your AIS `userDid` (printed by `npx aismemory whoami`).
The plugin reads your private key from `~/.aismemory/keys/<userDid>.json`
and authenticates silently from then on.

## Config

Config file: `$HERMES_HOME/ais.json`

| Key | Default | Description |
|-----|---------|-------------|
| `user_did` | (required) | Your AIS userDid |
| `key_path` | `~/.aismemory/keys/<userDid>.json` | Override for non-standard install |
| `domain` | `ais.agentsandswarms.ai` | AIS domain (override for self-hosted / staging) |
| `telemetry` | `false` | Opt-in PostHog adoption pings |

## Identity scoping

The plugin creates one AIS agent per Hermes profile, named
`hermes:<profile>` (e.g. `hermes:coder`). Memories scope to that
agent only — your AIS account's other agents (the one paired with
Claude Code, for instance) stay isolated from Hermes traffic.
`metadata.managedBy = "hermes-memory-plugin"` is stamped on each
managed agent so direct-API dashboards can filter them out.

## Tools

| Tool | What it does |
|---|---|
| `ais_recall` | Search the user's memories for context relevant to a query. |
| `ais_remember` | Persist a durable fact. |
| `ais_full_context` | Pull the agent's full snapshot. |
| `ais_dream` | Trigger memory synthesis. |

Auto behaviors:
- `prefetch` queues a background recall before each turn.
- `sync_turn` persists every user/assistant pair as a `conversation` memory.
- `on_session_end` triggers a dream cycle for synthesis.

## Source

Implementation lives at [rudedoggg/hermes-memory-ais](https://github.com/rudedoggg/hermes-memory-ais).
File issues there. The wrapper module in this directory exists only
to integrate with Hermes's plugin discovery; no logic lives in it.

## License

MIT.
