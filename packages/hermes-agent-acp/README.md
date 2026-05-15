# @nousresearch/hermes-agent-acp

ACP launcher for Hermes Agent.

This package is intended for clients such as Zed that install agents through the official ACP Registry. It launches the Python Hermes ACP server with:

```bash
uvx --from 'hermes-agent[acp]==0.13.0' hermes-acp
```

## Requirements

- Node.js 18+
- `uv` or `uvx` on PATH
- Hermes provider credentials configured with `hermes model`, or through Hermes' normal `~/.hermes/.env` / `~/.hermes/config.yaml` setup

## Commands

```bash
npx @nousresearch/hermes-agent-acp@0.13.0 --version
npx @nousresearch/hermes-agent-acp@0.13.0 --check
npx @nousresearch/hermes-agent-acp@0.13.0 --setup
npx @nousresearch/hermes-agent-acp@0.13.0
```

Normal no-argument mode reserves stdout for ACP JSON-RPC traffic. Diagnostics are emitted on stderr by Hermes.
