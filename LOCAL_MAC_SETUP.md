# Local macOS setup notes

This repo is bootstrapped for an isolated local run on this Mac.

## What was installed

- Python: project-local `venv/` via `uv` with Python 3.11
- Python extras: `messaging,cli,web,mcp,pty,honcho,acp,voice`
- Node deps: `npm install`
- Isolated Hermes home: `.hermes-home/`

## Why not `.[all]`

`uv sync --all-extras --locked` currently fails on Python 3.11 because the `all`
extra includes `dev`, which pulls `yc-bench`, and that dependency is gated to
Python 3.12+.

## Local commands

```bash
cd /Users/steven/.openclaw/workspace/repos/hermes-agent
chmod +x scripts/run-local-hermes.sh
scripts/run-local-hermes.sh --help
scripts/run-local-hermes.sh doctor
scripts/run-local-hermes.sh status
scripts/run-local-hermes.sh gateway --help
```

## Auth findings

Hermes supports two different OpenAI-ish paths:

1. `openai-codex` via OAuth device code flow, no API key required.
2. Standard OpenAI-compatible/API-key providers via `.env` or config.

For Codex OAuth, Hermes stores tokens in `HERMES_HOME/auth.json`, not in
`~/.codex/auth.json`, though it can optionally import existing Codex CLI creds.

### Exact interactive step required

To authorize Codex/OpenAI OAuth interactively:

```bash
cd /Users/steven/.openclaw/workspace/repos/hermes-agent
HERMES_HOME="$PWD/.hermes-home" ./scripts/run-local-hermes.sh auth add openai-codex --type oauth --no-browser
```

Hermes will then print:
- a browser URL: `https://auth.openai.com/codex/device`
- a one-time code to enter there

After approval, Hermes saves tokens to `.hermes-home/auth.json`.
Then set the active provider/model:

```bash
./scripts/run-local-hermes.sh config set model.provider openai-codex
./scripts/run-local-hermes.sh config set model.default openai-codex/gpt-5.4
```

## Verification summary

Verified locally:
- CLI entrypoint loads
- `doctor`, `status`, `gateway --help`, and `chat -q` error handling run under isolated `HERMES_HOME`
- Missing auth is the remaining blocker for an actual model-backed chat session
