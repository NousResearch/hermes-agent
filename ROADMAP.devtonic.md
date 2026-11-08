# devtonic-agent roadmap

The autonomous improvement loop works these items top-down. When an item lands, its PR checks it off and links to it.

## Phase 0: Perfect the fork (current)
- [ ] Get a fast, reliable local test baseline: document which `scripts/run_tests.sh` targets pass on macOS and Linux, and record known-flaky tests in `docs/devtonic/test-baseline.md`
- [ ] Expand Devtonic CI coverage step by step (core agent loop, tools, cron, cli) while keeping it under 15 minutes on `ubuntu-latest`
- [ ] Add a `scripts/devtonic/sync-upstream.sh` helper plus docs: merge upstream/main, run the baseline, open a PR
- [ ] Audit install: document a minimal clean install (`uv sync` + entrypoint) that doesn't touch `~/.hermes` of an existing upstream Hermes install (`DEVTONIC_HOME` / profile isolation)
- [ ] Identify and document every file path, env var, and CLI flag that Devtonic currently relies on (`hermes --cli -z`, `--continue`, `~/.hermes/logs/agent.log`, `cron/jobs.json`, `processes.json`), and add regression tests pinning them

## Phase 1: Embeddable API
- [ ] `devtonic/` package: stable Python API `run_turn(session, message) -> async iterator[Event]` wrapping the agent loop
- [ ] Event schema (`docs/devtonic/events.md`): `turn.start`, `thinking`, `tool.call`, `tool.result`, `message.delta`, `turn.end`, `error`
- [ ] `hermes serve --devtonic` local server: HTTP + WebSocket, token-auth, 127.0.0.1 only
- [ ] Sessions, cron, and config endpoints
- [ ] Version handshake endpoint (`/v1/info`) for host apps

## Phase 2: Devtonic integration
- [ ] devtonic-web Chat panel consumes the event stream instead of log tailing
- [ ] Installer: `devtonic-agent install` that Devtonic can call non-interactively

## Phase 3: Standalone app
- [ ] Spike: Tauri shell that bundles devtonic-web UI and launches the embedded agent server
- [ ] Packaging for macOS / Linux / Windows
