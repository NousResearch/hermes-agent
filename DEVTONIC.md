# devtonic-agent

**devtonic-agent** is an MIT fork of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent). It's maintained as the agent runtime for [Devtonic](https://github.com/devtonic0/devtonic-web).

## Why a fork

Devtonic currently drives Hermes by spawning `hermes --cli -z …` per turn and tailing log files. That works, but it's fragile. The goal of this fork is an agent that:

1. **Installs cleanly next to Devtonic.** It needs one predictable install location and a version handshake, so Devtonic can detect, install, and upgrade it.
2. **Exposes a structured, embeddable API.** Streaming turns, tool events, sessions, and cron over a local HTTP/WS server or stdio JSON-RPC, so a host app never has to scrape logs.
3. **Eventually runs inside a standalone Devtonic app.** The current Electron shell gets replaced by a native shell (Tauri is the leading candidate) with this agent embedded.

## Ground rules

- **Stay mergeable with upstream.** Prefer additive modules (`devtonic/` package, new CLI subcommands) over invasive edits to core files. Sync upstream weekly (`git fetch upstream && git merge upstream/main`).
- **Every change lands as a PR** with docs and passing `Devtonic CI`.
- Upstream credit and license stay intact (see `LICENSE`).

## Plan

See [ROADMAP.devtonic.md](./ROADMAP.devtonic.md).
