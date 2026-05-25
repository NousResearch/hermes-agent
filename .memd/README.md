# memd bundle

This directory contains the memd configuration for `hermes-agent`.

## Quick Start

1. Set up the bundle:
   - `memd setup --output /home/aparcedodev/.hermes/hermes-agent/.memd`
2. Check readiness and repair drift when needed:
   - `memd doctor --output /home/aparcedodev/.hermes/hermes-agent/.memd`
   - `memd doctor --output /home/aparcedodev/.hermes/hermes-agent/.memd --repair`
3. Inspect the active config:
   - `memd config --output /home/aparcedodev/.hermes/hermes-agent/.memd`
4. Refresh the live wake-up surface:
   - `memd wake --output /home/aparcedodev/.hermes/hermes-agent/.memd --route auto --intent current_task --write`
5. Launch an agent profile:
   - `.memd/agents/codex.sh`
   - `.memd/agents/claude-code.sh`
   - `.memd/agents/agent-zero.sh`
   - `.memd/agents/hermes.sh`
   - `.memd/agents/openclaw.sh`
   - `.memd/agents/opencode.sh`
6. Inspect the compact working-memory view when needed:
   - `memd resume --output /home/aparcedodev/.hermes/hermes-agent/.memd --route auto --intent current_task`
7. Before memory-dependent answers, run bundle-aware recall:
   - `memd lookup --output /home/aparcedodev/.hermes/hermes-agent/.memd --query "..."`

## Commands

- `memd commands --output /home/aparcedodev/.hermes/hermes-agent/.memd`
- `memd commands --output /home/aparcedodev/.hermes/hermes-agent/.memd --summary`
- `memd commands --output /home/aparcedodev/.hermes/hermes-agent/.memd --json`
- `memd setup --output /home/aparcedodev/.hermes/hermes-agent/.memd`
- `memd doctor --output /home/aparcedodev/.hermes/hermes-agent/.memd`
- `memd config --output /home/aparcedodev/.hermes/hermes-agent/.memd`

The same catalog is written to `COMMANDS.md` in the bundle root.

## Notes

- Prefer the built `memd` binary during normal multi-session use; `cargo run` adds avoidable compile/cache contention.
- `env` and `env.ps1` export the same bundle defaults if you want to wire another harness manually.
- Automatic short-term capture is enabled by default and writes bundle state under `state/last-resume.json`.
- `wake.md` is the startup live-memory surface; `mem.md` is the deeper compact memory view.
- Add `--semantic` only when you want deeper LightRAG fallback.
- For Codex, start from `.memd/wake.md`, then use `memd lookup --output /home/aparcedodev/.hermes/hermes-agent/.memd --query "..."` before memory-dependent answers.
- For Claude Code, import `.memd/agents/CLAUDE_IMPORTS.md` from your project `CLAUDE.md`, then use `/memory` to verify the memd files are loaded.
