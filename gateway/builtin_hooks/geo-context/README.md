# geo-context hook

Fires on `session:start` and `session:reset`. Reads the live Soul/Memory/User-Profile/Today state from Gabriel's Geo macOS app via `geo-mcp-bridge`, optionally summarizes Today's record with Claude Haiku, and writes the consolidated result to `~/.hermes/memories/MEMORY.md`. Hermes's own memory-injection pipeline picks that file up at session start so the agent always begins a conversation with current real-world context.

Requirements:
- Geo.app must be running (the bridge dials its Unix socket).
- `hermes auth add anthropic --type oauth` for Haiku summarization.

Safe to leave installed forever — silently no-ops when Geo.app is closed.
