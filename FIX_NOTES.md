# TOKEN COUNTER FIX NOTES

## The Problem
The token counter in the TUI was completely frozen. It did not update during streaming AND it did not update after the turn was complete. The AI knew the usage, but the information never reached the UI.

## The Solution
A 'heartbeat' system was implemented:
1. **run_agent.py**: Added `usage_callback` to `AIAgent`. Triggered it the moment `chunk.usage` is detected during streaming.
2. **tui_gateway/server.py**: Wired `usage_callback` to emit a `usage.delta` event.
3. **ui-tui/src/app/createGatewayEventHandler.ts**: Added a handler for `usage.delta` that calls `patchUiState` to update the counter live.

## Crucial Setup
The project is located in `/home/nishant/hermes-agent-contrib`.
An editable install was performed: `pip install -e . --break-system-packages`
This ensures the running TUI uses the code in the contrib folder, not a static snapshot in site-packages.

## Verification
Restart the TUI and send a message. The token counter in the footer should move while the AI is talking.
