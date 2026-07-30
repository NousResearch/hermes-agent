# AbletonMCP research

Primary source: <https://github.com/ahujasid/ableton-mcp>

## Shape

AbletonMCP has two parts:

1. `AbletonMCP_Remote_Script/__init__.py` — Ableton MIDI Remote Script. It
   creates a local socket server inside Ableton Live.
2. `MCP_Server/server.py` — Python stdio MCP server. It connects MCP tool calls
   to the Remote Script.

Upstream's canonical MCP command is:

```json
{
  "mcpServers": {
    "AbletonMCP": {
      "command": "uvx",
      "args": ["ableton-mcp"]
    }
  }
}
```

For Hermes:

```bash
hermes mcp add ableton --command uvx --env ABLETON_MCP_DISABLE_TELEMETRY=true --args ableton-mcp
```

`--env` must appear before `--args` because Hermes's `--args` consumes the
remaining command line.

## Remote Script setup

Upstream locations:

- macOS:
  - `Contents/App-Resources/MIDI Remote Scripts/` inside the Ableton app bundle
  - `~/Library/Preferences/Ableton/Live XX/User Remote Scripts`
- Windows:
  - `C:\Users\<User>\AppData\Roaming\Ableton\Live x.x.x\Preferences\User Remote Scripts`
  - `C:\ProgramData\Ableton\Live XX\Resources\MIDI Remote Scripts\`
  - `C:\Program Files\Ableton\Live XX\Resources\MIDI Remote Scripts\`

Create `AbletonMCP/` under the Remote Scripts dir, copy `__init__.py`, restart
Ableton, then Preferences → Link, Tempo & MIDI → Control Surface:
`AbletonMCP`, Input/Output: `None`.

The Remote Script socket is observed in community docs as `127.0.0.1:9877`.

## Tool surface

Tool names from upstream `MCP_Server/server.py` include:

- `get_session_info`
- `get_track_info`
- `create_midi_track`
- `set_track_name`
- `create_clip`
- `create_audio_clip`
- `add_notes_to_clip`
- `set_clip_name`
- `set_tempo`
- `load_instrument_or_effect`
- `fire_clip`
- `stop_clip`
- `stop_playback`
- `get_browser_tree`
- `get_browser_items_at_path`
- `load_drum_kit`
- `set_arrangement_time`
- `get_arrangement_clips`
- `duplicate_to_arrangement`

Capabilities advertised by upstream: track manipulation, MIDI/audio clips,
instruments/effects, Arrangement View composition, session/transport control.

## Telemetry

Upstream collects anonymous telemetry unless disabled. Disable by default in
Hermes instructions with one of:

- `ABLETON_MCP_DISABLE_TELEMETRY=true`
- `DISABLE_TELEMETRY=true`
- `MCP_DISABLE_TELEMETRY=true`

The skill uses `ABLETON_MCP_DISABLE_TELEMETRY=true`.

## GitHub usage checks

Observed real-world usage agrees that `uvx ableton-mcp` is the canonical entry:

- `ahujasid/ableton-mcp`
- `dnkrow/claude-code-ableton-mac`
- `ParkerRex/synthia`
- `patrickking67/producer`

Some forks warn about package/version confusion, but for the upstream repo the
published PyPI package exposes console script `ableton-mcp =
MCP_Server.server:main`.
