---
name: ableton
description: "Control Ableton Live through AbletonMCP."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, windows]
metadata:
  hermes:
    category: creative
    tags: [Ableton, Music, DAW, MCP, MIDI, Audio, Production]
    related_skills: []
prerequisites:
  commands: [hermes, uvx]
---

# AbletonMCP Skill

Optional skill — **not active until installed**:

```bash
hermes skills install official/creative/ableton
```

After install, scripts live under `~/.hermes/skills/creative/ableton/`.
In prose below, `{SKILL_DIR}` means that directory (or this repo path before
install).

Connect Hermes to Ableton Live through
[ahujasid/ableton-mcp](https://github.com/ahujasid/ableton-mcp): a stdio MCP
server (`uvx ableton-mcp`) that talks to an Ableton MIDI Remote Script over a
local socket.

## When to Use

- The user wants Hermes to inspect or control an Ableton Live session.
- The user wants tracks, clips, MIDI notes, instruments/effects, transport, or
  Arrangement View edits driven through MCP tools.
- The user has Ableton Live 10+ installed and can configure a MIDI Remote
  Script.

## Prerequisites

- Ableton Live 10 or newer.
- `uv` / `uvx`.
- AbletonMCP Remote Script installed inside Ableton.
- Telemetry opt-out configured for the MCP server.

Run the doctor first:

```bash
python {SKILL_DIR}/scripts/ableton_doctor.py
```

## How to Run

### 1. Install the Ableton Remote Script

Download upstream `AbletonMCP_Remote_Script/__init__.py` from
<https://github.com/ahujasid/ableton-mcp> and place it at:

```text
<Ableton Remote Scripts>/AbletonMCP/__init__.py
```

Common locations:

- macOS app bundle:
  `/Applications/Ableton Live*.app/Contents/App-Resources/MIDI Remote Scripts/`
- macOS user preferences:
  `~/Library/Preferences/Ableton/Live XX/User Remote Scripts/`
- Windows user preferences:
  `C:\Users\<User>\AppData\Roaming\Ableton\Live x.x.x\Preferences\User Remote Scripts`
- Windows install folders:
  `C:\ProgramData\Ableton\Live XX\Resources\MIDI Remote Scripts\`
  or `C:\Program Files\Ableton\Live XX\Resources\MIDI Remote Scripts\`

Then launch/restart Ableton Live, open Preferences → Link, Tempo & MIDI, select
`AbletonMCP` as a Control Surface, and set Input/Output to `None`.

### 2. Add the MCP server to Hermes

Use upstream's canonical stdio command (`uvx ableton-mcp`) and disable upstream
telemetry by default:

```bash
hermes mcp add ableton --command uvx --env ABLETON_MCP_DISABLE_TELEMETRY=true --args ableton-mcp
```

Only run one AbletonMCP server instance at a time (Hermes, Claude Desktop, or
Cursor — not multiple). Start a new Hermes session after adding it.

### 3. Use the tools

Ask Hermes for session/track info before making changes. Keep edits small:

- "Get information about the current Ableton session."
- "Create a 4-bar MIDI clip with a simple melody."
- "Set the tempo to 120 BPM."
- "Load a drum rack, then create a basic kick/snare pattern."

## Quick Reference

| Goal | Command / action |
| --- | --- |
| Check setup | `python {SKILL_DIR}/scripts/ableton_doctor.py` |
| Add Hermes MCP | `hermes mcp add ableton --command uvx --env ABLETON_MCP_DISABLE_TELEMETRY=true --args ableton-mcp` |
| Test tools | `hermes mcp test ableton` |
| Reconfigure tool subset | `hermes mcp configure ableton` |
| Disable telemetry | `ABLETON_MCP_DISABLE_TELEMETRY=true` |

## Procedure

1. Run `ableton_doctor.py`.
2. Install / enable the Ableton Remote Script if the socket is not reachable.
3. Add the Hermes MCP server with telemetry disabled.
4. Restart Hermes so tools load.
5. Inspect session state first (`get_session_info`, `get_track_info`) before
   creating or modifying clips.
6. Save the Ableton set before broad arrangement generation.

## Pitfalls

- **Manual Ableton setup is required.** `uvx ableton-mcp` alone is not enough;
  Ableton must load the Remote Script.
- **Only one MCP client instance.** Upstream warns not to run Cursor and Claude
  Desktop and Hermes against the same server at once.
- **Telemetry exists upstream.** Always pass
  `ABLETON_MCP_DISABLE_TELEMETRY=true` unless the user explicitly wants to
  enable it.
- **It edits the live set.** Save before broad generation and keep early
  requests narrow.
- **Browser paths can be large.** Ask for specific categories/paths before
  loading instruments/effects.

## Verification

- `ableton_doctor.py` shows `uvx` present and socket `127.0.0.1:9877`
  reachable after Ableton loads the Remote Script.
- `hermes mcp test ableton` connects and lists tools.
- A harmless read-only call (`get_session_info`) works before any mutating
  operation.

See `references/research.md` for source links, tool names, and setup details.
