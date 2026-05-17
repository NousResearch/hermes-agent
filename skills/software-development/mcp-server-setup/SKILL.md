---
name: mcp-server-setup
description: Configure Hermes Agent MCP servers for Godot and Blender automation. Use when setting up, troubleshooting, or reconfiguring MCP connections for VR development workflows.
category: software-development
version: 1.0.0
author: Hermes VR DevKit
license: MIT
metadata:
  hermes:
    tags: [mcp, hermes, godot, blender, config, setup]
    related_skills: [godot-quest-dev, blender-godot-pipeline, godot-xr-interactions, quest-native-toolchain]
---

# MCP Server Setup for VR Development

Configure Hermes Agent to control both Godot and Blender via MCP, enabling AI-driven asset creation and scene assembly for Quest VR projects.

## When to Use

- Setting up Hermes MCP for the first time on a new machine
- Godot-MCP or Blender-MCP tools are not appearing in Hermes
- MCP connection fails with "socket hang up" or "connection refused"
- Rebuilding the MCP server after an update
- Moving the dev environment to a new path or user

## What MCP Enables

Without MCP, the agent must ask you to click buttons in Blender/Godot. With MCP:
- **Blender**: Create objects, apply materials, sculpt terrain, export GLB, take viewport screenshots -- all programmatic
- **Godot**: Create scenes, add nodes, set transforms, attach scripts, run the game -- all programmatic

The agent drives the full pipeline: Blender asset creation -> GLB export -> Godot import -> scene assembly -> export APK -> deploy to Quest.

## Quick Config

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  godot:
    command: node
    args: [$HOME/.local/share/godot-mcp/dist/index.js]
    connect_timeout: 30
    timeout: 120

  blender:
    command: uvx
    args: [blender-mcp]
```

After editing config, restart Hermes (`/reload` or relaunch) to discover tools.

Verify:
```bash
hermes mcp test godot
hermes mcp test blender
hermes mcp list
```

## Godot-MCP Setup

### 1. Build the Server

```bash
git clone --depth 1 https://github.com/ee0pdt/Godot-MCP.git
cd Godot-MCP/server
npm install && npm run build
```

### 2. Install the Godot Addon

```bash
mkdir -p /path/to/your-project/addons
cp -r Godot-MCP/addons/godot_mcp /path/to/your-project/addons/
```

In Godot editor: **Project -> Project Settings -> Plugins -> enable Godot MCP**.

### 3. Deploy Server to Persistent Location

```bash
mkdir -p ~/.local/share/godot-mcp
cp -r Godot-MCP/server/dist Godot-MCP/server/node_modules \
  Godot-MCP/server/package.json ~/.local/share/godot-mcp/
```

### 4. Critical Fix: WebSocket Protocol (Godot 4.5)

Godot 4.5's `WebSocketPeer` does not negotiate subprotocols. The upstream client sends `protocol: 'json'`, causing handshake failure.

Apply the fix:
```bash
sed -i "/protocol: 'json',/d" ~/.local/share/godot-mcp/dist/utils/godot_connection.js
```

Or use the provided script:
```bash
./mcp-servers/godot-mcp/fix-protocol.sh
```

Run this after every `npm run build`.

### 5. Hermes Config

```yaml
mcp_servers:
  godot:
    command: node
    args: [$HOME/.local/share/godot-mcp/dist/index.js]
    connect_timeout: 30
    timeout: 120
```

### 6. Usage

1. Open Godot editor with your project
2. Enable the Godot MCP plugin (it starts WebSocket server on port 9080)
3. Restart Hermes to discover tools
4. Tools appear as `mcp_godot_*`

See `references/godot-mcp-protocol-fix.md` for the full protocol fix details and troubleshooting.

## Blender-MCP Setup

### 1. Install the Server

```bash
# Via uv (recommended)
uvx blender-mcp

# Or via pip
pip install blender-mcp
```

### 2. Install the Blender Addon

```bash
mkdir -p ~/.config/blender/4.3/scripts/addons
cp /path/to/blender-mcp/addon.py ~/.config/blender/4.3/scripts/addons/
```

In Blender: **Edit -> Preferences -> Add-ons -> Install -> select addon.py -> Enable "Interface: Blender MCP"**.

In the 3D View sidebar (press N), click **"Connect to Claude"**.

### 3. Hermes Config

```yaml
mcp_servers:
  blender:
    command: uvx
    args: [blender-mcp]
```

### 4. Usage

1. Open Blender
2. Enable the Blender MCP addon
3. Click "Connect to Claude" in the sidebar
4. Restart Hermes to discover tools
5. Tools appear as `mcp_blender_*`

See `references/blender-mcp-install.md` for detailed addon troubleshooting.

## Tool Naming Convention

MCP tools are prefixed with the server name:
- Godot: `mcp_godot_create_scene`, `mcp_godot_create_node`, ...
- Blender: `mcp_blender_create_object`, `mcp_blender_execute_blender_code`, ...

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Tools not appearing after config edit | Restart Hermes with `/reload` or relaunch |
| "socket hang up" on Godot-MCP connect | Run `fix-protocol.sh` to remove `protocol: 'json'` |
| "Connection refused" on Godot-MCP | Ensure Godot editor is open and MCP plugin enabled (check Output panel for "MCP server started on port 9080") |
| Stale Godot-MCP data from previous session | `kill $(pgrep -f "godot-mcp/dist/index.js")` then reconnect |
| Blender addon greyed out | Restart Blender -- cache may be stale after manual file copy |
| Blender glTF export fails | `sudo pip install numpy --break-system-packages` (Blender uses system Python) |
| `hermes mcp test` hangs | Increase `connect_timeout` to 60 or 90 seconds |
| `update_node_property` changes vanish | Call `save_scene` after modifications; `update_node_property` does not mark scene modified |
| `execute_editor_script` returns no data | Use `get_node_properties` for data retrieval, or write to temp file |

## Security Notes

- Hermes does NOT pass your full shell environment to MCP subprocesses. Only safe baseline variables (PATH, HOME, USER, LANG, etc.) are inherited.
- API keys and secrets are excluded unless explicitly added via the `env` config key.
- Credential patterns in error messages are automatically redacted before being shown to the LLM.

## References

- `references/godot-mcp-protocol-fix.md` -- Full protocol fix details and Godot-MCP troubleshooting
- `references/blender-mcp-install.md` -- Blender addon install, numpy fix, and connection troubleshooting
