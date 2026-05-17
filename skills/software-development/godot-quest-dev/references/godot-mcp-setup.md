# Godot-MCP Setup and Usage

Godot-MCP enables real-time control of the Godot editor via MCP tools. Create scenes, add nodes, set properties, attach scripts -- all programmatically.

## Installation

### 1. Clone and Build

```bash
git clone --depth 1 https://github.com/ee0pdt/Godot-MCP.git
cd Godot-MCP/server
npm install && npm run build
```

### 2. Install Godot Addon

```bash
mkdir -p /path/to/your-project/addons
cp -r Godot-MCP/addons/godot_mcp /path/to/your-project/addons/
```

In Godot editor: **Project -> Project Settings -> Plugins -> enable Godot MCP**. The WebSocket server starts automatically on port 9080 (check Output panel).

### 3. Copy Server to Persistent Location

```bash
mkdir -p ~/.local/share/godot-mcp
cp -r Godot-MCP/server/dist Godot-MCP/server/node_modules \
  Godot-MCP/server/package.json ~/.local/share/godot-mcp/
```

### 4. Hermes Config

```yaml
mcp_servers:
  godot:
    command: node
    args: [/home/YOUR_USER/.local/share/godot-mcp/dist/index.js]
    connect_timeout: 30
    timeout: 120
```

Restart Hermes (`/reload`) to discover tools.

## Critical Fix: WebSocket Protocol Handshake (Godot 4.5)

Godot 4.5's `WebSocketPeer` does not negotiate subprotocols. The upstream client sends `protocol: 'json'`, causing handshake failure.

**Fix script:**
```bash
#!/bin/bash
SERVER_DIR="${1:-$HOME/.local/share/godot-mcp}"
sed -i "/protocol: 'json',/d" "$SERVER_DIR/dist/utils/godot_connection.js"
echo "Patched: $SERVER_DIR/dist/utils/godot_connection.js"
```

Run after every `npm run build`.

## Available Tools

| Tool | Purpose |
|------|---------|
| `mcp_godot_create_scene` | New `.tscn` with root node type |
| `mcp_godot_open_scene` | Load existing scene |
| `mcp_godot_create_node` | Add node by type + name |
| `mcp_godot_delete_node` | Remove from tree (marks scene modified) |
| `mcp_godot_update_node_property` | Set any property (does NOT mark modified) |
| `mcp_godot_get_node_properties` | Read node state (returns actual data) |
| `mcp_godot_list_nodes` | Scene tree dump (swallows data -- see workaround) |
| `mcp_godot_create_script` | New GDScript |
| `mcp_godot_edit_script` | Modify source |
| `mcp_godot_execute_editor_script` | Run GDScript in editor context (does NOT return data) |
| `mcp_godot_save_scene` | Persist to disk |
| `mcp_godot_get_current_scene` | Active scene info |

## Data Retrieval Workarounds

`execute_editor_script` and `list_nodes` return generic success messages without the actual data. To get real data:

**Option A:** Use `get_node_properties` (returns JSON state).

**Option B:** Write to a temp file in the script, then read it back:

```gdscript
var f = FileAccess.open("/tmp/godot_debug.json", FileAccess.WRITE)
f.store_string(JSON.stringify({"dome_size": str(aabb.size)}))
f.close()
```

Then read `/tmp/godot_debug.json` via `read_file`.

**Option C:** For complex queries, use `get_node_properties` on a parent to get child lists, or inspect `.tscn` files directly (they're plain text).

## Persistence Rules

| Operation | Persists? | Notes |
|-----------|-----------|-------|
| `delete_node` | Yes | Calls `_mark_scene_modified()` internally |
| `update_node_property` | No | Must call `save_scene` after, or changes vanish on reload |
| `execute_editor_script` + `queue_free()` | No | Use `delete_node` instead |
| `execute_editor_script` + property set | No | Must mark scene unsaved explicitly or call `save_scene` |

**Always call `save_scene` after modifications.**

## Scene Debugging via execute_editor_script

For deep inspection that `get_node_properties` can't reach (AABB bounds, mesh class, material transparency, finding duplicates):

```gdscript
# Example: inspect dome mesh and scene bounds
var dome = get_tree().edited_scene_root.find_child("SkyDome", true, false)
var aabb = dome.get_aabb()
# Write to temp file since execute_editor_script doesn't return data
var f = FileAccess.open("/tmp/godot_debug.json", FileAccess.WRITE)
f.store_string(JSON.stringify({"dome_size": str(aabb.size)}))
f.close()
```

```gdscript
# Example: find duplicate nodes
var dupes = []
for c in get_tree().edited_scene_root.get_children():
    for cc in c.get_children():
        if cc.name.ends_with("_001"):
            dupes.append(cc.name)
result = dupes
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "socket hang up" on connect | Run `fix-protocol.sh` -- remove `protocol: 'json'` |
| Tools not appearing after config | Restart Hermes with `/reload` or relaunch |
| Stale data from previous session | `kill $(pgrep -f "godot-mcp/dist/index.js")` then reconnect |
| Scene changes not persisting | `update_node_property` doesn't mark modified -- call `save_scene` or use `delete_node` |
| Script timeout | Break complex scripts into smaller queries |
| Editor not responding | Ensure Godot MCP plugin is enabled and WebSocket server started (check Output panel) |
