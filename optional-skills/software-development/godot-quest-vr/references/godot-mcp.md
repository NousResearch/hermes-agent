# Godot-MCP setup

Live control of the Godot editor over MCP (create scenes/nodes, set
properties, run editor scripts). Optional — the rest of this skill works
without it.

Upstream server: https://github.com/ee0pdt/Godot-MCP

## Build and install server

```bash
git clone --depth 1 https://github.com/ee0pdt/Godot-MCP.git
cd Godot-MCP/server
npm install && npm run build

mkdir -p "${HOME}/.local/share/godot-mcp"
cp -r dist node_modules package.json "${HOME}/.local/share/godot-mcp/"
```

## Protocol fix (Godot 4.5)

Godot 4.5 `WebSocketPeer` does **not** negotiate subprotocols. Upstream
clients that pass `protocol: 'json'` fail the handshake ("socket hang up").

From this skill directory:

```bash
bash scripts/fix-godot-mcp-protocol.sh "${HOME}/.local/share/godot-mcp"
```

Re-run after every `npm run build`. To survive rebuilds, also strip the line
from the TypeScript source under `Godot-MCP/server/src/` and rebuild.

## Project addon

```bash
mkdir -p /path/to/your-project/addons
cp -r /path/to/Godot-MCP/addons/godot_mcp /path/to/your-project/addons/
```

Godot → Project → Project Settings → Plugins → enable **Godot MCP**.
The editor listens on **ws://127.0.0.1:9080** by default.

## Hermes config

Hermes expands **`${VAR}`** placeholders in MCP config. Bare `$HOME` is **not**
expanded and is passed literally to Node.

```yaml
mcp_servers:
  godot:
    command: node
    args: ["${HOME}/.local/share/godot-mcp/dist/index.js"]
    connect_timeout: 30
    timeout: 120
```

Apply with a new session or `/reload-mcp`, then:

```bash
hermes mcp test godot
```

## Session order

1. Open Godot with the project; plugin enabled (port 9080 listening).
2. Ensure protocol fix applied on the Node server build.
3. Start / reload Hermes so tools register (`mcp_godot_*` / server-prefixed names).

## Tool pitfalls

Some editor tools return generic success without payloads. Prefer
property getters, explicit saves, and verifying on disk with `read_file`
after mutations. Always save the scene after structural edits.
