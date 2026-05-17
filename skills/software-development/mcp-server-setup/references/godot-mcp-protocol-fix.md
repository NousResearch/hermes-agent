# Godot-MCP WebSocket Protocol Fix

## The Problem

Godot 4.5's `WebSocketPeer` does **not** negotiate subprotocols. The upstream Godot-MCP TypeScript client sends `protocol: 'json'` in the WebSocket constructor:

```typescript
// In godot_connection.ts (upstream)
this.ws = new WebSocket(url, {
  protocol: 'json',  // <-- This breaks Godot 4.5
});
```

This causes the handshake to fail with `Error: socket hang up` because Godot's server rejects the connection when it sees an unsupported subprotocol.

## The Fix

Remove the `protocol` option from the WebSocket constructor:

```bash
sed -i "/protocol: 'json',/d" ~/.local/share/godot-mcp/dist/utils/godot_connection.js
```

## Persistent Fix (Survives Rebuilds)

Patch the TypeScript source so the fix survives `npm run build`:

```bash
sed -i "/protocol: 'json',/d" /path/to/Godot-MCP/server/src/utils/godot_connection.ts
cd /path/to/Godot-MCP/server && npm run build
cp -r dist ~/.local/share/godot-mcp/
```

## Fix Script

A convenience script is provided in this repo:

```bash
./mcp-servers/godot-mcp/fix-protocol.sh [path/to/dist]
```

Run this after every `npm run build`.

## Verification

After applying the fix:
1. Ensure Godot editor is open with the MCP plugin enabled
2. Check Output panel for "MCP server started on port 9080"
3. Restart Hermes (`/reload`)
4. Run `hermes mcp test godot`
5. Should show "Connected" and list available tools

## Why This Happens

Godot 4.5 changed WebSocket behavior to strictly validate subprotocols. Earlier Godot versions silently ignored unsupported protocols. The upstream Godot-MCP client was written against Godot 4.4 behavior.

## Upstream Status

As of the last check, this fix is not yet merged upstream. Track the issue at: https://github.com/ee0pdt/Godot-MCP/issues
