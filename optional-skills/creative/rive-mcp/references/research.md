# Rive integration research

## Official Rive desktop MCP

Source: <https://rive.mintlify.dev/docs/editor/ai/mcp>

- Official MCP is currently documented as available in the desktop Editor for
  Windows and macOS.
- The endpoint is local HTTP:

```json
{
  "mcpServers": {
    "rive": {
      "url": "http://127.0.0.1:9791/mcp"
    }
  }
}
```

- Rive desktop / Early Access app must be open.
- User should have a Rive file open and an artboard created.
- Rive docs describe a confirmation flow: after prompt processing, type
  `End Prompt` to allow changes.
- Supported feature areas include files/artboards, scene hierarchy edits,
  shapes/layout/components/assets, animations, state machines, View Models,
  data binding, Luau scripts, and WGSL shaders.

## RiveMCP

Source: <https://github.com/paradoxsyn/rivemcp-releases>

- Third-party standalone MCP server for programmatically creating/editing Rive
  animations.
- Recommended config:

```json
{
  "mcpServers": {
    "rivemcp": {
      "command": "npx",
      "args": ["-y", "rivemcp"]
    }
  }
}
```

- Exposes 139+ tools across project, shapes, styling, animation, state
  machines, text, images, bones, nested artboards, events, data binding,
  advanced layout/scroll/audio/scripting, HLAPI, sprites, import/export, edit.
- Can export runtime `.riv` and editor `.rev`.
- 3 free exports per machine; license required after that via
  `RIVEMCP_LICENSE_KEY`.

## Non-paths

- Official Rive runtime export is an editor action, not an official CLI export
  command.
- Do not claim headless official Rive support; use RiveMCP if the user needs
  headless generation.
